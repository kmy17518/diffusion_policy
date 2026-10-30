"""Small, native training entrypoint; legacy Hydra workspaces remain unchanged."""

import argparse
import contextlib
from contextlib import contextmanager
import copy
import fcntl
from functools import partial
import json
import os
from pathlib import Path
import random
import time
import uuid

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.data import DataLoader, Sampler

from diffusion_policy.b1k.dataset import (NORMALIZATIONS, B1KLeRobotDataset, expand_task_groups, images_to_float,
                                          parse_settle_steps, resolve_episode_split, resolve_val_split)
from diffusion_policy.b1k.model import GOAL_STEM_INITS, IMAGE_AUGMENTS, POLICY_TARGETS, ModelConfig, build_policy, load_checkpoint
from diffusion_policy.b1k.robot import (ACTION_REPRESENTATIONS, CAMERAS, GRIPPER_STATES, RESIZE_MODES, absolute_actions,
                                        goal_key, proprio_dim, task_family)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.model.diffusion.ema_model import EMAModel


class StepBatchSampler(Sampler):
    """O(batch_size) frame-uniform sampling, exactly addressable across resume.

    `batch_size` is the global optimizer batch: every step draws it from SeedSequence([seed, step]) and, under DDP
    (`world_size` > 1), rank `rank` yields its contiguous slice of `batch_size / world_size` samples, in chunks of
    `loader_batch_size` (one loader task each).
    """

    def __init__(self, length, batch_size, start_step, max_steps, seed, loader_batch_size=None, rank=0, world_size=1):
        if world_size < 1 or not 0 <= rank < world_size or batch_size % world_size:
            raise ValueError(f'Invalid rank {rank} / world_size {world_size} for batch size {batch_size}')
        self.length, self.batch_size = length, batch_size
        self.start_step, self.max_steps, self.seed = start_step, max_steps, seed
        self.rank, self.world_size = rank, world_size
        self.rank_batch_size = batch_size // world_size
        self.loader_batch_size = loader_batch_size or self.rank_batch_size

    def __iter__(self):
        lo = self.rank * self.rank_batch_size
        for step in range(self.start_step, self.max_steps):
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, step]))
            indices = rng.integers(self.length, size=self.batch_size).tolist()[lo:lo + self.rank_batch_size]
            for start in range(0, self.rank_batch_size, self.loader_batch_size):
                yield indices[start:start + self.loader_batch_size]

    def __len__(self):
        chunks = (self.rank_batch_size + self.loader_batch_size - 1) // self.loader_batch_size
        return max(0, self.max_steps - self.start_step) * chunks


class LossModule(torch.nn.Module):
    """`policy.compute_loss` as a forward, so DistributedDataParallel can hook the policy's gradients."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, batch):
        return self.policy.compute_loss(batch)


def distributed_env():
    """(rank, local_rank, world_size) from torchrun's environment (0, 0, 1 outside it)."""
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError(f'Invalid RANK={rank} / WORLD_SIZE={world_size}')
    return rank, local_rank, world_size


def stop_requested(stop_file, device, distributed):
    """Whether `stop_file` exists on any rank; under DDP every rank gets the same answer (one all-reduce)."""
    if stop_file is None:
        return False
    flag = torch.tensor(float(Path(stop_file).exists()), device=device)
    if distributed:
        import torch.distributed as dist
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


def concatenate_batches(batches):
    if isinstance(batches[0], dict):
        return {key: concatenate_batches([batch[key] for batch in batches]) for key in batches[0]}
    if batches[0].is_pinned():
        shape = (sum(len(batch) for batch in batches), *batches[0].shape[1:])
        output = torch.empty(shape, dtype=batches[0].dtype, pin_memory=True)
        return torch.cat(batches, dim=0, out=output)
    return torch.cat(batches, dim=0)


def training_batches(loader, batch_size):
    """Group loader chunks into batches of exactly `batch_size` samples (an optimizer batch, or a
    micro-batch under gradient accumulation)."""
    chunks, count = [], 0
    for batch in loader:
        chunks.append(batch)
        count += len(batch['action'])
        if count > batch_size:
            raise ValueError('Loader chunks crossed a batch boundary')
        if count == batch_size:
            combined = chunks[0] if len(chunks) == 1 else concatenate_batches(chunks)
            chunks, count = [], 0
            yield combined
            del combined
    if chunks:
        raise ValueError('Incomplete batch from loader')


def configure_cpu_threads(threads):
    import cv2
    import pyarrow as pa
    from threadpoolctl import threadpool_limits

    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = str(threads)
    torch.set_num_threads(threads)
    cv2.setNumThreads(threads)
    pa.set_cpu_count(threads)
    pa.set_io_thread_count(threads)
    threadpool_limits(limits=threads)


def seed_worker(worker_id, cpu_threads=1):
    seed = torch.initial_seed() % 2 ** 32
    random.seed(seed)
    np.random.seed(seed)
    configure_cpu_threads(cpu_threads)


@contextmanager
def output_lock(output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'run.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f'Another trainer holds {output / "run.lock"}') from error
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_save(checkpoint, path):
    if path.exists():
        raise FileExistsError(f'Will not overwrite checkpoint {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp')
    try:
        with temporary.open('wb') as stream:
            torch.save(checkpoint, stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def publish_full_checkpoint(output, path):
    queue = output / 'export_queue' / 'full'
    queue.mkdir(parents=True, exist_ok=True)
    destination = queue / path.name
    os.link(path, destination)
    sync_directory(queue)
    # Uploaders hardlink before reading; old staged inodes survive queue pruning.
    for previous in queue.glob('step-*.pt'):
        if previous != destination:
            previous.unlink(missing_ok=True)
    sync_directory(queue)
    sync_directory(queue.parent)


def prune_checkpoints(output, limit):
    if limit:
        paths = sorted(output.glob('step-*.pt'))
        for path in paths[:-limit]:
            path.unlink()
        sync_directory(output)


def build_optimizer(policy, args):
    common = {'learning_rate': args.learning_rate, 'betas': tuple(args.betas)}
    if args.optimizer == 'upstream':
        if args.variant == 'transformer_hybrid_image':
            optimizer = policy.get_optimizer(transformer_weight_decay=args.weight_decay,
                                             obs_encoder_weight_decay=args.obs_encoder_weight_decay, **common)
        elif args.variant == 'transformer_lowdim':
            optimizer = policy.get_optimizer(weight_decay=args.weight_decay, **common)
        else:
            raise ValueError('--optimizer upstream requires a transformer variant')
    else:
        optimizer = torch.optim.AdamW((p for p in policy.parameters() if p.requires_grad),
                                      lr=args.learning_rate, weight_decay=args.weight_decay, betas=tuple(args.betas))
    return optimizer


class EMAUpdater:
    """EMAModel.step with the same arithmetic issued as a few multi-tensor kernels.

    Upstream loops over every parameter with `mul_` and `add_(alpha=)` (850 launches for this
    model); `_foreach_mul_` / `_foreach_add_` perform exactly those two elementwise operations
    per tensor. BatchNorm parameters and frozen parameters are copied, as upstream does. (AdamW
    itself already uses PyTorch's multi-tensor implementation on CUDA by default; its fused
    kernel measured slower for these 425 tensors.)
    """

    def __init__(self, ema, policy):
        self.ema = ema
        self.averaged, self.copied = [], []
        for module, ema_module in zip(policy.modules(), ema.averaged_model.modules()):
            for param, ema_param in zip(module.parameters(recurse=False), ema_module.parameters(recurse=False)):
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) or not param.requires_grad:
                    self.copied.append((param, ema_param))
                else:
                    self.averaged.append((param, ema_param))

    @torch.no_grad()
    def step(self):
        ema = self.ema
        ema.decay = ema.get_decay(ema.optimization_step)
        if self.averaged:
            targets = [ema_param for _, ema_param in self.averaged]
            torch._foreach_mul_(targets, ema.decay)
            torch._foreach_add_(targets, [param.data for param, _ in self.averaged], alpha=1 - ema.decay)
        for param, ema_param in self.copied:
            ema_param.copy_(param.data)
        ema.optimization_step += 1


def compile_policy(policy, mode):
    """Compile the two hot modules in place without changing the module tree or state_dict keys."""
    for module in (policy.obs_encoder, policy.model) if hasattr(policy, 'obs_encoder') else (policy.model,):
        module.forward = torch.compile(module.forward, mode=mode)


def wandb_metadata(args):
    return {key: getattr(args, f'wandb_{key}') for key in ('project', 'entity', 'name', 'id', 'mode')}


def initialize_wandb(args, output, checkpoint, config):
    saved = (checkpoint or {}).get('wandb') or {}
    metadata_path = output / 'wandb.json'
    if metadata_path.exists():
        local = json.loads(metadata_path.read_text())
        if saved.get('id') and local.get('id') != saved['id']:
            raise ValueError('Checkpoint W&B run ID does not match output directory')
        saved = saved or local
    for key in ('id', 'project', 'entity', 'name'):
        requested = getattr(args, f'wandb_{key}')
        if saved.get(key) is not None:
            if key in ('id', 'project', 'entity') and requested is not None and requested != saved[key]:
                raise ValueError(f'--wandb-{key} conflicts with the resumed W&B run')
            if requested is None:
                setattr(args, f'wandb_{key}', saved[key])
    if args.wandb_mode == 'disabled':
        return None
    if args.wandb_mode == 'online' and not os.environ.get('WANDB_API_KEY'):
        raise RuntimeError('W&B online requires WANDB_API_KEY in the environment; refusing offline fallback')
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError('W&B logging requested but wandb is not installed; install requirements-b1k.txt') from error
    args.wandb_id = args.wandb_id or uuid.uuid4().hex[:12]
    args.wandb_project = args.wandb_project or 'diffusion_policy_b1k'
    args.wandb_name = args.wandb_name or output.name
    temporary = metadata_path.with_suffix('.tmp')
    with temporary.open('w') as stream:
        json.dump(wandb_metadata(args), stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(metadata_path)
    sync_directory(output)
    run = None
    try:
        run = wandb.init(**wandb_metadata(args), dir=str(output), resume='allow',
                         config={'model': config.to_dict(), 'training': {
                             key: value for key, value in vars(args).items() if not key.startswith('wandb_')}},
                         settings=wandb.Settings(init_timeout=60))
        if run is None or (args.wandb_mode == 'online' and run.settings.mode != 'online'):
            raise RuntimeError('W&B did not establish the requested online run')
        run.define_metric('step')
        run.define_metric('*', step_metric='step')
        return run
    except Exception as error:
        if run is not None:
            run.finish(exit_code=1)
        raise RuntimeError(f'W&B {args.wandb_mode} initialization/authentication failed; '
                           'check WANDB_API_KEY, project/entity access and connectivity') from error


@torch.no_grad()
def sync_batchnorm_buffers(policy, averaged_policy):
    for module, averaged_module in zip(policy.modules(), averaged_policy.modules()):
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            for name, buffer in module.named_buffers(recurse=False):
                getattr(averaged_module, name).copy_(buffer)


def source_commit():
    """Git commit of this checkout (None outside a repository); recorded with every run for provenance."""
    import subprocess
    try:
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parent,
                                capture_output=True, text=True, check=False, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def conditioning_record(config, task_map, dataset_root=None, episodes=None, episode_split=None, settle_steps=None):
    """Resolved conditioning configuration saved with config.json and every checkpoint (plan section 9)."""
    goal = config.goal_fusion != 'none'
    return {
        'regime': config.regime, 'source_commit': source_commit(), 'dataset_root': str(dataset_root) if dataset_root else None,
        'task_map': task_map, 'episodes': episodes, 'data_split': episode_split['subset'] if episode_split else 'all',
        'settle_steps': settle_steps['spec'] if settle_steps else 'all',
        'language': {'implementation': config.language_conditioning,
                     'prompt_source': config.prompt_source if config.language_conditioning != 'none' else None,
                     'encoder': 'openai/clip-vit-large-patch14' if config.language_conditioning == 'clip_film' else None,
                     'encoder_trainability': 'frozen_text_encoder_cached' if config.language_conditioning != 'none' else None,
                     'on_goal_encoder': config.language_on_goal_encoder if goal and config.goal_fusion == 'late' else None},
        'goal': {'fusion': config.goal_fusion, 'views': list(config.goal_views) if goal else [],
                 'encoder': config.goal_encoder if goal else None, 'role_embedding': 'condition_position' if goal and config.goal_fusion == 'late' else None,
                 'source': config.goal_source if goal else None,
                 'sampling_policy': 'fixed_terminal_frame_per_episode' if goal else None,
                 'hindsight_probability': 0.0, 'dropout_probability': 0.0},
        'task_onehot': config.task_onehot, 'denoiser': config.variant,
        'pretrained_encoder': config.encoder_weights, 'auxiliary_pose_weight': 0.0, 'guidance_scale': 1.0,
    }


def selection_record(dataset, args):
    """Training data selection saved with full and evaluation checkpoints; resume rebuilds and compares it."""
    record = {'task_names': list(dataset.task_map.values()), 'max_episodes': args.max_episodes,
              'episode_indices': [row['episode_index'] for row in dataset.episodes], 'episode_split': args.episode_split,
              'settle_steps': dataset.settle}
    if getattr(args, 'val_split', None):  # the held-out episodes of an ACT-comparable run (absent otherwise)
        record['val_split'] = args.val_split
    if getattr(dataset, 'frame_cache_formats', ()):
        record['frame_cache_formats'] = sorted(dataset.frame_cache_formats)
    return record


def save_checkpoint(output, policy, ema, optimizer, lr_scheduler, config, dataset, step, args):
    checkpoint = {
        'format': 'diffusion_policy_b1k_v1', 'checkpoint_type': 'full',
        'config': config.to_dict(), 'task_map': dataset.task_map,
        'conditioning': conditioning_record(config, dataset.task_map, dataset.root, len(dataset.episodes), args.episode_split,
                                            dataset.settle),
        'normalizer': policy.normalizer.state_dict(),
        'model': policy.state_dict(), 'ema_model': ema.averaged_model.state_dict(),
        'ema_step': ema.optimization_step, 'optimizer': optimizer.state_dict(), 'step': step,
        'wandb': wandb_metadata(args),
        'training': {'seed': args.seed, 'batch_size': args.batch_size,
                     'learning_rate': args.learning_rate, 'weight_decay': args.weight_decay,
                     'optimizer': args.optimizer, 'betas': tuple(args.betas),
                     'obs_encoder_weight_decay': args.obs_encoder_weight_decay,
                     'lr_scheduler': args.lr_scheduler, 'lr_warmup_steps': args.lr_warmup_steps,
                     'lr_schedule_steps': args.lr_schedule_steps, 'ema_power': args.ema_power,
                     'grad_clip': args.grad_clip, 'grad_accumulation': args.grad_accumulation},
        'lr_scheduler': lr_scheduler.state_dict(),
        'selection': selection_record(dataset, args),
        'dataset_fingerprint': dataset.fingerprint(),
        'rng': {'torch': torch.get_rng_state(),
                'numpy': (np.random.get_state()[0], torch.from_numpy(np.random.get_state()[1].astype(np.int64)),
                          *np.random.get_state()[2:]),
                'python': random.getstate(),
                'cuda': torch.cuda.get_rng_state_all() if policy.device.type == 'cuda' else None},
    }
    if config.language_conditioning == 'clip_film':
        from diffusion_policy.b1k.language import validate_language_cache
        checkpoint['language'] = validate_language_cache(dataset.language, dataset.task_map, config.prompt_source)
    path = output / f'step-{step:08d}.pt'
    atomic_save(checkpoint, path)
    publish_full_checkpoint(output, path)
    latest = output / '.latest.tmp'
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(path.name)
    latest.replace(output / 'latest.pt')
    sync_directory(output)
    prune_checkpoints(output, args.save_total_limit)
    return path


def export_checkpoint(output, ema, config, task_map, step, language=None, selection=None):
    checkpoint = {
        'format': 'diffusion_policy_b1k_v1', 'checkpoint_type': 'eval',
        'config': config.to_dict(), 'task_map': task_map,
        'conditioning': conditioning_record(config, task_map, episode_split=(selection or {}).get('episode_split'),
                                            settle_steps=(selection or {}).get('settle_steps')),
        'selection': selection, 'normalizer': ema.averaged_model.normalizer.state_dict(),
        'ema_model': ema.averaged_model.state_dict(), 'step': step,
    }
    if config.language_conditioning == 'clip_film':
        from diffusion_policy.b1k.language import validate_language_cache
        checkpoint['language'] = validate_language_cache(language, task_map, config.prompt_source)
    path = output / 'export_queue' / 'eval' / f'step-{step:08d}.pt'
    atomic_save(checkpoint, path)
    sync_directory(path.parent.parent)
    sync_directory(output)
    return path


def gpu_memory_metrics(device):
    if device.type != 'cuda':
        return {key: 0 for key in ('gpu_allocated_bytes', 'gpu_reserved_bytes', 'gpu_peak_allocated_bytes')}
    return {'gpu_allocated_bytes': torch.cuda.memory_allocated(device),
            'gpu_reserved_bytes': torch.cuda.memory_reserved(device),
            'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated(device)}


def validation_batches(length, val_batches, val_batch_size, seed):
    """The index lists of the validation batches: fixed draws (seed, batch) over the validation sequences, the same at
    every validation of a run (and across runs with the same split and seed)."""
    batches = []
    for index in range(val_batches):
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), 1_000_003, index]))
        batches.append(rng.integers(length, size=val_batch_size).tolist())
    return batches


@torch.no_grad()
def evaluate_validation(policy, val_dataset, config, args, device, autocast, attention, rank=0, world_size=1):
    """Held-out metrics of the EMA policy on the validation split, logged per `--val-every` steps:

    - val/loss: the denoising MSE of fixed-seed noise and timesteps on the validation batches;
    - val/l1: L1 between the denoised full-horizon trajectory (`predict_action`, matched initial noise) and the recorded
      actions in normalized units, over the steps inside the episode (edge-padded steps left out);
    - val/l1_action_units[/<family>]: the same L1 after decoding both to absolute actions in action units (relative
      actions added to the current state's reference), overall and per task family -- ACT's `val/l1_action_units`;
    - val/goal_l1_own, val/goal_l1_other, val/goal_gap, val/goal_max_dpred (goal-conditioned policies, the first
      `--val-goal-batches` batches): the normalized L1 with the sample's own goal image and with the goal of another
      validation episode of the same task, gap = other - own, and the largest prediction change (normalized units);
    - timing/val_s.
    Under DDP each rank evaluates the batches with index % world_size == rank and the sums are all-reduced. The RNG
    states and the policy's mode are restored afterwards.
    """
    import torch.distributed as dist
    started = time.monotonic()
    cuda = device.type == 'cuda'
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone() if cuda else None
    was_training = policy.training
    goal_pairs = {goal_key(view): view for view in config.goal_views} if config.goal_fusion != 'none' else {}
    family_of = {int(task): task_family(name) for task, name in val_dataset.task_map.items()}
    families = sorted(set(family_of.values()))
    by_task = {}
    for position, episode in enumerate(val_dataset.episodes):
        by_task.setdefault(int(episode['task_index']), []).append(position)
    batches = validation_batches(len(val_dataset), args.val_batches, args.val_batch_size, args.seed)
    mine = [(index, indices) for index, indices in enumerate(batches) if index % world_size == rank]
    loader = DataLoader(val_dataset, batch_sampler=[indices for _, indices in mine], num_workers=0, pin_memory=cuda)
    # sums: loss, loss count, l1 sum, l1 count, units sum, units count, per family (units sum, count), goal own, goal
    # other, goal count, goal max dpred
    sums = torch.zeros(10 + 2 * len(families), dtype=torch.float64, device=device)
    normalizer = policy.normalizer['action']
    proprio = proprio_dim(config.gripper_state)
    To = config.n_obs_steps
    try:
        policy.eval()
        for (batch_index, _), batch in zip(mine, loader):
            meta = batch.pop('meta')
            valid = batch.pop('action_valid').to(device)
            batch = dict_apply(batch, lambda value: value.to(device, non_blocking=True))
            batch['obs'] = images_to_float(batch['obs'])
            target = batch['action']
            torch.manual_seed(args.seed + batch_index)
            if cuda:
                torch.cuda.manual_seed(args.seed + batch_index)
            with attention(), autocast:
                loss = policy.compute_loss({'obs': dict(batch['obs']), 'action': target})
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite validation loss in batch {batch_index}')
            sums[0] += float(loss)
            sums[1] += 1
            torch.manual_seed(args.seed)
            if cuda:
                torch.cuda.manual_seed(args.seed)
            with attention(), autocast:
                prediction = policy.predict_action(dict(batch['obs']))['action_pred'].float()
            mask = valid.unsqueeze(-1).float()
            n_pred = normalizer.normalize(prediction)
            n_target = normalizer.normalize(target.float())
            sums[2] += float(((n_pred - n_target).abs() * mask).sum())
            sums[3] += float(mask.sum()) * target.shape[-1]
            # action units: both decoded to absolute actions (relative ones with the current state's reference)
            state = batch['obs']['state'][:, To - 1, :proprio].float().cpu().numpy().astype(np.float64)
            pred_np, target_np = prediction.cpu().numpy().astype(np.float64), target.float().cpu().numpy().astype(np.float64)
            if config.action_representation == 'relative':
                from diffusion_policy.b1k.robot import action_delta
                delta = action_delta(config.gripper_action)
                pred_np, target_np = absolute_actions(pred_np, state, delta), absolute_actions(target_np, state, delta)
            error = np.abs(pred_np - target_np) * valid.cpu().numpy()[..., None]
            per_sample = error.sum((1, 2))
            counts = valid.cpu().numpy().sum(1) * target.shape[-1]
            sums[4] += float(per_sample.sum())
            sums[5] += float(counts.sum())
            tasks = meta['task'].tolist()
            for sample, task in enumerate(tasks):
                slot = 10 + 2 * families.index(family_of[int(task)])
                sums[slot] += float(per_sample[sample])
                sums[slot + 1] += float(counts[sample])
            if goal_pairs and batch_index < args.val_goal_batches:
                rng = np.random.default_rng(np.random.SeedSequence([int(args.seed), 7, batch_index]))
                episodes = meta['episode'].tolist()
                swapped = dict(batch['obs'])
                for key, view in goal_pairs.items():
                    column = list(val_dataset.goal_views).index(view)
                    others = []
                    for task, position in zip(tasks, episodes):
                        candidates = [p for p in by_task[int(task)] if p != position] or [position]
                        others.append(val_dataset.goal_table[int(rng.choice(candidates)), column])
                    goals = torch.from_numpy(np.ascontiguousarray(np.moveaxis(np.stack(others), -1, 1))).to(device)
                    swapped[key] = images_to_float(goals)
                torch.manual_seed(args.seed)
                if cuda:
                    torch.cuda.manual_seed(args.seed)
                with attention(), autocast:
                    other = policy.predict_action(swapped)['action_pred'].float()
                n_other = normalizer.normalize(other)
                sums[6] += float(((n_pred - n_target).abs() * mask).sum())
                sums[7] += float(((n_other - n_target).abs() * mask).sum())
                sums[8] += float(mask.sum()) * target.shape[-1]
                sums[9] = max(float(sums[9]), float(((n_pred - n_other).abs() * mask).max()))
    finally:
        policy.train(was_training)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng, device)
    if world_size > 1:
        peak = sums[9].clone()
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)
        dist.all_reduce(peak, op=dist.ReduceOp.MAX)
        sums[9] = peak
    if cuda:
        torch.cuda.synchronize(device)
    s = sums.tolist()
    metrics = {'val/loss': s[0] / max(1, s[1]), 'val/l1': s[2] / max(1, s[3]), 'val/l1_action_units': s[4] / max(1, s[5])}
    if len(families) > 1:
        for index, family in enumerate(families):
            metrics[f'val/l1_action_units/{family}'] = s[10 + 2 * index] / max(1, s[11 + 2 * index])
    if goal_pairs and s[8] > 0:
        metrics.update({'val/goal_l1_own': s[6] / s[8], 'val/goal_l1_other': s[7] / s[8],
                        'val/goal_gap': (s[7] - s[6]) / s[8], 'val/goal_max_dpred': s[9]})
    metrics['timing/val_s'] = time.monotonic() - started
    return metrics


class ExplicitLanguageChoice(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        setattr(namespace, f'{self.dest}_explicit', True)


ExplicitChoice = ExplicitLanguageChoice


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--dataset-path', '--dataset-root', dest='dataset_path', required=True)
    result.add_argument('--task-names', nargs='+',
                        help='Tasks or task groups of <dataset>/isg_meta/task_groups.json (default: every local task; '
                             'with an episode split, every task of the split)')
    result.add_argument('--episode-split', default='auto', action=ExplicitChoice,
                        help='auto: <dataset>/isg_meta/train_split.json if present, else every episode of the selected '
                             'tasks; none; or an isg-episode-split/v1 file. Trains on the split\'s train episodes of the '
                             'selected tasks (default: all tasks of the split). Resume reuses the checkpoint\'s split; '
                             'an explicit value must select the same episodes')
    result.add_argument('--settle-steps', default='all', action=ExplicitChoice,
                        help='Frames kept after each episode\'s program (the recorded settle window, where the robot '
                             'holds still), cut at load time using <dataset>/isg_meta/settle_windows.json: all (default: '
                             'every recorded frame), a number of frames (0 keeps the program only) or a decimal fraction '
                             'of the program length (0.2), capped at the recorded window. Goal images stay the last '
                             'recorded frame. Resume reuses the checkpoint\'s value; an explicit value must equal it')
    result.add_argument('--output-dir', required=True)
    result.add_argument('--max-steps', type=int, default=100000)
    result.add_argument('--batch-size', type=int, default=64)
    result.add_argument('--loader-batch-size', type=int, help='Decode chunks per worker; optimizer batch size is unchanged')
    result.add_argument('--num-workers', type=int, default=4)
    result.add_argument('--prefetch-factor', type=int, default=1)
    result.add_argument('--worker-cpu-threads', type=int, default=1)
    result.add_argument('--device', default='cuda')
    result.add_argument('--variant', choices=list(POLICY_TARGETS), default='unet_image')
    result.add_argument('--conditioning', choices=['global', 'local', 'inpainting'], default='global')
    result.add_argument('--language-conditioning', choices=['none', 'clip_film'], default='none',
                        action=ExplicitLanguageChoice)
    result.add_argument('--prompt-source', choices=['task_name', 'task_description'], default='task_name',
                        action=ExplicitLanguageChoice)
    result.add_argument('--pred-action-steps-only', action='store_true')
    result.add_argument('--prediction-type', choices=['epsilon', 'sample'], default='epsilon')
    result.add_argument('--kernel-size', type=int, default=5)
    result.add_argument('--n-groups', type=int, default=8)
    result.add_argument('--cond-predict-scale', action=argparse.BooleanOptionalAction, default=True)
    result.add_argument('--n-layer', type=int, default=8)
    result.add_argument('--n-head', type=int, default=4)
    result.add_argument('--n-emb', type=int, default=256)
    result.add_argument('--n-cond-layers', type=int, default=0)
    result.add_argument('--p-drop-emb', type=float, default=0.0)
    result.add_argument('--p-drop-attn', type=float, default=0.3)
    result.add_argument('--causal-attn', action=argparse.BooleanOptionalAction, default=True)
    result.add_argument('--time-as-cond', action=argparse.BooleanOptionalAction, default=True)
    result.add_argument('--crop-shape', type=int, nargs=2)
    result.add_argument('--resize-shape', type=int, nargs=2)
    result.add_argument('--random-crop', action=argparse.BooleanOptionalAction, default=True)
    result.add_argument('--obs-encoder-group-norm', action=argparse.BooleanOptionalAction, default=True)
    result.add_argument('--eval-fixed-crop', action=argparse.BooleanOptionalAction, default=True)
    result.add_argument('--share-rgb-model', action=argparse.BooleanOptionalAction, default=True)
    result.add_argument('--imagenet-norm', action='store_true')
    result.add_argument('--encoder-weights', choices=['IMAGENET1K_V1'])
    result.add_argument('--freeze-encoder', action='store_true')
    result.add_argument('--gripper-state', choices=GRIPPER_STATES, default='sum', action=ExplicitChoice,
                        help='Gripper proprioception: sum (default) adds the two finger positions of each gripper into '
                             'one opening, giving a 23-value state in the action\'s layout; fingers keeps both (25 '
                             'values). Resume uses the checkpoint\'s layout (fingers for checkpoints without one)')
    result.add_argument('--resume', type=Path)
    result.add_argument('--horizon', type=int, default=16)
    result.add_argument('--n-obs-steps', type=int, default=2)
    result.add_argument('--n-action-steps', type=int, default=8)
    result.add_argument('--image-size', type=int, default=96)
    result.add_argument('--cameras', choices=list(CAMERAS), nargs='+', default=list(CAMERAS))
    result.add_argument('--down-dims', type=int, nargs='+', default=[256, 512, 1024])
    result.add_argument('--diffusion-step-embed-dim', type=int, default=256)
    result.add_argument('--num-train-timesteps', type=int, default=100)
    result.add_argument('--num-inference-steps', type=int, default=100)
    result.add_argument('--scheduler', choices=['ddpm', 'ddim'], default='ddpm')
    result.add_argument('--no-clip-sample', action='store_true')
    result.add_argument('--frame-selection', choices=('pi', 'legacy'), default='pi', action=ExplicitChoice)
    result.add_argument('--gripper-action', choices=('absolute', 'delta'), default='absolute', action=ExplicitChoice)
    result.add_argument('--stats-estimator', choices=('auto', 'exact', 'sampled', 'legacy'), default='exact', action=ExplicitChoice)
    result.add_argument('--stats-sample-size', type=int, default=10_000_000, action=ExplicitChoice)
    result.add_argument('--stats-file', help='Portable precomputed JSON statistics (validated against selection and recipe)')
    result.add_argument('--learning-rate', type=float, default=1e-4)
    result.add_argument('--weight-decay', type=float, default=1e-6)
    result.add_argument('--optimizer', choices=['adamw', 'upstream'], default='adamw')
    result.add_argument('--obs-encoder-weight-decay', type=float, default=1e-6)
    result.add_argument('--betas', type=float, nargs=2, default=[0.9, 0.999])
    result.add_argument('--lr-scheduler', choices=['constant', 'cosine'], default='constant',
                        help="upstream Diffusion Policy schedule: 'cosine' = linear warmup then cosine decay to 0 "
                             'over --lr-schedule-steps optimizer steps (stepped after every optimizer step)')
    result.add_argument('--lr-warmup-steps', type=int, default=0)
    result.add_argument('--lr-schedule-steps', type=int,
                        help='Length of the cosine schedule in optimizer steps (default: --max-steps). Fixed at '
                             'the first launch and restored on resume, so a later --max-steps change does not '
                             'reshape the schedule')
    result.add_argument('--ema-power', type=float, default=2 / 3,
                        help='EMAModel decay exponent: decay = 1 - (1 + step)^-power, clipped to 0.9999 '
                             '(upstream default 2/3; the RoboCasa recipe uses 0.75)')
    result.add_argument('--grad-clip', type=float, default=1.0,
                        help='Gradient-norm clipping threshold; 0 disables clipping (upstream trains unclipped). '
                             'Non-finite gradients are rejected either way')
    result.add_argument('--grad-accumulation', type=int, default=1,
                        help='Split each optimizer batch into this many equal micro-batches (forward/backward '
                             'each, one optimizer/EMA/schedule step per --batch-size samples). The samples per '
                             'step are the same seeded draw as without accumulation; only peak memory changes')
    result.add_argument('--task-onehot', action=argparse.BooleanOptionalAction, default=False,
                        help='Append the one-hot task id to the 25-D state (the v1 behavior). Off by default: '
                             'with several tasks the policy then needs --language-conditioning clip_film')
    result.add_argument('--save-every', type=int, default=5000)
    result.add_argument('--save-first-step', action='store_true')
    result.add_argument('--save-total-limit', type=int, default=0, help='Retained local full checkpoints; 0 keeps all')
    result.add_argument('--export-every', type=int, default=10000, help='EMA evaluation export interval; 0 disables')
    result.add_argument('--wandb-project')
    result.add_argument('--wandb-entity')
    result.add_argument('--wandb-name')
    result.add_argument('--wandb-id')
    result.add_argument('--wandb-mode', choices=['disabled', 'offline', 'online'], default='disabled')
    result.add_argument('--seed', type=int, default=42)
    result.add_argument('--cpu-threads', type=int, default=2)
    result.add_argument('--episode-cache-size', type=int, default=8)
    result.add_argument('--parquet-cache-mb', type=int, default=256)
    result.add_argument('--video-max-open', type=int, default=32,
                        help='Open decoders kept per worker; cover the files a task spans to avoid reopen costs')
    result.add_argument('--frame-cache', type=Path,
                        help='Pixel-exact resized-frame cache built by scripts/b1k/build_frame_cache.py')
    result.add_argument('--matmul-precision', choices=['highest', 'high', 'medium'], default='highest',
                        help='torch float32 matmul precision; high enables TF32 tensor cores for the '
                             'transformer matmuls (cuDNN convolutions already default to TF32)')
    result.add_argument('--autocast', choices=['none', 'bf16'], default='none',
                        help='bf16 autocast for the forward pass and loss; parameters, gradients, '
                             'optimizer and EMA stay FP32')
    result.add_argument('--compile', choices=['none', 'default', 'reduce-overhead', 'max-autotune-no-cudagraphs'],
                        default='none', help='torch.compile the observation encoder and denoiser')
    result.add_argument('--sdpa-backend', choices=['math', 'auto'], default='math',
                        help='Attention kernels for training: math (plain matmul/softmax, best for the short '
                             'action/observation sequences of these policies) or PyTorch auto-selection')
    result.add_argument('--multi-tensor-ema', action=argparse.BooleanOptionalAction, default=True,
                        help='EMA update through multi-tensor kernels (same arithmetic as EMAModel.step)')
    result.add_argument('--cudnn-benchmark', action=argparse.BooleanOptionalAction, default=True,
                        help='Let cuDNN time convolution algorithms once for the fixed batch shapes')
    result.add_argument('--film-init', choices=['random', 'identity'], default='random',
                        help='clip_film only, fresh runs only (resume keeps checkpoint weights): random keeps '
                             'nn.Linear initialization of the FiLM projections; identity zeroes them so beta = '
                             'gamma = 0 and every conditioned block starts as the identity')
    result.add_argument('--film-recompute', action=argparse.BooleanOptionalAction, default=True,
                        help='clip_film only: recompute the FiLM ResNet blocks during backward '
                             '(torch.utils.checkpoint) instead of storing their activations. Same forward '
                             'values and gradients; trades about a third of the language step-time overhead '
                             'for activation memory when disabled')
    # --- conditioning regime and goal-image conditioning (b1k.md "Goal-image conditioning") ---
    result.add_argument('--regime', choices=['none', 'language', 'image', 'image_language'], default=None,
                        action=ExplicitLanguageChoice,
                        help='Declared conditioning regime: none (N: observation only, no task id), language (L),'
                             ' image (I) or image_language (LI). Fills defaults (language: clip_film; image: late'
                             ' head-goal fusion) and rejects disagreeing flags; recorded in the checkpoint')
    result.add_argument('--goal-fusion', choices=['none', 'early', 'late'], default='none', action=ExplicitLanguageChoice,
                        help='Goal-image conditioning for the hybrid image policies: early (goal channel-stacked with'
                             ' its camera before the ResNet stem, zero-initialised goal stem) or late (goal encoded by'
                             ' the camera encoder into a 64-D feature: an extra condition token for the transformer,'
                             ' appended to the global condition for the U-Net)')
    result.add_argument('--goal-views', choices=list(CAMERAS), nargs='+', default=None, action=ExplicitLanguageChoice,
                        help='Cameras whose goal image is supplied (default with --goal-fusion: head)')
    result.add_argument('--goal-source', choices=['episode_last', 'goal_key'], default='episode_last',
                        action=ExplicitLanguageChoice,
                        help='episode_last: the last frame of the episode\'s camera; goal_key: the dataset\'s'
                             ' observation.goal_rgb.<camera> stream')
    result.add_argument('--goal-encoder', choices=['shared_base', 'separate_base'], default='shared_base',
                        action=ExplicitLanguageChoice,
                        help='Late fusion: encode the goal with the camera\'s own encoder (default) or a separate copy')
    result.add_argument('--language-on-goal-encoder', action=argparse.BooleanOptionalAction, default=False,
                        help='Late fusion with clip_film: also FiLM-modulate the goal pass (default: identity FiLM)')
    result.add_argument('--goal-stem-init', default='zero', action=ExplicitChoice, metavar='zero|random|copy:SCALE',
                        help='Early fusion: the goal half of the paired stem starts at zero (default), at a fresh '
                             'nn.Conv2d draw, or as the camera stem\'s weights times SCALE (ACT\'s copy:0.5)')
    result.add_argument('--max-episodes', type=int, help='Explicit small-data smoke/debug subset')
    # --- the ACT-comparable recipe (b1k.md "ACT-comparable recipe"): pixels, actions, augmentation, validation, DDP ---
    result.add_argument('--resize-mode', choices=RESIZE_MODES, default='dp', action=ExplicitChoice,
                        help='dp (default): aspect-preserving cv2 letterbox; act: ACT\'s whole-frame antialiased '
                             'bilinear resize -- required with an ACT-built --frame-cache (act_b1k_frame_cache_v1), '
                             'so served frames and goal frames match the cached pixels')
    result.add_argument('--action-representation', choices=ACTION_REPRESENTATIONS, default='absolute', action=ExplicitChoice,
                        help='absolute (default): the recorded commands; relative: joints and grippers relative to the '
                             'current state (ACT / openpi pi05 gripper deltas), decoded back at serving time')
    result.add_argument('--normalization', choices=NORMALIZATIONS, default='limits', action=ExplicitChoice,
                        help='limits (default): min/max to [-1, 1]; zscore; quantile: pi05\'s (x - q01) / (q99 - q01 + '
                             '1e-6) * 2 - 1 over the training episodes (actions: every valid chunk step)')
    result.add_argument('--image-augment', choices=IMAGE_AUGMENTS, default='none', action=ExplicitChoice,
                        help='pi05: openpi pi05_b1k\'s training augmentation on the GPU (crop 95%% + resize + rotate '
                             '+-5 deg + color jitter for the head and its goal, color jitter for the wrists; one draw per '
                             'sample shared over the observation steps); training batches only; needs no --crop-shape')
    result.add_argument('--val-fraction', type=float, default=None,
                        help='Hold out this fraction of each task\'s episodes for validation (ACT\'s per-task seeded '
                             'split, --seed; default 0.1 when --val-every > 0, else 0); 0 trains on everything')
    result.add_argument('--val-every', type=int, default=0,
                        help='Validate the EMA policy on the held-out episodes every N optimizer steps (ACT-comparable '
                             'runs: 10000; 0, the default, disables validation)')
    result.add_argument('--val-batches', type=int, default=16, help='Validation batches per validation')
    result.add_argument('--val-batch-size', type=int, default=64)
    result.add_argument('--val-goal-batches', type=int, default=2,
                        help='Goal-conditioned policies: validation batches that also run the goal-swap probe')
    result.add_argument('--stop-file', type=Path,
                        help='Stop cleanly when this file exists: a full checkpoint is saved and the trainer exits 0 '
                             '(every DDP rank agrees through an all-reduce)')
    return result


def resolve_regime(args):
    """Fill the regime's defaults and reject disagreeing flags (fresh runs; ModelConfig.validate re-checks)."""
    if args.goal_fusion != 'none' and args.goal_views is None:
        args.goal_views = ['head']
    if args.goal_fusion == 'none' and args.goal_views:
        raise ValueError('--goal-views require --goal-fusion early or late')
    args.goal_views = tuple(args.goal_views or ())
    if args.regime is None:
        return
    wants_language = args.regime in ('language', 'image_language')
    wants_goal = args.regime in ('image', 'image_language')
    if wants_language and not getattr(args, 'language_conditioning_explicit', False):
        args.language_conditioning = 'clip_film'
    if wants_goal and not getattr(args, 'goal_fusion_explicit', False):
        args.goal_fusion = 'late'
        args.goal_views = args.goal_views or ('head',)
    if wants_language != (args.language_conditioning != 'none'):
        raise ValueError(f'--regime {args.regime} {"requires" if wants_language else "excludes"} language conditioning')
    if wants_goal != (args.goal_fusion != 'none'):
        raise ValueError(f'--regime {args.regime} {"requires" if wants_goal else "excludes"} --goal-fusion early|late')
    if args.task_onehot:
        print(f'WARNING: --task-onehot with --regime {args.regime}: the task id enters the state (task-ID experiment)', flush=True)


def main(argv=None):
    args = parser().parse_args(argv)
    rank, local_rank, world_size = distributed_env()
    device = torch.device(args.device)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('--device must be cpu, cuda or an indexed CUDA device')
    if world_size > 1:
        if device.type == 'cuda':
            if device.index is not None:
                raise ValueError('Under torchrun give --device cuda; each rank takes its LOCAL_RANK device')
            device = torch.device(f'cuda:{local_rank}')
            torch.cuda.set_device(device)
        if args.batch_size % world_size:
            raise ValueError(f'--batch-size ({args.batch_size}) must be divisible by WORLD_SIZE ({world_size})')
        if (args.batch_size // world_size) % args.grad_accumulation:
            raise ValueError('--grad-accumulation must divide each rank\'s share of the batch')
    args.rank, args.local_rank, args.world_size = rank, local_rank, world_size
    if args.val_every < 0 or args.val_batches < 1 or args.val_batch_size < 1 or args.val_goal_batches < 0:
        raise ValueError('--val-every and --val-goal-batches must be nonnegative, --val-batches and --val-batch-size positive')
    if args.val_fraction is None:
        args.val_fraction = 0.1 if args.val_every > 0 else 0.0
    if not 0.0 <= args.val_fraction < 1.0:
        raise ValueError('--val-fraction must be in [0, 1)')
    if args.val_every > 0 and args.val_fraction == 0.0:
        raise ValueError('--val-every needs a validation split: --val-fraction > 0 (or --val-every 0)')
    if (min(args.max_steps, args.batch_size, args.save_every, args.cpu_threads,
            args.worker_cpu_threads, args.prefetch_factor, args.video_max_open) < 1 or
            min(args.num_workers, args.save_total_limit, args.export_every) < 0):
        raise ValueError('Steps, batch size, save interval, prefetch and CPU threads must be positive; '
                         'workers, retention and export interval nonnegative')
    if args.loader_batch_size is not None and args.loader_batch_size < 1:
        raise ValueError('--loader-batch-size must be positive')
    if args.grad_accumulation < 1 or args.batch_size % args.grad_accumulation:
        raise ValueError('--grad-accumulation must be positive and divide --batch-size')
    if (args.grad_accumulation > 1 and args.loader_batch_size is not None and
            (args.batch_size // args.grad_accumulation) % args.loader_batch_size):
        # Without accumulation a shorter final chunk is fine; micro-batches must be whole chunks.
        raise ValueError('--loader-batch-size must divide the micro-batch (--batch-size / --grad-accumulation)')
    if (args.lr_warmup_steps < 0 or args.grad_clip < 0 or not args.ema_power > 0 or
            (args.lr_schedule_steps is not None and args.lr_schedule_steps < 1)):
        raise ValueError('--lr-warmup-steps and --grad-clip must be nonnegative, --ema-power positive, '
                         '--lr-schedule-steps positive')
    if args.lr_scheduler == 'cosine' and args.lr_warmup_steps >= (args.lr_schedule_steps or args.max_steps):
        raise ValueError('--lr-warmup-steps must be smaller than the cosine schedule length')
    if args.learning_rate <= 0 or min(args.weight_decay, args.obs_encoder_weight_decay) < 0:
        raise ValueError('Learning rate must be positive and weight decay nonnegative')
    if not all(0 <= beta < 1 for beta in args.betas):
        raise ValueError('AdamW betas must be in [0, 1)')
    args.settle_steps = parse_settle_steps(args.settle_steps)
    output = Path(args.output_dir).resolve()
    if output.is_relative_to(Path(args.dataset_path).resolve()):
        raise ValueError('--output-dir must not be inside the read-only dataset')
    distributed = world_size > 1
    if distributed:
        import torch.distributed as dist
        dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo', rank=rank, world_size=world_size,
                                timeout=__import__('datetime').timedelta(hours=2))
    try:
        if rank == 0:
            with output_lock(output):
                if any(path.name != 'run.lock' for path in output.iterdir()) and args.resume is None:
                    raise FileExistsError(f'Output directory is nonempty: {output}; use --resume or a new directory')
                run_training(args, device, output)
        else:
            run_training(args, device, output)
    finally:
        if distributed:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.destroy_process_group()


def run_training(args, device, output):
    configure_cpu_threads(args.cpu_threads)
    checkpoint = load_checkpoint(args.resume) if args.resume else None
    args.requested_task_names = args.task_names
    args.task_names = expand_task_groups(args.dataset_path, args.task_names)
    if args.task_names != args.requested_task_names:
        print(json.dumps({'task_groups': {'requested': args.requested_task_names, 'tasks': args.task_names}}),
              flush=True)
    if checkpoint:
        if checkpoint.get('checkpoint_type') == 'eval' or 'optimizer' not in checkpoint:
            raise ValueError('Cannot resume training from an eval-only checkpoint; use a full checkpoint')
        config = ModelConfig(**checkpoint['config'])
        for key in ('language_conditioning', 'prompt_source'):
            if getattr(args, f'{key}_explicit', False) and getattr(args, key) != getattr(config, key):
                raise ValueError(f'--{key.replace("_", "-")} conflicts with the resumed checkpoint')
        if getattr(args, 'gripper_state_explicit', False) and args.gripper_state != config.gripper_state:
            raise ValueError(f'--gripper-state {args.gripper_state} conflicts with the resumed checkpoint '
                             f'({config.gripper_state})')
        args.gripper_state = config.gripper_state
        saved_tasks = checkpoint['selection']['task_names']
        if args.task_names is not None and set(args.task_names) != set(saved_tasks):
            raise ValueError(f'--task-names {args.requested_task_names} select {args.task_names}, but the checkpoint '
                             f'trained on {saved_tasks}')
        if args.task_names is None:
            args.task_names = saved_tasks
        if args.max_episodes is None:
            args.max_episodes = checkpoint['selection']['max_episodes']
        split = checkpoint['selection'].get('episode_split')
        if getattr(args, 'episode_split_explicit', False):
            requested = resolve_episode_split(args.dataset_path, args.episode_split, args.task_names)
            if (requested or {}).get('episodes') != (split or {}).get('episodes'):
                raise ValueError('--episode-split selects other episodes than the resumed checkpoint')
        saved_settle = (checkpoint['selection'].get('settle_steps') or {}).get('spec', 'all')
        if getattr(args, 'settle_steps_explicit', False) and args.settle_steps != saved_settle:
            raise ValueError(f'--settle-steps {args.settle_steps} conflicts with the resumed checkpoint ({saved_settle})')
        args.settle_steps = saved_settle
        for key in ('resize_mode', 'action_representation', 'normalization', 'image_augment', 'goal_stem_init', 'frame_selection',
                'gripper_action', 'stats_estimator', 'stats_sample_size'):
            if getattr(args, f'{key}_explicit', False) and getattr(args, key) != getattr(config, key):
                raise ValueError(f'--{key.replace("_", "-")} {getattr(args, key)} conflicts with the resumed checkpoint '
                                 f'({getattr(config, key)})')
        val_split = checkpoint['selection'].get('val_split')
        if val_split is not None and args.val_fraction and (
                abs(val_split['val_fraction'] - args.val_fraction) > 1e-9 or val_split['seed'] != args.seed):
            raise ValueError('--val-fraction / --seed differ from the resumed checkpoint\'s validation split')
        for key, value in {'optimizer': 'adamw', 'betas': (0.9, 0.999), 'obs_encoder_weight_decay': 1e-6,
                           # Checkpoints predating these options trained with a constant learning rate,
                           # EMA power 2/3 and gradient clipping at 1.0.
                           'lr_scheduler': 'constant', 'lr_warmup_steps': 0, 'lr_schedule_steps': None,
                           'ema_power': 2 / 3, 'grad_clip': 1.0, 'grad_accumulation': 1,
                           **checkpoint['training']}.items():
            setattr(args, key, value)
    else:
        resolve_regime(args)
        split = resolve_episode_split(args.dataset_path, args.episode_split, args.task_names)
        if split:
            args.task_names = split['tasks']
        values = {key: getattr(args, key) for key in ModelConfig.__dataclass_fields__ if hasattr(args, key)}
        for key in ('cameras', 'down_dims', 'crop_shape', 'resize_shape', 'goal_views'):
            if values[key] is not None:
                values[key] = tuple(values[key])
        config = ModelConfig(**values, clip_sample=not (args.no_clip_sample or args.normalization == 'quantile'))
        # the per-task held-out episodes (ACT's split: the same episodes for every policy at this seed)
        val_split = resolve_val_split(args.dataset_path, args.task_names, split, args.val_fraction, args.seed)
    args.episode_split = split
    args.val_split = val_split
    rank, world_size = getattr(args, 'rank', 0), getattr(args, 'world_size', 1)
    if split and rank == 0:
        print(json.dumps({'episode_split': {'name': split['name'], 'file': split['file'],
                                            'episodes': len(split['episodes']), 'tasks': split['tasks']}}), flush=True)
    if val_split and rank == 0:
        print(json.dumps({'val_split': {'val_fraction': val_split['val_fraction'], 'seed': val_split['seed'],
                                        'train_episodes': len(val_split['train_episodes']),
                                        'val_episodes': len(val_split['val_episodes']), 'tasks': len(val_split['tasks'])}}),
              flush=True)
    config.validate()
    args.variant = config.variant
    args.language_conditioning = config.language_conditioning
    args.prompt_source = config.prompt_source
    for key in ('regime', 'goal_fusion', 'goal_source', 'goal_encoder', 'resize_mode', 'action_representation',
                'normalization', 'image_augment', 'goal_stem_init', 'frame_selection',
                'gripper_action', 'stats_estimator', 'stats_sample_size'):
        if checkpoint and getattr(args, f'{key}_explicit', False) and getattr(args, key) != getattr(config, key):
            raise ValueError(f'--{key.replace("_", "-")} conflicts with the resumed checkpoint')
        setattr(args, key, getattr(config, key))
    if checkpoint and getattr(args, 'goal_views_explicit', False) and tuple(args.goal_views) != tuple(config.goal_views):
        raise ValueError('--goal-views conflicts with the resumed checkpoint')
    args.goal_views = tuple(config.goal_views)
    if args.optimizer == 'upstream' and not config.variant.startswith('transformer_'):
        raise ValueError('--optimizer upstream requires a transformer variant')
    if checkpoint and checkpoint['step'] >= args.max_steps:
        raise ValueError(f'--max-steps is a total target and must exceed checkpoint step {checkpoint["step"]}')
    distributed = world_size > 1
    is_main = rank == 0
    wandb_run = initialize_wandb(args, output, checkpoint, config) if is_main else None
    dataset = val_dataset = None
    exit_code = 1
    try:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        train_episodes = val_split['train_episodes'] if val_split else (split['episodes'] if split else None)
        dataset_kwargs = dict(config.dataset_kwargs(), episode_cache_size=args.episode_cache_size,
                              parquet_cache_mb=args.parquet_cache_mb, image_dtype='uint8', frame_cache=args.frame_cache,
                              video_max_open=args.video_max_open, settle_steps=args.settle_steps)
        dataset = B1KLeRobotDataset(args.dataset_path, args.task_names, max_episodes=args.max_episodes,
                                    episodes=train_episodes, **dataset_kwargs)
        if checkpoint and (dataset.task_map != checkpoint['task_map'] or
                           [row['episode_index'] for row in dataset.episodes] != checkpoint['selection']['episode_indices'] or
                           checkpoint['dataset_fingerprint'] not in (dataset.fingerprint(), dataset.legacy_fingerprint())):
            raise ValueError('Resume dataset selection, metadata or file fingerprint does not match checkpoint')
        dataset.prepare_language(checkpoint.get('language') if checkpoint else None)
        if val_split and args.val_every > 0:
            val_dataset = B1KLeRobotDataset(args.dataset_path, args.task_names, episodes=val_split['val_episodes'],
                                            return_meta=True, **dataset_kwargs)
            if val_dataset.task_map != dataset.task_map:
                raise ValueError('The validation episodes do not cover the training tasks')
            if config.language_conditioning == 'clip_film':
                val_dataset.prepare_language(dataset.language)
        if is_main:
            print(json.dumps({'episodes': len(dataset.episodes), 'sequences': len(dataset),
                              'val_episodes': len(val_dataset.episodes) if val_dataset else 0,
                              'val_sequences': len(val_dataset) if val_dataset else 0,
                              'frame_cache_formats': sorted(dataset.frame_cache_formats),
                              'task_map': dataset.task_map, 'config': config.to_dict()}), flush=True)
        policy = build_policy(config, dataset.task_map, initialize_encoder=checkpoint is None)
        if checkpoint:
            policy.load_state_dict(checkpoint['model'])
        else:
            if is_main:
                print(f'Computing {config.normalization} normalization statistics of {config.action_representation} '
                      'actions, streaming parquet once per file (no video).', flush=True)
                from diffusion_policy.b1k.stats_v2 import load_or_compute
                stats = load_or_compute(dataset, config.normalization, args.stats_file, output)
                policy.set_normalizer(dataset.get_normalizer(config.normalization, stats=stats))
            if distributed:  # rank 0's statistics for every rank (the same bytes, computed once)
                import torch.distributed as dist
                payload = [policy.normalizer.state_dict() if is_main else None]
                dist.broadcast_object_list(payload, src=0)
                if not is_main:
                    policy.normalizer.load_state_dict(payload[0])
        if args.film_init == 'identity':
            if config.language_conditioning != 'clip_film':
                raise ValueError('--film-init identity requires --language-conditioning clip_film')
            if checkpoint is None:
                from diffusion_policy.model.vision.clip_film import identity_initialize_film
                print(f'Identity FiLM initialization: zeroed {identity_initialize_film(policy)} projections', flush=True)
        if not args.film_recompute:
            # Runtime choice, not model configuration: no parameters or state_dict keys depend on it, so a
            # checkpoint trained either way resumes either way.
            from diffusion_policy.model.vision.clip_film import ResNet18FiLM
            for module in policy.modules():
                if isinstance(module, ResNet18FiLM):
                    module.checkpoint_blocks = False
        policy.to(device).train()
        if config.freeze_encoder:
            policy.obs_encoder.eval().requires_grad_(False)
        policy.normalizer.requires_grad_(False)
        for name, parameter in policy.named_parameters():
            if name.rpartition('.')[2] == '_dummy_variable' and parameter.numel() == 0:
                # ModuleAttrMixin's device/dtype probes: empty, never trained; DDP would wait for their gradients
                parameter.requires_grad_(False)
        if distributed:  # rank 0's initial weights for every rank (a fresh start draws them per process)
            import torch.distributed as dist
            for tensor in policy.state_dict().values():
                if torch.is_tensor(tensor) and tensor.numel():
                    dist.broadcast(tensor, src=0)
        ema = EMAModel(copy.deepcopy(policy), power=args.ema_power)
        ema_updater = EMAUpdater(ema, policy) if args.multi_tensor_ema else None
        optimizer = build_optimizer(policy, args)
        if args.lr_schedule_steps is None:
            args.lr_schedule_steps = args.max_steps
        # Upstream Diffusion Policy's schedule (diffusers' get_scheduler copy): constant, or linear warmup then
        # cosine decay to zero over lr_schedule_steps; stepped once per optimizer step.
        lr_scheduler = get_scheduler(args.lr_scheduler, optimizer, num_warmup_steps=args.lr_warmup_steps,
                                     num_training_steps=args.lr_schedule_steps)
        step = 0
        if checkpoint:
            ema.averaged_model.load_state_dict(checkpoint['ema_model'])
            ema.optimization_step = checkpoint['ema_step']
            if 'lr_scheduler' in checkpoint:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            step = checkpoint['step']
            torch.set_rng_state(checkpoint['rng']['torch'])
            numpy_rng = checkpoint['rng']['numpy']
            np.random.set_state((numpy_rng[0], numpy_rng[1].numpy().astype(np.uint32), *numpy_rng[2:]))
            random.setstate(checkpoint['rng']['python'])
            if checkpoint['rng']['cuda'] is not None and device.type == 'cuda' and not distributed:
                torch.cuda.set_rng_state_all(checkpoint['rng']['cuda'])
            del checkpoint
        if distributed and (rank > 0 or step):
            # the saved RNG state is rank 0's; every other rank (and rank 0's CUDA stream under DDP) takes its own seed
            torch.manual_seed(args.seed + 1_000 * (step + 1) + rank)
            if device.type == 'cuda':
                torch.cuda.manual_seed(args.seed + 1_000 * (step + 1) + rank)
        elif distributed:
            torch.manual_seed(args.seed + rank)
            if device.type == 'cuda':
                torch.cuda.manual_seed(args.seed + rank)
        torch.set_float32_matmul_precision(args.matmul_precision)
        torch.backends.cudnn.benchmark = args.cudnn_benchmark
        if args.compile != 'none':
            # The attention backend gets baked into the traced graphs, and neither the AOT-autograd
            # nor the Inductor cache key includes the backend flags; keep the artifacts apart.
            torch.compiler.config.cache_key_tag = f'b1k-sdpa-{args.sdpa_backend}'
            compile_policy(policy, args.compile)
        trainer = LossModule(policy)
        if distributed:
            from torch.nn.parallel import DistributedDataParallel
            trainer = DistributedDataParallel(trainer, device_ids=[device.index] if device.type == 'cuda' else None,
                                              broadcast_buffers=False, find_unused_parameters=False)
        autocast = torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.autocast == 'bf16')
        attention = (partial(sdpa_kernel, [SDPBackend.MATH]) if args.sdpa_backend == 'math'
                     else contextlib.nullcontext)
        goal_pairs = {goal_key(view): view for view in config.goal_views} if config.goal_fusion != 'none' else {}
        augment = config.image_augment == 'pi05'
        if augment:
            from diffusion_policy.b1k.augment import augment_obs
        # The sampler draws the global optimizer batch per step exactly as without accumulation; each rank's loader
        # delivers its share in micro-batches (one worker task each unless --loader-batch-size subdivides them further).
        rank_batch = args.batch_size // world_size
        micro_batch = rank_batch // args.grad_accumulation
        from diffusion_policy.b1k.b1k_data_core import PiFrameBatchSampler
        sampler = (PiFrameBatchSampler(len(dataset), args.batch_size, step, args.max_steps, args.seed,
                                      loader_batch_size=args.loader_batch_size or micro_batch,
                                      rank=rank, world_size=world_size) if config.frame_selection == 'pi' else
                   StepBatchSampler(len(dataset), args.batch_size, step, args.max_steps, args.seed,
                                    args.loader_batch_size or micro_batch, rank, world_size))
        loader = DataLoader(
            dataset, batch_sampler=sampler,
            num_workers=args.num_workers, pin_memory=device.type == 'cuda',
            worker_init_fn=partial(seed_worker, cpu_threads=args.worker_cpu_threads),
            generator=torch.Generator().manual_seed(args.seed + rank),
            **({'multiprocessing_context': 'spawn', 'prefetch_factor': args.prefetch_factor} if args.num_workers else {}))
        if is_main:
            (output / 'config.json').write_text(json.dumps({
                'model': config.to_dict(), 'tasks': dataset.task_map, 'training': vars(args), 'settle_steps': dataset.settle,
                'val_split': val_split,
                'conditioning': conditioning_record(config, dataset.task_map, dataset.root, len(dataset.episodes), split,
                                                    dataset.settle)},
                indent=2, default=str))
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        stopped = False
        with (output / 'train.jsonl').open('a') if is_main else contextlib.nullcontext() as log:
            start = wait_start = time.monotonic()
            data_wait_s = compute_s = 0.0
            micro_losses = []
            optimizer.zero_grad(set_to_none=True)
            for batch in training_batches(loader, micro_batch):
                compute_start = time.monotonic()
                data_wait_s += compute_start - wait_start
                batch = dict_apply(batch, lambda value: value.to(device, non_blocking=True))
                # Images travel as uint8; this reproduces the float32 [0, 1] values bit for bit.
                batch['obs'] = images_to_float(batch['obs'])
                if augment:  # training batches only, in [0, 1], before the policy's normalizer; one draw per micro-batch
                    batch['obs'] = augment_obs(batch['obs'], config.cameras, goal_pairs, args.seed, step,
                                               rank * args.grad_accumulation + len(micro_losses),
                                               compiled=device.type == 'cuda' and args.compile != 'none')
                last_micro = len(micro_losses) == args.grad_accumulation - 1
                sync = trainer.no_sync() if distributed and not last_micro else contextlib.nullcontext()
                with sync, attention():
                    with autocast:
                        loss = trainer(batch)
                    # Equal micro-batches: the summed gradient equals that of the full-batch mean loss.
                    (loss / args.grad_accumulation if args.grad_accumulation > 1 else loss).backward()
                # Checked after backward so the CPU can enqueue it without a device sync in between.
                if not torch.isfinite(loss):
                    raise RuntimeError(f'Non-finite loss at step {step}')
                micro_losses.append(loss.detach().clone())
                del batch, loss
                if len(micro_losses) < args.grad_accumulation:
                    compute_s += time.monotonic() - compute_start
                    wait_start = time.monotonic()
                    continue
                # max_norm=inf leaves the gradients untouched but still computes the norm and rejects non-finite ones.
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    policy.parameters(), args.grad_clip if args.grad_clip > 0 else float('inf'), error_if_nonfinite=True)
                learning_rate = optimizer.param_groups[0]['lr']
                optimizer.step()
                lr_scheduler.step()
                if ema_updater is not None:
                    ema_updater.step()
                else:
                    ema.step(policy)
                # Upstream EMA updates parameters only, not BatchNorm running statistics.
                sync_batchnorm_buffers(policy, ema.averaged_model)
                step += 1
                loss_tensor = torch.stack(micro_losses).sum() / len(micro_losses)
                if distributed:  # the step's loss over the whole global batch
                    import torch.distributed as dist
                    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                    loss_tensor /= world_size
                loss_value = loss_tensor.item()
                grad_value = float(grad_norm)
                optimizer.zero_grad(set_to_none=True)
                micro_losses = []
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                compute_s += time.monotonic() - compute_start
                checkpoint_start = time.monotonic()
                stopped = stop_requested(args.stop_file, device, distributed)
                validation = {}
                if val_dataset is not None and args.val_every and (step % args.val_every == 0 or stopped or step == args.max_steps):
                    validation = evaluate_validation(ema.averaged_model, val_dataset, config, args, device, autocast,
                                                     attention, rank, world_size)
                if is_main:
                    if args.export_every and step % args.export_every == 0:
                        path = export_checkpoint(output, ema, config, dataset.task_map, step, dataset.language,
                                                 selection_record(dataset, args))
                        print(f'Evaluation export: {path}', flush=True)
                    if (step % args.save_every == 0 or step == args.max_steps or (args.save_first_step and step == 1)
                            or stopped):
                        path = save_checkpoint(output, policy, ema, optimizer, lr_scheduler, config, dataset, step, args)
                        print(f'Checkpoint: {path}', flush=True)
                    record = {'step': step, 'loss': loss_value, 'grad_norm': grad_value,
                              'elapsed_s': time.monotonic() - start, 'data_wait_s': data_wait_s,
                              'compute_s': compute_s, 'step_s': data_wait_s + compute_s,
                              'checkpoint_s': time.monotonic() - checkpoint_start,
                              'samples_per_s': args.batch_size / (data_wait_s + compute_s),
                              'learning_rate': learning_rate, **gpu_memory_metrics(device), **validation}
                    if stopped:
                        record['stopped'] = True
                    print(json.dumps(record), flush=True)
                    log.write(json.dumps(record) + '\n')
                    log.flush()
                    if wandb_run is not None:
                        wandb_run.log(record, step=step)
                del grad_norm
                data_wait_s = compute_s = 0.0
                wait_start = time.monotonic()
                if stopped:
                    if is_main:
                        print(f'Stop file {args.stop_file}: stopped after saving step {step}', flush=True)
                    break
        if distributed:  # ranks leave together (rank 0 may still be writing the last checkpoint)
            import torch.distributed as dist
            dist.barrier()
        exit_code = 0
    finally:
        if dataset is not None:
            dataset.close()
        if val_dataset is not None:
            val_dataset.close()
        if wandb_run is not None:
            wandb_run.finish(exit_code=exit_code)


if __name__ == '__main__':
    main()
