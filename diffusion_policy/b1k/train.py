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

from diffusion_policy.b1k.dataset import B1KLeRobotDataset, images_to_float
from diffusion_policy.b1k.model import POLICY_TARGETS, ModelConfig, build_policy, load_checkpoint
from diffusion_policy.b1k.robot import CAMERAS
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.model.diffusion.ema_model import EMAModel


class StepBatchSampler(Sampler):
    """O(batch_size) frame-uniform sampling, exactly addressable across resume."""

    def __init__(self, length, batch_size, start_step, max_steps, seed, loader_batch_size=None):
        self.length, self.batch_size = length, batch_size
        self.start_step, self.max_steps, self.seed = start_step, max_steps, seed
        self.loader_batch_size = loader_batch_size or batch_size

    def __iter__(self):
        for step in range(self.start_step, self.max_steps):
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, step]))
            indices = rng.integers(self.length, size=self.batch_size).tolist()
            for start in range(0, self.batch_size, self.loader_batch_size):
                yield indices[start:start + self.loader_batch_size]

    def __len__(self):
        chunks = (self.batch_size + self.loader_batch_size - 1) // self.loader_batch_size
        return max(0, self.max_steps - self.start_step) * chunks


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


def save_checkpoint(output, policy, ema, optimizer, lr_scheduler, config, dataset, step, args):
    checkpoint = {
        'format': 'diffusion_policy_b1k_v1', 'checkpoint_type': 'full',
        'config': config.to_dict(), 'task_map': dataset.task_map,
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
        'selection': {'task_names': list(dataset.task_map.values()),
                      'max_episodes': args.max_episodes,
                      'episode_indices': [row['episode_index'] for row in dataset.episodes]},
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


def export_checkpoint(output, ema, config, task_map, step, language=None):
    checkpoint = {
        'format': 'diffusion_policy_b1k_v1', 'checkpoint_type': 'eval',
        'config': config.to_dict(), 'task_map': task_map,
        'normalizer': ema.averaged_model.normalizer.state_dict(),
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


class ExplicitLanguageChoice(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        setattr(namespace, f'{self.dest}_explicit', True)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--dataset-path', '--dataset-root', dest='dataset_path', required=True)
    result.add_argument('--task-names', nargs='+')
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
    result.add_argument('--max-episodes', type=int, help='Explicit small-data smoke/debug subset')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    device = torch.device(args.device)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('--device must be cpu, cuda or an indexed CUDA device')
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
    output = Path(args.output_dir).resolve()
    if output.is_relative_to(Path(args.dataset_path).resolve()):
        raise ValueError('--output-dir must not be inside the read-only dataset')
    with output_lock(output):
        if any(path.name != 'run.lock' for path in output.iterdir()) and args.resume is None:
            raise FileExistsError(f'Output directory is nonempty: {output}; use --resume or a new directory')
        run_training(args, device, output)


def run_training(args, device, output):
    configure_cpu_threads(args.cpu_threads)
    checkpoint = load_checkpoint(args.resume) if args.resume else None
    if checkpoint:
        if checkpoint.get('checkpoint_type') == 'eval' or 'optimizer' not in checkpoint:
            raise ValueError('Cannot resume training from an eval-only checkpoint; use a full checkpoint')
        config = ModelConfig(**checkpoint['config'])
        for key in ('language_conditioning', 'prompt_source'):
            if getattr(args, f'{key}_explicit', False) and getattr(args, key) != getattr(config, key):
                raise ValueError(f'--{key.replace("_", "-")} conflicts with the resumed checkpoint')
        if args.task_names is None:
            args.task_names = checkpoint['selection']['task_names']
        if args.max_episodes is None:
            args.max_episodes = checkpoint['selection']['max_episodes']
        for key, value in {'optimizer': 'adamw', 'betas': (0.9, 0.999), 'obs_encoder_weight_decay': 1e-6,
                           # Checkpoints predating these options trained with a constant learning rate,
                           # EMA power 2/3 and gradient clipping at 1.0.
                           'lr_scheduler': 'constant', 'lr_warmup_steps': 0, 'lr_schedule_steps': None,
                           'ema_power': 2 / 3, 'grad_clip': 1.0, 'grad_accumulation': 1,
                           **checkpoint['training']}.items():
            setattr(args, key, value)
    else:
        values = {key: getattr(args, key) for key in ModelConfig.__dataclass_fields__ if hasattr(args, key)}
        for key in ('cameras', 'down_dims', 'crop_shape', 'resize_shape'):
            if values[key] is not None:
                values[key] = tuple(values[key])
        config = ModelConfig(**values, clip_sample=not args.no_clip_sample)
    config.validate()
    args.variant = config.variant
    args.language_conditioning = config.language_conditioning
    args.prompt_source = config.prompt_source
    if args.optimizer == 'upstream' and not config.variant.startswith('transformer_'):
        raise ValueError('--optimizer upstream requires a transformer variant')
    if checkpoint and checkpoint['step'] >= args.max_steps:
        raise ValueError(f'--max-steps is a total target and must exceed checkpoint step {checkpoint["step"]}')
    wandb_run = initialize_wandb(args, output, checkpoint, config)
    dataset = None
    exit_code = 1
    try:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        dataset = B1KLeRobotDataset(
            args.dataset_path, args.task_names, **config.dataset_kwargs(),
            episode_cache_size=args.episode_cache_size, parquet_cache_mb=args.parquet_cache_mb,
            max_episodes=args.max_episodes, image_dtype='uint8', frame_cache=args.frame_cache,
            video_max_open=args.video_max_open)
        if checkpoint and (dataset.task_map != checkpoint['task_map'] or
                           [row['episode_index'] for row in dataset.episodes] != checkpoint['selection']['episode_indices'] or
                           dataset.fingerprint() != checkpoint['dataset_fingerprint']):
            raise ValueError('Resume dataset selection, metadata or file fingerprint does not match checkpoint')
        dataset.prepare_language(checkpoint.get('language') if checkpoint else None)
        print(json.dumps({'episodes': len(dataset.episodes), 'sequences': len(dataset),
                          'task_map': dataset.task_map, 'config': config.to_dict()}), flush=True)
        policy = build_policy(config, dataset.task_map, initialize_encoder=checkpoint is None)
        if checkpoint:
            policy.load_state_dict(checkpoint['model'])
        else:
            print('Computing exact selected-frame limits, streaming parquet once per file (no video).', flush=True)
            policy.set_normalizer(dataset.get_normalizer())
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
        policy.to(args.device).train()
        if config.freeze_encoder:
            policy.obs_encoder.eval().requires_grad_(False)
        policy.normalizer.requires_grad_(False)
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
            if checkpoint['rng']['cuda'] is not None and device.type == 'cuda':
                torch.cuda.set_rng_state_all(checkpoint['rng']['cuda'])
            del checkpoint
        torch.set_float32_matmul_precision(args.matmul_precision)
        torch.backends.cudnn.benchmark = args.cudnn_benchmark
        if args.compile != 'none':
            # The attention backend gets baked into the traced graphs, and neither the AOT-autograd
            # nor the Inductor cache key includes the backend flags; keep the artifacts apart.
            torch.compiler.config.cache_key_tag = f'b1k-sdpa-{args.sdpa_backend}'
            compile_policy(policy, args.compile)
        autocast = torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.autocast == 'bf16')
        attention = (partial(sdpa_kernel, [SDPBackend.MATH]) if args.sdpa_backend == 'math'
                     else contextlib.nullcontext)
        # The sampler draws the optimizer batch per step exactly as without accumulation; the loader delivers
        # it in micro-batches (one worker task each unless --loader-batch-size subdivides them further).
        micro_batch = args.batch_size // args.grad_accumulation
        loader = DataLoader(
            dataset, batch_sampler=StepBatchSampler(len(dataset), args.batch_size, step, args.max_steps,
                                                    args.seed, args.loader_batch_size or micro_batch),
            num_workers=args.num_workers, pin_memory=device.type == 'cuda',
            worker_init_fn=partial(seed_worker, cpu_threads=args.worker_cpu_threads),
            generator=torch.Generator().manual_seed(args.seed),
            **({'multiprocessing_context': 'spawn', 'prefetch_factor': args.prefetch_factor} if args.num_workers else {}))
        (output / 'config.json').write_text(json.dumps({
            'model': config.to_dict(), 'tasks': dataset.task_map, 'training': vars(args)}, indent=2, default=str))
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        with (output / 'train.jsonl').open('a') as log:
            start = wait_start = time.monotonic()
            data_wait_s = compute_s = 0.0
            micro_losses = []
            optimizer.zero_grad(set_to_none=True)
            for batch in training_batches(loader, micro_batch):
                compute_start = time.monotonic()
                data_wait_s += compute_start - wait_start
                batch = dict_apply(batch, lambda value: value.to(args.device, non_blocking=True))
                # Images travel as uint8; this reproduces the float32 [0, 1] values bit for bit.
                batch['obs'] = images_to_float(batch['obs'])
                with attention():
                    with autocast:
                        loss = policy.compute_loss(batch)
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
                loss_value = (torch.stack(micro_losses).sum() / len(micro_losses)).item()
                grad_value = float(grad_norm)
                optimizer.zero_grad(set_to_none=True)
                micro_losses = []
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                compute_s += time.monotonic() - compute_start
                checkpoint_start = time.monotonic()
                if args.export_every and step % args.export_every == 0:
                    path = export_checkpoint(output, ema, config, dataset.task_map, step, dataset.language)
                    print(f'Evaluation export: {path}', flush=True)
                if step % args.save_every == 0 or step == args.max_steps or (args.save_first_step and step == 1):
                    path = save_checkpoint(output, policy, ema, optimizer, lr_scheduler, config, dataset, step, args)
                    print(f'Checkpoint: {path}', flush=True)
                record = {'step': step, 'loss': loss_value, 'grad_norm': grad_value,
                          'elapsed_s': time.monotonic() - start, 'data_wait_s': data_wait_s,
                          'compute_s': compute_s, 'step_s': data_wait_s + compute_s,
                          'checkpoint_s': time.monotonic() - checkpoint_start,
                          'samples_per_s': args.batch_size / (data_wait_s + compute_s),
                          'learning_rate': learning_rate, **gpu_memory_metrics(device)}
                print(json.dumps(record), flush=True)
                log.write(json.dumps(record) + '\n')
                log.flush()
                if wandb_run is not None:
                    wandb_run.log(record, step=step)
                del grad_norm
                data_wait_s = compute_s = 0.0
                wait_start = time.monotonic()
        exit_code = 0
    finally:
        if dataset is not None:
            dataset.close()
        if wandb_run is not None:
            wandb_run.finish(exit_code=exit_code)


if __name__ == '__main__':
    main()
