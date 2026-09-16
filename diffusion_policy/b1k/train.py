"""Small, native training entrypoint; legacy Hydra workspaces remain unchanged."""

import argparse
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
from torch.utils.data import DataLoader, Sampler

from diffusion_policy.b1k.dataset import B1KLeRobotDataset
from diffusion_policy.b1k.model import POLICY_TARGETS, ModelConfig, build_policy, load_checkpoint
from diffusion_policy.b1k.robot import CAMERAS
from diffusion_policy.common.pytorch_util import dict_apply
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
    chunks, count = [], 0
    for batch in loader:
        chunks.append(batch)
        count += len(batch['action'])
        if count > batch_size:
            raise ValueError('Loader chunks crossed an optimizer-step boundary')
        if count == batch_size:
            combined = chunks[0] if len(chunks) == 1 else concatenate_batches(chunks)
            chunks, count = [], 0
            yield combined
            del combined
    if chunks:
        raise ValueError('Incomplete optimizer batch from loader')


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
            return policy.get_optimizer(transformer_weight_decay=args.weight_decay,
                                        obs_encoder_weight_decay=args.obs_encoder_weight_decay, **common)
        if args.variant == 'transformer_lowdim':
            return policy.get_optimizer(weight_decay=args.weight_decay, **common)
        raise ValueError('--optimizer upstream requires a transformer variant')
    return torch.optim.AdamW((p for p in policy.parameters() if p.requires_grad),
                             lr=args.learning_rate, weight_decay=args.weight_decay, betas=tuple(args.betas))


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


def save_checkpoint(output, policy, ema, optimizer, config, dataset, step, args):
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
                     'obs_encoder_weight_decay': args.obs_encoder_weight_decay},
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
    result.add_argument('--max-episodes', type=int, help='Explicit small-data smoke/debug subset')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    device = torch.device(args.device)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('--device must be cpu, cuda or an indexed CUDA device')
    if (min(args.max_steps, args.batch_size, args.save_every, args.cpu_threads,
            args.worker_cpu_threads, args.prefetch_factor) < 1 or
            min(args.num_workers, args.save_total_limit, args.export_every) < 0):
        raise ValueError('Steps, batch size, save interval, prefetch and CPU threads must be positive; '
                         'workers, retention and export interval nonnegative')
    if args.loader_batch_size is not None and args.loader_batch_size < 1:
        raise ValueError('--loader-batch-size must be positive')
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
        for key, value in {'optimizer': 'adamw', 'betas': (0.9, 0.999),
                           'obs_encoder_weight_decay': 1e-6, **checkpoint['training']}.items():
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
            max_episodes=args.max_episodes)
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
        policy.to(args.device).train()
        if config.freeze_encoder:
            policy.obs_encoder.eval().requires_grad_(False)
        policy.normalizer.requires_grad_(False)
        ema = EMAModel(copy.deepcopy(policy))
        optimizer = build_optimizer(policy, args)
        step = 0
        if checkpoint:
            ema.averaged_model.load_state_dict(checkpoint['ema_model'])
            ema.optimization_step = checkpoint['ema_step']
            optimizer.load_state_dict(checkpoint['optimizer'])
            step = checkpoint['step']
            torch.set_rng_state(checkpoint['rng']['torch'])
            numpy_rng = checkpoint['rng']['numpy']
            np.random.set_state((numpy_rng[0], numpy_rng[1].numpy().astype(np.uint32), *numpy_rng[2:]))
            random.setstate(checkpoint['rng']['python'])
            if checkpoint['rng']['cuda'] is not None and device.type == 'cuda':
                torch.cuda.set_rng_state_all(checkpoint['rng']['cuda'])
            del checkpoint
        loader = DataLoader(
            dataset, batch_sampler=StepBatchSampler(len(dataset), args.batch_size, step, args.max_steps,
                                                    args.seed, args.loader_batch_size),
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
            for batch in training_batches(loader, args.batch_size):
                compute_start = time.monotonic()
                data_wait_s = compute_start - wait_start
                batch = dict_apply(batch, lambda value: value.to(args.device, non_blocking=True))
                optimizer.zero_grad(set_to_none=True)
                loss = policy.compute_loss(batch)
                if not torch.isfinite(loss):
                    raise RuntimeError(f'Non-finite loss at step {step}')
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                ema.step(policy)
                # Upstream EMA updates parameters only, not BatchNorm running statistics.
                sync_batchnorm_buffers(policy, ema.averaged_model)
                step += 1
                loss_value, grad_value = loss.item(), float(grad_norm)
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                compute_s = time.monotonic() - compute_start
                checkpoint_start = time.monotonic()
                if args.export_every and step % args.export_every == 0:
                    path = export_checkpoint(output, ema, config, dataset.task_map, step, dataset.language)
                    print(f'Evaluation export: {path}', flush=True)
                if step % args.save_every == 0 or step == args.max_steps or (args.save_first_step and step == 1):
                    path = save_checkpoint(output, policy, ema, optimizer, config, dataset, step, args)
                    print(f'Checkpoint: {path}', flush=True)
                record = {'step': step, 'loss': loss_value, 'grad_norm': grad_value,
                          'elapsed_s': time.monotonic() - start, 'data_wait_s': data_wait_s,
                          'compute_s': compute_s, 'step_s': data_wait_s + compute_s,
                          'checkpoint_s': time.monotonic() - checkpoint_start,
                          'samples_per_s': args.batch_size / (data_wait_s + compute_s),
                          'learning_rate': optimizer.param_groups[0]['lr'], **gpu_memory_metrics(device)}
                print(json.dumps(record), flush=True)
                log.write(json.dumps(record) + '\n')
                log.flush()
                if wandb_run is not None:
                    wandb_run.log(record, step=step)
                del batch, loss, grad_norm
                wait_start = time.monotonic()
        exit_code = 0
    finally:
        if dataset is not None:
            dataset.close()
        if wandb_run is not None:
            wandb_run.finish(exit_code=exit_code)


if __name__ == '__main__':
    main()
