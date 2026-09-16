"""Small, native training entrypoint; legacy Hydra workspaces remain unchanged."""

import argparse
import copy
import json
import os
from pathlib import Path
import random
import time

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

    def __init__(self, length, batch_size, start_step, max_steps, seed):
        self.length, self.batch_size = length, batch_size
        self.start_step, self.max_steps, self.seed = start_step, max_steps, seed

    def __iter__(self):
        for step in range(self.start_step, self.max_steps):
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, step]))
            yield rng.integers(self.length, size=self.batch_size).tolist()

    def __len__(self):
        return max(0, self.max_steps - self.start_step)


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2 ** 32
    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)


@torch.no_grad()
def sync_batchnorm_buffers(policy, averaged_policy):
    for module, averaged_module in zip(policy.modules(), averaged_policy.modules()):
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            for name, buffer in module.named_buffers(recurse=False):
                getattr(averaged_module, name).copy_(buffer)


def save_checkpoint(output, policy, ema, optimizer, config, dataset, step, args):
    checkpoint = {
        'format': 'diffusion_policy_b1k_v1',
        'config': config.to_dict(), 'task_map': dataset.task_map,
        'normalizer': policy.normalizer.state_dict(),
        'model': policy.state_dict(), 'ema_model': ema.averaged_model.state_dict(),
        'ema_step': ema.optimization_step, 'optimizer': optimizer.state_dict(), 'step': step,
        'training': {'seed': args.seed, 'batch_size': args.batch_size,
                     'learning_rate': args.learning_rate, 'weight_decay': args.weight_decay},
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
    path = output / f'step-{step:08d}.pt'
    if path.exists():
        raise FileExistsError(f'Will not overwrite checkpoint {path}')
    temporary = output / '.checkpoint.tmp'
    torch.save(checkpoint, temporary)
    temporary.replace(path)
    latest = output / '.latest.tmp'
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(path.name)
    latest.replace(output / 'latest.pt')
    return path


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--dataset-path', '--dataset-root', dest='dataset_path', required=True)
    result.add_argument('--task-names', nargs='+')
    result.add_argument('--output-dir', required=True)
    result.add_argument('--max-steps', type=int, default=100000)
    result.add_argument('--batch-size', type=int, default=64)
    result.add_argument('--num-workers', type=int, default=4)
    result.add_argument('--device', default='cuda')
    result.add_argument('--variant', choices=list(POLICY_TARGETS), default='unet_image')
    result.add_argument('--conditioning', choices=['global', 'local', 'inpainting'], default='global')
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
    result.add_argument('--save-every', type=int, default=5000)
    result.add_argument('--seed', type=int, default=42)
    result.add_argument('--cpu-threads', type=int, default=4)
    result.add_argument('--episode-cache-size', type=int, default=8)
    result.add_argument('--parquet-cache-mb', type=int, default=256)
    result.add_argument('--max-episodes', type=int, help='Explicit small-data smoke/debug subset')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    device = torch.device(args.device)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('--device must be cpu, cuda or an indexed CUDA device')
    if min(args.max_steps, args.batch_size, args.save_every, args.cpu_threads) < 1 or args.num_workers < 0:
        raise ValueError('Steps, batch size, save interval and CPU threads must be positive; workers nonnegative')
    output = Path(args.output_dir).resolve()
    if output.is_relative_to(Path(args.dataset_path).resolve()):
        raise ValueError('--output-dir must not be inside the read-only dataset')
    if output.exists() and any(output.iterdir()) and args.resume is None:
        raise FileExistsError(f'Output directory is nonempty: {output}; use --resume or a new directory')
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.cpu_threads)
    checkpoint = load_checkpoint(args.resume) if args.resume else None
    if checkpoint:
        config = ModelConfig(**checkpoint['config'])
        if args.task_names is None:
            args.task_names = checkpoint['selection']['task_names']
        if args.max_episodes is None:
            args.max_episodes = checkpoint['selection']['max_episodes']
        for key, value in checkpoint['training'].items():
            setattr(args, key, value)
    else:
        values = {key: getattr(args, key) for key in ModelConfig.__dataclass_fields__ if hasattr(args, key)}
        for key in ('cameras', 'down_dims', 'crop_shape', 'resize_shape'):
            if values[key] is not None:
                values[key] = tuple(values[key])
        config = ModelConfig(**values, clip_sample=not args.no_clip_sample)
    config.validate()
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
    optimizer = torch.optim.AdamW((p for p in policy.parameters() if p.requires_grad),
                                 lr=args.learning_rate, weight_decay=args.weight_decay)
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
        if checkpoint['rng']['cuda'] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(checkpoint['rng']['cuda'])
    if step >= args.max_steps:
        raise ValueError(f'--max-steps is a total target and must exceed checkpoint step {step}')
    loader = DataLoader(
        dataset, batch_sampler=StepBatchSampler(len(dataset), args.batch_size, step, args.max_steps, args.seed),
        num_workers=args.num_workers, pin_memory=str(args.device).startswith('cuda'),
        worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(args.seed),
        **({'multiprocessing_context': 'spawn', 'prefetch_factor': 2} if args.num_workers else {}))
    (output / 'config.json').write_text(json.dumps({'model': config.to_dict(), 'tasks': dataset.task_map}, indent=2))
    log_path = output / 'train.jsonl'
    try:
        with log_path.open('a') as log:
            start = time.monotonic()
            for batch in loader:
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
                record = {'step': step, 'loss': loss.item(), 'grad_norm': float(grad_norm),
                          'elapsed_s': time.monotonic() - start}
                print(json.dumps(record), flush=True)
                log.write(json.dumps(record) + '\n')
                log.flush()
                if step % args.save_every == 0 or step == args.max_steps:
                    path = save_checkpoint(output, policy, ema, optimizer, config, dataset, step, args)
                    print(f'Checkpoint: {path}', flush=True)
    finally:
        dataset.close()


if __name__ == '__main__':
    main()
