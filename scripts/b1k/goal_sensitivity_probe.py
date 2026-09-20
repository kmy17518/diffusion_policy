#!/usr/bin/env python3
"""Offline goal-sensitivity diagnostic for a goal-conditioned Diffusion Policy checkpoint (plan section 12, quick check).

For sampled training sequences it denoises the action trajectory (EMA weights, matched initial noise) with (a) the
episode's own goal image and (b) the goal image of an episode of the other task, and reports the normalized L1 to
the recorded actions plus the maximum prediction change. Own < other shows the policy reads the goal; it is NOT
goal-following evidence (no rollout, training sequences). Language-conditioned checkpoints keep the sequence's own
prompt in both cases (language held constant, goal swapped).

    goal_sensitivity_probe.py CHECKPOINT --dataset-path ROOT [--frame-cache DIR] [--samples 6] [--seed 0]
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from diffusion_policy.b1k.dataset import B1KLeRobotDataset, images_to_float
from diffusion_policy.b1k.model import ModelConfig, goal_key, load_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('checkpoint')
    parser.add_argument('--dataset-path', required=True)
    parser.add_argument('--frame-cache')
    parser.add_argument('--samples', type=int, default=6)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    policy, checkpoint = load_policy(args.checkpoint, args.device)
    config = ModelConfig(**checkpoint['config'])
    if config.goal_fusion == 'none':
        raise SystemExit('Not a goal-conditioned checkpoint')
    conditioning = checkpoint.get('conditioning', {})
    print('conditioning:', {k: conditioning.get(k) for k in ('regime', 'source_commit')}, '| goal', config.goal_fusion,
          list(config.goal_views), '| language', config.language_conditioning)
    dataset = B1KLeRobotDataset(args.dataset_path, list(checkpoint['task_map'].values()), **config.dataset_kwargs(),
                                image_dtype='uint8', frame_cache=args.frame_cache, episode_cache_size=len(checkpoint['task_map']) * 200)
    if dataset.task_map != checkpoint['task_map']:
        raise SystemExit(f'Dataset task map {dataset.task_map} differs from the checkpoint {checkpoint["task_map"]}')
    if config.language_conditioning != 'none':
        dataset.prepare_language(checkpoint.get('language'))
    rng = np.random.default_rng(args.seed)
    by_task = {}
    for position, ep in enumerate(dataset.episodes):
        by_task.setdefault(int(ep['task_index']), []).append(position)
    keys = [goal_key(view) for view in config.goal_views]
    rows = []
    device = torch.device(args.device)
    for position in rng.choice(len(dataset.episodes), min(args.samples, len(dataset.episodes)), replace=False):
        ep = dataset.episodes[position]
        task = int(ep['task_index'])
        other_tasks = [t for t in by_task if t != task] or [task]
        other_position = int(rng.choice(by_task[int(rng.choice(other_tasks))]))
        start = int(dataset.sampler.ends[position - 1]) if position else 0
        index = start + int(rng.integers(0, int(dataset.sampler.ends[position]) - start))
        item = dataset[index]
        other = dataset[int(dataset.sampler.ends[other_position - 1]) if other_position else 0]
        obs = images_to_float({k: v[None].to(device) for k, v in item['obs'].items()})
        swapped = {**obs, **images_to_float({k: other['obs'][k][None].to(device) for k in keys})}
        with torch.no_grad():
            torch.manual_seed(args.seed)
            own = policy.predict_action(obs)['action_pred'][0].cpu()
            torch.manual_seed(args.seed)
            different = policy.predict_action(swapped)['action_pred'][0].cpu()
        target = item['action']
        rows.append((int(ep['episode_index']), task, (own - target).abs().mean().item(),
                     (different - target).abs().mean().item(), (own - different).abs().max().item()))
    for r in rows:
        print('episode %4d task %d | L1 own goal %.4f | other-task goal %.4f | max |dpred| %.3f' % r)
    print('mean L1: own goal %.4f | other-task goal %.4f (%d samples; normalized action units)' % (
        np.mean([r[2] for r in rows]), np.mean([r[3] for r in rows]), len(rows)))


if __name__ == '__main__':
    main()
