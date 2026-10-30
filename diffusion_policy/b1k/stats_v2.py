"""Portable DP statistics over exactly the valid steps of its sampled chunks."""
import json
from pathlib import Path
import numpy as np
from .b1k_data_core import StreamingStats, valid_chunk_count
from .robot import action_reference, proprio_dim


def compute_stats(dataset, method):
    if method not in ('zscore', 'quantile'):
        raise ValueError('v2 statistics require zscore or quantile')
    lengths = np.array([ep['length'] for ep in dataset.episodes], dtype=np.int64)
    offsets = np.arange(dataset.horizon) - (dataset.n_obs_steps - 1)
    if dataset.frame_selection != 'pi':
        raise ValueError('v2 DP statistics require frame_selection=pi; use stats_estimator=legacy for old recipes')
    frames, steps = int(lengths.sum()), valid_chunk_count(lengths, offsets)
    options = dict(estimator=dataset.stats_estimator, sample_size=dataset.stats_sample_size,
                   quantiles=method == 'quantile')
    with StreamingStats(proprio_dim(dataset.gripper_state), frames, **options) as states, \
         StreamingStats(23, steps, **options) as actions:
        for _, state, action in dataset.iter_episode_lowdim():
            states.update(state)
            reference = action_reference(state, dataset.action_delta) if dataset.action_representation == 'relative' else None
            for start in range(0, len(action), 2048):
                anchors = np.arange(start, min(start + 2048, len(action)))
                indices = anchors[:, None] + offsets
                valid = (indices >= 0) & (indices < len(action))
                values = action[np.clip(indices, 0, len(action)-1)].astype(np.float64)
                if reference is not None:
                    values -= reference[anchors, None, :]
                actions.update(values[valid])
        result = dict(version=2, fingerprint=dataset.fingerprint(), method=method,
                      action_representation=dataset.action_representation, gripper_action=dataset.gripper_action,
                      horizon=dataset.horizon, n_obs_steps=dataset.n_obs_steps, frame_selection=dataset.frame_selection,
                      count=frames, action_count=steps, std_eps=1e-6, std_correction=0, min_range=0, quantile_eps=1e-6,
                      state=states.finish(), action=actions.finish(),
                      estimator={'state':states.metadata(), 'action':actions.metadata()})
    if dataset.action_representation == 'relative':
        result['action_delta'] = dataset.action_delta
    if dataset.task_onehot:
        identity = {'min':0., 'max':1., 'mean':0., 'std':1., 'q01':0., 'q99':1.}
        for key, values in result['state'].items():
            result['state'][key] = np.r_[values, np.full(len(dataset.task_map), identity[key])]
    return result


def load_or_compute(dataset, method, stats_file, output):
    if method == 'limits':
        if stats_file:
            raise ValueError('--stats-file is for zscore/quantile')
        return None
    path = Path(stats_file) if stats_file else Path(output) / (
        f'stats-{dataset.fingerprint()}-{method}-{dataset.action_representation}-{dataset.gripper_action}'
        f'-h{dataset.horizon}-o{dataset.n_obs_steps}-{dataset.stats_estimator}-{dataset.stats_sample_size}.json')
    expected = dict(fingerprint=dataset.fingerprint(), method=method,
                    action_representation=dataset.action_representation, gripper_action=dataset.gripper_action,
                    horizon=dataset.horizon, n_obs_steps=dataset.n_obs_steps, frame_selection=dataset.frame_selection)
    if path.is_file():
        stats = json.loads(path.read_text())
        if any(stats.get(k) != v for k,v in expected.items()):
            raise ValueError(f'statistics selection/recipe mismatch: {path}')
        expected_dim=proprio_dim(dataset.gripper_state)+(len(dataset.task_map) if dataset.task_onehot else 0)
        if len(stats['state']['mean']) != expected_dim:
            raise ValueError(f'statistics state/task-onehot dimension mismatch: {path}')
        return stats
    if stats_file:
        raise FileNotFoundError(path)
    stats = dataset.compute_action_stats(method)
    if dataset.stats_estimator != 'legacy':
        from .b1k_cache_v2 import atomic_json
        serializable = json.loads(json.dumps(stats, default=lambda x: np.asarray(x).tolist()))
        atomic_json(path, serializable)
    return stats


def main(argv=None):
    """Precompute stats without RGB files; output loads in the real RGB trainer."""
    import argparse
    from .dataset import B1KLeRobotDataset, expand_task_groups, resolve_episode_split, resolve_val_split
    from .b1k_cache_v2 import atomic_json
    p=argparse.ArgumentParser(description=main.__doc__)
    p.add_argument('--dataset-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--task-names',nargs='+')
    p.add_argument('--episode-split',default='auto')
    p.add_argument('--settle-steps',default='0')
    p.add_argument('--val-fraction',type=float,default=.1)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--horizon',type=int,default=16)
    p.add_argument('--n-obs-steps',type=int,default=2)
    p.add_argument('--n-action-steps',type=int,default=8)
    p.add_argument('--normalization',choices=['quantile','zscore'],default='quantile')
    p.add_argument('--action-representation',choices=['relative','absolute'],default='relative')
    p.add_argument('--gripper-action',choices=['absolute','delta'],default='absolute')
    p.add_argument('--estimator',choices=['exact','sampled','auto'],default='exact')
    p.add_argument('--sample-size',type=int,default=10_000_000)
    p.add_argument('--task-onehot',action='store_true')
    args=p.parse_args(argv);root=args.dataset_root.resolve();output=args.output.resolve()
    if root == output or root in output.parents:p.error('statistics output must be outside the source dataset')
    if output.exists():raise FileExistsError(output)
    tasks=expand_task_groups(root,args.task_names)
    episode_split=resolve_episode_split(root,args.episode_split,tasks)
    split=resolve_val_split(root,tasks,episode_split,args.val_fraction,args.seed)
    episodes=split['train_episodes'] if split else (episode_split['episodes'] if episode_split else None)
    dataset=B1KLeRobotDataset(root,tasks,episodes=episodes,settle_steps=args.settle_steps,observation_mode='lowdim',
        horizon=args.horizon,n_obs_steps=args.n_obs_steps,n_action_steps=args.n_action_steps,frame_selection='pi',
        gripper_state='sum',task_onehot=args.task_onehot,gripper_action=args.gripper_action,
        action_representation=args.action_representation,stats_estimator=args.estimator,stats_sample_size=args.sample_size)
    try:
        stats=compute_stats(dataset,args.normalization)
        output.parent.mkdir(parents=True,exist_ok=True)
        atomic_json(output,json.loads(json.dumps(stats,default=lambda x:np.asarray(x).tolist())))
        print(json.dumps({'frames':stats['count'],'valid_steps':stats['action_count'],'fingerprint':stats['fingerprint']}))
        print(f'RESULT: DP STATS READY {output}')
    finally:dataset.close()


if __name__=='__main__':main()
