"""Streaming equivalent of LinearNormalizer.fit(mode='limits')."""

import numpy as np

from diffusion_policy.common.normalize_util import get_image_range_normalizer, get_range_normalizer_from_stat
from diffusion_policy.model.common.normalizer import LinearNormalizer


class RunningStats:
    def __init__(self):
        self.count = 0

    def update(self, values):
        values = np.asarray(values, dtype=np.float64)
        if not len(values):
            return
        if not np.isfinite(values).all():
            raise ValueError('Normalization input contains non-finite values')
        count = len(values)
        mean = values.mean(axis=0)
        m2 = np.square(values - mean).sum(axis=0)
        low, high = values.min(axis=0), values.max(axis=0)
        if self.count:
            delta = mean - self.mean
            total = self.count + count
            self.m2 += m2 + delta ** 2 * (self.count * count / total)
            self.mean += delta * count / total
            self.low = np.minimum(self.low, low)
            self.high = np.maximum(self.high, high)
        else:
            self.mean, self.m2, self.low, self.high = mean, m2, low, high
        self.count += count

    def stats(self):
        if self.count < 2:
            raise ValueError('At least two selected frames are required for normalization')
        return {key: value.astype(np.float32) for key, value in {
            'min': self.low, 'max': self.high, 'mean': self.mean,
            'std': np.sqrt(self.m2 / (self.count - 1)),
        }.items()}


def field_from_stats(stats, method, eps=1e-6):
    """A SingleFieldLinearNormalizer (`x * scale + offset`) for one group's statistics dict (`min`, `max`, `mean`,
    `std` and, for quantile, `q01` / `q99`; one value per dimension):

    - zscore:   (x - mean) / std                       scale = 1 / std,             offset = -mean / std
    - quantile: (x - q01) / (q99 - q01 + eps) * 2 - 1   scale = 2 / (q99 - q01 + eps), offset = -1 - q01 * scale
      (openpi / ACT's pi05 quantile normalization; `unnormalize` is the exact inverse)
    The statistics travel in `input_stats` (q01 / q99 included for quantile) and so in every checkpoint."""
    from diffusion_policy.model.common.normalizer import SingleFieldLinearNormalizer

    def array(key):
        return np.asarray(stats[key], dtype=np.float64)
    if method == 'zscore':
        scale = 1.0 / array('std')
        offset = -array('mean') * scale
        keys = ('min', 'max', 'mean', 'std')
    elif method == 'quantile':
        scale = 2.0 / (array('q99') - array('q01') + eps)
        offset = -1.0 - array('q01') * scale
        keys = ('min', 'max', 'mean', 'std', 'q01', 'q99')
    else:
        raise ValueError(f'field_from_stats serves zscore or quantile, not {method!r}')
    to32 = lambda values: np.asarray(values, dtype=np.float32)
    return SingleFieldLinearNormalizer.create_manual(to32(scale), to32(offset), {key: to32(stats[key]) for key in keys})


def normalizer_from_stats(stats, cameras, proprio_size, task_onehot=True):
    """LinearNormalizer of `B1KLeRobotDataset.compute_action_stats` output: `state` and `action` with the statistics'
    method; a one-hot task suffix of the state (`task_onehot`, beyond `proprio_size`) keeps its literal values; the
    cameras get the image range normalizer."""
    normalizer = LinearNormalizer()
    method = stats['method']
    state = field_from_stats(stats['state'], method, stats.get('quantile_eps', 1e-6))
    if task_onehot and len(stats['state']['mean']) > proprio_size:
        state.params_dict['scale'].data[proprio_size:] = 1
        state.params_dict['offset'].data[proprio_size:] = 0
    normalizer['state'] = state
    normalizer['action'] = field_from_stats(stats['action'], method, stats.get('quantile_eps', 1e-6))
    for camera in cameras:
        normalizer[camera] = get_image_range_normalizer()
    normalizer.requires_grad_(False)
    return normalizer


def fit_normalizer(batches, cameras, proprio_size):
    stats = {key: RunningStats() for key in ('state', 'action')}
    for batch in batches:
        for key, tracker in stats.items():
            tracker.update(batch[key])
    normalizer = LinearNormalizer()
    for key, tracker in stats.items():
        field = get_range_normalizer_from_stat(tracker.stats(), range_eps=1e-4)
        if key == 'state':
            # Categorical channels retain their literal one-hot values.
            field.params_dict['scale'].data[proprio_size:] = 1
            field.params_dict['offset'].data[proprio_size:] = 0
        normalizer[key] = field
    for camera in cameras:
        normalizer[camera] = get_image_range_normalizer()
    normalizer.requires_grad_(False)
    return normalizer
