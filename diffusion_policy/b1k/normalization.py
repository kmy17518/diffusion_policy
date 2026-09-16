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


def fit_normalizer(batches, cameras):
    stats = {key: RunningStats() for key in ('state', 'action')}
    for batch in batches:
        for key, tracker in stats.items():
            tracker.update(batch[key])
    normalizer = LinearNormalizer()
    for key, tracker in stats.items():
        field = get_range_normalizer_from_stat(tracker.stats(), range_eps=1e-4)
        if key == 'state':
            # Categorical channels retain their literal one-hot values.
            field.params_dict['scale'].data[25:] = 1
            field.params_dict['offset'].data[25:] = 0
        normalizer[key] = field
    for camera in cameras:
        normalizer[camera] = get_image_range_normalizer()
    normalizer.requires_grad_(False)
    return normalizer
