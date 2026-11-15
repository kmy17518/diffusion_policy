"""isg-train224: pi05's LR schedule and prepared (precomputed) startup for Diffusion Policy."""
import json

import pytest
import torch

from test_b1k import root, make_dataset  # noqa: F401  (fixture)
from diffusion_policy.b1k.b1k_cache_v2 import build_group
from diffusion_policy.b1k.b1k_data_core import pi_lr_factor
from diffusion_policy.b1k.frame_cache import selected_videos


def _common(root, cache, stats):
    return ['--dataset-root', str(root), '--device', 'cpu', '--num-workers', '0', '--batch-size', '2',
            '--cpu-threads', '1', '--horizon', '4', '--n-action-steps', '3', '--image-size', '16', '--cameras', 'head',
            '--goal-views', 'head', '--goal-fusion', 'late', '--variant', 'unet_hybrid_image', '--resize-mode', 'act',
            '--frame-cache', str(cache), '--down-dims', '16', '32', '--diffusion-step-embed-dim', '16',
            '--num-train-timesteps', '4', '--num-inference-steps', '2', '--action-representation', 'relative',
            '--normalization', 'quantile', '--no-clip-sample', '--stats-file', str(stats)]


def test_prepare_only_then_require_prepared_with_pi_cosine_resume(root, tmp_path):
    from diffusion_policy.b1k.train import main
    native = make_dataset(root, cameras=['head'], resize_mode='act')
    cache = tmp_path / 'cache'
    for path in selected_videos(native):
        build_group([path], cache, root, (16, 16))
    native.close()
    stats = tmp_path / 'prepared' / 'stats.json'
    common = _common(root, cache, stats) + ['--lr-scheduler', 'pi_cosine', '--lr-warmup-steps', '2',
                                            '--lr_schedule.decay_steps', '6', '--learning-rate', '1e-4']
    with pytest.raises(FileNotFoundError, match='prepare-only'):
        main(common + ['--output-dir', str(tmp_path / 'early'), '--max-steps', '2', '--require-prepared'])
    main(common + ['--output-dir', str(tmp_path / 'prep'), '--max-steps', '4', '--prepare-only'])
    assert stats.is_file() and not (tmp_path / 'prep' / 'latest.pt').exists()
    assert list((cache.parent / (cache.name + '-dp-tables')).glob('*/complete.json'))
    full, split = tmp_path / 'full', tmp_path / 'split'
    main(common + ['--output-dir', str(full), '--max-steps', '4', '--require-prepared'])
    lrs = [json.loads(line)['learning_rate'] for line in (full / 'train.jsonl').read_text().splitlines()]
    assert lrs == pytest.approx([1e-4 * pi_lr_factor(i, 2, 6) for i in range(4)])
    main(common + ['--output-dir', str(split), '--max-steps', '2', '--require-prepared'])
    main(common + ['--output-dir', str(split), '--max-steps', '4', '--require-prepared', '--resume', str(split / 'latest.pt')])
    a = torch.load(full / 'latest.pt', weights_only=True)
    b = torch.load(split / 'latest.pt', weights_only=True)
    for field in ('model', 'ema_model'):
        for key in a[field]:
            torch.testing.assert_close(a[field][key], b[field][key], rtol=0, atol=0)
    assert b['training']['lr_scheduler'] == 'pi_cosine' and b['training']['lr_schedule_steps'] == 6
    assert b['config']['clip_sample'] is False and b['config']['gripper_action'] == 'absolute'
    # Missing sidecar tables are refused under --require-prepared (here: another cache directory, never prepared).
    other = tmp_path / 'cache2'
    for path in selected_videos(make_dataset(root, cameras=['head'], resize_mode='act')):
        build_group([path], other, root, (16, 16))
    with pytest.raises(FileNotFoundError, match='frame-cache tables'):
        main(_common(root, other, stats) + ['--output-dir', str(tmp_path / 'other'), '--max-steps', '2',
                                            '--require-prepared'])
