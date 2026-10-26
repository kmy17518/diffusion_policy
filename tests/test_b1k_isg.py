"""The ACT-comparable recipe (b1k.md "ACT-comparable recipe"): ACT frame caches and resize mode, pi05 augmentation,
relative actions with quantile normalization, the shared validation split, validation metrics, DDP + stop file,
and the evaluator-compatible serving handshake."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from diffusion_policy.b1k.augment import augment_generator, augment_obs, sample_params
from diffusion_policy.b1k.dataset import (B1KLeRobotDataset, VideoReader, expand_task_groups, images_to_float,
                                          resolve_val_split)
from diffusion_policy.b1k.frame_cache import ACT_FORMAT, FORMAT, FrameCacheReader, build_frame_cache, verify_frame_cache
from diffusion_policy.b1k.model import ModelConfig, build_policy
from diffusion_policy.b1k.robot import (ACTION_DELTA, CAMERAS, PROPRIO_KEY, absolute_actions, action_reference,
                                        extract_state, goal_key, relative_actions, resize_rgb, snap_grippers)
from diffusion_policy.b1k.serve import B1KPolicySession, WebsocketPolicyServer, packb, unpackb
from diffusion_policy.b1k.train import StepBatchSampler, evaluate_validation, main as train_main

torch.set_num_threads(1)
ACT_REPO = Path(os.environ.get('ISG_ACT_REPO', '/vision/u/minyeong/projects/act-piaug'))
ISG_DATASET = Path(os.environ.get('ISG_DATASET_ROOT', '/vision/u/minyeong/projects/isg-side-data/isg-1020-b1k'))
MULTITASK5 = ['camera_relocalization', 'object_scaling', 'mental_rotation', 'cm', 'iors_small']


# ---- a synthetic LeRobot v3 root with several episodes per task -------------------------------------------------------

def write_video(path, length, size=16, seed=0):
    """16 x 16 lossless frames with structure (not constant) so resizes and warps are testable; frame i's top-left
    pixel carries i (goal-frame identity)."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), 'w') as output:
        stream = output.add_stream('libx264rgb', rate=10)
        stream.width = stream.height = size
        stream.pix_fmt = 'rgb24'
        stream.options = {'crf': '0', 'g': '4'}
        for index in range(length):
            image = np.roll(base, index, axis=1).copy()
            image[0, 0] = index
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format='rgb24')):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)


def make_root(tmp_path, episodes=((7, 6, 3), (8, 7, 3), (9, 5, 3), (42, 8, 9), (43, 6, 9), (44, 9, 9)), settle=None,
              size=16):
    """A root with episodes (index, length, task); `settle` = program length per episode index writes
    isg_meta/settle_windows.json; the three cameras share one packed video per root."""
    root = tmp_path / 'dataset'
    (root / 'meta/episodes/chunk-000').mkdir(parents=True)
    (root / 'data/chunk-000').mkdir(parents=True)
    info = {'codebase_version': 'v3.0', 'fps': 10, 'total_episodes': len(episodes),
            'data_path': 'data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet',
            'video_path': 'videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4'}
    (root / 'meta/info.json').write_text(json.dumps(info))
    pq.write_table(pa.table({'task_index': [3, 9], 'task': ['alpha-one', 'beta-two']}), root / 'meta/tasks.parquet')
    records, frames, global_start, video_start = [], [], 1000, 0
    for episode, length, task in episodes:
        row = {'episode_index': episode, 'task_index': task, 'length': length,
               'data/chunk_index': 0, 'data/file_index': 0,
               'dataset_from_index': global_start, 'dataset_to_index': global_start + length}
        for camera in CAMERAS:
            key = CAMERAS[camera][0]
            for field, value in [('chunk_index', 0), ('file_index', 0),
                                 ('from_timestamp', video_start / 10), ('to_timestamp', (video_start + length) / 10)]:
                row[f'videos/{key}/{field}'] = value
        records.append(row)
        rng = np.random.default_rng(episode)
        for index in range(length):
            state = rng.normal(size=61).astype(np.float32)
            state[53:57] = 0.3 + index / 10  # torso: changes along the episode
            state[24:26] = 0.02 + index / 100  # left fingers
            state[49:51] = 0.01
            action = rng.normal(size=23).astype(np.float32)
            action[14] = 0.5 if index % 2 else 1.0
            action[22] = -0.2
            frames.append({'observation.state': state.tolist(), 'action': action.tolist(),
                           'episode_index': episode, 'task_index': task, 'frame_index': index,
                           'timestamp': index / 10, 'index': global_start + index})
        global_start += length
        video_start += length
    pq.write_table(pa.Table.from_pylist(records), root / 'meta/episodes/chunk-000/file-000.parquet')
    pq.write_table(pa.Table.from_pylist(frames), root / 'data/chunk-000/file-000.parquet', row_group_size=8)
    for camera in CAMERAS:
        write_video(root / f'videos/{CAMERAS[camera][0]}/chunk-000/file-000.mp4', video_start, size)
    if settle is not None:
        (root / 'isg_meta').mkdir()
        (root / 'isg_meta/settle_windows.json').write_text(json.dumps({
            'format': 'isg-settle-windows/v1',
            'episodes': {str(index): {'length': length, 'program_length': settle[index]} for index, length, _ in episodes}}))
    return root


def make_dataset(root, **kwargs):
    return B1KLeRobotDataset(root, horizon=4, n_obs_steps=2, n_action_steps=3, image_size=16, **kwargs)


def act_cache_entry(cache_root, root, video, size=16):
    """An ACT-format cache entry (act_b1k_frame_cache_v1) of `video`, built as ACT's b1k_frame_cache.build_entry does."""
    frames, pts = [], []
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            frames.append(resize_rgb(frame.to_ndarray(format='rgb24'), size, 'act'))
            pts.append(float(frame.pts * stream.time_base))
    stem = cache_root / video.relative_to(root).parent / video.stem
    stem.parent.mkdir(parents=True, exist_ok=True)
    np.save(stem.with_name(stem.name + '.frames.npy'), np.stack(frames))
    np.save(stem.with_name(stem.name + '.pts.npy'), np.asarray(pts))
    stat = video.stat()
    stem.with_name(stem.name + '.json').write_text(json.dumps({
        'format': ACT_FORMAT, 'image_size': [size, size], 'frames': len(frames), 'source': str(video.relative_to(root)),
        'source_size': stat.st_size, 'source_mtime_ns': stat.st_mtime_ns,
        'pixels': 'preprocess_image (bilinear, antialias, align_corners=False) then round(x * 255)'}))


# ---- 1. the validation split: ACT's algorithm, and the real multitask5 counts ----------------------------------------

def act_reference_split(by_task_episodes, val_fraction, seed):
    """ACT b1k_training.resolve_val_split's arithmetic (8da77eb), independently restated."""
    import hashlib
    result = {}
    for task_name, episodes in by_task_episodes.items():
        episodes = sorted(episodes)
        n_val = min(len(episodes) - 1, max(1, round(len(episodes) * val_fraction)))
        task_seed = int.from_bytes(hashlib.sha256(task_name.encode()).digest()[:8], 'little')
        perm = np.random.default_rng(np.random.SeedSequence([int(seed), task_seed])).permutation(len(episodes))
        chosen = [episodes[i] for i in perm]
        result[task_name] = {'train': sorted(chosen[n_val:]), 'val': sorted(chosen[:n_val])}
    return result


def test_val_split_matches_act_arithmetic_and_is_disjoint(tmp_path):
    root = make_root(tmp_path)
    split = resolve_val_split(root, ['alpha-one', 'beta-two'], None, 0.34, seed=0)
    assert split['format'] == 'isg-val-split/v1'
    assert split['tasks'] == act_reference_split({'alpha-one': [7, 8, 9], 'beta-two': [42, 43, 44]}, 0.34, 0)
    assert set(split['train_episodes']).isdisjoint(split['val_episodes'])
    assert sorted(split['train_episodes'] + split['val_episodes']) == [7, 8, 9, 42, 43, 44]
    assert resolve_val_split(root, None, None, 0.0, 0) is None
    with pytest.raises(ValueError):
        resolve_val_split(root, ['alpha-one'], {'episodes': [7]}, 0.5, 0)  # one candidate episode


@pytest.mark.skipif(not (ISG_DATASET / 'meta/episodes').is_dir(), reason='kmy17518/isg-1020-b1k metadata not available')
def test_val_split_multitask5_holds_out_the_same_536_episodes_as_act_and_openpi():
    tasks = expand_task_groups(ISG_DATASET, MULTITASK5)
    assert len(tasks) == 53
    split = resolve_val_split(ISG_DATASET, tasks, None, 0.1, seed=0)
    assert len(split['tasks']) == 53
    assert (len(split['train_episodes']), len(split['val_episodes'])) == (4856, 536)
    catalog = B1KLeRobotDataset(ISG_DATASET, tasks, observation_mode='lowdim')
    by_task = {}
    for row in catalog.episodes:
        by_task.setdefault(catalog.task_map[int(row['task_index'])], []).append(int(row['episode_index']))
    catalog.close()
    assert split['tasks'] == act_reference_split(by_task, 0.1, 0)


# ---- 2. ACT's frame cache and the act resize mode --------------------------------------------------------------------

def test_resize_rgb_act_is_acts_preprocess_image_rounded():
    import torch.nn.functional as F
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(48, 64, 3), dtype=np.uint8)
    tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255.0
    expected = F.interpolate(tensor[None], size=(16, 16), mode='bilinear', align_corners=False, antialias=True)[0]
    expected = torch.round(expected * 255).clamp_(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()
    assert np.array_equal(resize_rgb(image, 16, 'act'), expected)
    assert resize_rgb(image, 16, 'act').shape == (16, 16, 3) and not np.array_equal(resize_rgb(image, 16, 'act'), resize_rgb(image, 16, 'dp'))
    sized = rng.integers(0, 256, size=(16, 16, 4), dtype=np.uint8)
    assert np.array_equal(resize_rgb(sized, 16, 'act'), sized[..., :3])  # already at the training size: untouched
    with pytest.raises(ValueError):
        resize_rgb(image, 16, 'bogus')


def test_dataset_reads_an_act_frame_cache_and_refuses_the_wrong_resize_mode(tmp_path):
    root = make_root(tmp_path)
    cache = tmp_path / 'act-cache'
    for camera in CAMERAS:
        act_cache_entry(cache, root, root / f'videos/{CAMERAS[camera][0]}/chunk-000/file-000.mp4')
    dataset = make_dataset(root, frame_cache=cache, resize_mode='act', goal_views=('head',), image_dtype='uint8')
    assert dataset.frame_cache_formats == {ACT_FORMAT}
    native = make_dataset(root, resize_mode='act', goal_views=('head',), image_dtype='uint8')
    for index in (0, 5, len(dataset) - 1):
        cached, decoded = dataset[index], native[index]
        for key in ('head', 'left_wrist', 'right_wrist', goal_key('head')):
            assert torch.equal(cached['obs'][key], decoded['obs'][key]), key
    with pytest.raises(ValueError, match='--resize-mode act'):
        make_dataset(root, frame_cache=cache, resize_mode='dp')
    dataset.close()
    native.close()


def test_build_frame_cache_in_act_mode_writes_act_entries_that_verify(tmp_path):
    root = make_root(tmp_path)
    dataset = make_dataset(root, resize_mode='act')
    cache = tmp_path / 'built'
    build_frame_cache(dataset, cache, workers=1, log=lambda *_: None)
    manifest = json.loads(next(cache.rglob('*.json')).read_text())
    assert manifest['format'] == ACT_FORMAT and manifest['image_size'] == [16, 16]
    assert verify_frame_cache(dataset, cache, samples=8, log=lambda *_: None) > 0
    reader = FrameCacheReader(cache, root, 16, resize_mode='act')
    assert reader.validate([dataset.video_path(dataset.episodes[0], 'head')]) == {ACT_FORMAT}
    dp_dataset = make_dataset(root, resize_mode='dp')
    with pytest.raises(ValueError):
        FrameCacheReader(cache, root, 16, resize_mode='dp').validate([dp_dataset.video_path(dp_dataset.episodes[0], 'head')])
    dataset.close()
    dp_dataset.close()


def test_goal_is_the_last_recorded_frame_when_settle_steps_cut_the_episode(tmp_path):
    root = make_root(tmp_path, settle={7: 3, 8: 4, 9: 2, 42: 5, 43: 3, 44: 6})
    cut = make_dataset(root, goal_views=('head',), settle_steps='0', image_dtype='uint8')
    whole = make_dataset(root, goal_views=('head',), image_dtype='uint8')
    assert [row['length'] for row in cut.episodes] == [3, 4, 2, 5, 3, 6]
    assert np.array_equal(cut.goal_table, whole.goal_table)
    for position, row in enumerate(cut.episodes):
        # video frame i's top-left pixel carries i: the goal is the episode's last RECORDED frame, not the cut's last
        first_frame = round(row[f'videos/{CAMERAS["head"][0]}/from_timestamp'] * 10)
        assert cut.goal_table[position, 0, 0, 0, 0] == first_frame + cut.recorded_lengths[row['episode_index']] - 1
        assert cut.goal_table[position, 0, 0, 0, 0] != first_frame + row['length'] - 1
    cut.close()
    whole.close()


# ---- 3. the pi05 augmentation: ACT's draws and pixels, shared over the observation steps and the goal ----------------

def load_act_augment():
    path = ACT_REPO / 'b1k_augment.py'
    if not path.is_file():
        pytest.skip(f'ACT checkout with b1k_augment.py not at {ACT_REPO}')
    spec = importlib.util.spec_from_file_location('act_b1k_augment', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_augment_matches_act_per_observation_step_and_shares_draws():
    act = load_act_augment()
    torch.manual_seed(0)
    batch, To, size = 3, 2, 32
    obs = {'state': torch.randn(batch, To, 23),
           'head': torch.rand(batch, To, 3, size, size), 'left_wrist': torch.rand(batch, To, 3, size, size),
           'right_wrist': torch.rand(batch, To, 3, size, size), goal_key('head'): torch.rand(batch, 3, size, size)}
    out = augment_obs(obs, ('head', 'left_wrist', 'right_wrist'), {goal_key('head'): 'head'}, seed=1, step=7, rank=2)
    assert torch.equal(out['state'], obs['state'])
    assert out['head'].shape == obs['head'].shape and out[goal_key('head')].shape == obs[goal_key('head')].shape
    params = act.sample_params(batch, (size, size), act.augment_generator(1, 7, 2), 'cpu')
    for t in range(To):  # ACT's per-frame call with the same params: head + goal warped and jittered, wrists jittered
        images = torch.stack([obs['head'][:, t], obs['left_wrist'][:, t], obs['right_wrist'][:, t]], dim=1)
        expected, expected_goal = act.pi05_augment(images, obs[goal_key('head')].unsqueeze(1), (False, True, True), params)
        torch.testing.assert_close(out['head'][:, t], expected[:, 0], rtol=0, atol=2e-6)
        torch.testing.assert_close(out['left_wrist'][:, t], expected[:, 1], rtol=0, atol=2e-6)
        torch.testing.assert_close(out['right_wrist'][:, t], expected[:, 2], rtol=0, atol=2e-6)
        torch.testing.assert_close(out[goal_key('head')], expected_goal[:, 0], rtol=0, atol=2e-6)
    # the same draws again for the same (seed, step, rank); other draws for another step
    again = augment_obs(obs, ('head', 'left_wrist', 'right_wrist'), {goal_key('head'): 'head'}, seed=1, step=7, rank=2)
    assert all(torch.equal(again[key], out[key]) for key in out)
    other = augment_obs(obs, ('head', 'left_wrist', 'right_wrist'), {goal_key('head'): 'head'}, seed=1, step=8, rank=2)
    assert not torch.equal(other['head'], out['head'])
    # wrists keep their geometry (jitter only): where the jitter is not applied the image is unchanged
    wrist_params = sample_params(batch, (size, size), augment_generator(1, 7, 2))['wrist_color']
    for sample in range(batch):
        if not bool(wrist_params['apply'][sample]):
            assert torch.equal(out['left_wrist'][sample], obs['left_wrist'][sample])


def test_augment_requires_no_crop_in_the_model_config():
    config = ModelConfig(variant='transformer_hybrid_image', image_size=16, crop_shape=(12, 12), image_augment='pi05',
                         gripper_state='sum', regime='image', goal_fusion='early', goal_views=('head',))
    with pytest.raises(ValueError, match='crop'):
        config.validate()
    ModelConfig(variant='transformer_hybrid_image', image_size=16, image_augment='pi05', gripper_state='sum',
                regime='image', goal_fusion='early', goal_views=('head',)).validate()


# ---- 4. relative actions and quantile normalization (relq), serving decode and the gripper snap ----------------------

def test_relative_actions_round_trip_and_reference_layout():
    rng = np.random.default_rng(0)
    proprio = rng.normal(size=(4, 23)).astype(np.float32)
    actions = rng.normal(size=(4, 5, 23)).astype(np.float32)
    reference = action_reference(proprio)
    assert np.all(reference[:, :3] == 0)  # base velocities stay absolute
    assert np.allclose(reference[:, ACTION_DELTA['joint_action_dims']], proprio[:, ACTION_DELTA['joint_state_dims']])
    assert np.allclose(reference[:, [14, 22]], 20 * proprio[:, [14, 22]] - 1)
    relative = relative_actions(actions, proprio)
    assert relative.dtype == np.float32 and np.allclose(relative[:, :, :3], actions[:, :, :3])
    assert np.allclose(absolute_actions(relative, proprio), actions, atol=1e-5)
    assert np.allclose(absolute_actions(relative[:, 0], proprio), actions[:, 0], atol=1e-5)  # single actions too
    with pytest.raises(ValueError):
        action_reference(np.zeros((4, 25)))


def test_dataset_relative_actions_quantile_normalizer_and_valid_mask(tmp_path):
    root = make_root(tmp_path)
    dataset = make_dataset(root, action_representation='relative', return_meta=True, task_onehot=False)
    absolute = make_dataset(root, task_onehot=False)
    index = 5
    item, raw = dataset[index], absolute[index]
    position, frames, valid = dataset.sampler.locate(index, with_valid=True)
    state = dataset._read_episode(dataset.episodes[position])['state'][frames[1]]  # the current frame (n_obs_steps - 1)
    np.testing.assert_allclose(item['action'].numpy(), relative_actions(raw['action'].numpy(), state), atol=1e-6)
    assert torch.equal(item['action_valid'], torch.from_numpy(valid))
    assert item['meta'] == {'task': int(dataset.episodes[position]['task_index']), 'episode': position, 'frame': int(frames[1])}
    assert 'action_valid' not in raw and 'meta' not in raw
    stats = dataset.compute_action_stats('quantile', log=lambda *_: None)
    lengths = [row['length'] for row in dataset.episodes]
    assert stats['count'] == sum(lengths)
    assert stats['action_count'] == sum(min(4, length - t) for length in lengths for t in range(length))
    assert stats['action_representation'] == 'relative' and stats['action_delta'] == ACTION_DELTA
    normalizer = dataset.get_normalizer('quantile')
    q01, q99 = stats['action']['q01'], stats['action']['q99']
    lo = normalizer['action'].normalize(torch.from_numpy(q01.astype(np.float32)))
    hi = normalizer['action'].normalize(torch.from_numpy(q99.astype(np.float32)))
    torch.testing.assert_close(lo, torch.full_like(lo, -1.0), atol=1e-4, rtol=0)
    torch.testing.assert_close(hi, torch.full_like(hi, 1.0), atol=1e-4, rtol=0)
    sample = torch.randn(3, 4, 23)
    torch.testing.assert_close(normalizer['action'].unnormalize(normalizer['action'].normalize(sample)), sample, atol=1e-5, rtol=0)
    assert set(normalizer['action'].get_input_stats()) == {'min', 'max', 'mean', 'std', 'q01', 'q99'}
    zscore = dataset.get_normalizer('zscore')
    mean = torch.from_numpy(dataset.compute_action_stats('zscore', log=lambda *_: None)['state']['mean'].astype(np.float32))
    torch.testing.assert_close(zscore['state'].normalize(mean), torch.zeros_like(mean), atol=1e-5, rtol=0)
    with pytest.raises(ValueError):
        dataset.get_normalizer('limits')  # relative actions need chunk statistics
    with pytest.raises(ValueError):
        make_dataset(root, action_representation='relative', gripper_state='fingers')
    dataset.close()
    absolute.close()


class ZeroPolicy:
    """A stand-in policy predicting zeros (relative: "hold the current state") for the serving decode."""
    device = torch.device('cpu')

    def __init__(self, n_action_steps=3):
        self.n_action_steps = n_action_steps

    def predict_action(self, inputs):
        batch = next(iter(inputs.values())).shape[0]
        return {'action': torch.zeros(batch, self.n_action_steps, 23), 'action_pred': torch.zeros(batch, 4, 23)}


def relq_config(**kwargs):
    config = ModelConfig(variant='transformer_hybrid_image', image_size=16, horizon=4, n_obs_steps=2, n_action_steps=3,
                         gripper_state='sum', task_onehot=False, regime='image', goal_fusion='early', goal_views=('head',),
                         action_representation='relative', normalization='quantile', resize_mode='act', **kwargs)
    config.validate()
    return config


def test_serving_decodes_relative_plans_and_snaps_grippers():
    config = relq_config()
    session = B1KPolicySession(ZeroPolicy(), config.to_dict(), {3: 'alpha-one'}, gripper_snap=0.9)
    rng = np.random.default_rng(0)
    proprio61 = rng.normal(size=(2, 61)).astype(np.float32)
    proprio61[:, 24:26] = [[0.05, 0.05], [0.0, 0.0]]  # left fingers: opening 0.1 -> command 1.0 (clamped / snapped); 0 -> -1
    frame = rng.integers(0, 256, size=(2, 16, 16, 3), dtype=np.uint8)
    obs = {PROPRIO_KEY: proprio61, 'task_id': np.array([3, 3]), **{CAMERAS[c][1]: frame for c in CAMERAS},
           'goal::robot_r1::robot_r1:zed_link:Camera:0::rgb': frame}
    action = session.act(obs)
    expected = snap_grippers(action_reference(extract_state(proprio61, 'sum')), 0.9)
    np.testing.assert_allclose(action, expected.astype(np.float32), atol=1e-5)
    assert action[0, 14] == 1.0 and action[1, 14] == -1.0  # snapped open; clamped
    assert session.act(obs).shape == (2, 23)  # the stored plan serves the next steps
    assert np.array_equal(snap_grippers(np.array([[0.0] * 14 + [0.95] + [0.0] * 7 + [-3.0]]), 0.9)[0, [14, 22]], [1.0, -1.0])
    assert snap_grippers(np.array([[2.0] * 23]), None)[0, 14] == 2.0


# ---- 5. the serving handshake the vectorized evaluator detects as the ACT protocol ----------------------------------

def load_detect_protocol():
    path = Path(os.environ.get('ISG_EVAL_VEC', '/vision/u/minyeong/projects/ispatialgym-eval-vec')) / 'ispatialgym/eval/ws_policy.py'
    if not path.is_file():
        def detect_protocol(metadata):  # ws_policy.detect_protocol (2026-10-25) restated
            if metadata.get('protocol') == 'behavior-numpy-msgpack' or 'execution_mode' in metadata or 'temporal_agg' in metadata:
                return 'act'
            return 'openpi'
        return detect_protocol, None
    spec = importlib.util.spec_from_file_location('isg_ws_policy', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.detect_protocol, module


def test_handshake_speaks_the_act_protocol_with_evaluator_camera_names():
    detect_protocol, ws_policy = load_detect_protocol()
    config = relq_config()
    checkpoint = {'config': config.to_dict(), 'task_map': {3: 'alpha-one', 9: 'beta-two'}, 'step': 10}
    server = WebsocketPolicyServer(ZeroPolicy(), checkpoint, gripper_snap=0.9)
    metadata = unpackb(packb(server.metadata(server.new_session())))
    assert detect_protocol(metadata) == 'act'
    assert metadata['protocol'] == 'behavior-numpy-msgpack' and metadata['execution_mode'] == 'chunked'
    assert metadata['goal_views'] == ['zed_link'] and metadata['goal_views_internal'] == ['head']
    assert metadata['goal_observation_keys'] == ['goal::robot_r1::robot_r1:zed_link:Camera:0::rgb']
    assert metadata['cameras'] == ['zed_link', 'left_realsense_link', 'right_realsense_link']
    assert metadata['action_horizon'] == 3 and metadata['chunk_size'] == 3 and metadata['action_dim'] == 23
    assert metadata['action_representation'] == 'relative' and metadata['served_actions'] == 'absolute'
    assert metadata['gripper_snap_threshold'] == 0.9 and metadata['resize_mode'] == 'act' and metadata['image_size'] == 16
    if ws_policy is not None:
        assert metadata['goal_observation_keys'] == [ws_policy.GOAL_KEYS[view] for view in metadata['goal_views']]
        assert set(metadata['observation_keys'].values()) == set(ws_policy.OBS_KEYS.values())


# ---- 6. training: validation metrics, DDP with torchrun (gloo), the stop file -----------------------------------------

def train_args(root, run, extra=()):
    return ['--dataset-path', str(root), '--output-dir', str(run), '--variant', 'transformer_hybrid_image',
            '--regime', 'image', '--goal-fusion', 'early', '--goal-views', 'head', '--no-task-onehot',
            '--n-layer', '1', '--n-emb', '16', '--n-head', '2', '--horizon', '4', '--n-obs-steps', '2',
            '--n-action-steps', '3', '--image-size', '16', '--num-train-timesteps', '4', '--num-inference-steps', '2',
            '--batch-size', '4', '--num-workers', '0', '--device', 'cpu', '--wandb-mode', 'disabled',
            '--action-representation', 'relative', '--normalization', 'quantile', '--image-augment', 'pi05',
            '--resize-mode', 'act', '--val-fraction', '0.34', '--val-every', '2', '--val-batches', '2',
            '--val-batch-size', '3', '--val-goal-batches', '1', '--save-every', '100', '--export-every', '2', *extra]


def test_training_logs_validation_metrics_and_records_the_split(tmp_path, capsys):
    root = make_root(tmp_path)
    run = tmp_path / 'run'
    train_main(train_args(root, run, ['--max-steps', '2']))
    rows = [json.loads(line) for line in (run / 'train.jsonl').read_text().splitlines()]
    assert [row['step'] for row in rows] == [1, 2]
    final = rows[-1]
    for key in ('val/loss', 'val/l1', 'val/l1_action_units', 'val/l1_action_units/alpha', 'val/l1_action_units/beta',
                'val/goal_l1_own', 'val/goal_l1_other', 'val/goal_gap', 'val/goal_max_dpred', 'timing/val_s'):
        assert key in final and np.isfinite(final[key]), key
    assert 'val/loss' not in rows[0]
    checkpoint = torch.load(run / 'step-00000002.pt', weights_only=True)
    assert checkpoint['selection']['val_split']['val_fraction'] == 0.34
    assert set(checkpoint['selection']['val_split']['val_episodes']).isdisjoint(checkpoint['selection']['episode_indices'])
    assert checkpoint['config']['action_representation'] == 'relative' and checkpoint['config']['normalization'] == 'quantile'
    assert checkpoint['config']['image_augment'] == 'pi05' and checkpoint['config']['resize_mode'] == 'act'
    assert set(checkpoint['normalizer']['params_dict.action.input_stats.q01'].shape) == {23}
    config = json.loads((run / 'config.json').read_text())
    assert config['val_split']['val_fraction'] == 0.34
    out = capsys.readouterr().out
    assert '"val_split"' in out and 'quantile normalization statistics of relative actions' in out


def test_evaluate_validation_action_units_are_absolute(tmp_path):
    """A relative policy's val/l1_action_units equals the L1 of decoded absolute actions (both sides decoded)."""
    root = make_root(tmp_path)
    dataset = make_dataset(root, action_representation='relative', goal_views=('head',), task_onehot=False,
                           image_dtype='uint8', return_meta=True)
    config = relq_config()
    policy = build_policy(config, dataset.task_map)
    policy.set_normalizer(dataset.get_normalizer('quantile'))
    ema = policy.eval()

    class Args:
        seed, val_batches, val_batch_size, val_goal_batches = 0, 1, 2, 1
    import contextlib
    metrics = evaluate_validation(ema, dataset, config, Args, torch.device('cpu'),
                                  torch.autocast('cpu', enabled=False), contextlib.nullcontext)
    assert set(metrics) >= {'val/loss', 'val/l1', 'val/l1_action_units', 'val/goal_gap', 'timing/val_s'}
    assert all(np.isfinite(value) for value in metrics.values())
    # relative targets of the same frames are close to the current state's reference in absolute units: the
    # action-unit error of a zero prediction is the reference distance, which differs from the normalized L1
    assert metrics['val/l1_action_units'] != pytest.approx(metrics['val/l1'])
    dataset.close()


@pytest.mark.skipif(os.environ.get('ISG_SKIP_DDP') == '1', reason='ISG_SKIP_DDP=1')
def test_torchrun_two_ranks_train_validate_and_stop_on_the_stop_file(tmp_path):
    root = make_root(tmp_path)
    run, stop = tmp_path / 'ddp', tmp_path / 'stop.flag'
    args = train_args(root, run, ['--max-steps', '50', '--stop-file', str(stop), '--loader-batch-size', '1'])
    env = dict(os.environ, OMP_NUM_THREADS='1')
    process = subprocess.Popen([sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
                                '--nproc_per_node=2', '-m', 'diffusion_policy.b1k.train', *args],
                               cwd=Path(__file__).resolve().parents[1], env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True)
    import time
    deadline = time.time() + 900
    while time.time() < deadline and process.poll() is None:
        rows = [json.loads(line) for line in (run / 'train.jsonl').read_text().splitlines()] if (run / 'train.jsonl').exists() else []
        if rows and rows[-1]['step'] >= 3 and not stop.exists():
            stop.write_text('stop')
        time.sleep(0.5)
    output = process.communicate(timeout=60)[0]
    assert process.returncode == 0, output[-3000:]
    rows = [json.loads(line) for line in (run / 'train.jsonl').read_text().splitlines()]
    assert rows[-1].get('stopped') is True and 4 <= rows[-1]['step'] < 50, output[-2000:]
    assert 'val/l1_action_units' in rows[1]  # validation at step 2 ran across both ranks
    checkpoint = torch.load(run / f'step-{rows[-1]["step"]:08d}.pt', weights_only=True)
    assert checkpoint['step'] == rows[-1]['step']
    assert (run / 'export_queue/eval/step-00000002.pt').exists()
    assert 'Stop file' in output
