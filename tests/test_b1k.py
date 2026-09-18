import asyncio
import copy
import json
import pickle
from pathlib import Path

import av
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import zarr

from diffusion_policy.b1k.dataset import B1KLeRobotDataset, EpisodeSequenceIndex, VideoReader, images_to_float
from diffusion_policy.b1k.frame_cache import FrameCacheReader, build_frame_cache, verify_frame_cache
from diffusion_policy.b1k.model import ModelConfig, build_policy, load_policy
from diffusion_policy.b1k.normalization import fit_normalizer
from diffusion_policy.b1k.robot import CAMERAS, PROPRIO_KEY, STATE_INDICES, extract_state, resize_rgb
from diffusion_policy.b1k.serve import B1KPolicySession, WebsocketPolicyServer, packb, unpackb
from diffusion_policy.b1k.train import StepBatchSampler, main as train_main, parser as train_parser
from diffusion_policy.b1k.variant_matrix import matrix_cases
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler
from diffusion_policy.model.common.normalizer import LinearNormalizer


torch.set_num_threads(1)


def write_video(path, length):
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), 'w') as output:
        stream = output.add_stream('libx264rgb', rate=10)
        stream.width = stream.height = 16
        stream.pix_fmt = 'rgb24'
        stream.options = {'crf': '0', 'g': '4'}
        for index in range(length):
            image = np.full((16, 16, 3), index, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format='rgb24')
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)


@pytest.fixture
def root(tmp_path):
    root = tmp_path / 'dataset'
    (root / 'meta/episodes/chunk-004').mkdir(parents=True)
    (root / 'data/chunk-004').mkdir(parents=True)
    info = {'codebase_version': 'v3.0', 'fps': 10, 'total_episodes': 999,
            'data_path': 'data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet',
            'video_path': 'videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4'}
    (root / 'meta/info.json').write_text(json.dumps(info))
    pq.write_table(pa.table({'task_index': [3, 9, 77], 'task': ['alpha', 'beta', 'absent']}), root / 'meta/tasks.parquet')
    records, frames = [], []
    global_start = 1000
    for episode, length, task in [(7, 5, 3), (42, 8, 9)]:
        row = {'episode_index': episode, 'task_index': task, 'length': length,
               'data/chunk_index': 4, 'data/file_index': 0,
               'dataset_from_index': global_start, 'dataset_to_index': global_start + length}
        for camera_index, camera in enumerate(CAMERAS):
            key = CAMERAS[camera][0]
            start = 5 + camera_index * 11 + (0 if episode == 7 else 9)
            for field, value in [('chunk_index', 4), ('file_index', 0),
                                 ('from_timestamp', start / 10), ('to_timestamp', (start + length) / 10)]:
                row[f'videos/{key}/{field}'] = value
        records.append(row)
        for index in range(length):
            state = np.arange(61, dtype=np.float32) + episode + index / 10
            action = np.arange(23, dtype=np.float32) + episode + index
            action[6] = 0
            frames.append({'observation.state': state.tolist(), 'action': action.tolist(),
                           'episode_index': episode, 'task_index': task, 'frame_index': index,
                           'timestamp': index / 10, 'index': global_start + index})
        global_start += length
    pq.write_table(pa.Table.from_pylist(records), root / 'meta/episodes/chunk-004/file-000.parquet')
    pq.write_table(pa.Table.from_pylist(frames), root / 'data/chunk-004/file-000.parquet', row_group_size=6)
    for camera in CAMERAS:
        key = CAMERAS[camera][0]
        write_video(root / f'videos/{key}/chunk-004/file-000.mp4', 50)
    return root


def make_dataset(root, **kwargs):
    return B1KLeRobotDataset(root, horizon=4, n_obs_steps=2, n_action_steps=3,
                             image_size=16, **kwargs)


@pytest.mark.parametrize('lengths', [[1, 2, 6], [4, 7], [13]])
@pytest.mark.parametrize('horizon,pad_before,pad_after', [(1, 0, 0), (4, 0, 0), (4, 1, 2), (4, 9, 9), (8, 2, 7)])
def test_index_equivalence(lengths, horizon, pad_before, pad_after):
    replay = ReplayBuffer.create_empty_zarr()
    for episode, length in enumerate(lengths):
        replay.add_episode({'state': (episode * 100 + np.arange(length))[:, None].astype(np.float32)})
    upstream = SequenceSampler(replay, horizon, pad_before, pad_after)
    lazy = EpisodeSequenceIndex(lengths, horizon, pad_before, pad_after)
    assert len(lazy) == len(upstream)
    for index in range(len(lazy)):
        episode, frames = lazy.locate(index)
        expected = upstream.sample_sequence(index)['state']
        np.testing.assert_array_equal(expected[:, 0], episode * 100 + frames)
    with pytest.raises(IndexError):
        lazy.locate(len(lazy))


def test_native_partial_root_and_boundaries(root):
    dataset = make_dataset(root)
    assert [row['episode_index'] for row in dataset.episodes] == [7, 42]
    assert dataset.task_map == {3: 'alpha', 9: 'beta'}
    assert not dataset._episodes and not dataset._row_groups
    for index in range(len(dataset)):
        episode_position, frames = dataset.sampler.locate(index)
        episode = dataset.episodes[episode_position]
        sample = dataset[index]
        assert sample['action'].shape == (4, 23)
        for camera_index, camera in enumerate(CAMERAS):
            offset = 5 + camera_index * 11 + (0 if episode['episode_index'] == 7 else 9)
            expected = offset + frames[:2]
            np.testing.assert_allclose(sample['obs'][camera][:, 0, 0, 0] * 255, expected, atol=0.01)
        expected_state = np.arange(61)[STATE_INDICES] + episode['episode_index'] + frames[:2, None] / 10
        np.testing.assert_allclose(sample['obs']['state'][:, :25], expected_state, rtol=1e-6)
        assert torch.all(sample['action'][:, 6] == 0)
        # Position n_obs_steps-1 is the current control timestep.
        assert sample['action'][1, 0] == episode['episode_index'] + frames[1]
    dataset.close()


def test_selection_caches_and_pickle(root):
    dataset = make_dataset(root, task_names=['beta'], cameras=['left_wrist'], episode_cache_size=1)
    assert len(dataset.episodes) == 1 and dataset.task_map == {9: 'beta'}
    sample = dataset[0]
    assert set(sample['obs']) == {'state', 'left_wrist'}
    assert len(dataset._video.frames) == 1
    restored = pickle.loads(pickle.dumps(dataset))
    assert not restored._episodes and not restored._video.frames
    torch.testing.assert_close(sample['action'], restored[0]['action'])
    with pytest.raises(ValueError, match='Unknown task'):
        make_dataset(root, task_names=['typo'])
    with pytest.raises(ValueError, match='No episodes'):
        make_dataset(root, task_names=['absent'])
    with pytest.raises(ValueError, match='remove a requested task'):
        make_dataset(root, task_names=['alpha', 'beta'], max_episodes=1)


def test_missing_camera_and_data_fail(root):
    (root / 'videos' / CAMERAS['head'][0] / 'chunk-004/file-000.mp4').unlink()
    with pytest.raises(FileNotFoundError, match='camera missing'):
        make_dataset(root)
    assert len(make_dataset(root, cameras=['left_wrist'])) > 0
    (root / 'data/chunk-004/file-000.parquet').unlink()
    with pytest.raises(FileNotFoundError, match='Selected episode'):
        make_dataset(root, task_names=['alpha'], cameras=['left_wrist'])


def test_video_tolerance_and_lru(root):
    dataset = make_dataset(root)
    reader = VideoReader(16, cache_frames=2, max_open=1)
    head = dataset.video_path(dataset.episodes[0], 'head')
    images = reader.read(head, [0.5, 0.6, 0.7])
    assert images[:, 0, 0, 0].tolist() == [5, 6, 7]
    assert len(reader.frames) == 2
    assert reader.read(head, [1.5])[0, 0, 0, 0] == 15
    assert reader.read(head, [0.2])[0, 0, 0, 0] == 2
    with pytest.raises(ValueError, match='timestamps not found'):
        reader.read(head, [0.555])
    reader.read(dataset.video_path(dataset.episodes[0], 'left_wrist'), [0.5])
    assert len(reader.containers) == 1
    reader.close()


def test_hdf5_ingestion_and_default_normalization(root, tmp_path):
    from diffusion_policy.dataset.robomimic_replay_image_dataset import RobomimicReplayImageDataset
    dataset = make_dataset(root, cameras=['head'])
    hdf5_path = tmp_path / 'default.hdf5'
    with h5py.File(hdf5_path, 'w') as output:
        for i, row in enumerate(dataset.episodes):
            data = dataset._read_episode(row)
            group = output.create_group(f'data/demo_{i}')
            group['actions'] = data['action']
            group['obs/robot_qpos'] = data['state']
            group['obs/head'] = dataset._video.read(dataset.video_path(row, 'head'),
                                                    data['timestamp'] + row[f'videos/{CAMERAS["head"][0]}/from_timestamp'])
    shape_meta = {'action': {'shape': [23]}, 'obs': {
        'robot_qpos': {'shape': [25], 'type': 'low_dim'},
        'head': {'shape': [3, 16, 16], 'type': 'rgb'}}}
    upstream = RobomimicReplayImageDataset(shape_meta, str(hdf5_path), horizon=4,
                                          n_obs_steps=2, pad_before=1, pad_after=2)
    assert len(dataset) == len(upstream)
    for index in range(len(dataset)):
        native, default = dataset[index], upstream[index]
        torch.testing.assert_close(native['action'], default['action'])
        torch.testing.assert_close(native['obs']['state'][:, :25], default['obs']['robot_qpos'])
        torch.testing.assert_close(native['obs']['head'], default['obs']['head'], atol=1 / 255, rtol=0)
    normalizer = upstream.get_normalizer()
    action = upstream[0]['action']
    torch.testing.assert_close(normalizer['action'].normalize(action), action)
    torch.testing.assert_close(normalizer['head'].normalize(torch.tensor([0., 1.])), torch.tensor([-1., 1.]))


def test_streaming_limits_match_zarr_and_no_clipping(root):
    dataset = make_dataset(root)
    batches = list(dataset.iter_lowdim(batch_size=3))
    normalizer = fit_normalizer(iter(batches), dataset.cameras)
    actions = np.concatenate([batch['action'] for batch in batches])
    state = np.concatenate([batch['state'] for batch in batches])
    upstream = LinearNormalizer()
    upstream.fit({'action': zarr.array(actions), 'state': zarr.array(state[:, :25])}, mode='limits')
    for key, values in [('action', actions), ('state', state)]:
        nvalues = normalizer[key].normalize(values)
        reference = upstream[key].normalize(values if key == 'action' else values[:, :25])
        torch.testing.assert_close(nvalues if key == 'action' else nvalues[:, :25], reference)
        for field in ('min', 'max', 'mean', 'std'):
            actual = normalizer[key].get_input_stats()[field]
            if key == 'state':
                actual = actual[:25]
            torch.testing.assert_close(actual, upstream[key].get_input_stats()[field], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(normalizer[key].unnormalize(nvalues), torch.from_numpy(values), rtol=1e-5, atol=1e-5)
    assert torch.all(normalizer['action'].normalize(actions)[:, 6] == 0)
    assert normalizer['action'].normalize(actions + 100).max() > 1
    torch.testing.assert_close(normalizer['state'].normalize(state)[:, 25:], torch.from_numpy(state[:, 25:]))


@pytest.mark.parametrize('scheduler', ['ddpm', 'ddim'])
def test_actual_policy_backward_and_inference(root, scheduler):
    dataset = make_dataset(root, cameras=['head'])
    config = ModelConfig(horizon=4, n_obs_steps=2, n_action_steps=3, cameras=('head',), image_size=16,
                         down_dims=(16, 32), diffusion_step_embed_dim=16, num_train_timesteps=4,
                         num_inference_steps=2, scheduler=scheduler)
    policy = build_policy(config, dataset.task_map)
    policy.set_normalizer(dataset.get_normalizer())
    sample = dataset[0]
    batch = {'obs': {key: value[None] for key, value in sample['obs'].items()}, 'action': sample['action'][None]}
    loss = policy.compute_loss(batch)
    loss.backward()
    assert np.isfinite(loss.item())
    assert any(p.grad is not None for p in policy.model.parameters())
    assert any(p.grad is not None for p in policy.obs_encoder.parameters())
    policy.eval()
    result = policy.predict_action(batch['obs'])
    torch.testing.assert_close(result['action'], result['action_pred'][:, 1:4])
    assert policy.noise_scheduler.config.clip_sample


@pytest.mark.parametrize('scheduler', ['ddpm', 'ddim'])
def test_scheduler_clipping_is_not_normalizer_clipping(scheduler):
    from diffusers import DDPMScheduler, DDIMScheduler
    cls = DDPMScheduler if scheduler == 'ddpm' else DDIMScheduler
    for clip in (True, False):
        noise = cls(num_train_timesteps=4, clip_sample=clip)
        noise.set_timesteps(4)
        result = noise.step(torch.zeros(1, 4, 23), noise.timesteps[0], torch.full((1, 4, 23), 10.))
        assert bool((result.pred_original_sample.abs() <= 1).all()) == clip


def test_near_constant_limits():
    values = np.zeros((3, 26), dtype=np.float32)
    values[:, 0] = [7, 7 + 1e-6, 7 + 2e-6]
    values[:, -1] = 1
    actions = values[:, :23].copy()
    normalizer = fit_normalizer([{'state': values, 'action': actions}], [])
    upstream = LinearNormalizer()
    upstream.fit(actions)
    torch.testing.assert_close(normalizer['action'].normalize(actions), upstream.normalize(actions))
    assert normalizer['action'].params_dict['scale'][0] == 1


class FakePolicy:
    device = torch.device('cpu')

    def __init__(self):
        self.calls = []

    def predict_action(self, obs):
        self.calls.append(copy.deepcopy(obs))
        actions = obs['state'][:, -1, :1, None].expand(-1, 3, 23).clone()
        actions += torch.arange(3)[None, :, None]
        return {'action': actions}


def observation(value, task=3, batch=1, rgba=False):
    return {PROPRIO_KEY: np.full((batch, 61), value, dtype=np.float32),
            CAMERAS['head'][1]: np.full((batch, 12, 16, 4 if rgba else 3), 128, dtype=np.uint8),
            'task_id': task}


def session(policy=None):
    return B1KPolicySession(policy or FakePolicy(), {'n_obs_steps': 2, 'n_action_steps': 3,
                                                    'cameras': ['head'], 'image_size': 16},
                            {3: 'alpha', 9: 'beta'}, action_horizon=2)


def test_session_history_replanning_reset_task_and_batch():
    wrapper = session()
    assert wrapper.act(observation(1))[0, 0] == 1
    assert wrapper.act(observation(2, rgba=True))[0, 0] == 2
    assert len(wrapper.policy.calls) == 1
    wrapper.act(observation(3))
    assert wrapper.policy.calls[-1]['state'][0, :, 0].tolist() == [2, 3]
    wrapper.act(observation(4, task=9))
    assert wrapper.policy.calls[-1]['state'][0, :, 0].tolist() == [4, 4]
    assert wrapper.act(observation(5, task=[3, 9], batch=2)).shape == (2, 23)
    wrapper.act(observation(6, task=[9, 9], batch=2))
    assert wrapper.policy.calls[-1]['state'].shape[0] == 1
    wrapper.reset()
    wrapper.act(observation(7))
    assert wrapper.policy.calls[-1]['state'][0, :, 0].tolist() == [7, 7]


@pytest.mark.parametrize('task', [999, 3.5, [3, 9, 3], '3'])
def test_session_rejects_task_ids(task):
    with pytest.raises(ValueError):
        session().act(observation(1, task=task))


def test_single_task_default_and_multitask_requirement():
    obs = observation(1)
    del obs['task_id']
    with pytest.raises(ValueError, match='requires task_id'):
        session().act(obs)
    wrapper = B1KPolicySession(FakePolicy(), session().config, {3: 'alpha'})
    assert wrapper.act(obs).shape == (1, 23)
    with pytest.raises(ValueError, match='Unknown task_id'):
        wrapper.act(observation(1, task=9))


def test_messagepack_roundtrip_and_rejection():
    data = {'array': np.arange(12, dtype=np.float32).reshape(3, 4), 'scalar': np.int64(9)}
    result = unpackb(packb(data))
    np.testing.assert_array_equal(result['array'], data['array'])
    assert result['scalar'] == 9
    with pytest.raises(ValueError, match='Unsupported dtype'):
        packb(np.array([object()], dtype=object))


def test_rgb_rgba_padding_and_state_mapping():
    image = np.full((8, 16, 4), 255, dtype=np.uint8)
    resized = resize_rgb(image, 16)
    assert resized.shape == (16, 16, 3)
    assert not resized[:4].any() and (resized[4:12] == 255).all()
    np.testing.assert_array_equal(extract_state(np.arange(61)), STATE_INDICES)
    with pytest.raises(ValueError):
        resize_rgb(image.astype(np.float32), 16)


def test_bounded_sampler_resume():
    all_batches = list(StepBatchSampler(210916774, 2, 0, 5, 42))
    assert all_batches[3:] == list(StepBatchSampler(210916774, 2, 3, 5, 42))
    assert all(len(batch) == 2 for batch in all_batches)


def test_train_resume_self_contained(root, tmp_path):
    output = tmp_path / 'run'
    args = ['--dataset-root', str(root), '--output-dir', str(output), '--device', 'cpu',
            '--num-workers', '0', '--batch-size', '1', '--cpu-threads', '1', '--task-onehot',
            '--horizon', '4', '--n-action-steps', '3', '--image-size', '16', '--cameras', 'head',
            '--down-dims', '16', '32', '--diffusion-step-embed-dim', '16',
            '--num-train-timesteps', '4', '--num-inference-steps', '2']
    train_main(args + ['--max-steps', '2'])
    train_main(args + ['--max-steps', '3', '--resume', str(output / 'latest.pt')])
    uninterrupted = tmp_path / 'uninterrupted'
    full_args = args.copy()
    full_args[full_args.index('--output-dir') + 1] = str(uninterrupted)
    train_main(full_args + ['--max-steps', '3'])
    resumed_state = torch.load(output / 'latest.pt', weights_only=True)
    full_state = torch.load(uninterrupted / 'latest.pt', weights_only=True)
    for field in ('model', 'ema_model'):
        for key in resumed_state[field]:
            torch.testing.assert_close(resumed_state[field][key], full_state[field][key], rtol=0, atol=0)
    with pytest.raises(FileExistsError, match='nonempty'):
        train_main(args + ['--max-steps', '3'])
    metadata = root / 'meta/info.json'
    info = json.loads(metadata.read_text())
    info['fps'] = 20
    metadata.write_text(json.dumps(info))
    with pytest.raises(ValueError, match='fingerprint'):
        train_main(args + ['--max-steps', '4', '--resume', str(output / 'latest.pt')])
    root.rename(root.with_name('dataset-unavailable'))
    policy, checkpoint = load_policy(output)
    assert checkpoint['step'] == checkpoint['ema_step'] == 3
    assert checkpoint['optimizer']['state']
    assert set(checkpoint['task_map']) == {3, 9}
    wrapper = B1KPolicySession(policy, checkpoint['config'], checkpoint['task_map'])
    assert wrapper.act(observation(1)).shape == (1, 23)


def test_train_rejects_dataset_output(root):
    with pytest.raises(ValueError, match='read-only dataset'):
        train_main(['--dataset-path', str(root), '--output-dir', str(root / 'forbidden'), '--max-steps', '1'])
    assert not (root / 'forbidden').exists()


@pytest.mark.parametrize('kwargs', [{'down_dims': ()}, {'down_dims': (16,)}, {'down_dims': (15, 32)},
                                   {'horizon': 15}, {'n_groups': 0}, {'num_inference_steps': 101}])
def test_invalid_model_configuration(kwargs):
    with pytest.raises(ValueError):
        build_policy(ModelConfig(**kwargs), {3: 'alpha'})


def test_spawn_loader(root):
    from torch.utils.data import DataLoader
    dataset = make_dataset(root, cameras=['head'])
    sample = next(iter(DataLoader(dataset, batch_size=2, num_workers=1, multiprocessing_context='spawn')))
    assert sample['obs']['head'].shape == (2, 2, 3, 16, 16)


def test_real_websocket_protocol_and_isolation():
    from websockets.asyncio.client import connect
    from websockets.asyncio.server import serve
    from diffusion_policy.b1k.serve import health_check

    async def exercise():
        policy = FakePolicy()
        server = WebsocketPolicyServer(policy, {'config': session().config, 'task_map': {3: 'alpha', 9: 'beta'}},
                                       action_horizon=2)
        server.inference_lock = asyncio.Lock()
        async with serve(server.handler, '127.0.0.1', 0, process_request=health_check) as running:
            port = running.sockets[0].getsockname()[1]
            reader, writer = await asyncio.open_connection('127.0.0.1', port)
            writer.write(b'GET /healthz HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n')
            await writer.drain()
            assert b'200 OK' in await reader.read()
            writer.close()
            await writer.wait_closed()
            async with connect(f'ws://127.0.0.1:{port}') as first, connect(f'ws://127.0.0.1:{port}') as second:
                import msgpack
                from diffusion_policy.b1k.serve import unpack_array
                handshake = msgpack.unpackb(await first.recv(), object_hook=unpack_array)
                assert handshake['action_dim'] == 23
                assert set(handshake['task_map']) == {'3', '9'}
                await second.recv()
                for client, value in [(first, 1), (second, 10), (first, 2), (second, 20), (first, 3)]:
                    await client.send(packb(observation(value)))
                    action = unpackb(await client.recv())['action']
                    assert action.shape == (1, 23) and action.dtype == np.float32
                assert policy.calls[-1]['state'][0, :, 0].tolist() == [2, 3]
                await first.send(packb({'reset': True}))
                with pytest.raises(asyncio.TimeoutError):
                    await asyncio.wait_for(first.recv(), timeout=0.05)
                await first.send(packb(observation(30, task=[3, 9], batch=2)))
                assert unpackb(await first.recv())['action'].shape == (2, 23)
    asyncio.run(exercise())


@pytest.mark.parametrize('case', [name for name in matrix_cases() if not name.startswith('unet_video')])
def test_variant_branches_real_classes(root, case):
    from dataclasses import replace
    from diffusion_policy.b1k.model import POLICY_TARGETS
    config = matrix_cases()[case]
    # Network initialization is audited separately on real B1K data.
    config = replace(config, encoder_weights=None)
    dataset = B1KLeRobotDataset(root, **config.dataset_kwargs())
    policy = build_policy(config, dataset.task_map)
    assert type(policy).__name__ == POLICY_TARGETS[config.variant].rsplit('.', 1)[-1]
    policy.set_normalizer(dataset.get_normalizer())
    sample = dataset[1]
    obs = sample['obs']
    if config.lowdim:
        assert obs.shape == (config.obs_steps, 27)
        obs = obs[None].repeat(2, 1, 1)
    else:
        assert obs['state'].shape == (config.obs_steps, 27)
        obs = {key: value[None].repeat(2, *([1] * value.ndim)) for key, value in obs.items()}
    loss = policy.compute_loss({'obs': obs, 'action': sample['action'][None].repeat(2, 1, 1)})
    loss.backward()
    assert torch.isfinite(loss)
    gradients = [parameter.grad for parameter in policy.model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(grad).all() for grad in gradients)
    if case == 'unet_image-trainable-batchnorm-ddpm':
        from diffusion_policy.b1k.train import sync_batchnorm_buffers
        from diffusion_policy.model.diffusion.ema_model import EMAModel
        averaged = EMAModel(copy.deepcopy(policy))
        policy.compute_loss({'obs': obs, 'action': sample['action'][None].repeat(2, 1, 1)})
        averaged.step(policy)
        sync_batchnorm_buffers(policy, averaged.averaged_model)
        buffers = dict(averaged.averaged_model.named_buffers())
        for key, value in policy.named_buffers():
            if 'running_' in key or 'num_batches_tracked' in key:
                torch.testing.assert_close(value, buffers[key])
    if config.freeze_encoder:
        assert not policy.obs_encoder.training
        assert all(not param.requires_grad and param.grad is None for param in policy.obs_encoder.parameters())
    if config.variant == 'transformer_hybrid_image' and config.conditioning == 'inpainting':
        assert all(parameter.grad is None for parameter in policy.obs_encoder.parameters())
    policy.eval()
    inference = {'obs': obs[:, :config.n_obs_steps]} if config.lowdim else {
        key: value[:, :config.n_obs_steps] for key, value in obs.items()}
    with torch.no_grad():
        prediction = policy.predict_action(inference)
    assert prediction['action'].shape == (2, config.n_action_steps, 23)
    assert torch.isfinite(prediction['action']).all()
    if config.pred_action_steps_only:
        torch.testing.assert_close(prediction['action'], prediction['action_pred'])
    else:
        torch.testing.assert_close(prediction['action'], prediction['action_pred'][:, 1:3])
    if config.variant == 'unet_lowdim':
        assert policy.oa_step_convention is True
    if config.imagenet_norm:
        torch.testing.assert_close(policy.normalizer['head'].normalize(torch.tensor([0., 1.])),
                                   torch.tensor([0., 1.]))
    dataset.close()


def test_lowdim_reader_no_video_and_full_horizon(root):
    import shutil
    shutil.rmtree(root / 'videos')
    dataset = make_dataset(root, observation_mode='lowdim', obs_steps=4)
    assert not dataset.cameras
    sample = dataset[2]
    _, frames = dataset.sampler.locate(2)
    assert isinstance(sample['obs'], torch.Tensor) and sample['obs'].shape == (4, 27)
    expected = np.arange(61)[STATE_INDICES] + 7 + frames[:, None] / 10
    np.testing.assert_allclose(sample['obs'][:, :25], expected, rtol=1e-6)
    assert set(dataset.get_normalizer().params_dict) == {'obs', 'action'}
    assert not dataset._video.frames and not dataset._video.containers
    assert sample['action'][1, 0] == 7 + frames[1]
    dataset.close()


@pytest.mark.parametrize('scheduler', ['ddpm', 'ddim'])
def test_video_missing_source_is_explicit(scheduler):
    with pytest.raises(ModuleNotFoundError, match='authentic upstream sources') as caught:
        build_policy(ModelConfig(variant='unet_video', scheduler=scheduler), {3: 'alpha'})
    assert caught.value.name == 'diffusion_policy.model.obs_encoder'


def test_old_checkpoint_variant_defaults_and_offline_encoder_restore(root, tmp_path, monkeypatch):
    from diffusion_policy.model.vision import model_getter
    config = ModelConfig(horizon=4, n_action_steps=2, cameras=('head',), image_size=32,
                         down_dims=(16, 32), diffusion_step_embed_dim=16,
                         num_train_timesteps=4, num_inference_steps=2)
    policy = build_policy(config, {3: 'alpha', 9: 'beta'})
    policy.set_normalizer(make_dataset(root, cameras=['head']).get_normalizer())
    old_fields = ('horizon', 'n_obs_steps', 'n_action_steps', 'image_size', 'cameras', 'down_dims',
                  'diffusion_step_embed_dim', 'n_groups', 'num_train_timesteps', 'num_inference_steps',
                  'scheduler', 'clip_sample')
    old_config = {key: config.to_dict()[key] for key in old_fields}
    checkpoint = {'format': 'diffusion_policy_b1k_v1', 'config': old_config,
                  'task_map': {3: 'alpha', 9: 'beta'}, 'ema_model': policy.state_dict()}
    path = tmp_path / 'old.pt'
    torch.save(checkpoint, path)
    restored, _ = load_policy(path)
    assert type(restored).__name__ == 'DiffusionUnetImagePolicy'
    for key, value in policy.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value)
    checkpoint['config'].update(encoder_weights='IMAGENET1K_V1')
    torch.save(checkpoint, path)
    getter = model_getter.get_resnet
    seen = []

    def no_download(name, weights=None, **kwargs):
        seen.append(weights)
        assert weights is None
        return getter(name, weights=None, **kwargs)

    monkeypatch.setattr(model_getter, 'get_resnet', no_download)
    load_policy(path)
    assert seen == [None]


def test_lowdim_session_needs_no_rgb(root):
    config = ModelConfig(variant='unet_lowdim', horizon=4, n_action_steps=2,
                         down_dims=(16, 32), diffusion_step_embed_dim=16,
                         num_train_timesteps=4, num_inference_steps=2)
    dataset = B1KLeRobotDataset(root, **config.dataset_kwargs())
    policy = build_policy(config, dataset.task_map)
    policy.set_normalizer(dataset.get_normalizer())
    wrapper = B1KPolicySession(policy, config.to_dict(), dataset.task_map)
    obs = {PROPRIO_KEY: np.ones((2, 61), np.float32), 'task_id': [3, 9]}
    for _ in range(3):
        assert wrapper.act(obs).shape == (2, 23)
    wrapper.reset()
    assert wrapper.act(obs).shape == (2, 23)
    dataset.close()


def test_center_crop_is_applied():
    from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
    meta = {'obs': {'head': {'type': 'rgb', 'shape': [3, 32, 32]}}}
    encoder = MultiImageObsEncoder(meta, torch.nn.Flatten(), crop_shape=(16, 16), random_crop=False)
    image = torch.arange(3 * 32 * 32).reshape(1, 3, 32, 32).float()
    torch.testing.assert_close(encoder({'head': image}), image[:, :, 8:24, 8:24].flatten(1))


def test_transformer_short_causal_trajectory_matches_masks():
    from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
    model = TransformerForDiffusion(23, 23, horizon=4, n_obs_steps=2, cond_dim=27,
                                    n_layer=1, n_head=2, n_emb=16, causal_attn=True)
    result = model(torch.randn(2, 2, 23), torch.tensor([1, 2]), torch.randn(2, 2, 27))
    assert result.shape == (2, 2, 23) and torch.isfinite(result).all()
    result.sum().backward()


@pytest.mark.parametrize('kwargs', [
    {'variant': 'missing'}, {'variant': 'unet_image', 'conditioning': 'local'},
    {'variant': 'unet_lowdim', 'conditioning': 'local', 'pred_action_steps_only': True},
    {'variant': 'unet_image', 'pred_action_steps_only': True},
    {'variant': 'transformer_lowdim', 'time_as_cond': False},
    {'variant': 'transformer_lowdim', 'n_emb': 15},
    {'variant': 'unet_image', 'encoder_weights': 'r3m'},
    {'variant': 'unet_hybrid_image', 'freeze_encoder': True},
])
def test_variant_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        build_policy(ModelConfig(**kwargs), {3: 'alpha'})


@pytest.mark.parametrize('case', [
    'unet_image-independent-imagenet-ddpm', 'unet_image-pretrained-frozen-ddim',
    'unet_hybrid_image-global-ddpm', 'transformer_hybrid_image-action-only-ddim',
    'unet_lowdim-local-ddpm', 'transformer_lowdim-inpainting-ddim',
])
def test_variant_resume_exact(root, tmp_path, case):
    from dataclasses import replace
    from diffusion_policy.b1k.variant_matrix import config_flags
    config = replace(matrix_cases()[case], encoder_weights=None)
    output, uninterrupted = tmp_path / 'resumed', tmp_path / 'full'
    args = ['--dataset-path', str(root), '--device', 'cpu', '--num-workers', '0',
            '--batch-size', '2', '--cpu-threads', '1', *config_flags(config)]
    train_main(args + ['--output-dir', str(output), '--max-steps', '2'])
    train_main(args + ['--output-dir', str(output), '--max-steps', '3', '--resume', str(output)])
    train_main(args + ['--output-dir', str(uninterrupted), '--max-steps', '3'])
    resumed = torch.load(output / 'latest.pt', weights_only=True)
    full = torch.load(uninterrupted / 'latest.pt', weights_only=True)
    for field in ('model', 'ema_model'):
        for key, value in resumed[field].items():
            torch.testing.assert_close(value, full[field][key], rtol=0, atol=0)
    assert resumed['config']['variant'] == config.variant
    assert resumed['step'] == resumed['ema_step'] == 3


@pytest.mark.parametrize('variant', ['unet_lowdim', 'transformer_lowdim', 'transformer_hybrid_image'])
def test_action_only_training_uses_current_row(root, variant, monkeypatch):
    config = matrix_cases()[f'{variant}-action-only-ddpm']
    dataset = B1KLeRobotDataset(root, **config.dataset_kwargs())
    policy = build_policy(config, dataset.task_map)
    policy.set_normalizer(dataset.get_normalizer())
    sample = dataset[2]
    obs = sample['obs'][None] if config.lowdim else {key: value[None] for key, value in sample['obs'].items()}
    action = sample['action'][None]
    original = policy.noise_scheduler.add_noise
    captured = []

    def record_trajectory(trajectory, noise, timesteps):
        captured.append(trajectory.detach().clone())
        return original(trajectory, noise, timesteps)

    monkeypatch.setattr(policy.noise_scheduler, 'add_noise', record_trajectory)
    policy.compute_loss({'obs': obs, 'action': action})
    assert len(captured) == 1
    torch.testing.assert_close(captured[0], policy.normalizer['action'].normalize(action)[:, 1:3])
    dataset.close()


@pytest.mark.parametrize('selected', ['', '0,1', '-1', 'all', 'MIG-example'])
def test_cuda_matrix_requires_one_explicit_gpu(monkeypatch, selected):
    from diffusion_policy.b1k.variant_matrix import require_idle_gpu
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', selected)
    with pytest.raises(RuntimeError, match='exactly one'):
        require_idle_gpu()


@pytest.mark.parametrize('gpu,apps,allowed', [
    ('GPU-example, 0, 0', '', True), ('GPU-example, 256, 0', '', True),
    ('GPU-example, 257, 0', '', False), ('GPU-example, 0, 1', '', False),
    ('GPU-example, 0, 0', '1234', False), ('GPU-example, N/A, 0', '', False),
])
def test_cuda_matrix_idle_gate_without_cuda(monkeypatch, gpu, apps, allowed):
    from types import SimpleNamespace
    from diffusion_policy.b1k.variant_matrix import require_idle_gpu
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    commands = []

    def query(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(stdout=apps if '--query-compute-apps=pid' in command else gpu)

    monkeypatch.setattr('subprocess.run', query)
    if allowed:
        assert require_idle_gpu()['uuid'] == 'GPU-example'
    else:
        with pytest.raises(RuntimeError):
            require_idle_gpu()
    assert len(commands) == 2 and all(command[0] == 'nvidia-smi' for command in commands)
    assert not torch.cuda.is_initialized()


def test_cuda_matrix_query_failure_is_closed(monkeypatch):
    from diffusion_policy.b1k.variant_matrix import require_idle_gpu
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')

    def unavailable(*args, **kwargs):
        raise FileNotFoundError('nvidia-smi')

    monkeypatch.setattr('subprocess.run', unavailable)
    with pytest.raises(RuntimeError, match='Cannot verify'):
        require_idle_gpu()


def test_dataset_denial_guard(root):
    from diffusion_policy.b1k.variant_matrix import deny_dataset_access
    with deny_dataset_access(root):
        with pytest.raises(AssertionError, match='Dataset access'):
            (root / 'meta/info.json').read_text()
        with pytest.raises(AssertionError, match='Dataset unavailable'):
            make_dataset(root)
        with pytest.raises(AssertionError, match='Parquet unavailable'):
            pq.ParquetFile(root / 'data/chunk-004/file-000.parquet')
        with pytest.raises(AssertionError, match='Video unavailable'):
            av.open(str(root / 'videos/unused.mp4'))
    assert json.loads((root / 'meta/info.json').read_text())['fps'] == 10


def test_bulk_dataset_fetch_preserves_samples_and_order(root):
    dataset = make_dataset(root, episode_cache_size=1)
    indices = [len(dataset) - 1, 2, 0, len(dataset) - 1, 1, 4]
    expected = [dataset[index] for index in indices]
    dataset.close()
    actual = dataset.__getitems__(indices)
    for item, reference in zip(actual, expected):
        torch.testing.assert_close(item['action'], reference['action'], rtol=0, atol=0)
        for key, value in item['obs'].items():
            torch.testing.assert_close(value, reference['obs'][key], rtol=0, atol=0)
    dataset.close()


def longrun_args(root, output):
    return ['--dataset-path', str(root), '--output-dir', str(output), '--device', 'cpu',
            '--num-workers', '0', '--batch-size', '3', '--cpu-threads', '1', '--task-onehot',
            '--variant', 'transformer_lowdim', '--n-layer', '1', '--n-head', '2', '--n-emb', '16',
            '--horizon', '4', '--n-action-steps', '3', '--num-train-timesteps', '4',
            '--num-inference-steps', '2']


@pytest.mark.parametrize('limit,expected', [(0, [1, 2, 4, 5]), (1, [5]), (3, [2, 4, 5])])
def test_longrun_retention_export_and_resume(root, tmp_path, limit, expected):
    from diffusion_policy.b1k.model import load_checkpoint
    output = tmp_path / 'longrun'
    args = longrun_args(root, output) + ['--save-every', '2', '--export-every', '2',
                                       '--save-first-step', '--save-total-limit', str(limit)]
    train_main(args + ['--max-steps', '5'])
    assert [int(path.stem.split('-')[1]) for path in sorted(output.glob('step-*.pt'))] == expected
    queue = output / 'export_queue'
    assert [path.name for path in (queue / 'full').glob('*.pt')] == ['step-00000005.pt']
    assert (queue / 'full/step-00000005.pt').stat().st_ino == (output / 'latest.pt').stat().st_ino
    assert sorted(path.name for path in (queue / 'eval').glob('*.pt')) == ['step-00000002.pt', 'step-00000004.pt']
    full = load_checkpoint(output)
    assert full['checkpoint_type'] == 'full' and full['optimizer']['state']
    evaluation = load_checkpoint(queue / 'eval/step-00000004.pt')
    assert set(evaluation) == {'format', 'checkpoint_type', 'config', 'task_map', 'normalizer', 'ema_model', 'step'}
    assert evaluation['checkpoint_type'] == 'eval' and evaluation['step'] == 4
    with pytest.raises(ValueError, match='eval-only'):
        train_main(args + ['--max-steps', '6', '--resume', str(queue / 'eval/step-00000004.pt')])
    records = [json.loads(line) for line in (output / 'train.jsonl').read_text().splitlines()]
    assert [record['step'] for record in records] == list(range(1, 6))
    for record in records:
        assert record['step_s'] == record['compute_s'] + record['data_wait_s']
        assert record['samples_per_s'] > 0 and record['checkpoint_s'] >= 0
        assert record['gpu_allocated_bytes'] == record['gpu_reserved_bytes'] == record['gpu_peak_allocated_bytes'] == 0
    train_main(args + ['--max-steps', '6', '--resume', str(output), '--loader-batch-size', '2'])
    assert load_checkpoint(output)['step'] == 6
    root.rename(root.with_name('removed-dataset'))
    policy, evaluation = load_policy(queue / 'eval/step-00000004.pt')
    wrapper = B1KPolicySession(policy, evaluation['config'], evaluation['task_map'])
    result = wrapper.act({PROPRIO_KEY: np.ones((2, 61), np.float32), 'task_id': [3, 9]})
    assert result.shape == (2, 23) and np.isfinite(result).all()


def test_full_queue_hardlink_survives_pruning(root, tmp_path, monkeypatch):
    import os
    from diffusion_policy.b1k import train
    output = tmp_path / 'run'
    staged = tmp_path / 'uploader-held.pt'
    original = train.publish_full_checkpoint

    def handoff(output, path):
        if path.name == 'step-00000002.pt':
            os.link(output / 'export_queue/full/step-00000001.pt', staged)
        original(output, path)

    monkeypatch.setattr(train, 'publish_full_checkpoint', handoff)
    train_main(longrun_args(root, output) + ['--max-steps', '2', '--save-every', '1', '--save-total-limit', '1'])
    assert not (output / 'step-00000001.pt').exists()
    assert not (output / 'export_queue/full/step-00000001.pt').exists()
    assert torch.load(staged, weights_only=True)['step'] == 1


def test_output_lock_rejects_concurrent_writer(tmp_path):
    from diffusion_policy.b1k.train import output_lock
    output = tmp_path / 'run'
    with output_lock(output):
        with pytest.raises(RuntimeError, match='Another trainer holds'):
            train_main(longrun_args(tmp_path / 'missing', output) + ['--max-steps', '1'])
    with output_lock(output):
        assert (output / 'run.lock').exists()


@pytest.mark.parametrize('chunk', [1, 2, 3, 5, 8])
def test_chunked_sampler_preserves_optimizer_batches(chunk):
    from diffusion_policy.b1k.train import training_batches
    sampler = StepBatchSampler(101, 5, 0, 4, 42, loader_batch_size=chunk)
    indices = list(sampler)
    assert len(indices) == len(sampler)
    batches = [{'obs': {'state': torch.tensor(items)[:, None]}, 'action': torch.tensor(items)[:, None]}
               for items in indices]
    restored = list(training_batches(batches, 5))
    expected = list(StepBatchSampler(101, 5, 0, 4, 42))
    assert [batch['action'][:, 0].tolist() for batch in restored] == expected
    assert [batch['obs']['state'][:, 0].tolist() for batch in restored] == expected
    chunks_per_step = (5 + chunk - 1) // chunk
    assert indices[2 * chunks_per_step:] == list(StepBatchSampler(101, 5, 2, 4, 42, chunk))


def test_chunked_training_and_upstream_resume_exact(root, tmp_path):
    output, full = tmp_path / 'resumed', tmp_path / 'full'
    recipe = ['--optimizer', 'upstream', '--weight-decay', '0.001', '--betas', '0.9', '0.95']
    args = longrun_args(root, output) + recipe
    train_main(args + ['--max-steps', '2', '--loader-batch-size', '2', '--num-workers', '1'])
    train_main(longrun_args(root, output) + ['--max-steps', '3', '--resume', str(output)])
    train_main(longrun_args(root, full) + recipe + ['--max-steps', '3'])
    resumed, reference = [torch.load(path / 'latest.pt', weights_only=True) for path in (output, full)]
    for field in ('model', 'ema_model'):
        for key, value in resumed[field].items():
            torch.testing.assert_close(value, reference[field][key], rtol=0, atol=0)
    assert resumed['training']['optimizer'] == 'upstream'
    assert resumed['training']['betas'] == (0.9, 0.95)
    assert [group['weight_decay'] for group in resumed['optimizer']['param_groups']] == [0.001, 0.0]
    for key, values in resumed['optimizer']['state'].items():
        for name, value in values.items():
            torch.testing.assert_close(value, reference['optimizer']['state'][key][name], rtol=0, atol=0)


def test_upstream_image_optimizer_matches_parameter_groups(root):
    from diffusion_policy.b1k.train import build_optimizer, parser
    config = ModelConfig(variant='transformer_hybrid_image', horizon=4, n_action_steps=2,
                         cameras=('head',), image_size=32, n_layer=1, n_head=2, n_emb=16)
    policy = build_policy(config, {3: 'alpha'})
    args = parser().parse_args(['--dataset-path', str(root), '--output-dir', 'unused',
                               '--variant', config.variant, '--optimizer', 'upstream',
                               '--weight-decay', '0.001', '--obs-encoder-weight-decay', '0.000002',
                               '--betas', '0.9', '0.95'])
    optimizer = build_optimizer(policy, args)
    reference = policy.get_optimizer(0.001, 0.000002, 1e-4, (0.9, 0.95))
    for actual, expected in zip(optimizer.param_groups, reference.param_groups):
        assert actual['weight_decay'] == expected['weight_decay']
        assert actual['betas'] == expected['betas']
        assert [id(param) for param in actual['params']] == [id(param) for param in expected['params']]
    assert [id(param) for param in optimizer.param_groups[-1]['params']] == [id(param) for param in policy.obs_encoder.parameters()]


def test_longrun_defaults_and_cpu_thread_limits(monkeypatch):
    from diffusion_policy.b1k.train import configure_cpu_threads, parser, seed_worker
    import cv2
    import os
    args = parser().parse_args(['--dataset-path', 'unused', '--output-dir', 'unused'])
    assert args.save_total_limit == 0 and args.export_every == 10000 and not args.save_first_step
    assert args.wandb_mode == 'disabled' and args.prefetch_factor == 1 and args.cpu_threads == 2
    assert args.loader_batch_size is None and args.optimizer == 'adamw' and args.betas == [0.9, 0.999]
    configure_cpu_threads(2)
    assert torch.get_num_threads() == pa.cpu_count() == pa.io_thread_count() == cv2.getNumThreads() == 2
    seed_worker(0)
    assert torch.get_num_threads() == pa.cpu_count() == pa.io_thread_count() == cv2.getNumThreads() == 1
    assert os.environ['OMP_NUM_THREADS'] == '1'


def test_wandb_mock_online_metrics_and_resume(root, tmp_path, monkeypatch):
    from types import SimpleNamespace
    import sys
    runs, calls = [], []

    def initialize(**kwargs):
        calls.append(kwargs)
        run = SimpleNamespace(settings=SimpleNamespace(mode='online'), records=[], definitions=[], exits=[])
        run.log = lambda record, step: run.records.append((copy.deepcopy(record), step))
        run.define_metric = lambda *args, **kwargs: run.definitions.append((args, kwargs))
        run.finish = lambda exit_code: run.exits.append(exit_code)
        runs.append(run)
        return run

    monkeypatch.setenv('WANDB_API_KEY', 'test-key')
    monkeypatch.setitem(sys.modules, 'wandb', SimpleNamespace(init=initialize, Settings=lambda **kwargs: kwargs))
    output = tmp_path / 'logged'
    args = longrun_args(root, output) + ['--wandb-mode', 'online']
    train_main(args + ['--max-steps', '1', '--wandb-project', 'project', '--wandb-entity', 'entity', '--wandb-name', 'name'])
    first = torch.load(output / 'latest.pt', weights_only=True)
    train_main(args + ['--max-steps', '2', '--resume', str(output)])
    second = torch.load(output / 'latest.pt', weights_only=True)
    assert first['wandb'] == second['wandb'] == json.loads((output / 'wandb.json').read_text())
    assert calls[0]['id'] == calls[1]['id'] == first['wandb']['id']
    assert calls[1]['project'] == 'project' and calls[1]['entity'] == 'entity' and calls[1]['resume'] == 'allow'
    assert [run.records[0][1] for run in runs] == [1, 2]
    assert all(run.exits == [0] and len(run.definitions) == 2 for run in runs)
    assert {'step', 'loss', 'step_s', 'data_wait_s', 'gpu_allocated_bytes'} <= set(runs[0].records[0][0])
    with pytest.raises(ValueError, match='conflicts'):
        train_main(args + ['--max-steps', '3', '--resume', str(output), '--wandb-id', 'different'])


@pytest.mark.parametrize('failure', ['missing-key', 'authentication', 'offline-fallback'])
def test_wandb_online_fails_before_dataset(tmp_path, monkeypatch, failure):
    from types import SimpleNamespace
    import sys
    from diffusion_policy.b1k import train
    monkeypatch.setenv('WANDB_API_KEY', 'test-key')
    if failure == 'missing-key':
        monkeypatch.delenv('WANDB_API_KEY')

    def initialize(**kwargs):
        if failure == 'authentication':
            raise RuntimeError('invalid credentials')
        return SimpleNamespace(settings=SimpleNamespace(mode='offline'), finish=lambda **kwargs: None)

    monkeypatch.setitem(sys.modules, 'wandb', SimpleNamespace(init=initialize, Settings=lambda **kwargs: kwargs))
    monkeypatch.setattr(train, 'B1KLeRobotDataset', lambda *args, **kwargs: pytest.fail('Dataset opened before W&B auth'))
    with pytest.raises(RuntimeError, match='W&B'):
        train_main(longrun_args(tmp_path / 'missing', tmp_path / 'run') + ['--max-steps', '1', '--wandb-mode', 'online'])


@pytest.mark.parametrize('flag', ['--save-total-limit', '--export-every', '--prefetch-factor', '--worker-cpu-threads', '--loader-batch-size'])
def test_longrun_invalid_flags(tmp_path, flag):
    with pytest.raises(ValueError):
        train_main(longrun_args(tmp_path / 'missing', tmp_path / 'run') + [flag, '-1'])


def test_atomic_save_failure_keeps_previous_checkpoint(tmp_path, monkeypatch):
    from diffusion_policy.b1k.train import atomic_save
    path = tmp_path / 'step-00000001.pt'
    atomic_save({'step': 1}, path)
    with pytest.raises(FileExistsError):
        atomic_save({'step': 9}, path)

    def interrupted(checkpoint, stream):
        stream.write(b'partial')
        raise OSError('disk full')

    monkeypatch.setattr(torch, 'save', interrupted)
    with pytest.raises(OSError, match='disk full'):
        atomic_save({'step': 2}, tmp_path / 'step-00000002.pt')
    assert list(tmp_path.iterdir()) == [path]
    assert torch.load(path, weights_only=True)['step'] == 1


# --- throughput path: uint8 transport, frame cache, seek, fused optimizer/EMA, compile-friendly model ---

def sequential_frames(path, image_size):
    """Ground truth: decode a whole video in order, returning {pts_seconds: resized frame}."""
    frames = {}
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            if frame.pts is not None:
                frames[round(float(frame.pts * stream.time_base), 6)] = resize_rgb(frame.to_ndarray(format='rgb24'), image_size)
    return frames


def test_images_to_float_is_bit_exact():
    values = np.arange(256, dtype=np.uint8)
    reference = torch.from_numpy(values.astype(np.float32) / 255.)
    converted = images_to_float({'head': torch.from_numpy(values).reshape(1, 1, 16, 16), 'state': torch.ones(2)})
    assert converted['head'].dtype == torch.float32 and converted['state'].dtype == torch.float32
    assert torch.equal(converted['head'].flatten(), reference)
    assert torch.equal(images_to_float(torch.ones(3)), torch.ones(3))
    if torch.cuda.is_available():
        assert torch.equal(images_to_float(torch.from_numpy(values).cuda()).cpu(), reference)


def test_uint8_image_transport_matches_float_samples(root):
    reference, compact = make_dataset(root), make_dataset(root, image_dtype='uint8')
    for index in range(len(reference)):
        expected, sample = reference[index], compact[index]
        for camera in CAMERAS:
            assert sample['obs'][camera].dtype == torch.uint8 and sample['obs'][camera].shape == expected['obs'][camera].shape
        restored = images_to_float(sample['obs'])
        for key, value in expected['obs'].items():
            assert torch.equal(restored[key], value), key
        assert torch.equal(sample['action'], expected['action'])
    with pytest.raises(ValueError, match='image_dtype'):
        make_dataset(root, image_dtype='float16')
    reference.close()
    compact.close()


def test_video_reader_seek_lead_returns_exact_frames(root):
    dataset = make_dataset(root)
    reader = VideoReader(16, max_open=1)
    for camera in CAMERAS:
        path = dataset.video_path(dataset.episodes[0], camera)
        truth = sequential_frames(path, 16)
        timestamps = sorted(truth)
        # every frame alone (keyframes included, where the forward lead avoids an extra GOP), and pairs
        for t in timestamps:
            np.testing.assert_array_equal(reader.read(path, [t])[0], truth[t])
        for first, second in zip(timestamps, timestamps[1:]):
            np.testing.assert_array_equal(reader.read(path, [first, second]), np.stack([truth[first], truth[second]]))
        # seeking cannot overshoot: the last frame requested right after a fresh open
        reader.close()
        np.testing.assert_array_equal(reader.read(path, [timestamps[-1]])[0], truth[timestamps[-1]])
    reader.close()
    dataset.close()


def test_frame_cache_is_pixel_exact_and_validated(root, tmp_path):
    dataset = make_dataset(root)
    cache = tmp_path / 'cache'
    with pytest.raises(ValueError, match='inside the read-only dataset'):
        build_frame_cache(dataset, root / 'cache', workers=1, log=lambda *_: None)
    videos = build_frame_cache(dataset, cache, workers=2, log=lambda *_: None)
    assert len(videos) == 3
    for camera in CAMERAS:
        path = dataset.video_path(dataset.episodes[0], camera)
        stem = cache / 'videos' / CAMERAS[camera][0] / 'chunk-004'
        assert {p.name for p in stem.iterdir()} == {'file-000.frames.npy', 'file-000.pts.npy', 'file-000.json'}
        truth = sequential_frames(path, 16)
        frames = np.load(stem / 'file-000.frames.npy', mmap_mode='r')
        pts = np.load(stem / 'file-000.pts.npy')
        assert frames.shape == (len(truth), 16, 16, 3) and frames.dtype == np.uint8
        np.testing.assert_allclose(pts, sorted(truth), atol=1e-6)
        manifest = json.loads((stem / 'file-000.json').read_text())
        assert manifest['frames'] == len(truth) and manifest['image_size'] == 16
        assert manifest['source'] == str(path.relative_to(root))
    assert verify_frame_cache(dataset, cache, samples=len(dataset), log=lambda *_: None) == len(dataset) * 3 * 2
    # a second build is a no-op (valid entries are skipped) and the reader matches the native decoder exactly
    before = {p: p.stat().st_mtime_ns for p in cache.rglob('*.npy')}
    build_frame_cache(dataset, cache, workers=1, log=lambda *_: None)
    assert {p: p.stat().st_mtime_ns for p in cache.rglob('*.npy')} == before
    cached = make_dataset(root, frame_cache=cache)
    assert isinstance(cached._video, FrameCacheReader)
    for index in range(len(dataset)):
        expected, sample = dataset[index], cached[index]
        for key, value in expected['obs'].items():
            assert torch.equal(sample['obs'][key], value), key
        assert torch.equal(sample['action'], expected['action'])
    restored = pickle.loads(pickle.dumps(cached))
    assert isinstance(restored._video, FrameCacheReader) and not restored._video.entries
    assert torch.equal(restored[1]['obs']['head'], dataset[1]['obs']['head'])
    reader = FrameCacheReader(cache, root, 16)
    head = dataset.video_path(dataset.episodes[0], 'head')
    with pytest.raises(ValueError, match='timestamps not found'):
        reader.read(head, [0.555])
    with pytest.raises(ValueError, match='timestamps not found'):
        reader.read(head, [99.0])
    with pytest.raises(ValueError, match='Stale or mismatched'):
        FrameCacheReader(cache, root, 32).read(head, [0.5])
    cached.close()
    dataset.close()
    # a modified source video invalidates its entry for readers and datasets alike
    import os
    stat = head.stat()
    os.utime(head, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))
    with pytest.raises(ValueError, match='Stale or mismatched'):
        make_dataset(root, frame_cache=cache)
    with pytest.raises(FileNotFoundError, match='No frame cache entry'):
        make_dataset(root, frame_cache=tmp_path / 'empty', cameras=['left_wrist'])


def test_frame_cache_training_matches_native_training(root, tmp_path):
    dataset = make_dataset(root, cameras=['head'])
    cache = tmp_path / 'cache'
    build_frame_cache(dataset, cache, workers=1, log=lambda *_: None)
    dataset.close()
    args = ['--dataset-root', str(root), '--device', 'cpu', '--num-workers', '0', '--batch-size', '2',
            '--cpu-threads', '1', '--task-onehot', '--horizon', '4', '--n-action-steps', '3', '--image-size', '16',
            '--cameras', 'head', '--down-dims', '16', '32', '--diffusion-step-embed-dim', '16',
            '--num-train-timesteps', '4', '--num-inference-steps', '2', '--max-steps', '2']
    train_main(args + ['--output-dir', str(tmp_path / 'native')])
    train_main(args + ['--output-dir', str(tmp_path / 'cached'), '--frame-cache', str(cache)])
    native, cached = [torch.load(tmp_path / name / 'latest.pt', weights_only=True) for name in ('native', 'cached')]
    for field in ('model', 'ema_model'):
        for key, value in native[field].items():
            torch.testing.assert_close(cached[field][key], value, rtol=0, atol=0)
    losses = [[json.loads(line)['loss'] for line in (tmp_path / name / 'train.jsonl').read_text().splitlines()]
              for name in ('native', 'cached')]
    assert losses[0] == losses[1]
    config = json.loads((tmp_path / 'cached' / 'config.json').read_text())
    assert config['training']['frame_cache'] == str(cache)


def test_ema_updater_matches_upstream_step():
    from diffusion_policy.b1k.train import EMAUpdater
    from diffusion_policy.model.diffusion.ema_model import EMAModel
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(4, 6), torch.nn.BatchNorm1d(6), torch.nn.Linear(6, 2))
    model[2].weight.requires_grad_(False)
    upstream, fused = EMAModel(copy.deepcopy(model)), EMAModel(copy.deepcopy(model))
    updater = EMAUpdater(fused, model)
    assert len(updater.averaged) == 3 and len(updater.copied) == 3  # linear0 w/b + linear2 bias | BN w/b + frozen weight
    for _ in range(5):
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(torch.randn_like(parameter))
        upstream.step(model)
        updater.step()
        assert upstream.optimization_step == fused.optimization_step and upstream.decay == fused.decay
        for reference, value in zip(upstream.averaged_model.parameters(), fused.averaged_model.parameters()):
            assert torch.equal(reference, value)


@pytest.mark.parametrize('kwargs', [{'cond_dim': 10, 'causal_attn': True}, {'time_as_cond': False, 'causal_attn': True},
                                    {'cond_dim': 10, 'causal_attn': False}])
def test_transformer_causal_hint_matches_mask_detection(kwargs):
    from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
    torch.manual_seed(0)
    model = TransformerForDiffusion(input_dim=5, output_dim=5, horizon=6, n_obs_steps=2, n_layer=2, n_head=2,
                                    n_emb=16, p_drop_attn=0.0, **kwargs).eval()
    sample, timestep = torch.randn(3, 6, 5), torch.tensor([1, 2, 3])
    cond = torch.randn(3, 2, 10) if kwargs.get('cond_dim') else None
    with torch.no_grad():
        hinted = model(sample, timestep, cond)
        # reference: let nn.Transformer* detect causality from the mask itself (upstream behaviour)
        if model.encoder_only:
            reference_module, name = model.encoder, 'is_causal'
        else:
            reference_module, name = model.decoder, 'tgt_is_causal'
        original = reference_module.forward
        seen = {}

        def detect(*args, **call_kwargs):
            seen[name] = call_kwargs.pop(name)
            return original(*args, **call_kwargs)

        reference_module.forward = detect
        reference = model(sample, timestep, cond)
        reference_module.forward = original
    assert seen[name] == kwargs['causal_attn']
    torch.testing.assert_close(hinted, reference, rtol=1e-5, atol=1e-5)
    # a strictly upper-triangular future token must not influence causal outputs
    if kwargs['causal_attn'] and not model.encoder_only:
        perturbed = sample.clone()
        perturbed[:, -1] += 10
        with torch.no_grad():
            assert torch.allclose(model(perturbed, timestep, cond)[:, :-1], hinted[:, :-1], atol=1e-5)


def test_crop_randomizer_samples_on_device_within_bounds():
    from diffusion_policy.model.vision.crop_randomizer import sample_random_image_crops
    torch.manual_seed(0)
    images = torch.arange(2 * 3 * 8 * 8, dtype=torch.float32).reshape(2, 3, 8, 8)
    crops, indices = sample_random_image_crops(images, 5, 6, num_crops=1)
    assert crops.shape == (2, 1, 3, 5, 6) and indices.device == images.device
    assert bool((indices[..., 0] >= 0).all()) and bool((indices[..., 0] < 3).all())
    assert bool((indices[..., 1] >= 0).all()) and bool((indices[..., 1] < 2).all())
    for image, crop, (h, w) in zip(images, crops[:, 0], indices[:, 0].tolist()):
        assert torch.equal(crop, image[:, h:h + 5, w:w + 6])


def test_train_precision_and_compile_flags_are_recorded(root, tmp_path):
    output = tmp_path / 'run'
    train_main(longrun_args(root, output) + ['--max-steps', '1', '--matmul-precision', 'high',
                                             '--no-multi-tensor-ema', '--autocast', 'none'])
    training = json.loads((output / 'config.json').read_text())['training']
    assert training['matmul_precision'] == 'high' and training['multi_tensor_ema'] is False
    assert training['autocast'] == 'none' and training['compile'] == 'none' and training['sdpa_backend'] == 'math'
    assert torch.get_float32_matmul_precision() == 'high'
    torch.set_float32_matmul_precision('highest')
    # the multi-tensor EMA path trains identically to upstream's per-parameter loop
    reference, multi = tmp_path / 'ema-upstream', tmp_path / 'ema-multi'
    train_main(longrun_args(root, reference) + ['--max-steps', '3', '--no-multi-tensor-ema'])
    train_main(longrun_args(root, multi) + ['--max-steps', '3'])
    states = [torch.load(path / 'latest.pt', weights_only=True) for path in (reference, multi)]
    assert states[0]['ema_step'] == states[1]['ema_step'] == 3
    for key, value in states[0]['ema_model'].items():
        assert torch.equal(states[1]['ema_model'][key], value), key


def _small_transformer_flags(**overrides):
    from dataclasses import replace
    from diffusion_policy.b1k.variant_matrix import config_flags, matrix_cases
    return config_flags(replace(matrix_cases()['transformer_hybrid_image-global-ddpm'], encoder_weights=None, **overrides))


def test_task_onehot_is_optional_off_by_default_in_cli_and_legacy_in_checkpoints(root, tmp_path):
    # Dataclass keeps the v1 behavior so checkpoints without the field still get their one-hot channels.
    legacy = {key: value for key, value in ModelConfig().to_dict().items() if key != 'task_onehot'}
    assert ModelConfig(**legacy).task_onehot is True
    assert train_parser().parse_args(['--dataset-path', 'x', '--output-dir', 'y']).task_onehot is False
    with_onehot = make_dataset(root, task_names=['alpha'])
    without = make_dataset(root, task_names=['alpha'], task_onehot=False)
    assert with_onehot[0]['obs']['state'].shape[-1] == 26 and without[0]['obs']['state'].shape[-1] == 25
    assert without.get_normalizer()['state'].params_dict['scale'].shape == (25,)
    assert build_policy(ModelConfig(task_onehot=False, **{k: v for k, v in _small_config_dict().items()}),
                        {3: 'alpha'}).obs_encoder is not None
    with pytest.raises(ValueError, match='no task conditioning'):
        build_policy(ModelConfig(task_onehot=False, **_small_config_dict()), {3: 'alpha', 9: 'beta'})
    # Several tasks are fine again once language carries the task.
    build_policy(ModelConfig(task_onehot=False, language_conditioning='clip_film', **_small_config_dict()),
                 {3: 'alpha', 9: 'beta'})
    output = tmp_path / 'no-onehot'
    train_main(['--dataset-path', str(root), '--task-names', 'alpha', '--device', 'cpu', '--num-workers', '0',
                '--batch-size', '2', '--cpu-threads', '1', '--max-steps', '1', '--output-dir', str(output),
                *_small_transformer_flags(task_onehot=False)])
    policy, checkpoint = load_policy(output)
    assert checkpoint['config']['task_onehot'] is False
    assert policy.normalizer['state'].params_dict['scale'].shape == (25,)
    session = B1KPolicySession(policy, checkpoint['config'], checkpoint['task_map'])
    actions = session.act(observation(0.5, task=3))
    assert actions.shape[-1] == 23 and np.isfinite(actions).all()
    with pytest.raises(ValueError, match='Unknown task_id'):
        session.act(observation(0.5, task=77))


def _small_config_dict():
    from dataclasses import replace
    from diffusion_policy.b1k.variant_matrix import matrix_cases
    config = replace(matrix_cases()['transformer_hybrid_image-global-ddpm'], encoder_weights=None).to_dict()
    return {key: value for key, value in config.items() if key not in ('task_onehot', 'language_conditioning')}


def test_cosine_schedule_ema_power_and_grad_clip_are_applied_recorded_and_resumed_exactly(root, tmp_path):
    import math
    lr, warmup, length = 1e-4, 2, 6
    args = ['--dataset-path', str(root), '--task-names', 'alpha', '--device', 'cpu', '--num-workers', '0',
            '--batch-size', '2', '--cpu-threads', '1', '--learning-rate', str(lr), '--lr-scheduler', 'cosine',
            '--lr-warmup-steps', str(warmup), '--lr-schedule-steps', str(length), '--ema-power', '0.75',
            '--grad-clip', '0', *_small_transformer_flags(task_onehot=False)]
    full, resumed = tmp_path / 'full', tmp_path / 'resumed'
    train_main(args + ['--output-dir', str(full), '--max-steps', '4'])
    train_main(args + ['--output-dir', str(resumed), '--max-steps', '2'])
    # Optimizer flags are restored from the checkpoint on resume; pass a different --max-steps to check that
    # the schedule length stays what the first launch fixed.
    train_main(args[:args.index('--lr-scheduler')] + ['--output-dir', str(resumed), '--max-steps', '4',
                                                      '--resume', str(resumed), *_small_transformer_flags(task_onehot=False)])

    def expected(step):  # upstream get_cosine_schedule_with_warmup, one scheduler step per optimizer step
        current = step - 1
        if current < warmup:
            return lr * current / warmup
        progress = (current - warmup) / max(1, length - warmup)
        return lr * max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    for path in (full, resumed):
        rows = [json.loads(line) for line in (path / 'train.jsonl').read_text().splitlines()]
        logged = {row['step']: row['learning_rate'] for row in rows}
        for step in range(1, 5):
            assert math.isclose(logged[step], expected(step), rel_tol=1e-9, abs_tol=1e-12), (path.name, step)
    full_state = torch.load(full / 'latest.pt', weights_only=True)
    resumed_state = torch.load(resumed / 'latest.pt', weights_only=True)
    for field in ('model', 'ema_model'):
        for key, value in full_state[field].items():
            torch.testing.assert_close(value, resumed_state[field][key], rtol=0, atol=0)
    training = resumed_state['training']
    assert training['lr_scheduler'] == 'cosine' and training['lr_warmup_steps'] == warmup
    assert training['lr_schedule_steps'] == length and training['ema_power'] == 0.75 and training['grad_clip'] == 0.0
    assert resumed_state['lr_scheduler']['last_epoch'] == 4
    assert json.loads((full / 'config.json').read_text())['training']['grad_clip'] == 0.0


def test_checkpoints_without_schedule_fields_resume_with_constant_lr_defaults(root, tmp_path):
    output = tmp_path / 'legacy'
    args = ['--dataset-path', str(root), '--task-names', 'alpha', '--device', 'cpu', '--num-workers', '0',
            '--batch-size', '2', '--cpu-threads', '1', '--output-dir', str(output), *_small_transformer_flags()]
    train_main(args + ['--max-steps', '1'])
    checkpoint = torch.load(output / 'step-00000001.pt', weights_only=True)
    for key in ('lr_scheduler', 'lr_warmup_steps', 'lr_schedule_steps', 'ema_power', 'grad_clip'):
        del checkpoint['training'][key]
    del checkpoint['lr_scheduler']
    torch.save(checkpoint, output / 'step-00000001.pt')
    train_main(args + ['--max-steps', '2', '--resume', str(output)])
    resumed = torch.load(output / 'latest.pt', weights_only=True)
    assert resumed['step'] == 2
    assert resumed['training']['lr_scheduler'] == 'constant' and resumed['training']['ema_power'] == 2 / 3
    assert resumed['training']['grad_clip'] == 1.0 and resumed['training']['lr_warmup_steps'] == 0
    rows = [json.loads(line) for line in (output / 'train.jsonl').read_text().splitlines()]
    assert all(math_isclose(row['learning_rate'], 1e-4) for row in rows)


def math_isclose(a, b):
    import math
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)


def test_grad_accumulation_keeps_the_step_batch_resumes_exactly_and_is_recorded(root, tmp_path):
    # The optimizer batch of a step is the same seeded draw whether or not it is delivered in micro-batches.
    full = [chunk for chunk in StepBatchSampler(1000, 8, 0, 3, 42)]
    micro = [chunk for chunk in StepBatchSampler(1000, 8, 0, 3, 42, loader_batch_size=2)]
    assert len(full) == 3 and len(micro) == 12
    for step in range(3):
        assert sum(micro[step * 4:(step + 1) * 4], []) == full[step]
    args = ['--dataset-path', str(root), '--task-names', 'alpha', '--device', 'cpu', '--num-workers', '0',
            '--batch-size', '4', '--grad-accumulation', '2', '--cpu-threads', '1', *_small_transformer_flags()]
    resumed, uninterrupted = tmp_path / 'resumed', tmp_path / 'full'
    train_main(args + ['--output-dir', str(resumed), '--max-steps', '2'])
    train_main(args[:args.index('--grad-accumulation')] + args[args.index('--grad-accumulation') + 2:]
               + ['--output-dir', str(resumed), '--max-steps', '3', '--resume', str(resumed)])
    train_main(args + ['--output-dir', str(uninterrupted), '--max-steps', '3'])
    a = torch.load(resumed / 'latest.pt', weights_only=True)
    b = torch.load(uninterrupted / 'latest.pt', weights_only=True)
    for field in ('model', 'ema_model'):
        for key, value in a[field].items():
            torch.testing.assert_close(value, b[field][key], rtol=0, atol=0)
    assert a['step'] == a['ema_step'] == 3 and a['training']['grad_accumulation'] == 2
    assert a['training']['batch_size'] == 4
    rows = [json.loads(line) for line in (uninterrupted / 'train.jsonl').read_text().splitlines()]
    assert [row['step'] for row in rows] == [1, 2, 3]            # one record per optimizer step, not per micro-batch
    assert all(abs(row['samples_per_s'] * row['step_s'] - 4) < 1e-6 for row in rows)
    assert json.loads((uninterrupted / 'config.json').read_text())['training']['grad_accumulation'] == 2
    with pytest.raises(ValueError, match='divide --batch-size'):
        train_main(args + ['--output-dir', str(tmp_path / 'bad'), '--max-steps', '1', '--batch-size', '3'])
    with pytest.raises(ValueError, match='divide the micro-batch'):
        train_main(args + ['--output-dir', str(tmp_path / 'bad2'), '--max-steps', '1', '--loader-batch-size', '3'])
