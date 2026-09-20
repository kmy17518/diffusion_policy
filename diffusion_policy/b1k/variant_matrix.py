"""Repeatable real-data train/resume/WebSocket audit of upstream diffusion variants."""

import argparse
import asyncio
from contextlib import contextmanager
from dataclasses import replace
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

import numpy as np
import pyarrow.parquet as pq
import torch
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from diffusion_policy.b1k.dataset import B1KLeRobotDataset
from diffusion_policy.b1k.model import ModelConfig, POLICY_TARGETS, build_policy, load_checkpoint, load_policy
from diffusion_policy.b1k.robot import CAMERAS, PROPRIO_KEY
from diffusion_policy.b1k.serve import WebsocketPolicyServer, health_check, packb, unpackb


def require_idle_gpu():
    """Fail closed without initializing CUDA; use the queried physical GPU UUID."""
    selected = os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()
    if not selected or ',' in selected or not (selected.isdecimal() or selected.startswith('GPU-')):
        raise RuntimeError('CUDA matrix requires exactly one explicit CUDA_VISIBLE_DEVICES index or GPU UUID')
    common = ['nvidia-smi', '--id', selected, '--format=csv,noheader,nounits']
    try:
        gpu = subprocess.run(common + ['--query-gpu=uuid,memory.used,utilization.gpu'],
                             capture_output=True, text=True, check=True, timeout=15).stdout.strip()
        apps = subprocess.run(common + ['--query-compute-apps=pid'],
                              capture_output=True, text=True, check=True, timeout=15).stdout.strip()
        lines = gpu.splitlines()
        if len(lines) != 1:
            raise ValueError('Expected one physical GPU')
        uuid, memory, utilization = [value.strip() for value in lines[0].split(',')]
        memory, utilization = int(memory), int(utilization)
        if not uuid.startswith('GPU-') or memory < 0 or utilization < 0:
            raise ValueError('Invalid GPU status')
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        raise RuntimeError('Cannot verify GPU idleness; refusing CUDA matrix') from error
    if apps or memory > 256 or utilization != 0:
        raise RuntimeError(f'GPU {uuid} is not idle: compute PIDs={apps!r}, memory={memory} MiB, utilization={utilization}%')
    os.environ['CUDA_VISIBLE_DEVICES'] = uuid
    return {'uuid': uuid, 'memory_mib': memory, 'utilization_percent': utilization, 'compute_pids': []}


@contextmanager
def deny_dataset_access(root):
    """Guard Python file opens and the native readers during restore/serving."""
    root = Path(root).resolve()
    state = {'active': True}

    def audit(event, args):
        if state['active'] and event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(args[0])).resolve()
            if path.is_relative_to(root):
                raise AssertionError(f'Dataset access during restore/serve: {path}')

    sys.addaudithook(audit)
    try:
        with patch.object(B1KLeRobotDataset, '__init__', side_effect=AssertionError('Dataset unavailable')), \
                patch('pyarrow.parquet.ParquetFile', side_effect=AssertionError('Parquet unavailable')), \
                patch('pyarrow.parquet.read_table', side_effect=AssertionError('Parquet unavailable')), \
                patch('av.open', side_effect=AssertionError('Video unavailable')):
            yield
    finally:
        state['active'] = False


def matrix_cases():
    """Branch coverage, not the Cartesian product of all training hyperparameters."""
    base = ModelConfig(horizon=4, n_obs_steps=2, n_action_steps=2, image_size=32,
                       cameras=('head',), down_dims=(16, 32), diffusion_step_embed_dim=16,
                       num_train_timesteps=4, num_inference_steps=2,
                       n_layer=1, n_head=2, n_emb=32, p_drop_attn=0.1)
    cases = {}
    for variant in POLICY_TARGETS:
        modes = ['global'] if variant == 'unet_video' else ['global', 'inpainting']
        if variant == 'unet_lowdim':
            modes += ['local']
        for scheduler in ('ddpm', 'ddim'):
            for mode in modes:
                config = replace(base, variant=variant, scheduler=scheduler, conditioning=mode)
                if 'hybrid' in variant:
                    config = replace(config, crop_shape=(28, 28))
                cases[f'{variant}-{mode}-{scheduler}'] = config
            if variant in ('unet_lowdim', 'transformer_lowdim', 'transformer_hybrid_image'):
                config = replace(cases[f'{variant}-global-{scheduler}'], pred_action_steps_only=True)
                cases[f'{variant}-action-only-{scheduler}'] = config
    for scheduler in ('ddpm', 'ddim'):
        cases[f'unet_image-independent-imagenet-{scheduler}'] = replace(
            base, scheduler=scheduler, cameras=('head', 'left_wrist'), share_rgb_model=False,
            imagenet_norm=True, resize_shape=(40, 40), crop_shape=(32, 32))
        cases[f'unet_image-pretrained-frozen-{scheduler}'] = replace(
            base, scheduler=scheduler, encoder_weights='IMAGENET1K_V1', freeze_encoder=True,
            obs_encoder_group_norm=False, imagenet_norm=True, random_crop=False,
            resize_shape=(40, 40), crop_shape=(32, 32))
        for variant in ('transformer_lowdim', 'transformer_hybrid_image'):
            cases[f'{variant}-encoder-only-{scheduler}'] = replace(
                base, variant=variant, scheduler=scheduler, conditioning='inpainting',
                time_as_cond=False, causal_attn=False)
            cases[f'{variant}-cond-encoder-{scheduler}'] = replace(
                base, variant=variant, scheduler=scheduler, n_cond_layers=1, causal_attn=False)
        cases[f'unet_lowdim-sample-target-{scheduler}'] = replace(
            base, variant='unet_lowdim', scheduler=scheduler, prediction_type='sample',
            cond_predict_scale=False, kernel_size=3)
    cases['unet_image-trainable-batchnorm-ddpm'] = replace(base, obs_encoder_group_norm=False)
    return cases


def config_flags(config):
    flags = []
    for key, value in config.to_dict().items():
        if value is None:
            continue
        flag = '--' + key.replace('_', '-')
        if key == 'clip_sample':
            if not value:
                flags.append('--no-clip-sample')
        elif key in ('freeze_encoder', 'imagenet_norm', 'pred_action_steps_only'):
            if value:
                flags.append(flag)
        elif isinstance(value, bool):
            flags.append(flag if value else '--no-' + key.replace('_', '-'))
        elif isinstance(value, (tuple, list)):
            if value:  # an empty selection (e.g. goal_views without goal fusion) is the flag's default
                flags.extend([flag, *map(str, value)])
        else:
            flags.extend([flag, str(value)])
    return flags


def real_observations(dataset):
    episode = dataset.episodes[0]
    file = pq.ParquetFile(dataset.data_path(episode))
    rows = []
    for batch in file.iter_batches(columns=['episode_index', 'observation.state', 'frame_index'], batch_size=65536):
        table = batch.to_pydict()
        rows.extend((frame, state) for ep, state, frame in zip(
            table['episode_index'], table['observation.state'], table['frame_index'])
                    if ep == episode['episode_index'] and frame < 3)
        if len(rows) >= 3:
            break
    rows.sort()
    if len(rows) != 3:
        raise ValueError('Real-data audit needs at least three frames in the first selected episode')
    lowdim = dataset._read_episode(episode)
    observations = []
    for frame, state in rows:
        obs = {PROPRIO_KEY: np.asarray(state, dtype=np.float32), 'task_id': episode['task_index']}
        for camera in dataset.cameras:
            key, wire = CAMERAS[camera]
            timestamp = lowdim['timestamp'][frame] + episode[f'videos/{key}/from_timestamp']
            obs[wire] = dataset._video.read(dataset.video_path(episode, camera), [timestamp])[0]
        observations.append(obs)
    return observations


async def websocket_roundtrip(policy, checkpoint, observations):
    server = WebsocketPolicyServer(policy, checkpoint, action_horizon=2)
    server.inference_lock = asyncio.Lock()
    calls = []
    original_predict = policy.predict_action

    def counted_predict(obs):
        values = obs['obs'] if 'obs' in obs else obs['state']
        calls.append({'batch': len(values), 'history': values[:, :, 0].detach().cpu().tolist()})
        return original_predict(obs)

    policy.predict_action = counted_predict
    replies = []
    try:
        async with serve(server.handler, '127.0.0.1', 0, process_request=health_check) as running:
            port = running.sockets[0].getsockname()[1]
            reader, writer = await asyncio.open_connection('127.0.0.1', port)
            writer.write(b'GET /healthz HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n')
            await writer.drain()
            assert b'200 OK' in await reader.read()
            writer.close()
            await writer.wait_closed()
            async with connect(f'ws://127.0.0.1:{port}') as first, connect(f'ws://127.0.0.1:{port}') as second:
                handshake = unpackb(await first.recv())
                assert handshake['policy'] == type(policy).__name__
                assert handshake['variant'] == checkpoint['config']['variant']
                assert handshake['action_dim'] == 23
                assert unpackb(await second.recv()) == handshake

                async def request(client, obs, batch=1):
                    await client.send(packb(obs))
                    action = unpackb(await asyncio.wait_for(client.recv(), 120))['action']
                    assert action.shape == (batch, 23) and action.dtype == np.float32
                    assert np.isfinite(action).all()
                    replies.append({'shape': list(action.shape), 'finite': True})

                await request(first, observations[0])
                await request(first, observations[1])
                assert len(calls) == 1
                await request(first, observations[2])
                assert len(calls) == 2
                await request(second, observations[0])
                assert len(calls) == 3
                await first.send(packb({'reset': True}))
                try:
                    await asyncio.wait_for(first.recv(), 0.05)
                except asyncio.TimeoutError:
                    pass
                else:
                    raise AssertionError('Reset unexpectedly replied')
                await request(first, observations[1])
                assert len(calls) == 4 and calls[-1]['history'][0][0] == calls[-1]['history'][0][1]
                batch = {key: np.stack([value, value]) if isinstance(value, np.ndarray) else [value, value]
                         for key, value in observations[0].items()}
                await request(first, batch, batch=2)
                assert len(calls) == 5 and calls[-1]['batch'] == 2
                await request(first, batch, batch=2)
                assert len(calls) == 5
                await request(first, batch, batch=2)
                assert len(calls) == 6
        return {'health': True, 'handshake': handshake, 'replies': replies, 'model_calls': calls,
                'reset_no_reply': True, 'replanning': True, 'batching': True, 'client_isolation': True}
    finally:
        policy.predict_action = original_predict


def inventory():
    import yaml
    root = Path(__file__).resolve().parents[1] / 'config'
    recipes = []
    for path in sorted(root.glob('train_diffusion*.yaml')):
        config = yaml.safe_load(path.read_text())
        recipes.append({'file': str(path.relative_to(root)), 'config': config})
    tasks = [{'file': str(path.relative_to(root)), 'config': yaml.safe_load(path.read_text())}
             for path in sorted((root / 'task').glob('*.yaml'))]
    return {'policy_targets': POLICY_TARGETS, 'diffusion_recipes': recipes, 'task_configs': tasks,
            'outside_dp_scope': ['BET', 'IBC', 'robomimic imitation policy algorithms'],
            'video_missing_sources': ['model/obs_encoder/temporal_aggregator.py',
                                      'model/obs_encoder/video_core.py', 'model/ibc/global_avgpool.py']}


def run_case(name, config, args):
    started = time.monotonic()
    output = args.output_dir / name
    output.mkdir(parents=True, exist_ok=True)
    evidence = {'case': name, 'config': config.to_dict(), 'device': args.device,
                'default_widths_tested': False, 'dataset': str(args.dataset_path),
                'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                'expected_class': POLICY_TARGETS[config.variant].rsplit('.', 1)[-1]}
    try:
        if config.variant == 'unet_video':
            try:
                importlib.import_module('diffusion_policy.policy.diffusion_unet_video_policy')
            except ModuleNotFoundError as error:
                evidence['direct_import_error'] = {'type': type(error).__name__, 'message': str(error), 'name': error.name}
            try:
                build_policy(config, {0: 'construction-probe'})
            except ModuleNotFoundError as error:
                evidence.update(status='blocked_missing_upstream_source', error=str(error), error_type=type(error).__name__)
                return evidence
            raise AssertionError('Video source availability changed; audit and implement the authentic recipe')
        command = [sys.executable, '-m', 'diffusion_policy.b1k.train',
                   '--dataset-path', str(args.dataset_path), '--output-dir', str(output / 'run'),
                   '--device', args.device, '--num-workers', '0', '--batch-size', '2',
                   '--cpu-threads', str(args.cpu_threads), '--max-episodes', str(args.max_episodes),
                   '--task-names', *args.task_names, *config_flags(config)]
        train = command + ['--max-steps', '2']
        resume = command + ['--max-steps', '3', '--resume', str(output / 'run/step-00000002.pt')]
        evidence['commands'] = [train, resume]
        for stage, cmd in [('train', train), ('resume', resume)]:
            if args.device == 'cuda':
                evidence.setdefault('gpu_idle_checks', {})[stage] = require_idle_gpu()
            env = os.environ.copy()
            if stage == 'resume':
                env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
            with (output / f'{stage}.log').open('w') as log:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, check=True)
        checkpoint = load_checkpoint(output / 'run')
        step2 = load_checkpoint(output / 'run/step-00000002.pt')
        assert checkpoint['step'] == checkpoint['ema_step'] == 3
        assert step2['step'] == step2['ema_step'] == 2
        assert checkpoint['optimizer']['state'] and step2['optimizer']['state']
        changed = [key for key in checkpoint['model'] if checkpoint['model'][key].numel()
                   and not torch.equal(checkpoint['model'][key], step2['model'][key])]
        assert any(key.startswith('model.') for key in changed)
        if config.freeze_encoder:
            assert not any(key.startswith('obs_encoder.') for key in changed)
        if not config.obs_encoder_group_norm:
            buffers = [key for key in checkpoint['model'] if 'running_' in key or 'num_batches_tracked' in key]
            assert buffers
            for key in buffers:
                torch.testing.assert_close(checkpoint['model'][key], checkpoint['ema_model'][key], rtol=0, atol=0)
            evidence['ema_batchnorm_buffers_match'] = True
        assert ModelConfig(**checkpoint['config']).to_dict() == config.to_dict()
        dataset = B1KLeRobotDataset(args.dataset_path, args.task_names, max_episodes=args.max_episodes,
                                   **config.dataset_kwargs())
        try:
            observations = real_observations(dataset)
            evidence['fingerprint'] = dataset.fingerprint()
            evidence['episode_ids'] = [row['episode_index'] for row in dataset.episodes]
        finally:
            dataset.close()
        del dataset
        with deny_dataset_access(args.dataset_path), patch(
                'torch.hub.download_url_to_file', side_effect=AssertionError('Restore must not download weights')):
            policy, restored = load_policy(output / 'run', 'cpu')
            assert type(policy).__name__ == evidence['expected_class']
            assert type(policy.noise_scheduler).__name__ == ('DDPMScheduler' if config.scheduler == 'ddpm' else 'DDIMScheduler')
            if args.device == 'cuda':
                evidence.setdefault('gpu_idle_checks', {})['serve'] = require_idle_gpu()
            policy.to(args.device)
            try:
                evidence['wire'] = asyncio.run(websocket_roundtrip(policy, restored, observations))
            finally:
                if args.device == 'cuda':
                    policy.to('cpu')
                    torch.cuda.empty_cache()
            evidence['dataset_access_denied_during_restore_and_serve'] = True
        records = [json.loads(line) for line in (output / 'run/train.jsonl').read_text().splitlines()]
        assert [record['step'] for record in records] == [1, 2, 3]
        assert all(np.isfinite(record['loss']) and np.isfinite(record['grad_norm']) for record in records)
        evidence.update(status='passed', records=records, checkpoint_steps=[2, 3],
                        optimizer_states=len(checkpoint['optimizer']['state']),
                        changed_model_tensors=len(changed), actual_class=type(policy).__name__,
                        restore_downloads_blocked=True, frozen_encoder_unchanged=config.freeze_encoder,
                        cuda_initialized=torch.cuda.is_initialized())
    except Exception as error:
        evidence.update(status='failed', error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    finally:
        evidence['elapsed_s'] = time.monotonic() - started
        (output / 'evidence.json').write_text(json.dumps(evidence, indent=2))
    return evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-path', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--task-names', nargs='+', default=['turning_on_radio'])
    parser.add_argument('--max-episodes', type=int, default=2)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--cpu-threads', type=int, default=2)
    parser.add_argument('--cases', nargs='+', help='Exact case IDs; omit for the full matrix')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--case-worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    elif not args.list:
        require_idle_gpu()
    torch.set_num_threads(args.cpu_threads)
    cases = matrix_cases()
    if args.list:
        print(json.dumps({name: config.to_dict() for name, config in cases.items()}, indent=2))
        return
    if args.output_dir.resolve().is_relative_to(args.dataset_path.resolve()):
        raise ValueError('Evidence must not be written inside the read-only dataset')
    if args.cases:
        unknown = set(args.cases) - cases.keys()
        if unknown:
            raise ValueError(f'Unknown cases: {sorted(unknown)}')
        cases = {name: cases[name] for name in args.cases}
    if args.case_worker:
        if len(cases) != 1 or args.device != 'cuda':
            raise ValueError('CUDA worker requires exactly one case')
        name, config = next(iter(cases.items()))
        result = run_case(name, config, args)
        if result['status'] == 'failed':
            raise RuntimeError(f'{name} failed; see {args.output_dir / name / "evidence.json"}')
        return
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError('Use a new output directory to preserve existing matrix evidence')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'inventory.json').write_text(json.dumps(inventory(), indent=2))
    results = []
    for name, config in cases.items():
        if args.device == 'cuda':
            require_idle_gpu()
            # A fresh worker releases its CUDA context before the next idle check.
            command = [sys.executable, '-m', 'diffusion_policy.b1k.variant_matrix',
                       '--dataset-path', str(args.dataset_path), '--output-dir', str(args.output_dir),
                       '--task-names', *args.task_names, '--max-episodes', str(args.max_episodes),
                       '--cpu-threads', str(args.cpu_threads), '--device', 'cuda',
                       '--cases', name, '--case-worker']
            subprocess.run(command, check=True)
            result = json.loads((args.output_dir / name / 'evidence.json').read_text())
        else:
            result = run_case(name, config, args)
        results.append(result)
        (args.output_dir / 'summary.json').write_text(json.dumps(results, indent=2))
        print(json.dumps({'case': name, 'status': result['status'], 'elapsed_s': result['elapsed_s']}), flush=True)
        if result['status'] == 'failed':
            raise RuntimeError(f'{name} failed; see {args.output_dir / name / "evidence.json"}')


if __name__ == '__main__':
    main()
