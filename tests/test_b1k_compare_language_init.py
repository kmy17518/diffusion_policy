import copy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from diffusion_policy.b1k import compare_language_init as compare
from diffusion_policy.b1k import language
from diffusion_policy.b1k.dataset import B1KLeRobotDataset
from diffusion_policy.b1k.model import load_policy
from diffusion_policy.b1k.robot import CAMERAS
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.vision.clip_film import FiLMLayer
from test_b1k import root


def small_config(**kwargs):
    return replace(compare.production_config(), n_layer=1, n_emb=16, n_head=2, horizon=4,
                   n_action_steps=2, image_size=64, crop_shape=(56, 56),
                   num_train_timesteps=4, num_inference_steps=4, **kwargs)


def identity_normalizer(config):
    result = LinearNormalizer()
    for key in ('state', 'action', *config.cameras):
        result[key] = SingleFieldLinearNormalizer.create_identity()
    return result


def fake_batch(config, seed=27):
    generator = torch.Generator().manual_seed(seed)
    return {'obs': {'state': torch.randn(2, 2, 26, generator=generator),
                    'lang_emb': torch.randn(2, 2, 768, generator=generator),
                    **{camera: torch.randn(2, 2, 3, config.image_size, config.image_size, generator=generator)
                       for camera in config.cameras}},
            'action': torch.randn(2, config.horizon, 23, generator=generator)}


@pytest.fixture
def arms():
    config = small_config()
    policies, proof = compare.build_arms(config, {3: 'alpha'}, identity_normalizer(config))
    return config, policies, proof


def test_production_defaults_and_import_safe(monkeypatch):
    config = compare.production_config()
    assert (config.n_layer, config.n_emb, config.n_head) == (12, 512, 8)
    assert (config.image_size, config.crop_shape, config.n_obs_steps, config.horizon, config.n_action_steps) == (
        96, (86, 86), 2, 16, 8)
    assert config.num_train_timesteps == config.num_inference_steps == 100
    assert config.scheduler == 'ddpm' and tuple(config.cameras) == tuple(CAMERAS)
    args = compare.parser().parse_args(['--dataset-path', 'unused', '--output-dir', '/tmp/unused'])
    assert (args.max_steps, args.batch_size, args.seed, args.eval_every) == (1000, 512, 42, 250)
    assert (args.eval_samples, args.holdout_every, args.expected_episodes) == (128, 10, 200)
    assert args.wandb_mode == 'disabled'
    monkeypatch.setattr(sys, 'argv', ['pytest', '--unknown-cli-option'])
    path = Path(__file__).parents[1] / 'scripts/b1k/compare_language_init.py'
    spec = importlib.util.spec_from_file_location('safe_compare_cli', path)
    spec.loader.exec_module(importlib.util.module_from_spec(spec))


def test_all_cameras_parameters_buffers_and_film_only_difference(arms):
    config, policies, proof = arms
    assert proof['all_shared_exact'] and proof['identity_shared_exact'] and proof['zero_language_columns']
    assert proof['baseline_shared_sha256'] == proof['conditioned_shared_sha256']
    baseline = policies['baseline'].state_dict()
    assert {entry['source'] for entry in proof['mapping']} == set(baseline)
    for arm in ('random_film', 'identity_film'):
        state = policies[arm].state_dict()
        for entry in proof['mapping']:
            source, target = baseline[entry['source']], state[entry['target']]
            if entry['source'] == 'model.cond_obs_emb.weight':
                target = torch.cat([target[:, item['target'][0]:item['target'][1]]
                                    for item in proof['layout']['columns']], dim=1)
            torch.testing.assert_close(source, target, rtol=0, atol=0)
        assert torch.count_nonzero(state['model.cond_obs_emb.weight'][:, 218:]) == 0
        assert policies[arm].model.input_emb.weight.data_ptr() != policies['baseline'].model.input_emb.weight.data_ptr()
    assert proof['layout']['baseline'] == {'state': [0, 26], 'head': [26, 90],
                                          'left_wrist': [90, 154], 'right_wrist': [154, 218]}
    assert proof['layout']['language_columns'] == [218, 986]
    for camera in config.cameras:
        target_names = [entry['target'] for entry in proof['mapping'] if f'.encoders.{camera}.' in entry['target']]
        assert any('.stem.0.weight' in key for key in target_names)
        assert any('.blocks.7.block.conv2.weight' in key for key in target_names)
        assert any('.pool.0.projection.weight' in key for key in target_names)
        assert any('.pool.0.pos_x' in key for key in target_names)
        assert any('.pool.1.weight' in key for key in target_names)
    assert len(proof['random_identity_differences']) == 3 * 8 * 2
    assert all('.film.lang_proj.' in key for key in proof['random_identity_differences'])
    films = [module for module in policies['identity_film'].modules() if isinstance(module, FiLMLayer)]
    assert len(films) == 24
    assert all(torch.count_nonzero(parameter) == 0 for film in films for parameter in film.parameters())


def test_feature_mapping_handles_inserted_language_and_reordered_state(arms):
    config, policies, _ = arms
    baseline, identity = policies['baseline'], policies['identity_film']
    shapes = identity.obs_encoder.shapes
    identity.obs_encoder.shapes = {key: shapes[key] for key in ('lang_emb', 'head', 'left_wrist', 'state', 'right_wrist')}
    proof = compare.copy_shared_state(baseline, identity)
    assert proof['layout']['conditioned']['state'] == [896, 922]
    assert proof['layout']['language_columns'] == [0, 768]
    parity = compare.initial_parity({'baseline': baseline, 'random_film': identity, 'identity_film': identity},
                                    fake_batch(config), proof['layout'], 91)
    assert parity['train']['identity_features_max_abs'] == 0
    shapes = identity.obs_encoder.shapes
    identity.obs_encoder.shapes = {key: shapes[key] for key in reversed(shapes)}
    with pytest.raises(ValueError, match='Camera execution order'):
        compare.feature_layout(baseline, identity)


def test_initial_train_eval_parity_and_rng_restore(arms):
    config, policies, proof = arms
    rng = torch.get_rng_state().clone()
    states = {arm: compare.state_hash(policy.state_dict()) for arm, policy in policies.items()}
    result = compare.initial_parity(policies, fake_batch(config), proof['layout'], 137)
    assert torch.equal(rng, torch.get_rng_state())
    for mode in ('train', 'eval'):
        assert result[mode]['paired_noise_timesteps_exact'] and result[mode]['paired_rng_exact']
        assert result[mode]['identity_prediction_max_abs'] <= 2e-6
        assert result[mode]['loss']['identity_film'] == pytest.approx(result[mode]['loss']['baseline'], abs=1e-7)
    assert result['train']['noisy_input_sha256'] != result['eval']['noisy_input_sha256']
    assert states == {arm: compare.state_hash(policy.state_dict()) for arm, policy in policies.items()}


def test_missing_keypoint_rng_draw_is_detected(arms):
    config, policies, proof = arms
    for encoder in policies['identity_film'].obs_encoder.encoders.values():
        pool = encoder.pool[0]
        forward = pool.forward

        def no_training_draw(value, pool=pool, forward=forward):
            training = pool.training
            pool.training = False
            try:
                return forward(value)
            finally:
                pool.training = training

        pool.forward = no_training_draw
    with pytest.raises(RuntimeError, match='RNG consumption differs'):
        compare.initial_parity(policies, fake_batch(config), proof['layout'], 73)


def test_two_updates_pair_random_inputs_and_evaluation_restores_rng(arms):
    config, policies, proof = arms
    batch = fake_batch(config)
    optimizers = {arm: torch.optim.AdamW((p for p in policy.parameters() if p.requires_grad),
                                       lr=1e-4, weight_decay=1e-6) for arm, policy in policies.items()}
    emas = {arm: EMAModel(copy.deepcopy(policy)) for arm, policy in policies.items()}
    before = {arm: policy.model.head.weight.detach().clone() for arm, policy in policies.items()}
    captures, crop_inputs, hooks = {}, {}, []
    for arm, policy in policies.items():
        def capture(module, inputs, arm=arm):
            captures[arm] = (inputs[0].detach().clone(), inputs[1].detach().clone())
        hooks.append(policy.model.register_forward_pre_hook(capture))
        for camera in config.cameras:
            stem = (policy.obs_encoder.obs_nets[camera].backbone.nets[0] if arm == 'baseline'
                    else policy.obs_encoder.encoders[camera].stem[0])
            def capture_crop(module, inputs, arm=arm, camera=camera):
                crop_inputs[arm, camera] = inputs[0].detach().clone()
            hooks.append(stem.register_forward_pre_hook(capture_crop))
    records, step_inputs = [], []
    try:
        for step in (1, 2):
            compare.train_step(policies, optimizers, emas, batch, 1700 + step, step,
                               lambda arm, row: records.append(row))
            for arm in compare.ARMS[1:]:
                for actual, expected in zip(captures[arm], captures['baseline']):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                for camera in config.cameras:
                    torch.testing.assert_close(crop_inputs[arm, camera], crop_inputs['baseline', camera], rtol=0, atol=0)
            step_inputs.append(captures['baseline'][0])
    finally:
        for hook in hooks:
            hook.remove()
    assert not torch.equal(*step_inputs)
    assert len(records) == 6 and all(not row['finite_fail'] for row in records)
    assert all(row['grad_norm'] > 0 and isinstance(row['clip_applied'], bool) for row in records)
    for arm, policy in policies.items():
        assert not torch.equal(before[arm], policy.model.head.weight)
        assert emas[arm].optimization_step == 2
        assert optimizers[arm].state_dict()['state']
    for encoder in policies['identity_film'].obs_encoder.encoders.values():
        assert any(torch.count_nonzero(parameter) for parameter in encoder.blocks[0].film.parameters())
    assert torch.count_nonzero(policies['identity_film'].model.cond_obs_emb.weight[:, 218:]) > 0
    rng = torch.get_rng_state().clone()
    first = compare.evaluate(policies, [batch], 77)
    assert torch.equal(rng, torch.get_rng_state())
    second = compare.evaluate(policies, [batch], 77)
    assert first == second
    assert all(policy.training for policy in policies.values())


def test_nonfinite_loss_is_logged_before_failure(arms):
    config, policies, _ = arms
    policies = {'baseline': policies['baseline']}
    optimizers = {'baseline': torch.optim.AdamW(policies['baseline'].parameters())}
    records = []
    batch = fake_batch(config)
    batch['action'].fill_(float('nan'))
    with pytest.raises(FloatingPointError, match='Non-finite loss'):
        compare.train_step(policies, optimizers, {}, batch, 19, 1, lambda arm, row: records.append(row))
    assert len(records) == 1 and records[0]['finite_fail'] and records[0]['status'] == 'failed'
    assert records[0]['loss'] is None
    json.dumps(records[0], allow_nan=False)


@pytest.fixture
def radio_root(root):
    for path in (root / 'meta/episodes/chunk-004/file-000.parquet', root / 'data/chunk-004/file-000.parquet'):
        table = pq.read_table(path)
        table = table.set_column(table.schema.get_field_index('task_index'), 'task_index', pa.array([3] * len(table)))
        pq.write_table(table, path)
    return root


@pytest.fixture
def fake_clip(monkeypatch):
    calls = []
    def encode(prompts):
        calls.append(prompts)
        return torch.linspace(-0.8, 0.8, 768).expand(len(prompts), -1).clone(), language.CLIP_REVISION
    monkeypatch.setattr(language, 'encode_clip_prompts', encode)
    return calls


def test_split_uses_one_based_tenth_and_no_stats_leak(radio_root):
    dataset = B1KLeRobotDataset(radio_root, ['alpha'], **small_config(cameras=('head',)).dataset_kwargs())
    train, heldout = compare.split_dataset(dataset, 2)
    try:
        assert [row['episode_index'] for row in train.episodes] == [7]
        assert [row['episode_index'] for row in heldout.episodes] == [42]
        assert train.task_map == heldout.task_map == {3: 'alpha'}
        assert len(train) + len(heldout) == len(dataset)
        normalizer = train.get_normalizer()
        assert normalizer['action'].get_input_stats()['max'][0] == 11
        assert heldout.get_normalizer()['action'].get_input_stats()['min'][0] == 42
        dataset.episodes = [dict(dataset.episodes[0], episode_index=index) for index in range(200)]
        left, right = compare.split_dataset(dataset)
        try:
            assert [row['episode_index'] for row in right.episodes] == list(range(9, 200, 10))
            assert len(left.episodes) == 180 and len(right.episodes) == 20
        finally:
            left.close()
            right.close()
    finally:
        dataset.close()
        train.close()
        heldout.close()


def test_runner_artifacts_and_reload(radio_root, tmp_path, fake_clip):
    output = tmp_path / 'comparison'
    args = compare.parser().parse_args([
        '--dataset-path', str(radio_root), '--output-dir', str(output), '--device', 'cpu',
        '--num-workers', '0', '--cpu-threads', '1', '--max-steps', '2', '--batch-size', '2',
        '--eval-every', '1', '--eval-samples', '2', '--eval-batch-size', '2', '--holdout-every', '2',
        '--task-name', 'alpha', '--expected-episodes', '2'])
    result = compare.run_comparison(args, config=small_config(cameras=('head',)))
    assert result['status'] == 'completed' and result['completed_steps'] == 2
    assert len(fake_clip) == 1
    required = ('summary.json', 'manifest.json', 'initialization.json', 'split.json', 'evaluation_manifest.json',
                'normalizer.pt', 'language.pt', 'metrics.jsonl', 'eval.jsonl')
    assert all((output / name).exists() for name in required)
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['physical_batch_size'] == manifest['effective_batch_size'] == 2
    assert manifest['production_batch_size'] == 8960
    assert len(manifest['git_commit']) == 40 and len(manifest['harness_sha256']) == 64
    eval_manifest = json.loads((output / 'evaluation_manifest.json').read_text())
    assert eval_manifest['episode_indices'] == [42]
    assert manifest['split']['train_episode_indices'] == [7]
    assert manifest['split']['heldout_episode_indices'] == [42]
    metrics = [json.loads(row) for row in (output / 'metrics.jsonl').read_text().splitlines()]
    evaluations = [json.loads(row) for row in (output / 'eval.jsonl').read_text().splitlines()]
    assert [row['step'] for row in metrics] == [1] * 3 + [2] * 3
    assert [row['step'] for row in evaluations] == [0] * 3 + [1] * 3 + [2] * 3
    assert all(row['status'] == 'ok' and not row['finite_fail'] for row in metrics)
    for arm in compare.ARMS:
        checkpoint_path = output / arm / 'final.pt'
        policy, checkpoint = load_policy(checkpoint_path)
        assert checkpoint['step'] == checkpoint['ema_step'] == 2
        assert checkpoint['checkpoint_type'] == 'comparison_final' and checkpoint['resume_supported'] is False
        assert checkpoint['optimizer']['state']
        assert ('language' in checkpoint) == (arm != 'baseline')
        assert result['arms'][arm]['last100_count'] == result['arms'][arm]['last500_count'] == 2
        assert result['arms'][arm]['last100_mean'] == pytest.approx(
            sum(row['loss'] for row in metrics if row['arm'] == arm) / 2)
        assert len((output / arm / 'train.jsonl').read_text().splitlines()) == 2
        assert len((output / arm / 'eval.jsonl').read_text().splitlines()) == 3
        del policy, checkpoint
    with pytest.raises(FileExistsError, match='never resumed'):
        compare.run_comparison(args, config=small_config(cameras=('head',)))
    assert not (output / 'wandb.json').exists()


def test_fixed_evaluation_covers_every_episode_and_is_reproducible(radio_root):
    dataset = B1KLeRobotDataset(radio_root, ['alpha'], **small_config(cameras=('head',)).dataset_kwargs())
    try:
        batches, manifest = compare.fixed_evaluation(dataset, 5, 2, 31)
        repeated, other = compare.fixed_evaluation(dataset, 5, 2, 31)
        assert manifest == other
        assert manifest['episode_indices'] == [7, 42]
        assert [entry['episode_index'] for entry in manifest['samples']].count(7) == 3
        assert [entry['episode_index'] for entry in manifest['samples']].count(42) == 2
        assert [len(batch['action']) for batch in batches] == [2, 2, 1]
        for left, right in zip(batches, repeated):
            torch.testing.assert_close(left['action'], right['action'], rtol=0, atol=0)
        with pytest.raises(ValueError, match='at least one sample'):
            compare.fixed_evaluation(dataset, 1, 1, 31)
    finally:
        dataset.close()


def test_wandb_separate_new_handles(tmp_path, monkeypatch):
    calls, handles = [], []
    def initialize(**kwargs):
        calls.append(kwargs)
        handle = SimpleNamespace(settings=SimpleNamespace(mode='offline'), define_metric=lambda *a, **kw: None,
                                 finish=lambda **kwargs: None)
        handles.append(handle)
        return handle
    monkeypatch.setitem(sys.modules, 'wandb', SimpleNamespace(init=initialize, Settings=lambda **kwargs: kwargs))
    args = compare.parser().parse_args(['--dataset-path', 'unused', '--output-dir', str(tmp_path),
                                       '--wandb-mode', 'offline'])
    result = compare.initialize_wandb(args, tmp_path, {'purpose': 'test'})
    assert list(result.values()) == handles and len(calls) == 3
    assert len({call['id'] for call in calls}) == 3
    assert all(call['reinit'] == 'create_new' and call['resume'] == 'never' for call in calls)
    assert all(call['group'] == 'dp-init-20260917' for call in calls)
    assert set(json.loads((tmp_path / 'wandb.json').read_text())) == set(compare.ARMS)
