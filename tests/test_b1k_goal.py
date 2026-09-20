"""Goal-image conditioning regressions: data contract, encoder equivalence, early/late fusion, masks, lifecycle."""

from dataclasses import replace
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from diffusion_policy.b1k.dataset import B1KLeRobotDataset
from diffusion_policy.b1k.model import ModelConfig, build_policy, load_policy
from diffusion_policy.b1k.robot import CAMERAS, GOAL_OBS_KEYS, GOAL_VIDEO_KEYS, goal_key
from diffusion_policy.b1k.serve import B1KPolicySession
from diffusion_policy.b1k.train import main as train_main, parser, resolve_regime
from diffusion_policy.b1k.variant_matrix import config_flags
from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from diffusion_policy.model.vision.clip_film import FiLMHybridObsEncoder, PairedConv2d
from test_b1k import make_dataset, observation, root, write_video  # noqa: F401  (fixture)


TASKS = {3: 'alpha', 9: 'beta'}


def small_config(**kwargs):
    # 64 px / 56 px crops: the spatial softmax needs a >= 2x2 feature grid to depend on the image at all
    config = dict(variant='transformer_hybrid_image', horizon=4, n_obs_steps=2, n_action_steps=2, cameras=('head', 'left_wrist'),
                  image_size=64, crop_shape=(56, 56), n_layer=1, n_head=2, n_emb=16, num_train_timesteps=4,
                  num_inference_steps=2, task_onehot=False, regime='none')
    config.update(kwargs)
    return ModelConfig(**config)


def goal_config(fusion='late', **kwargs):
    return small_config(**{'regime': 'image', 'goal_fusion': fusion, 'goal_views': ('head',), **kwargs})


def observations(config, batch=2, seed=0):
    generator = torch.Generator().manual_seed(seed)
    size = config.image_size
    obs = {'state': torch.randn(batch, config.n_obs_steps, 25, generator=generator)}
    for camera in config.cameras:
        obs[camera] = torch.rand(batch, config.n_obs_steps, 3, size, size, generator=generator)
    for key in config.goal_keys:
        obs[key] = torch.rand(batch, 3, size, size, generator=generator)
    if config.language_conditioning != 'none':
        obs['lang_emb'] = torch.randn(batch, config.n_obs_steps, 768, generator=generator)
    return obs


def add_goal_streams(root):
    """Give the fixture root dedicated goal streams (frame i has value i, 50 frames) with per-episode offsets."""
    episodes = pq.read_table(root / 'meta/episodes/chunk-004/file-000.parquet')
    for camera in CAMERAS:
        key = GOAL_VIDEO_KEYS[camera]
        write_video(root / f'videos/{key}/chunk-004/file-000.mp4', 50)
        rows = episodes.num_rows
        episodes = episodes.append_column(f'videos/{key}/chunk_index', pa.array([4] * rows))
        episodes = episodes.append_column(f'videos/{key}/file_index', pa.array([0] * rows))
        episodes = episodes.append_column(f'videos/{key}/from_timestamp', pa.array([3.0, 4.0]))  # frames 30, 40
        episodes = episodes.append_column(f'videos/{key}/to_timestamp', pa.array([3.5, 4.8]))
    pq.write_table(episodes, root / 'meta/episodes/chunk-004/file-000.parquet')


def test_dataset_goal_table_is_the_episode_last_frame(root):
    dataset = make_dataset(root, goal_views=('head', 'right_wrist'), task_onehot=False, image_dtype='uint8')
    assert dataset.goal_table.shape == (2, 2, 16, 16, 3) and dataset.goal_table.dtype == np.uint8
    # fixture: camera c of episode e shows frames start..start+length-1 with start = 5 + 11 c + (9 if e == 42)
    expected = {0: {'head': 9, 'right_wrist': 31}, 1: {'head': 21, 'right_wrist': 43}}
    for position in range(2):
        for view, camera in enumerate(['head', 'right_wrist']):
            np.testing.assert_array_equal(dataset.goal_table[position, view], expected[position][camera])
    item = dataset[len(dataset) - 1]  # a sequence of episode 42
    assert item['obs']['goal_head'].shape == (3, 16, 16) and item['obs']['goal_head'].dtype == torch.uint8
    assert int(item['obs']['goal_head'][0, 0, 0]) == 21 and int(item['obs']['goal_right_wrist'][0, 0, 0]) == 43
    assert item['obs']['head'].shape == (2, 3, 16, 16) and item['obs']['state'].shape == (2, 25)
    normalizer = dataset.get_normalizer()
    assert 'goal_head' in normalizer.params_dict and 'goal_right_wrist' in normalizer.params_dict
    normalized = normalizer.normalize({'goal_head': item['obs']['goal_head'][None].float() / 255})
    torch.testing.assert_close(normalized['goal_head'].min(), torch.tensor(21 / 255 * 2 - 1))
    floating = make_dataset(root, goal_views=('head',))[0]['obs']  # library default: float images like the cameras
    assert floating['goal_head'].dtype == floating['head'].dtype == torch.float32
    torch.testing.assert_close(floating['goal_head'], torch.full((3, 16, 16), 9 / 255))
    plain = make_dataset(root)
    assert plain.goal_table is None and 'goal_head' not in plain[0]['obs']
    with pytest.raises(ValueError, match='goal_views'):
        make_dataset(root, goal_views=('head', 'head'))
    with pytest.raises(ValueError, match='goal_views'):
        make_dataset(root, goal_views=('head',), cameras=('left_wrist',))
    with pytest.raises(ValueError, match='goal_source'):
        make_dataset(root, goal_views=('head',), goal_source='terminal')
    # the frame cache path yields the same goal table
    from diffusion_policy.b1k.frame_cache import build_frame_cache
    cache = root.parent / 'cache'
    build_frame_cache(plain, cache, workers=1, log=lambda *_: None)
    cached = make_dataset(root, goal_views=('head',), frame_cache=cache)
    np.testing.assert_array_equal(cached.goal_table[:, 0], dataset.goal_table[:, 0])


def test_dataset_goal_key_source_reads_the_dedicated_stream(root):
    with pytest.raises(ValueError, match='no metadata for goal stream'):
        make_dataset(root, goal_views=('head',), goal_source='goal_key')
    add_goal_streams(root)
    dataset = make_dataset(root, goal_views=('head',), goal_source='goal_key')
    assert [int(dataset.goal_table[p, 0, 0, 0, 0]) for p in range(2)] == [30, 40]
    assert dataset.goal_source == 'goal_key'


def test_reimplemented_hybrid_encoder_matches_robomimic_without_language_or_goal():
    """The goal/language encoder with both switches off is robomimic's hybrid encoder (same parameters, same output)."""
    torch.manual_seed(1)
    config = small_config()
    reference = build_policy(config, TASKS).obs_encoder
    shape_meta = {'obs': {'state': {'shape': [25], 'type': 'low_dim'},
                          **{camera: {'shape': [3, 64, 64], 'type': 'rgb'} for camera in config.cameras}}}
    ours = FiLMHybridObsEncoder(shape_meta, crop_shape=(56, 56), group_norm=True, eval_fixed_crop=True, language=False)
    assert ours.output_shape() == list(reference.output_shape()) and ours.goal_feature_dim == 0
    assert sum(p.numel() for p in ours.parameters()) == sum(p.numel() for p in reference.parameters())
    assert not any(isinstance(module, PairedConv2d) for module in ours.modules())
    for camera in config.cameras:
        original, mine = reference.obs_nets[camera], ours.encoders[camera]
        mine.stem.load_state_dict(torch.nn.Sequential(*list(original.backbone.nets.children())[:4]).state_dict())
        blocks = [block for stage in list(original.backbone.nets.children())[4:] for block in stage]
        for block, source in zip(mine.blocks, blocks):
            block.block.load_state_dict(source.state_dict())
            assert block.film is None
        mine.pool[0].projection.load_state_dict(original.pool.nets.state_dict())
        mine.pool[1].load_state_dict(original.nets[-1].state_dict())
    obs = {'state': torch.randn(3, 25), **{camera: torch.rand(3, 3, 64, 64) for camera in config.cameras}}
    for training in (False, True):
        reference.train(training)
        ours.train(training)
        torch.manual_seed(5)
        expected = reference(obs)
        torch.manual_seed(5)
        torch.testing.assert_close(ours(obs), expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match='no FiLM'):
        ours.encoders['head'](torch.rand(1, 3, 56, 56), torch.randn(1, 768))


def test_early_fusion_stem_is_the_base_stem_at_initialization_and_for_absent_goals():
    torch.manual_seed(2)
    early = build_policy(goal_config('early'), TASKS)
    torch.manual_seed(2)
    plain = build_policy(replace(goal_config('early'), regime='none', goal_fusion='none', goal_views=()), TASKS)
    encoder, base = early.obs_encoder, plain.obs_encoder
    stem = encoder.encoders['head'].stem[0]
    assert isinstance(stem, PairedConv2d) and torch.count_nonzero(stem.goal_weight) == 0
    assert not isinstance(encoder.encoders['left_wrist'].stem[0], PairedConv2d)
    assert encoder.output_shape() == list(base.output_shape()) and encoder.goal_feature_dim == 0
    assert sum(p.numel() for p in early.parameters()) == sum(p.numel() for p in plain.parameters()) + stem.goal_weight.numel()
    # robomimic's encoder (plain) and ours name parameters differently: compare through a reimplemented mirror
    mirror = FiLMHybridObsEncoder({'obs': {'state': {'shape': [25], 'type': 'low_dim'},
                                           'head': {'shape': [3, 64, 64], 'type': 'rgb'},
                                           'left_wrist': {'shape': [3, 64, 64], 'type': 'rgb'}}},
                                  crop_shape=(56, 56), language=False)
    state = {key: value for key, value in encoder.state_dict().items() if not key.endswith('goal_weight')}
    mirror.load_state_dict(state)
    encoder.eval()
    mirror.eval()
    obs = {'state': torch.randn(2, 25), 'head': torch.rand(2, 3, 64, 64), 'left_wrist': torch.rand(2, 3, 64, 64),
           'goal_head': torch.rand(2, 3, 64, 64)}
    with torch.no_grad():
        torch.testing.assert_close(encoder(obs), mirror(obs))
        stem.goal_weight.normal_()
        assert not torch.allclose(encoder(obs), mirror(obs))
        torch.testing.assert_close(encoder({**obs, 'goal_head': torch.zeros_like(obs['goal_head'])}), mirror(obs))
        paired = torch.cat([obs['head'], obs['goal_head']], dim=1)
        expected = torch.nn.functional.conv2d(obs['head'], stem.weight, None, 2, 3) + \
            torch.nn.functional.conv2d(obs['goal_head'], stem.goal_weight, None, 2, 3)
        torch.testing.assert_close(stem(paired), expected, atol=1e-5, rtol=1e-5)
    with pytest.raises(ValueError, match='row by row'):
        encoder({**obs, 'goal_head': obs['goal_head'][:1]})
    with pytest.raises(ValueError, match='shared_base'):
        build_policy(goal_config('early', goal_encoder='separate_base'), TASKS)


def test_late_fusion_goal_token_is_visible_to_every_action_and_maskable():
    model = TransformerForDiffusion(input_dim=5, output_dim=5, horizon=6, n_obs_steps=2, cond_dim=7, n_layer=1, n_head=2,
                                    n_emb=8, causal_attn=True, obs_as_cond=True, goal_cond_dim=3)
    assert model.T_cond == 4 and model.cond_pos_emb.shape == (1, 4, 8)
    mask = model.memory_mask
    assert mask.shape == (6, 4) and torch.all(mask[:, -1] == 0), 'goal column must be fully visible'
    assert mask[0, 2] == float('-inf') and mask[1, 2] == 0  # observation columns keep the time-based rule
    model.eval()
    sample, cond = torch.randn(2, 6, 5), torch.randn(2, 2, 7)
    goal_a, goal_b = torch.randn(2, 3), torch.randn(2, 3)
    with torch.no_grad():
        assert not torch.allclose(model(sample, 1, cond, goal=goal_a), model(sample, 1, cond, goal=goal_b))
        invalid = torch.zeros(2, dtype=torch.bool)
        torch.testing.assert_close(model(sample, 1, cond, goal=goal_a, goal_valid=invalid),
                                   model(sample, 1, cond, goal=goal_b, goal_valid=invalid))
        mixed = torch.tensor([True, False])
        out_a, out_b = model(sample, 1, cond, goal=goal_a, goal_valid=mixed), model(sample, 1, cond, goal=goal_b, goal_valid=mixed)
        torch.testing.assert_close(out_a[1], out_b[1])
        assert not torch.allclose(out_a[0], out_b[0])
    with pytest.raises(ValueError, match='pass goal features'):
        model(sample, 1, cond)
    plain = TransformerForDiffusion(input_dim=5, output_dim=5, horizon=6, n_obs_steps=2, cond_dim=7, n_layer=1, n_head=2,
                                    n_emb=8, causal_attn=True, obs_as_cond=True)
    assert plain.T_cond == 3 and plain.memory_mask.shape == (6, 3)
    with pytest.raises(ValueError, match='no goal condition'):
        plain(sample, 1, cond, goal=goal_a)
    with pytest.raises(ValueError, match='requires observation conditioning'):
        TransformerForDiffusion(input_dim=5, output_dim=5, horizon=6, n_layer=1, n_head=2, n_emb=8, goal_cond_dim=3)
    # the condition-encoder path also masks an absent goal
    encoded = TransformerForDiffusion(input_dim=5, output_dim=5, horizon=6, n_obs_steps=2, cond_dim=7, n_layer=1, n_head=2,
                                      n_emb=8, causal_attn=True, obs_as_cond=True, goal_cond_dim=3, n_cond_layers=1).eval()
    with torch.no_grad():
        torch.testing.assert_close(encoded(sample, 1, cond, goal=goal_a, goal_valid=invalid),
                                   encoded(sample, 1, cond, goal=goal_b, goal_valid=invalid))


@pytest.mark.parametrize('variant', ['transformer_hybrid_image', 'unet_hybrid_image'])
def test_late_fusion_policy_encodes_the_goal_once_per_sample_and_is_goal_sensitive(variant):
    extra = {} if variant == 'transformer_hybrid_image' else dict(down_dims=(16, 32), diffusion_step_embed_dim=16)
    torch.manual_seed(4)
    config = goal_config('late', variant=variant, **extra)
    policy = build_policy(config, TASKS)
    from diffusion_policy.b1k.normalization import fit_normalizer
    normalizer = fit_normalizer([{'state': np.random.randn(50, 25).astype(np.float32),
                                  'action': np.random.randn(50, 23).astype(np.float32)}], config.cameras)
    normalizer['goal_head'] = normalizer['head']
    policy.set_normalizer(normalizer)
    obs = observations(config)
    assert policy.obs_encoder.goal_feature_dim == 64
    if variant == 'transformer_hybrid_image':
        assert policy.model.goal_as_cond and policy.model.T_cond == 4
    else:
        assert policy.model.global_cond_dim if hasattr(policy.model, 'global_cond_dim') else True
    calls = []
    original = policy.obs_encoder.encode_goals

    def spy(obs_dict):
        result = original(obs_dict)
        calls.append(result.shape)
        return result
    policy.obs_encoder.encode_goals = spy
    policy.eval()
    with torch.no_grad():
        torch.manual_seed(7)
        first = policy.predict_action(obs)['action']
        torch.manual_seed(7)
        same = policy.predict_action(obs)['action']
        torch.manual_seed(7)
        other = policy.predict_action({**obs, 'goal_head': torch.rand_like(obs['goal_head'])})['action']
    torch.testing.assert_close(first, same)
    assert not torch.allclose(first, other)
    # one (B, 64) goal encoding per predict_action call, outside the denoising loop (2 inference steps each)
    assert calls == [(2, 64)] * 3
    policy.train()
    loss = policy.compute_loss({'obs': obs, 'action': torch.randn(2, 4, 23)})
    loss.backward()
    assert torch.isfinite(loss) and calls[-1] == (2, 64)
    with pytest.raises(ValueError, match='needs'):
        policy.compute_loss({'obs': {key: value for key, value in obs.items() if key != 'goal_head'}, 'action': torch.randn(2, 4, 23)})
    with pytest.raises(ValueError, match='no observation-history axis'):
        policy.compute_loss({'obs': {**obs, 'goal_head': obs['goal_head'][:, None]}, 'action': torch.randn(2, 4, 23)})


def test_language_on_goal_encoder_false_runs_identity_film():
    from diffusion_policy.model.vision.clip_film import identity_initialize_film
    outputs = {}
    for on_goal in (False, True):
        torch.manual_seed(6)
        config = goal_config('late', regime='image_language', language_conditioning='clip_film', language_on_goal_encoder=on_goal)
        policy = build_policy(config, TASKS)
        identity_initialize_film(policy)
        policy.eval()
        obs = observations(config)
        last = {camera: obs[camera][:, -1] for camera in config.cameras}
        with torch.no_grad():
            outputs[on_goal] = policy.obs_encoder.encode_goals({**last, 'goal_head': obs['goal_head'], 'lang_emb': obs['lang_emb'][:, -1]})
    torch.testing.assert_close(outputs[False], outputs[True])
    torch.manual_seed(6)
    config = goal_config('late', regime='image_language', language_conditioning='clip_film')
    policy = build_policy(config, TASKS).eval()
    obs = observations(config)
    last = {camera: obs[camera][:, -1] for camera in config.cameras}
    with torch.no_grad():
        goal_a = policy.obs_encoder.encode_goals({**last, 'goal_head': obs['goal_head'], 'lang_emb': obs['lang_emb'][:, -1]})
        goal_b = policy.obs_encoder.encode_goals({**last, 'goal_head': obs['goal_head'], 'lang_emb': obs['lang_emb'][:, -1] + 1})
        torch.testing.assert_close(goal_a, goal_b)  # language does not reach the goal pass by default
        camera_a = policy.obs_encoder({**{k: v[:, -1] for k, v in obs.items() if k != 'goal_head'}})
        camera_b = policy.obs_encoder({**{k: v[:, -1] for k, v in obs.items() if k != 'goal_head'}, 'lang_emb': obs['lang_emb'][:, -1] + 1})
    assert not torch.allclose(camera_a[:, :-768], camera_b[:, :-768])  # ... but it modulates the observation pass


def test_config_and_regime_validation():
    with pytest.raises(ValueError, match='goal_views require'):
        small_config(goal_views=('head',)).validate()
    with pytest.raises(ValueError, match='distinct cameras'):
        small_config(regime='image', goal_fusion='late', goal_views=('right_wrist',)).validate()
    with pytest.raises(ValueError, match='hybrid image policies'):
        small_config(variant='unet_image', regime='image', goal_fusion='late', goal_views=('head',)).validate()
    with pytest.raises(ValueError, match='global observation conditioning'):
        small_config(regime='image', goal_fusion='late', goal_views=('head',), conditioning='inpainting').validate()
    with pytest.raises(ValueError, match='disagree'):
        small_config(regime='image').validate()
    with pytest.raises(ValueError, match='disagree'):
        small_config(regime='none', goal_fusion='late', goal_views=('head',)).validate()
    with pytest.raises(ValueError, match='language_on_goal_encoder'):
        small_config(language_on_goal_encoder=True).validate()
    with pytest.raises(ValueError, match='Several tasks'):
        build_policy(small_config(regime=None), TASKS)
    build_policy(small_config(regime='none'), TASKS)  # strict N: two tasks, no task signal, declared

    def resolve(*flags):
        args = parser().parse_args(['--dataset-path', 'x', '--output-dir', 'y', *flags])
        resolve_regime(args)
        return args
    image = resolve('--regime', 'image')
    assert image.goal_fusion == 'late' and image.goal_views == ('head',) and image.language_conditioning == 'none'
    language = resolve('--regime', 'language')
    assert language.language_conditioning == 'clip_film' and language.goal_fusion == 'none'
    early = resolve('--regime', 'image_language', '--goal-fusion', 'early', '--goal-views', 'head', 'left_wrist')
    assert early.goal_fusion == 'early' and early.goal_views == ('head', 'left_wrist') and early.language_conditioning == 'clip_film'
    assert resolve().goal_views == () and resolve().regime is None
    for flags, message in [(['--regime', 'none', '--goal-fusion', 'late'], 'excludes'),
                           (['--regime', 'image', '--goal-fusion', 'none'], 'requires'),
                           (['--regime', 'language', '--language-conditioning', 'none'], 'requires'),
                           (['--regime', 'image', '--language-conditioning', 'clip_film'], 'excludes'),
                           (['--goal-views', 'head'], 'require --goal-fusion')]:
        with pytest.raises(ValueError, match=message):
            resolve(*flags)


@pytest.mark.parametrize('fusion', ['late', 'early'])
def test_train_resume_serve_goal_conditioned_policy(root, tmp_path, fusion):
    config = goal_config(fusion, cameras=('head',), image_size=64, crop_shape=(56, 56))
    args = ['--dataset-path', str(root), '--device', 'cpu', '--num-workers', '0', '--batch-size', '2', '--cpu-threads', '1',
            '--export-every', '2', '--seed', '3', *config_flags(config)]
    full, split = tmp_path / 'full', tmp_path / 'split'
    train_main(args + ['--output-dir', str(full), '--max-steps', '2'])
    train_main(args + ['--output-dir', str(split), '--max-steps', '1'])
    config_json = json.loads((split / 'config.json').read_text())
    assert config_json['model']['goal_fusion'] == fusion and config_json['model']['goal_views'] == ['head']
    assert config_json['conditioning']['regime'] == 'image' and config_json['conditioning']['goal']['fusion'] == fusion
    assert config_json['conditioning']['task_onehot'] is False and 'source_commit' in config_json['conditioning']
    for option in (['--goal-fusion', 'none' if fusion == 'late' else 'late'], ['--goal-source', 'goal_key'], ['--regime', 'none']):
        with pytest.raises(ValueError, match='conflicts with the resumed checkpoint'):
            train_main(args + ['--output-dir', str(split), '--resume', str(split / 'latest.pt'), '--max-steps', '2', *option])
    train_main(args + ['--output-dir', str(split), '--resume', str(split / 'latest.pt'), '--max-steps', '2'])
    resumed = torch.load(split / 'latest.pt', weights_only=True)
    expected = torch.load(full / 'latest.pt', weights_only=True)
    assert resumed['conditioning']['goal']['views'] == ['head'] and resumed['config']['regime'] == 'image'
    for field in ('model', 'ema_model'):
        assert resumed[field].keys() == expected[field].keys()
        for key in resumed[field]:
            torch.testing.assert_close(resumed[field][key], expected[field][key], rtol=0, atol=0)
    exported = split / 'export_queue/eval/step-00000002.pt'
    assert exported.is_file()
    root.rename(root.with_name('dataset-unavailable'))
    policy, checkpoint = load_policy(exported)
    assert checkpoint['conditioning']['goal']['fusion'] == fusion
    session = B1KPolicySession(policy, checkpoint['config'], checkpoint['task_map'])
    obs = observation(1.0, task=3)
    with pytest.raises(ValueError, match='needs goal'):
        session.act(obs)
    goal_key_wire = GOAL_OBS_KEYS['head']
    obs[goal_key_wire] = np.full((16, 16, 3), 40, dtype=np.uint8)
    torch.manual_seed(11)
    action = B1KPolicySession(policy, checkpoint['config'], checkpoint['task_map']).act(obs)
    assert action.shape == (1, 23) and np.isfinite(action).all()
    # image regime: the task id selects nothing in the network
    torch.manual_seed(11)
    swapped = B1KPolicySession(policy, checkpoint['config'], checkpoint['task_map']).act({**obs, 'task_id': 9})
    np.testing.assert_array_equal(swapped, action)
    torch.manual_seed(11)
    other = B1KPolicySession(policy, checkpoint['config'], checkpoint['task_map']).act(
        {**obs, goal_key_wire: np.full((16, 16, 3), 200, dtype=np.uint8)})
    assert not np.array_equal(other, action)
    torch.manual_seed(11)
    fixed = B1KPolicySession(policy, checkpoint['config'], checkpoint['task_map'],
                             fixed_goals={'head': np.full((16, 16, 3), 40, np.uint8)}).act(observation(1.0, task=3))
    np.testing.assert_array_equal(fixed, action)
