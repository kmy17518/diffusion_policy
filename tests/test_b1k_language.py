import asyncio
import copy
import json
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from diffusion_policy.b1k import language
from diffusion_policy.b1k.dataset import B1KLeRobotDataset
from diffusion_policy.b1k.model import ModelConfig, build_policy, load_checkpoint, load_policy
from diffusion_policy.b1k.serve import B1KPolicySession
from diffusion_policy.b1k.train import main as train_main, parser
from diffusion_policy.model.vision.clip_film import FiLMLayer, FiLMResidualBlock, ResNet18FiLM
from test_b1k import observation, root


VARIANTS = ['transformer_hybrid_image', 'unet_hybrid_image', 'unet_image']


class FakeTokenizer:
    bos_token_id = 1000
    eos_token_id = pad_token_id = 1001

    def __call__(self, text, add_special_tokens=True, truncation=False, **kwargs):
        assert truncation is False
        ids = [int(word) for word in text.split()]
        if not add_special_tokens:
            return {'input_ids': ids}
        ids = [self.bos_token_id, *ids, self.eos_token_id]
        return {'input_ids': torch.tensor([ids]), 'attention_mask': torch.ones(1, len(ids), dtype=torch.long)}


class FakeTextEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(1))
        self.config = SimpleNamespace(max_position_embeddings=77, projection_dim=768,
                                      _commit_hash=language.CLIP_REVISION)
        self.calls = []

    def forward(self, input_ids, attention_mask):
        assert not self.training and not self.scale.requires_grad and not torch.is_grad_enabled()
        self.calls.append((input_ids.clone(), attention_mask.clone()))
        values = (input_ids.float() * attention_mask).sum(dim=1)
        return SimpleNamespace(text_embeds=values[:, None].expand(-1, 768) * self.scale)


@pytest.fixture
def fake_clip(monkeypatch):
    calls = []

    def encode(prompts):
        calls.append(list(prompts))
        result = torch.stack([torch.linspace(-0.8, 0.8, 768) + sum(prompt.encode()) / 10000 for prompt in prompts])
        return result, language.CLIP_REVISION

    monkeypatch.setattr(language, 'encode_clip_prompts', encode)
    return calls


def descriptions(root, rows=None):
    rows = rows if rows is not None else [
        {'task_index': 3, 'task_name': 'alpha', 'task': 'Turn on the radio receiver.'},
        {'task_index': 9, 'task_name': 'beta', 'task': 'Put the cans inside the trash can.'},
    ]
    (root / 'meta/tasks.jsonl').write_text('\n'.join(json.dumps(row) for row in rows))
    return rows


def small_config(variant, prompt_source='task_name', **kwargs):
    return ModelConfig(variant=variant, language_conditioning='clip_film', prompt_source=prompt_source,
                       horizon=4, n_action_steps=2, cameras=('head',), image_size=64, crop_shape=(56, 56),
                       down_dims=(16, 32), diffusion_step_embed_dim=16, n_layer=1, n_head=2, n_emb=16,
                       num_train_timesteps=4, num_inference_steps=2, **kwargs)


def test_prompt_source_exact_metadata(root, fake_clip):
    rows = descriptions(root)
    tasks = {9: 'beta', 3: 'alpha'}
    assert language.select_prompts(root, tasks, 'task_name') == ['alpha', 'beta']
    assert language.select_prompts(root, tasks, 'task_description') == [row['task'] for row in rows]
    assert language.select_prompts(root, {7: 'raw_SNAKE_case'}, 'task_name') == ['raw_SNAKE_case']
    cache = language.prepare_language(root, tasks, 'clip_film', 'task_description')
    assert len(fake_clip) == 1 and fake_clip[0] == cache['prompts']
    assert cache['revision'] == language.CLIP_REVISION and cache['model'] == language.CLIP_MODEL
    assert cache['embeddings'].shape == (2, 768) and not cache['embeddings'].requires_grad
    (root / 'meta/tasks.jsonl').unlink()
    restored = language.prepare_language(None, tasks, 'clip_film', 'task_description', cached=cache)
    torch.testing.assert_close(restored['embeddings'], cache['embeddings'])
    assert len(fake_clip) == 1
    with pytest.raises(ValueError, match='requires'):
        language.select_prompts(root, tasks, 'task_description')


@pytest.mark.parametrize('failure', ['duplicate_id', 'duplicate_name', 'conflicting_id', 'missing',
                                     'blank', 'nonstring', 'invalid_id', 'bad_json'])
def test_malformed_selected_prompts(root, failure):
    rows = descriptions(root)
    if failure == 'duplicate_id':
        rows.append(rows[0].copy())
    elif failure == 'duplicate_name':
        rows.append({**rows[0], 'task_index': 100})
    elif failure == 'conflicting_id':
        rows[0]['task_index'] = 100
    elif failure == 'missing':
        rows.pop()
    elif failure == 'blank':
        rows[0]['task'] = '   '
    elif failure == 'nonstring':
        rows[0]['task'] = None
    elif failure == 'invalid_id':
        rows[0]['task_index'] = '3'
    descriptions(root, rows)
    if failure == 'bad_json':
        (root / 'meta/tasks.jsonl').write_text('{')
    with pytest.raises(ValueError):
        language.select_prompts(root, {3: 'alpha', 9: 'beta'}, 'task_description')


def test_distinct_tasks_can_share_description(root, fake_clip):
    rows = descriptions(root)
    rows[1]['task'] = rows[0]['task']
    descriptions(root, rows)
    cache = language.prepare_language(root, {3: 'alpha', 9: 'beta'}, 'clip_film', 'task_description')
    assert cache['prompts'][0] == cache['prompts'][1]


@pytest.mark.parametrize('duplicate', ['task_index', 'task'])
def test_duplicate_parquet_tasks_rejected(root, duplicate):
    fields = {'task_index': [3, 9], 'task': ['alpha', 'beta']}
    fields[duplicate][1] = fields[duplicate][0]
    pq.write_table(pa.table(fields), root / 'meta/tasks.parquet')
    with pytest.raises(ValueError, match='Duplicate/conflicting'):
        B1KLeRobotDataset(root)


@pytest.mark.parametrize('length', [1, 75, 76, 151, 302])
def test_clip_short_standard_long_lossless_masked_chunks(length):
    tokenizer, model = FakeTokenizer(), FakeTextEncoder()
    content = list(range(1, length + 1))
    embeddings, revision = language.encode_clip_prompts(' '.join(map(str, content)).split('|'),
                                                       tokenizer=tokenizer, model=model)
    assert revision == language.CLIP_REVISION
    ids, masks = model.calls[0]
    assert len(ids) == (length + 74) // 75 and ids.shape[1] <= 77
    recovered = []
    sums = []
    for row, mask in zip(ids, masks):
        valid = row[mask.bool()].tolist()
        assert valid[0] == tokenizer.bos_token_id and valid[-1] == tokenizer.eos_token_id
        recovered += valid[1:-1]
        sums.append(sum(valid))
        assert torch.all(row[~mask.bool()] == tokenizer.pad_token_id)
    assert recovered == content
    torch.testing.assert_close(embeddings, torch.full((1, 768), float(np.mean(sums))))
    assert embeddings.norm() > 1 and not embeddings.requires_grad


def test_pretrained_loader_accepts_pinned_revision_with_missing_nested_commit(root, monkeypatch):
    import sys
    model, tokenizer, seen = FakeTextEncoder(), FakeTokenizer(), []
    model.config._commit_hash = None

    def load_model(name, revision):
        seen.append(('model', name, revision))
        return model

    def load_tokenizer(name, revision):
        seen.append(('tokenizer', name, revision))
        return tokenizer

    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(
        CLIPTextModelWithProjection=SimpleNamespace(from_pretrained=load_model),
        AutoTokenizer=SimpleNamespace(from_pretrained=load_tokenizer)))
    cache = language.prepare_language(root, {3: '1 2 3'}, 'clip_film')
    assert cache['revision'] == language.CLIP_REVISION
    assert seen == [('model', language.CLIP_MODEL, language.CLIP_REVISION),
                    ('tokenizer', language.CLIP_MODEL, language.CLIP_REVISION)]
    assert cache['embeddings'].shape == (1, 768)


def test_film_equation_and_each_residual_block():
    film = FiLMLayer(2)
    with torch.no_grad():
        film.lang_proj.weight.zero_()
        film.lang_proj.bias.copy_(torch.tensor([2., -3., 0.5, -0.5]))
    image = torch.tensor([[[[2.]], [[4.]]]])
    torch.testing.assert_close(film(image, torch.zeros(1, 768)), torch.tensor([[[[5.]], [[0.]]]]))
    backbone = ResNet18FiLM()
    assert len(backbone.blocks) == 8
    assert all(isinstance(block, FiLMResidualBlock) for block in backbone.blocks)
    assert [block.film.lang_proj.out_features for block in backbone.blocks] == [128, 128, 256, 256, 512, 512, 1024, 1024]


def test_checkpoint_recomputation_gradient_parity_and_batchnorm_guard():
    torch.manual_seed(2)
    checkpointed = ResNet18FiLM().train()
    direct = copy.deepcopy(checkpointed)
    direct.checkpoint_blocks = False
    image, text = torch.randn(2, 3, 32, 32), torch.randn(2, 768)
    left, right = checkpointed(image, text), direct(image, text)
    torch.testing.assert_close(left, right, rtol=0, atol=0)
    left.square().mean().backward()
    right.square().mean().backward()
    for actual, expected in zip(checkpointed.parameters(), direct.parameters()):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)
    assert not ResNet18FiLM(group_norm=False).checkpoint_blocks


@pytest.mark.parametrize('training', [False, True])
@pytest.mark.parametrize('image_shape', [(56, 56), (86, 86), (224, 192)])
def test_hybrid_pooling_matches_robomimic_exactly(training, image_shape):
    from robomimic.models.base_nets import SpatialSoftmax as ReferencePool
    from diffusion_policy.model.vision.clip_film import SpatialSoftmax
    height, width = [(size + 31) // 32 for size in image_shape]
    reference = ReferencePool([512, height, width], num_kp=32)
    actual = SpatialSoftmax(image_shape)
    actual.projection.load_state_dict(reference.nets.state_dict())
    actual.train(training)
    reference.train(training)
    image = torch.randn(2, 512, height, width, requires_grad=True)
    other = image.detach().clone().requires_grad_(True)
    torch.manual_seed(29)
    left = actual(image)
    torch.manual_seed(29)
    right = reference(other).flatten(1)
    torch.testing.assert_close(left, right, rtol=0, atol=0)
    left.square().sum().backward()
    right.square().sum().backward()
    torch.testing.assert_close(image.grad, other.grad, rtol=0, atol=0)
    torch.testing.assert_close(actual.projection.weight.grad, reference.nets.weight.grad, rtol=0, atol=0)


@pytest.mark.parametrize('training,fixed', [(False, True), (False, False), (True, True), (True, False)])
def test_hybrid_visual_core_zero_film_matches_upstream(training, fixed):
    from dataclasses import replace
    config = small_config('transformer_hybrid_image', eval_fixed_crop=fixed)
    old = build_policy(replace(config, language_conditioning='none'), {3: 'alpha', 9: 'beta'}).obs_encoder
    new = build_policy(config, {3: 'alpha', 9: 'beta'}).obs_encoder
    original, conditioned = old.obs_nets['head'], new.encoders['head']
    conditioned.stem.load_state_dict(torch.nn.Sequential(*list(original.backbone.nets.children())[:4]).state_dict())
    original_blocks = [block for stage in list(original.backbone.nets.children())[4:] for block in stage]
    for block, reference in zip(conditioned.blocks, original_blocks):
        block.block.load_state_dict(reference.state_dict())
        torch.nn.init.zeros_(block.film.lang_proj.weight)
        torch.nn.init.zeros_(block.film.lang_proj.bias)
    conditioned.pool[0].projection.load_state_dict(original.pool.nets.state_dict())
    conditioned.pool[1].load_state_dict(original.nets[-1].state_dict())
    old.train(training)
    new.train(training)
    obs = {'state': torch.randn(2, 27), 'head': torch.randn(2, 3, 64, 64), 'lang_emb': torch.randn(2, 768)}
    torch.manual_seed(97)
    expected = old(obs)
    torch.manual_seed(97)
    result = new(obs)
    torch.testing.assert_close(result[:, :-768], expected, rtol=0, atol=0)
    torch.testing.assert_close(result[:, -768:], obs['lang_emb'], rtol=0, atol=0)


@pytest.mark.parametrize('variant', VARIANTS)
def test_language_sensitivity_gradients_and_lowdim_concatenation(root, fake_clip, variant):
    config = small_config(variant)
    dataset = B1KLeRobotDataset(root, **config.dataset_kwargs())
    dataset.prepare_language()
    sample = dataset[0]
    policy = build_policy(config, dataset.task_map)
    policy.set_normalizer(dataset.get_normalizer())
    obs = {key: value[None].repeat(2, *([1] * value.ndim)) for key, value in sample['obs'].items()}
    policy.eval()
    normalized = policy.normalizer.normalize({key: value[:, 0] for key, value in obs.items()})
    torch.testing.assert_close(normalized['lang_emb'], obs['lang_emb'][:, 0], rtol=0, atol=0)
    with torch.no_grad():
        first = policy.obs_encoder(normalized)
        normalized['lang_emb'] = normalized['lang_emb'] + 1
        second = policy.obs_encoder(normalized)
    assert (first - second).abs().max() > 0.1
    # Hold state and language's concatenated slot fixed when measuring visual FiLM sensitivity.
    if variant == 'unet_image':
        assert not torch.allclose(first[:, :512], second[:, :512])
        torch.testing.assert_close(first[:, 512:1280], obs['lang_emb'][:, 0])
    else:
        assert not torch.allclose(first[:, 27:91], second[:, 27:91])
        torch.testing.assert_close(first[:, -768:], obs['lang_emb'][:, 0])
    assert sample['obs']['state'].shape[-1] == 27
    torch.manual_seed(17)
    action1 = policy.predict_action(obs)['action']
    torch.manual_seed(17)
    action2 = policy.predict_action({**obs, 'lang_emb': obs['lang_emb'] + 1})['action']
    assert not torch.allclose(action1, action2)
    policy.train()
    loss = policy.compute_loss({'obs': obs, 'action': sample['action'][None].repeat(2, 1, 1)})
    loss.backward()
    films = [module for module in policy.modules() if isinstance(module, FiLMLayer)]
    assert len(films) == 8 and torch.isfinite(loss)
    for film in films:
        gradient = film.lang_proj.weight.grad
        assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    dataset.close()


def test_film_init_identity_zeroes_projections_and_is_recorded(root, tmp_path, fake_clip):
    from dataclasses import replace
    from diffusion_policy.b1k.variant_matrix import config_flags
    from diffusion_policy.model.vision.clip_film import identity_initialize_film
    config = small_config('transformer_hybrid_image')
    torch.manual_seed(3)
    policy = build_policy(config, {3: 'alpha'})
    encoder = policy.obs_encoder.encoders['head']
    images = torch.rand(2, 3, 56, 56)
    assert not torch.equal(encoder(images, torch.randn(2, 768)), encoder(images, torch.randn(2, 768)))
    assert identity_initialize_film(policy) == 8           # one camera, eight residual blocks
    assert all(not module.lang_proj.weight.any() and not module.lang_proj.bias.any()
               for module in policy.modules() if isinstance(module, FiLMLayer))
    # beta = gamma = 0: the encoder no longer depends on the language embedding at all.
    torch.testing.assert_close(encoder(images, torch.randn(2, 768)), encoder(images, torch.randn(2, 768)), rtol=0, atol=0)
    args = ['--dataset-path', str(root), '--device', 'cpu', '--num-workers', '0', '--batch-size', '2',
            '--cpu-threads', '1', '--export-every', '0', '--max-steps', '1']
    output = tmp_path / 'identity'
    train_main(args + ['--output-dir', str(output), '--film-init', 'identity', *config_flags(config)])
    assert json.loads((output / 'config.json').read_text())['training']['film_init'] == 'identity'
    with pytest.raises(ValueError, match='film-init identity requires'):
        train_main(args + ['--output-dir', str(tmp_path / 'none'), '--film-init', 'identity',
                           *config_flags(replace(config, language_conditioning='none'))])
    assert parser().parse_args(['--dataset-path', 'unused', '--output-dir', 'unused']).film_init == 'random'


def test_film_recompute_flag_is_numerically_identical_and_recorded(root, tmp_path, fake_clip):
    from diffusion_policy.b1k.variant_matrix import config_flags
    config = small_config('transformer_hybrid_image')
    args = ['--dataset-path', str(root), '--device', 'cpu', '--num-workers', '0', '--batch-size', '2',
            '--cpu-threads', '1', '--export-every', '0', '--max-steps', '2', *config_flags(config)]
    recomputed, stored = tmp_path / 'recomputed', tmp_path / 'stored'
    train_main(args + ['--output-dir', str(recomputed)])
    train_main(args + ['--output-dir', str(stored), '--no-film-recompute'])
    first, second = load_checkpoint(recomputed), load_checkpoint(stored)
    for field in ('model', 'ema_model'):
        assert first[field].keys() == second[field].keys()
        for key, value in first[field].items():
            torch.testing.assert_close(value, second[field][key], rtol=0, atol=0)
    for path, expected in ((recomputed, True), (stored, False)):
        assert json.loads((path / 'config.json').read_text())['training']['film_recompute'] is expected
    # The switch is a runtime choice: a checkpoint trained one way resumes the other way.
    train_main(args[:args.index('--max-steps')] + ['--max-steps', '3', '--output-dir', str(stored),
                                                    '--resume', str(stored), *config_flags(config)])
    assert load_checkpoint(stored)['step'] == 3
    assert parser().parse_args(['--dataset-path', 'unused', '--output-dir', 'unused']).film_recompute is True


@pytest.mark.parametrize('variant', ['unet_lowdim', 'transformer_lowdim', 'unet_video'])
def test_language_rejects_unsupported_variants(variant):
    with pytest.raises(ValueError, match='clip_film supports only'):
        build_policy(ModelConfig(variant=variant, language_conditioning='clip_film'), {3: 'alpha'})


def test_language_rejects_detached_or_frozen_encoders():
    with pytest.raises(ValueError, match='detaches'):
        small_config('transformer_hybrid_image', conditioning='inpainting').validate()
    with pytest.raises(ValueError, match='trainable'):
        small_config('unet_image', freeze_encoder=True).validate()


@pytest.mark.parametrize('variant', VARIANTS)
def test_none_mode_checkpoint_keys_and_initialization_unchanged(variant):
    config = small_config(variant).to_dict()
    config['language_conditioning'] = 'none'
    legacy = {key: value for key, value in config.items() if key not in ('language_conditioning', 'prompt_source')}
    torch.manual_seed(21)
    before = build_policy(legacy, {3: 'alpha'})
    torch.manual_seed(21)
    after = build_policy(config, {3: 'alpha'})
    assert before.state_dict().keys() == after.state_dict().keys()
    for key, value in before.state_dict().items():
        torch.testing.assert_close(value, after.state_dict()[key], rtol=0, atol=0)
    assert not any(isinstance(module, FiLMLayer) for module in after.modules())
    args = parser().parse_args(['--dataset-path', 'unused', '--output-dir', 'unused'])
    assert args.language_conditioning == 'none' and args.prompt_source == 'task_name'


@pytest.mark.parametrize('prompt_source', ['task_name', 'task_description'])
@pytest.mark.parametrize('variant', VARIANTS)
def test_language_train_checkpoint_resume_and_dataset_free_serving(root, tmp_path, monkeypatch, fake_clip,
                                                                 variant, prompt_source):
    from diffusion_policy.b1k.variant_matrix import config_flags, deny_dataset_access, websocket_roundtrip
    config = small_config(variant, prompt_source)
    descriptions(root)
    output, full = tmp_path / 'resumed', tmp_path / 'full'
    args = ['--dataset-path', str(root), '--device', 'cpu', '--num-workers', '0', '--batch-size', '2',
            '--cpu-threads', '1', '--export-every', '1', *config_flags(config)]
    train_main(args + ['--output-dir', str(output), '--max-steps', '1'])
    initial = load_checkpoint(output)
    assert len(fake_clip) == 1
    for flag, value in [('--language-conditioning', 'none'),
                        ('--prompt-source', 'task_name' if prompt_source == 'task_description' else 'task_description')]:
        with pytest.raises(ValueError, match='conflicts'):
            train_main(args + ['--output-dir', str(output), '--max-steps', '2', '--resume', str(output), flag, value])
    resume_args = args.copy()
    for flag in ('--language-conditioning', '--prompt-source'):
        index = resume_args.index(flag)
        del resume_args[index:index + 2]
    (root / 'meta/tasks.jsonl').unlink()
    train_main(resume_args + ['--output-dir', str(output), '--max-steps', '2', '--resume', str(output)])
    assert len(fake_clip) == 1
    resumed = load_checkpoint(output)
    descriptions(root)
    train_main(args + ['--output-dir', str(full), '--max-steps', '2'])
    reference = load_checkpoint(full)
    for field in ('model', 'ema_model'):
        for key, value in resumed[field].items():
            torch.testing.assert_close(value, reference[field][key], rtol=0, atol=0)
    for key in initial['language']:
        if key == 'embeddings':
            torch.testing.assert_close(initial['language'][key], resumed['language'][key], rtol=0, atol=0)
        else:
            assert initial['language'][key] == resumed['language'][key]
    assert resumed['config']['prompt_source'] == prompt_source
    assert resumed['language']['prompts'] != resumed['language']['task_names'] or prompt_source == 'task_name'
    monkeypatch.setattr(language, 'encode_clip_prompts', lambda *args, **kwargs: pytest.fail('Text encoder used offline'))
    with deny_dataset_access(root):
        for path in [output, output / 'export_queue/eval/step-00000002.pt']:
            policy, checkpoint = load_policy(path)
            torch.testing.assert_close(checkpoint['language']['embeddings'], resumed['language']['embeddings'])
            if path != output:
                assert checkpoint['checkpoint_type'] == 'eval' and 'optimizer' not in checkpoint
                wire = []
                for step in range(3):
                    item = observation(step + 1)
                    wire.append({key: value[0] if isinstance(value, np.ndarray) else value for key, value in item.items()})
                evidence = asyncio.run(websocket_roundtrip(policy, checkpoint, wire))
                assert len(evidence['replies']) == 8 and evidence['client_isolation']
                assert evidence['handshake']['language_conditioning'] == 'clip_film'
                assert evidence['handshake']['prompt_source'] == prompt_source
            wrapper = B1KPolicySession(policy, checkpoint['config'], checkpoint['task_map'])
            first = wrapper.act(observation(1, task=[3, 9], batch=2))
            assert first.shape == (2, 23) and np.isfinite(first).all()
            torch.testing.assert_close(torch.from_numpy(wrapper.histories[0][-1]['lang_emb']),
                                       resumed['language']['embeddings'][0])
            wrapper.act(observation(2, task=[9, 3], batch=2))
            assert len(wrapper.histories[0]) == config.n_obs_steps
            torch.testing.assert_close(torch.from_numpy(wrapper.histories[0][0]['lang_emb']),
                                       resumed['language']['embeddings'][1])
            other = B1KPolicySession(policy, checkpoint['config'], checkpoint['task_map'])
            other.act(observation(3))
            assert other.histories[0][-1]['state'][0] != wrapper.histories[0][-1]['state'][0]


@pytest.mark.parametrize('variant,conditioning,shared,batchnorm,scheduler', [
    ('unet_image', 'inpainting', False, False, 'ddim'),
    ('unet_image', 'global', True, True, 'ddpm'),
    ('unet_hybrid_image', 'inpainting', True, False, 'ddim'),
    ('unet_hybrid_image', 'global', True, True, 'ddpm'),
    ('transformer_hybrid_image', 'global', True, True, 'ddim'),
])
def test_language_additional_image_branches(root, fake_clip, variant, conditioning, shared, batchnorm, scheduler):
    from dataclasses import replace
    config = replace(small_config(variant, conditioning=conditioning, share_rgb_model=shared,
                                   obs_encoder_group_norm=not batchnorm, scheduler=scheduler),
                     cameras=('head', 'left_wrist'))
    dataset = B1KLeRobotDataset(root, **config.dataset_kwargs())
    dataset.prepare_language()
    dataset.prepare_language()
    assert len(fake_clip) == 1
    policy = build_policy(config, dataset.task_map)
    policy.set_normalizer(dataset.get_normalizer())
    sample = dataset[0]
    obs = {key: value[None].repeat(2, *([1] * value.ndim)) for key, value in sample['obs'].items()}
    loss = policy.compute_loss({'obs': obs, 'action': sample['action'][None].repeat(2, 1, 1)})
    loss.backward()
    assert torch.isfinite(loss)
    for module in policy.modules():
        if isinstance(module, FiLMLayer):
            assert module.lang_proj.weight.grad is not None
    policy.eval()
    result = policy.predict_action({key: value[:, :config.n_obs_steps] for key, value in obs.items()})
    assert result['action'].shape == (2, 2, 23) and torch.isfinite(result['action']).all()
    dataset.close()


def test_cached_language_survives_spawn_loader_without_encoder(root, fake_clip):
    from torch.utils.data import DataLoader
    config = small_config('transformer_hybrid_image')
    dataset = B1KLeRobotDataset(root, **config.dataset_kwargs())
    dataset.prepare_language()
    sample = next(iter(DataLoader(dataset, batch_size=2, num_workers=1, multiprocessing_context='spawn')))
    assert sample['obs']['lang_emb'].shape == (2, 2, 768)
    torch.testing.assert_close(sample['obs']['lang_emb'][0, 0], dataset.language['embeddings'][0])
    assert len(fake_clip) == 1
    dataset.close()


@pytest.mark.parametrize('missing', [True, False])
def test_resume_rejects_missing_or_corrupt_language_before_dataset(tmp_path, monkeypatch, missing):
    from diffusion_policy.b1k import train
    config = small_config('transformer_hybrid_image')
    checkpoint = {'format': 'diffusion_policy_b1k_v1', 'config': config.to_dict(), 'task_map': {3: 'alpha'}}
    if not missing:
        checkpoint['language'] = {'format': 'corrupt'}
    path = tmp_path / 'bad.pt'
    torch.save(checkpoint, path)
    monkeypatch.setattr(language, 'encode_clip_prompts', lambda *args: pytest.fail('Missing cache re-encoded'))
    monkeypatch.setattr(train, 'B1KLeRobotDataset', lambda *args, **kwargs: pytest.fail('Dataset opened'))
    with pytest.raises(ValueError, match='language cache'):
        train_main(['--dataset-path', str(tmp_path / 'missing'), '--output-dir', str(tmp_path / 'output'),
                    '--device', 'cpu', '--resume', str(path), '--cpu-threads', '1'])


@pytest.mark.parametrize('field,value', [('revision', ''), ('revision', 'main'), ('prompt_source', 'task_description'),
                                         ('embeddings', torch.zeros(2, 512)),
                                         ('embeddings', torch.full((2, 768), float('nan'))),
                                         ('prompts', ['alpha', 'alpha']), ('task_ids', [9, 3])])
def test_invalid_checkpoint_language_rejected(root, fake_clip, field, value):
    cache = language.prepare_language(root, {3: 'alpha', 9: 'beta'}, 'clip_film')
    cache[field] = value
    with pytest.raises(ValueError):
        language.validate_language_cache(cache, {3: 'alpha', 9: 'beta'}, 'task_name')
