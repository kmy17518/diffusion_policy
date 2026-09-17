"""Paired, short DP initialization diagnostics; no rollout or action-error evaluation."""

import argparse
import copy
from dataclasses import dataclass, replace
from functools import partial
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
import uuid

import numpy as np
import torch
from torch.utils.data import DataLoader, default_collate

from diffusion_policy.b1k.dataset import B1KLeRobotDataset, EpisodeSequenceIndex
from diffusion_policy.b1k.language import LANGUAGE_KEY, validate_language_cache
from diffusion_policy.b1k.model import ModelConfig, build_policy
from diffusion_policy.b1k.train import (
    StepBatchSampler, atomic_save, configure_cpu_threads, output_lock, seed_worker,
    sync_batchnorm_buffers,
)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.normalizer import SingleFieldLinearNormalizer
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.vision.clip_film import FiLMLayer


ARMS = ('baseline', 'random_film', 'identity_film')
PRODUCTION_BATCH_SIZE = 8960


def production_config():
    return ModelConfig(variant='transformer_hybrid_image', n_layer=12, n_emb=512, n_head=8,
                       image_size=96, crop_shape=(86, 86), horizon=16, n_obs_steps=2,
                       n_action_steps=8, num_train_timesteps=100, num_inference_steps=100)


def tensor_hash(value):
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256(str((str(value.dtype), tuple(value.shape))).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def state_hash(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode())
        digest.update(tensor_hash(value).encode())
    return digest.hexdigest()


def named_tensors(module):
    return dict(list(module.named_parameters(remove_duplicate=False)) +
                list(module.named_buffers(remove_duplicate=False)))


def feature_layout(baseline, conditioned):
    """Read actual concatenation orders, not state-dict names or a prefix assumption."""
    old, new = baseline.obs_encoder, conditioned.obs_encoder
    old_cameras = [key for key in old.obs_shapes if old.obs_nets[key] is not None]
    new_cameras = [key for key in new.shapes if key in new.encoders]
    if old_cameras != new_cameras:
        raise ValueError('Camera execution order differs: paired crop/keypoint RNG would be invalid')
    layouts = []
    for shapes, cameras in ((old.obs_shapes, old_cameras), (new.shapes, list(new.encoders))):
        layout, offset = {}, 0
        for key, shape in shapes.items():
            if isinstance(shape, dict):
                shape = shape['shape']
            width = 64 if key in cameras else math.prod(shape)
            layout[key] = [offset, offset + width]
            offset += width
        layouts.append(layout)
    left, right = layouts
    if set(right) != set(left) | {LANGUAGE_KEY}:
        raise ValueError('Unexpected shared observation fields')
    for key in left:
        if left[key][1] - left[key][0] != right[key][1] - right[key][0]:
            raise ValueError(f'Feature width mismatch for {key}')
    if (max(end for _, end in left.values()) != baseline.obs_feature_dim or
            max(end for _, end in right.values()) != conditioned.obs_feature_dim):
        raise ValueError('Feature layout does not cover the encoder output')
    return {'baseline': left, 'conditioned': right,
            'columns': [{'feature': key, 'source': left[key], 'target': right[key]} for key in left],
            'language_columns': right[LANGUAGE_KEY], 'camera_execution_order': old_cameras}


@torch.no_grad()
def copy_shared_state(baseline, conditioned):
    """Map every shared tensor, including robomimic's duplicate state-dict aliases."""
    layout = feature_layout(baseline, conditioned)
    sources, targets = named_tensors(baseline), named_tensors(conditioned)
    target_names = {id(value): key for key, value in reversed(list(targets.items()))}
    mapped = {}

    def pair(source, target):
        if source.shape != target.shape:
            raise ValueError('Shared tensor shape mismatch')
        if id(source) in mapped and mapped[id(source)] is not target:
            raise ValueError('Conflicting shared tensor aliases')
        target.copy_(source)
        mapped[id(source)] = target

    def module_pair(source, target):
        left, right = named_tensors(source), named_tensors(target)
        if left.keys() != right.keys():
            raise ValueError('Shared module structure differs')
        for key in left:
            pair(left[key], right[key])

    for camera in layout['camera_execution_order']:
        old = baseline.obs_encoder.obs_nets[camera]
        new = conditioned.obs_encoder.encoders[camera]
        if len(old.backbone.nets) != 8 or len(new.blocks) != 8:
            raise ValueError('Expected the original ResNet18 backbone')
        for index in range(4):
            module_pair(old.backbone.nets[index], new.stem[index])
        for stage in range(4):
            for block in range(2):
                module_pair(old.backbone.nets[stage + 4][block], new.blocks[2 * stage + block].block)
        module_pair(old.pool.nets, new.pool[0].projection)
        for key in ('pos_x', 'pos_y', 'temperature'):
            pair(getattr(old.pool, key), getattr(new.pool[0], key))
        module_pair(old.nets[-1], new.pool[1])

    widened = 'model.cond_obs_emb.weight'
    for key, source in sources.items():
        if not key.startswith('obs_encoder.') and key != widened:
            if key not in targets:
                raise ValueError(f'Unmapped common tensor: {key}')
            pair(source, targets[key])
    old_weight, new_weight = sources[widened], targets[widened]
    new_weight.zero_()
    for entry in layout['columns']:
        a, b = entry['source']
        c, d = entry['target']
        new_weight[:, c:d].copy_(old_weight[:, a:b])
    if set(map(id, sources.values())) != set(mapped) | {id(old_weight)}:
        missing = [key for key, value in sources.items() if id(value) not in mapped and key != widened]
        raise ValueError(f'Unmapped baseline parameters/buffers: {missing}')
    allowed_extra = {id(value) for key, value in targets.items()
                     if '.film.lang_proj.' in key or key.startswith(f'normalizer.params_dict.{LANGUAGE_KEY}.')}
    if set(map(id, targets.values())) != {id(value) for value in mapped.values()} | allowed_extra | {id(new_weight)}:
        raise ValueError('Unexpected conditioned-only parameters/buffers')

    reconstructed = {}
    entries = []
    for key, source in sources.items():
        if key == widened:
            target = torch.cat([new_weight[:, entry['target'][0]:entry['target'][1]]
                                for entry in layout['columns']], dim=1)
            target_name = widened
        else:
            target = mapped[id(source)]
            target_name = target_names[id(target)]
        if not torch.equal(source, target):
            raise ValueError(f'Shared copy is not exact: {key}')
        reconstructed[key] = target
        entries.append({'source': key, 'target': target_name, 'shape': list(source.shape)})
    c, d = layout['language_columns']
    if torch.count_nonzero(new_weight[:, c:d]):
        raise ValueError('Language columns must start at zero')
    return {'layout': layout, 'mapping': entries, 'all_shared_exact': True,
            'baseline_shared_sha256': state_hash(sources),
            'conditioned_shared_sha256': state_hash(reconstructed),
            'zero_language_columns': True}


def build_arms(config, task_map, normalizer, seed=42):
    if (config.variant != 'transformer_hybrid_image' or config.conditioning != 'global' or
            not config.obs_encoder_group_norm or not config.eval_fixed_crop or config.pred_action_steps_only):
        raise ValueError('Comparison requires the global GroupNorm transformer hybrid recipe with fixed eval crops')
    # Construction is CPU-only; CLIP preparation and the sampler cannot change this seed.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        baseline = build_policy(replace(config, language_conditioning='none'), task_map)
        base_normalizer = copy.deepcopy(normalizer)
        if LANGUAGE_KEY in base_normalizer.params_dict:
            del base_normalizer.params_dict[LANGUAGE_KEY]
        baseline.set_normalizer(base_normalizer)
        random_film = build_policy(replace(config, language_conditioning='clip_film'), task_map)
        lang_normalizer = copy.deepcopy(base_normalizer)
        lang_normalizer[LANGUAGE_KEY] = SingleFieldLinearNormalizer.create_identity()
        random_film.set_normalizer(lang_normalizer)
        proof = copy_shared_state(baseline, random_film)
        identity_film = copy.deepcopy(random_film)
        for module in identity_film.modules():
            if isinstance(module, FiLMLayer):
                torch.nn.init.zeros_(module.lang_proj.weight)
                torch.nn.init.zeros_(module.lang_proj.bias)
    arms = dict(zip(ARMS, (baseline, random_film, identity_film)))
    for policy in arms.values():
        policy.normalizer.requires_grad_(False)
    random_state, identity_state = random_film.state_dict(), identity_film.state_dict()
    differences = [key for key in random_state if not torch.equal(random_state[key], identity_state[key])]
    if not differences or any('.film.lang_proj.' not in key for key in differences):
        raise ValueError('Language arms must differ only in FiLM initialization')
    proof.update({'seed': seed, 'random_identity_differences': differences,
                  'identity_shared_exact': True,
                  'initial_state_sha256': {arm: state_hash(policy.state_dict()) for arm, policy in arms.items()}})
    return arms, proof


@dataclass
class RNGState:
    cpu: torch.Tensor
    cuda: torch.Tensor | None
    device: torch.device

    @classmethod
    def capture(cls, device):
        device = torch.device(device)
        return cls(torch.get_rng_state(), torch.cuda.get_rng_state(device) if device.type == 'cuda' else None, device)

    @classmethod
    def seeded(cls, seed, device):
        device = torch.device(device)
        return cls(torch.Generator().manual_seed(seed).get_state(),
                   torch.Generator(device=device).manual_seed(seed).get_state() if device.type == 'cuda' else None,
                   device)

    def restore(self):
        torch.set_rng_state(self.cpu)
        if self.cuda is not None:
            torch.cuda.set_rng_state(self.cuda, self.device)

    def hashes(self):
        return {'cpu': tensor_hash(self.cpu), 'cuda': tensor_hash(self.cuda) if self.cuda is not None else None}


def arm_batch(batch, arm):
    if arm == 'baseline':
        return {**batch, 'obs': {key: value for key, value in batch['obs'].items() if key != LANGUAGE_KEY}}
    return batch


def assert_rng_equal(expected, actual):
    if expected.hashes() != actual.hashes():
        raise RuntimeError('Arm RNG consumption differs; crop/keypoint/dropout/noise pairing is invalid')


def initial_parity(arms, batch, layout, seed):
    device = arms['baseline'].device
    saved = RNGState.capture(device)
    modes = {arm: policy.training for arm, policy in arms.items()}
    report = {}
    try:
        for training in (False, True):
            results, states = {}, {}
            rng = RNGState.seeded(seed + int(training), device)
            for arm, policy in arms.items():
                policy.train(training)
                captured = {}

                def capture_inputs(module, inputs):
                    captured.update(noisy=inputs[0].detach().clone(), timesteps=inputs[1].detach().clone(),
                                    features=inputs[2].detach().clone())

                def capture_output(module, inputs, output):
                    captured['prediction'] = output.detach().clone()

                hooks = [policy.model.register_forward_pre_hook(capture_inputs),
                         policy.model.register_forward_hook(capture_output)]
                try:
                    rng.restore()
                    with torch.set_grad_enabled(training):
                        loss = policy.compute_loss(arm_batch(batch, arm))
                    captured['loss'] = float(loss.detach())
                    del loss
                finally:
                    for hook in hooks:
                        hook.remove()
                states[arm] = RNGState.capture(device)
                results[arm] = captured
            reference, identity = results['baseline'], results['identity_film']
            for arm in ARMS[1:]:
                assert_rng_equal(states['baseline'], states[arm])
                for key in ('noisy', 'timesteps'):
                    if not torch.equal(reference[key], results[arm][key]):
                        raise RuntimeError(f'Unpaired initial {key} in {arm}')
            shared = torch.cat([identity['features'][..., c:d]
                                for c, d in (entry['target'] for entry in layout['columns'])], dim=-1)
            torch.testing.assert_close(shared, reference['features'], rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(identity['prediction'], reference['prediction'], rtol=1e-5, atol=2e-6)
            if not math.isclose(reference['loss'], identity['loss'], rel_tol=1e-6, abs_tol=1e-7):
                raise RuntimeError('Identity FiLM does not reproduce the baseline initial loss')
            if not all(math.isfinite(result['loss']) for result in results.values()):
                raise RuntimeError('Non-finite initialization loss')
            report['train' if training else 'eval'] = {
                'loss': {arm: result['loss'] for arm, result in results.items()},
                'identity_prediction_max_abs': float((identity['prediction'] - reference['prediction']).abs().max()),
                'identity_features_max_abs': float((shared - reference['features']).abs().max()),
                'paired_noise_timesteps_exact': True, 'paired_rng_exact': True,
                'rng_before': rng.hashes(), 'rng_after': states['baseline'].hashes(),
                'noisy_input_sha256': tensor_hash(reference['noisy']),
                'timesteps_sha256': tensor_hash(reference['timesteps']),
            }
    finally:
        for arm, policy in arms.items():
            policy.train(modes[arm])
        saved.restore()
    return report


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def train_step(arms, optimizers, emas, batch, seed, step, emit, data_wait_s=0.0):
    device = arms['baseline'].device
    rng, after = RNGState.seeded(seed, device), None
    for arm, policy in arms.items():
        policy.train()
        optimizer = optimizers[arm]
        optimizer.zero_grad(set_to_none=True)
        synchronize(device)
        start = time.monotonic()
        record = {'arm': arm, 'step': step, 'rng_seed': seed, 'loss': None, 'grad_norm': None,
                  'clip_applied': False, 'finite_fail': False, 'data_wait_s': data_wait_s,
                  'learning_rate': optimizer.param_groups[0]['lr']}
        try:
            rng.restore()
            loss = policy.compute_loss(arm_batch(batch, arm))
            current = RNGState.capture(device)
            if after is not None:
                assert_rng_equal(after, current)
            after = current
            synchronize(device)
            record['forward_s'] = time.monotonic() - start
            if not torch.isfinite(loss):
                record['finite_fail'] = True
                raise FloatingPointError('Non-finite loss')
            record['loss'] = float(loss.detach())
            backward_start = time.monotonic()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            if not torch.isfinite(norm):
                record['finite_fail'] = True
                raise FloatingPointError('Non-finite gradient norm')
            record.update(grad_norm=float(norm), clip_applied=bool(norm > 1.0))
            synchronize(device)
            record['backward_s'] = time.monotonic() - backward_start
            update_start = time.monotonic()
            optimizer.step()
            emas[arm].step(policy)
            sync_batchnorm_buffers(policy, emas[arm].averaged_model)
            synchronize(device)
            record['update_s'] = time.monotonic() - update_start
            del loss, norm
        except Exception as error:
            record.update(status='failed', error_type=type(error).__name__, compute_s=time.monotonic() - start)
            emit(arm, record)
            raise
        record.update(status='ok', compute_s=time.monotonic() - start, paired_rng_exact=True)
        emit(arm, record)
        optimizer.zero_grad(set_to_none=True)


@torch.no_grad()
def evaluate(arms, batches, seed):
    device = arms['baseline'].device
    saved = RNGState.capture(device)
    modes = {arm: policy.training for arm, policy in arms.items()}
    totals, count = dict.fromkeys(arms, 0.0), 0
    try:
        for policy in arms.values():
            policy.eval()
        for index, cpu_batch in enumerate(batches):
            batch = dict_apply(cpu_batch, lambda value: value.to(device))
            size = len(batch['action'])
            rng, after = RNGState.seeded(seed + index, device), None
            for arm, policy in arms.items():
                rng.restore()
                loss = policy.compute_loss(arm_batch(batch, arm))
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Non-finite evaluation loss for {arm}')
                current = RNGState.capture(device)
                if after is not None:
                    assert_rng_equal(after, current)
                after = current
                totals[arm] += float(loss) * size
            count += size
    finally:
        for arm, policy in arms.items():
            policy.train(modes[arm])
        saved.restore()
    if not count:
        raise ValueError('Evaluation needs fixed held-out inputs')
    return {arm: total / count for arm, total in totals.items()}


def split_dataset(dataset, every=10):
    """Split before fitting stats or constructing frame-uniform training sampling."""
    if every < 2:
        raise ValueError('Holdout interval must be at least two')
    rows = sorted(dataset.episodes, key=lambda row: row['episode_index'])
    heldout = {row['episode_index'] for row in rows[every - 1::every]}
    if not heldout or len(heldout) == len(rows):
        raise ValueError('Need nonempty training and held-out episodes')
    subsets = []
    for is_eval in (False, True):
        subset = copy.copy(dataset)
        subset.episodes = [row for row in rows if (row['episode_index'] in heldout) == is_eval]
        ids = {row['task_index'] for row in subset.episodes}
        subset.task_map = {key: value for key, value in dataset.task_map.items() if key in ids}
        subset.sampler = EpisodeSequenceIndex([row['length'] for row in subset.episodes], dataset.horizon,
                                              dataset.sampler.pad_before, dataset.sampler.pad_after)
        if not len(subset):
            raise ValueError('Split has no valid sequences')
        subset._reset_cache()
        subsets.append(subset)
    if subsets[0].task_map != subsets[1].task_map:
        raise ValueError('Both splits must contain the same tasks')
    return tuple(subsets)


def fixed_evaluation(dataset, samples, batch_size, seed):
    rng = np.random.default_rng(seed)
    if samples < len(dataset.episodes):
        raise ValueError('Evaluation needs at least one sample per held-out episode')
    indices, with_replacement = [], False
    for position, end in enumerate(dataset.sampler.ends):
        start = int(dataset.sampler.ends[position - 1]) if position else 0
        count = samples // len(dataset.episodes) + int(position < samples % len(dataset.episodes))
        available = int(end) - start
        if available < 1:
            raise ValueError('Each held-out episode needs a valid sequence')
        with_replacement |= available < count
        indices.extend((rng.choice(available, size=count, replace=available < count) + start).tolist())
    rng.shuffle(indices)
    batches = [default_collate(dataset.__getitems__(indices[start:start + batch_size]))
               for start in range(0, samples, batch_size)]
    positions = []
    for index in indices:
        position, frames = dataset.sampler.locate(index)
        positions.append({'sequence_index': index, 'episode_index': dataset.episodes[position]['episode_index'],
                          'frames': frames.tolist()})
    return batches, {'samples': positions, 'sampling_seed': seed, 'batch_size': batch_size,
                     'sampling': 'equal episode allocation, random valid sequences within each episode',
                     'episode_indices': sorted({entry['episode_index'] for entry in positions}),
                     'with_replacement': with_replacement,
                     'inputs_sha256': [state_hash({**{f'obs.{key}': value for key, value in batch['obs'].items()},
                                                   'action': batch['action']}) for batch in batches]}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str) + '\n')


def initialize_wandb(args, output, metadata):
    if args.wandb_mode == 'disabled':
        return {}
    import os
    if args.wandb_mode == 'online' and not os.environ.get('WANDB_API_KEY'):
        raise RuntimeError('Online W&B requires WANDB_API_KEY')
    runs, identities = {}, {}
    try:
        import wandb
        for arm in ARMS:
            identity = {'id': uuid.uuid4().hex[:12], 'name': f'{output.name}-{arm}',
                        'group': args.wandb_group, 'project': args.wandb_project,
                        'entity': args.wandb_entity, 'mode': args.wandb_mode}
            identities[arm] = identity
            runs[arm] = wandb.init(**identity, dir=str(output / arm), resume='never', reinit='create_new',
                                   config={**metadata, 'arm': arm},
                                   settings=wandb.Settings(silent=True, console='off', init_timeout=60))
            if runs[arm] is None or (args.wandb_mode == 'online' and runs[arm].settings.mode != 'online'):
                raise RuntimeError('W&B mode mismatch')
            runs[arm].define_metric('step')
            runs[arm].define_metric('*', step_metric='step')
        write_json(output / 'wandb.json', identities)
    except Exception:
        for run in runs.values():
            run.finish(exit_code=1)
        raise RuntimeError('Comparison W&B initialization failed; check credentials and connectivity') from None
    return runs


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--dataset-path', '--dataset-root', dest='dataset_path', required=True)
    result.add_argument('--output-dir', required=True)
    result.add_argument('--max-steps', type=int, default=1000)
    result.add_argument('--batch-size', type=int, default=512)
    result.add_argument('--num-workers', type=int, default=8)
    result.add_argument('--device', default='cuda')
    result.add_argument('--seed', type=int, default=42)
    result.add_argument('--cpu-threads', type=int, default=2)
    result.add_argument('--eval-every', type=int, default=250, help='0 disables intermediate evaluations, not endpoints')
    result.add_argument('--eval-samples', type=int, default=128)
    result.add_argument('--eval-batch-size', type=int, default=16)
    result.add_argument('--holdout-every', type=int, default=10)
    result.add_argument('--task-name', default='turning_on_radio')
    result.add_argument('--expected-episodes', type=int, default=200)
    result.add_argument('--wandb-mode', choices=('disabled', 'offline', 'online'), default='disabled')
    result.add_argument('--wandb-project', default='b1k-challenge-2026-diffusion-policy')
    result.add_argument('--wandb-entity')
    result.add_argument('--wandb-group', default='dp-init-20260917')
    return result


def run_comparison(args, *, config=None):
    """Run parsed arguments; injectable model config keeps CPU tests small."""
    device = torch.device(args.device)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Device must be CPU or CUDA')
    if (min(args.max_steps, args.batch_size, args.cpu_threads, args.eval_samples,
            args.eval_batch_size, args.expected_episodes) < 1 or min(args.eval_every, args.num_workers) < 0):
        raise ValueError('Invalid comparison size, interval or worker count')
    output, root = Path(args.output_dir).resolve(), Path(args.dataset_path).resolve()
    if not output.is_relative_to('/tmp') or output.is_relative_to(root):
        raise ValueError('Output must be under /tmp and outside the read-only dataset')
    config = config or production_config()
    config.validate()
    configure_cpu_threads(args.cpu_threads)
    # Different GEMM widths must not silently use reduced-mantissa arithmetic.
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    with output_lock(output):
        if any(path.name != 'run.lock' for path in output.iterdir()):
            raise FileExistsError('Comparison output must be empty; runs are never resumed or overwritten')
        return _run(args, config, device, output)


def _run(args, config, device, output):
    datasets, runs, logs = [], {}, []
    completed = 0
    history = {arm: [] for arm in ARMS}
    summary = {'status': 'failed', 'completed_steps': 0, 'metric': 'heldout_denoising_mse', 'arms': {}}
    try:
        dataset = B1KLeRobotDataset(args.dataset_path, [args.task_name],
                                    **replace(config, language_conditioning='clip_film').dataset_kwargs(),
                                    episode_cache_size=args.expected_episodes)
        datasets.append(dataset)
        if len(dataset.episodes) != args.expected_episodes:
            raise ValueError(f'Expected {args.expected_episodes} selected episodes, got {len(dataset.episodes)}')
        train, heldout = split_dataset(dataset, args.holdout_every)
        datasets.extend((train, heldout))
        language = train.prepare_language()
        heldout.prepare_language(language)
        normalizer = train.get_normalizer()
        split = {'train_episode_indices': [row['episode_index'] for row in train.episodes],
                 'heldout_episode_indices': [row['episode_index'] for row in heldout.episodes],
                 'rule': f'every {args.holdout_every}th sorted episode (one-based)',
                 'normalization': 'exact selected TRAIN frames only; heldout excluded',
                 'train_frames': sum(row['length'] for row in train.episodes),
                 'heldout_frames': sum(row['length'] for row in heldout.episodes),
                 'train_fingerprint': train.fingerprint(), 'heldout_fingerprint': heldout.fingerprint(),
                 'normalizer_sha256': state_hash(normalizer.state_dict())}
        write_json(output / 'split.json', split)
        atomic_save(normalizer.state_dict(), output / 'normalizer.pt')
        atomic_save(language, output / 'language.pt')
        arms, proof = build_arms(config, train.task_map, normalizer, args.seed)
        for policy in arms.values():
            policy.to(device)
        batches, eval_manifest = fixed_evaluation(heldout, args.eval_samples, args.eval_batch_size, args.seed + 2000000)
        eval_seed = args.seed + 3000000
        eval_manifest.update(noise_timestep_seed=eval_seed, mode='eval; center crop; dropout disabled',
                             metric='heldout denoising MSE; not rollout success or action error')
        write_json(output / 'evaluation_manifest.json', eval_manifest)
        probe = dict_apply(batches[0], lambda value: value.to(device))
        proof['forward_parity'] = initial_parity(arms, probe, proof['layout'], args.seed + 4000000)
        write_json(output / 'initialization.json', proof)
        del probe
        git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                              cwd=Path(__file__).resolve().parents[2], text=True).strip()
        metadata = {'model': config.to_dict(), 'arguments': vars(args), 'split': split,
                    'git_commit': git_commit,
                    'harness_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    'optimizer': {'name': 'AdamW', 'learning_rate': 1e-4, 'weight_decay': 1e-6,
                                  'betas': [0.9, 0.999], 'gradient_clip_norm': 1.0},
                    'precision': 'float32; no AMP; no gradient accumulation; TF32 disabled',
                    'physical_batch_size': args.batch_size, 'effective_batch_size': args.batch_size,
                    'production_batch_size': PRODUCTION_BATCH_SIZE,
                    'diagnostic_batch_note': 'Short diagnostic batch intentionally differs from production 8960',
                    'rng': 'CPU and selected CUDA state restored before EACH arm; train seed=seed+1000000+step',
                    'sampler': 'one DataLoader; independent frame-uniform StepBatchSampler seed',
                    'language': {key: value for key, value in language.items() if key != 'embeddings'},
                    'language_sha256': tensor_hash(language['embeddings'])}
        write_json(output / 'manifest.json', metadata)
        optimizers = {arm: torch.optim.AdamW((p for p in policy.parameters() if p.requires_grad),
                                             lr=1e-4, weight_decay=1e-6, betas=(0.9, 0.999))
                      for arm, policy in arms.items()}
        emas = {arm: EMAModel(copy.deepcopy(policy)) for arm, policy in arms.items()}
        train_logs, eval_logs = {}, {}
        for arm in ARMS:
            (output / arm).mkdir()
            for name, group in (('train', train_logs), ('eval', eval_logs)):
                group[arm] = (output / arm / f'{name}.jsonl').open('w')
                logs.append(group[arm])
        metrics_log = (output / 'metrics.jsonl').open('w')
        evaluation_log = (output / 'eval.jsonl').open('w')
        logs.extend((metrics_log, evaluation_log))
        runs = initialize_wandb(args, output, metadata)

        def log_wandb(arm, record):
            if arm in runs:
                try:
                    runs[arm].log(record)
                except Exception:
                    raise RuntimeError('Comparison W&B logging failed') from None

        def emit(arm, record):
            for stream in (train_logs[arm], metrics_log):
                stream.write(json.dumps(record, allow_nan=False) + '\n')
                stream.flush()
            if record['status'] == 'ok':
                history[arm].append(record['loss'])
            log_wandb(arm, record)

        def evaluate_at(step):
            start = time.monotonic()
            online = evaluate(arms, batches, eval_seed)
            averaged = evaluate({arm: ema.averaged_model for arm, ema in emas.items()}, batches, eval_seed)
            result = {}
            for arm in ARMS:
                record = {'step': step, 'arm': arm, 'loss': online[arm], 'ema_loss': averaged[arm],
                          'samples': args.eval_samples, 'eval_s_all_arms': time.monotonic() - start,
                          'finite_fail': False}
                for stream in (eval_logs[arm], evaluation_log):
                    stream.write(json.dumps(record, allow_nan=False) + '\n')
                    stream.flush()
                log_wandb(arm, {**{key: value for key, value in record.items() if key not in ('loss', 'ema_loss')},
                                'heldout_denoising_loss': online[arm],
                                'heldout_ema_denoising_loss': averaged[arm]})
                result[arm] = record
            return result

        initial_eval = evaluate_at(0)
        loader = DataLoader(train, batch_sampler=StepBatchSampler(len(train), args.batch_size, 0,
                                                                  args.max_steps, args.seed),
                            num_workers=args.num_workers, pin_memory=device.type == 'cuda',
                            worker_init_fn=partial(seed_worker, cpu_threads=1),
                            generator=torch.Generator().manual_seed(args.seed),
                            **({'multiprocessing_context': 'spawn', 'prefetch_factor': 1} if args.num_workers else {}))
        final_eval = initial_eval
        wait_start = time.monotonic()
        for step, cpu_batch in enumerate(loader, 1):
            data_wait_s = time.monotonic() - wait_start
            batch = dict_apply(cpu_batch, lambda value: value.to(device, non_blocking=True))
            train_step(arms, optimizers, emas, batch, args.seed + 1000000 + step, step, emit, data_wait_s)
            completed = step
            if step == args.max_steps or (args.eval_every and step % args.eval_every == 0):
                final_eval = evaluate_at(step)
            del batch
            wait_start = time.monotonic()
        for arm, policy in arms.items():
            losses = history[arm]
            summary['arms'][arm] = {
                'last100_mean': float(np.mean(losses[-100:])), 'last100_count': min(100, len(losses)),
                'last500_mean': float(np.mean(losses[-500:])), 'last500_count': min(500, len(losses)),
                'initial_eval': initial_eval[arm], 'final_eval': final_eval[arm],
                'final_checkpoint': str(output / arm / 'final.pt')}
            arm_config = replace(config, language_conditioning='none' if arm == 'baseline' else 'clip_film')
            checkpoint = {'format': 'diffusion_policy_b1k_v1', 'checkpoint_type': 'comparison_final',
                          'resume_supported': False,
                          'purpose': 'Diagnostic final state; load_policy compatible, not a production resume checkpoint',
                          'config': arm_config.to_dict(), 'task_map': train.task_map,
                          'normalizer': policy.normalizer.state_dict(), 'model': policy.state_dict(),
                          'ema_model': emas[arm].averaged_model.state_dict(),
                          'ema_step': emas[arm].optimization_step, 'optimizer': optimizers[arm].state_dict(),
                          'step': completed, 'comparison': {'arm': arm, 'metadata': metadata,
                                                          'initial_state_sha256': proof['initial_state_sha256'][arm]},
                          'selection': {'episode_indices': split['train_episode_indices']},
                          'dataset_fingerprint': split['train_fingerprint']}
            if arm != 'baseline':
                checkpoint['language'] = validate_language_cache(language, train.task_map, 'task_name')
            atomic_save(checkpoint, output / arm / 'final.pt')
        summary['status'] = 'completed'
        return summary
    except Exception as error:
        summary['error_type'] = type(error).__name__
        summary['finite_fail'] = isinstance(error, FloatingPointError)
        raise
    finally:
        summary['completed_steps'] = completed
        write_json(output / 'summary.json', summary)
        for stream in logs:
            stream.close()
        for dataset in datasets:
            dataset.close()
        for run in runs.values():
            try:
                run.finish(exit_code=0 if summary['status'] == 'completed' else 1)
            except Exception:
                pass


def main(argv=None):
    return run_comparison(parser().parse_args(argv))


if __name__ == '__main__':
    main()
