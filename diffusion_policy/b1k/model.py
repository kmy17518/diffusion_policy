"""Construct upstream diffusion policies from checkpoint-safe primitives."""

from dataclasses import asdict, dataclass
import importlib

import torch
from diffusers import DDIMScheduler, DDPMScheduler

from diffusion_policy.b1k.robot import CAMERAS


POLICY_TARGETS = {
    'unet_image': 'diffusion_unet_image_policy.DiffusionUnetImagePolicy',
    'unet_hybrid_image': 'diffusion_unet_hybrid_image_policy.DiffusionUnetHybridImagePolicy',
    'transformer_hybrid_image': 'diffusion_transformer_hybrid_image_policy.DiffusionTransformerHybridImagePolicy',
    'unet_lowdim': 'diffusion_unet_lowdim_policy.DiffusionUnetLowdimPolicy',
    'transformer_lowdim': 'diffusion_transformer_lowdim_policy.DiffusionTransformerLowdimPolicy',
    'unet_video': 'diffusion_unet_video_policy.DiffusionUnetVideoPolicy',
}


@dataclass
class ModelConfig:
    horizon: int = 16
    n_obs_steps: int = 2
    n_action_steps: int = 8
    image_size: int = 96
    cameras: tuple = tuple(CAMERAS)
    down_dims: tuple = (256, 512, 1024)
    diffusion_step_embed_dim: int = 256
    n_groups: int = 8
    num_train_timesteps: int = 100
    num_inference_steps: int = 100
    scheduler: str = 'ddpm'
    clip_sample: bool = True
    # Missing fields in v1 checkpoints retain the original image U-Net semantics.
    variant: str = 'unet_image'
    conditioning: str = 'global'
    pred_action_steps_only: bool = False
    prediction_type: str = 'epsilon'
    kernel_size: int = 5
    cond_predict_scale: bool = True
    n_layer: int = 8
    n_head: int = 4
    n_emb: int = 256
    n_cond_layers: int = 0
    p_drop_emb: float = 0.0
    p_drop_attn: float = 0.3
    causal_attn: bool = True
    time_as_cond: bool = True
    crop_shape: tuple | None = None
    resize_shape: tuple | None = None
    random_crop: bool = True
    obs_encoder_group_norm: bool = True
    eval_fixed_crop: bool = True
    share_rgb_model: bool = True
    imagenet_norm: bool = False
    encoder_weights: str | None = None
    freeze_encoder: bool = False
    language_conditioning: str = 'none'
    prompt_source: str = 'task_name'
    # Append the one-hot task id to the 25-D state. True is the v1 behavior and therefore the dataclass
    # default (checkpoints without the field keep it); the trainer CLI defaults to --no-task-onehot.
    task_onehot: bool = True

    @property
    def lowdim(self):
        return self.variant.endswith('_lowdim')

    @property
    def obs_steps(self):
        return self.horizon if self.conditioning in ('local', 'inpainting') else self.n_obs_steps

    def dataset_kwargs(self):
        return {'horizon': self.horizon, 'n_obs_steps': self.n_obs_steps,
                'n_action_steps': self.n_action_steps,
                'cameras': () if self.lowdim else self.cameras, 'image_size': self.image_size,
                'observation_mode': 'lowdim' if self.lowdim else 'image',
                'obs_steps': self.obs_steps, 'imagenet_norm': self.imagenet_norm,
                'language_conditioning': self.language_conditioning, 'prompt_source': self.prompt_source,
                'task_onehot': self.task_onehot}

    def to_dict(self):
        return asdict(self)

    def validate(self):
        if self.variant not in POLICY_TARGETS:
            raise ValueError(f'Unknown diffusion variant {self.variant!r}')
        if self.language_conditioning not in ('none', 'clip_film'):
            raise ValueError('language_conditioning must be none or clip_film')
        if self.prompt_source not in ('task_name', 'task_description'):
            raise ValueError('prompt_source must be task_name or task_description')
        if self.language_conditioning == 'clip_film':
            if self.variant not in ('unet_image', 'unet_hybrid_image', 'transformer_hybrid_image'):
                raise ValueError('clip_film supports only unet_image, unet_hybrid_image and transformer_hybrid_image')
            if self.variant == 'transformer_hybrid_image' and self.conditioning != 'global':
                raise ValueError('clip_film transformer_hybrid_image requires global conditioning; '
                                 'upstream inpainting detaches the vision encoder')
            if self.freeze_encoder:
                raise ValueError('clip_film requires a trainable vision encoder (including FiLM)')
        if self.conditioning not in ('global', 'local', 'inpainting'):
            raise ValueError('conditioning must be global, local or inpainting')
        if self.conditioning == 'local' and self.variant != 'unet_lowdim':
            raise ValueError('Local observation conditioning is supported only by unet_lowdim')
        if self.pred_action_steps_only and (self.conditioning != 'global' or self.variant not in (
                'unet_lowdim', 'transformer_lowdim', 'transformer_hybrid_image')):
            raise ValueError('pred_action_steps_only requires a globally conditioned lowdim U-Net or transformer')
        if not 1 <= self.n_obs_steps <= self.horizon:
            raise ValueError('Invalid n_obs_steps')
        if not 1 <= self.n_action_steps <= self.horizon - self.n_obs_steps + 1:
            raise ValueError('n_action_steps exceeds available prediction after observation history')
        if not 1 <= self.num_inference_steps <= self.num_train_timesteps:
            raise ValueError('num_inference_steps must be within training diffusion timesteps')
        if self.scheduler not in ('ddpm', 'ddim') or self.prediction_type not in ('epsilon', 'sample'):
            raise ValueError('Need DDPM or DDIM with epsilon or sample prediction')
        if self.variant.startswith('unet'):
            if (len(self.down_dims) < 2 or self.n_groups < 1 or any(
                    width < self.n_groups or width % self.n_groups for width in self.down_dims)):
                raise ValueError('Need at least two positive U-Net widths divisible by n_groups')
            if self.diffusion_step_embed_dim < 4 or self.diffusion_step_embed_dim % 2:
                raise ValueError('diffusion_step_embed_dim must be even and at least four')
            divisor = 2 ** (len(self.down_dims) - 1)
            length = self.n_action_steps if self.pred_action_steps_only else self.horizon
            if length < divisor or length % divisor:
                raise ValueError(f'Diffusion trajectory length must be divisible by U-Net downsampling factor {divisor}')
            if self.kernel_size < 1 or self.kernel_size % 2 != 1:
                raise ValueError('kernel_size must be positive and odd')
        else:
            if self.n_head < 1 or self.n_emb < 4 or self.n_emb % 2 or self.n_emb % self.n_head:
                raise ValueError('n_emb must be even, >= 4 and divisible by positive n_head')
            if self.n_layer < 1 or self.n_cond_layers < 0:
                raise ValueError('Need n_layer >= 1 and n_cond_layers >= 0')
            if not self.time_as_cond and self.conditioning != 'inpainting':
                raise ValueError('time_as_cond=False requires inpainting (encoder-only transformer)')
            if not all(0 <= drop < 1 for drop in (self.p_drop_emb, self.p_drop_attn)):
                raise ValueError('Dropout probabilities must be in [0, 1)')
        if not self.lowdim:
            if (self.image_size < 16 or not self.cameras or len(set(self.cameras)) != len(self.cameras)
                    or set(self.cameras) - set(CAMERAS)):
                raise ValueError('Need image_size >= 16 and unique known RGB cameras')
            image_shape = self.resize_shape or (self.image_size, self.image_size)
            if len(image_shape) != 2 or min(image_shape) < 16:
                raise ValueError('resize_shape must have two dimensions >= 16')
            if self.crop_shape is not None and (len(self.crop_shape) != 2 or min(self.crop_shape) < 16
                    or any(c > s for c, s in zip(self.crop_shape, image_shape))):
                raise ValueError('crop_shape must fit the image and have dimensions >= 16')
        if self.encoder_weights not in (None, 'IMAGENET1K_V1'):
            raise ValueError('encoder_weights must be None or IMAGENET1K_V1')
        if self.variant != 'unet_image' and (self.encoder_weights or self.freeze_encoder or self.imagenet_norm
                or self.resize_shape is not None or not self.share_rgb_model or not self.random_crop):
            raise ValueError('Pretrained/resize/share/normalization encoder controls apply only to unet_image')


def build_policy(config, task_map, initialize_encoder=True):
    if isinstance(config, dict):
        config = ModelConfig(**config)
    config.validate()
    if not task_map:
        raise ValueError('Need a nonempty task map')
    module, name = POLICY_TARGETS[config.variant].rsplit('.', 1)
    # Optional vision dependencies must not affect lowdim or plain image policies.
    try:
        policy_type = getattr(importlib.import_module(f'diffusion_policy.policy.{module}'), name)
    except ModuleNotFoundError as error:
        if config.variant == 'unet_video' and error.name.startswith('diffusion_policy.model.obs_encoder'):
            raise ModuleNotFoundError(
                'unet_video cannot run: upstream model.obs_encoder.temporal_aggregator, '
                'model.obs_encoder.video_core (VideoCore/VideoResNet), and model.ibc.global_avgpool '
                'sources are absent from this checkout; obtain the authentic upstream sources first',
                name=error.name) from error
        raise
    if config.variant == 'unet_video':
        raise NotImplementedError('The upstream VideoCore/VideoResNet recipe requires verified model.obs_encoder sources')
    scheduler_type = DDPMScheduler if config.scheduler == 'ddpm' else DDIMScheduler
    scheduler_kwargs = {'variance_type': 'fixed_small'} if config.scheduler == 'ddpm' else {}
    scheduler = scheduler_type(num_train_timesteps=config.num_train_timesteps,
                               beta_start=0.0001, beta_end=0.02,
                               beta_schedule='squaredcos_cap_v2', clip_sample=config.clip_sample,
                               prediction_type=config.prediction_type, **scheduler_kwargs)
    common = dict(noise_scheduler=scheduler, horizon=config.horizon,
                  n_action_steps=config.n_action_steps, n_obs_steps=config.n_obs_steps,
                  num_inference_steps=config.num_inference_steps)
    unet = dict(diffusion_step_embed_dim=config.diffusion_step_embed_dim, down_dims=config.down_dims,
                n_groups=config.n_groups, kernel_size=config.kernel_size,
                cond_predict_scale=config.cond_predict_scale)
    transformer = {key: getattr(config, key) for key in (
        'n_layer', 'n_head', 'n_emb', 'n_cond_layers', 'p_drop_emb', 'p_drop_attn', 'causal_attn', 'time_as_cond')}
    global_cond = config.conditioning == 'global'
    if len(task_map) > 1 and not config.task_onehot and config.language_conditioning == 'none':
        raise ValueError('Several tasks but no task conditioning: enable task_onehot or clip_film language conditioning')
    obs_dim = 25 + (len(task_map) if config.task_onehot else 0)
    if config.lowdim:
        common.update(obs_dim=obs_dim, action_dim=23, pred_action_steps_only=config.pred_action_steps_only)
        input_dim = 23 + (obs_dim if config.conditioning == 'inpainting' else 0)
        if config.variant == 'unet_lowdim':
            from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
            model = ConditionalUnet1D(input_dim=input_dim,
                                      local_cond_dim=obs_dim if config.conditioning == 'local' else None,
                                      global_cond_dim=obs_dim * config.n_obs_steps if global_cond else None, **unet)
            # B1K row t contains the action at observation t, not t+1.
            return policy_type(model=model, obs_as_global_cond=global_cond,
                               obs_as_local_cond=config.conditioning == 'local', oa_step_convention=True, **common)
        from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
        model = TransformerForDiffusion(input_dim=input_dim, output_dim=input_dim, horizon=config.horizon,
                                        n_obs_steps=config.n_obs_steps, cond_dim=obs_dim if global_cond else 0,
                                        obs_as_cond=global_cond, **transformer)
        return policy_type(model=model, obs_as_cond=global_cond, **common)
    shape_meta = {
        'obs': {'state': {'shape': [obs_dim], 'type': 'low_dim'},
                **{camera: {'shape': [3, config.image_size, config.image_size], 'type': 'rgb'}
                   for camera in config.cameras}},
        'action': {'shape': [23]},
    }
    if config.language_conditioning == 'clip_film':
        from diffusion_policy.b1k.language import LANGUAGE_DIM, LANGUAGE_KEY
        shape_meta['obs'][LANGUAGE_KEY] = {'shape': [LANGUAGE_DIM], 'type': 'low_dim'}
    common['shape_meta'] = shape_meta
    if config.variant == 'unet_image':
        from diffusion_policy.common.pytorch_util import replace_submodules
        from diffusion_policy.model.vision.model_getter import get_resnet
        from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
        if config.language_conditioning == 'clip_film':
            from diffusion_policy.model.vision.clip_film import ResNet18FiLM
            backbone = ResNet18FiLM(group_norm=config.obs_encoder_group_norm,
                                    weights=config.encoder_weights if initialize_encoder else None)
        else:
            backbone = get_resnet('resnet18', weights=config.encoder_weights if initialize_encoder else None)
        if config.obs_encoder_group_norm:
            backbone = replace_submodules(backbone, lambda m: isinstance(m, torch.nn.BatchNorm2d),
                                          lambda m: torch.nn.GroupNorm(m.num_features // 16, m.num_features))
        if not config.obs_encoder_group_norm:
            # Shape probes must not update BatchNorm statistics or require a batch > 1.
            backbone.eval()
        if config.freeze_encoder:
            backbone.requires_grad_(False)
        encoder = MultiImageObsEncoder(
            shape_meta, backbone, share_rgb_model=config.share_rgb_model,
            resize_shape=config.resize_shape, crop_shape=config.crop_shape,
            random_crop=config.random_crop, imagenet_norm=config.imagenet_norm,
            language_key='lang_emb' if config.language_conditioning == 'clip_film' else None)
        policy = policy_type(obs_encoder=encoder, obs_as_global_cond=global_cond, **unet, **common)
        if config.freeze_encoder:
            policy.obs_encoder.eval().requires_grad_(False)
        else:
            policy.obs_encoder.train()
        return policy
    if config.language_conditioning == 'clip_film':
        from diffusion_policy.model.vision.clip_film import FiLMHybridObsEncoder
        common['obs_encoder'] = FiLMHybridObsEncoder(
            shape_meta, crop_shape=config.crop_shape, group_norm=config.obs_encoder_group_norm,
            eval_fixed_crop=config.eval_fixed_crop)
    hybrid = dict(crop_shape=config.crop_shape, obs_encoder_group_norm=config.obs_encoder_group_norm,
                  eval_fixed_crop=config.eval_fixed_crop)
    if config.variant == 'unet_hybrid_image':
        return policy_type(obs_as_global_cond=global_cond, **unet, **hybrid, **common)
    return policy_type(obs_as_cond=global_cond, pred_action_steps_only=config.pred_action_steps_only,
                       **transformer, **hybrid, **common)


def load_checkpoint(path, device='cpu'):
    from pathlib import Path
    path = Path(path)
    if path.is_dir():
        path = path / 'latest.pt'
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get('format') != 'diffusion_policy_b1k_v1':
        raise ValueError('Not a self-contained B1K checkpoint')
    config = ModelConfig(**checkpoint['config'])
    config.validate()
    if config.language_conditioning == 'clip_film':
        from diffusion_policy.b1k.language import validate_language_cache
        checkpoint['language'] = validate_language_cache(
            checkpoint.get('language'), checkpoint['task_map'], config.prompt_source)
    elif checkpoint.get('language') is not None:
        raise ValueError('Language cache is incompatible with language_conditioning=none')
    return checkpoint


def load_policy(path, device='cpu'):
    checkpoint = load_checkpoint(path, 'cpu')
    policy = build_policy(checkpoint['config'], checkpoint['task_map'], initialize_encoder=False)
    policy.load_state_dict(checkpoint['ema_model'])
    if 'language' in checkpoint:
        policy.language = checkpoint['language']
    policy.to(device).eval().requires_grad_(False)
    return policy, checkpoint
