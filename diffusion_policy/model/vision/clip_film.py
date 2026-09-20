"""Explicit hybrid (ResNet18 + SpatialSoftmax) image encoders with optional RoboCasa-style language FiLM and
optional goal-image conditioning.

`FiLMHybridObsEncoder` reproduces the robomimic `ObservationEncoder` the hybrid policies build (per camera:
ResNet18Conv -> SpatialSoftmax(32 keypoints) -> Linear(64, 64) -> ReLU; random crops in training, center crops in
eval; low-dim keys pass through; features concatenated in shape_meta order) and adds two independent switches:

- language (`language=True`): CLIP FiLM after every residual block (`FiLMLayer`, RoboCasa365-style), the
  `lang_emb` low-dim key of the observation dict supplying the embedding;
- goal images (`goal_fusion`): `early` channel-stacks each selected camera with its goal image before the ResNet
  stem (`PairedConv2d`: one 6-channel stem `[W_obs, W_goal]` with the goal half zero-initialised), so the paired
  input is cropped jointly and the encoder output shape is unchanged; `late` encodes the goal image with the
  camera's own encoder (`goal_encoder=shared_base`) or an identical copy (`separate_base`) into one 64-D feature
  per goal view, cropped with the same offsets as the current image it is paired with (`encode_goals`).
  With language, the goal pass runs the FiLM layers at the identity (gamma = beta = 0) unless
  `language_on_goal_encoder` is set.
"""

import math

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from diffusion_policy.b1k.language import LANGUAGE_DIM, LANGUAGE_KEY
from diffusion_policy.common.pytorch_util import replace_submodules
from diffusion_policy.model.vision.crop_randomizer import CropRandomizer, sample_random_image_crops
from diffusion_policy.model.vision.model_getter import get_resnet


GOAL_FUSIONS = ('none', 'early', 'late')
GOAL_ENCODERS = ('shared_base', 'separate_base')
GOAL_FEATURE_DIM = 64  # the hybrid encoder's per-camera feature width (SpatialSoftmax keypoints -> Linear(64, 64))


class FiLMLayer(nn.Module):
    def __init__(self, channels, language_dim=LANGUAGE_DIM):
        super().__init__()
        self.lang_proj = nn.Linear(language_dim, 2 * channels)

    def forward(self, x, lang_emb):
        beta, gamma = self.lang_proj(lang_emb).chunk(2, dim=-1)
        return torch.relu((1 + gamma[:, :, None, None]) * x + beta[:, :, None, None])


def identity_initialize_film(root):
    """Zero every FiLM projection under `root` so beta = gamma = 0 and each conditioned block starts as
    ReLU(x) = x, i.e. the encoder initially computes what the unconditioned ResNet18 would (the
    "identity FiLM" arm of compare_language_init). The projections still receive gradients from the
    first step. Returns the number of FiLM layers initialized."""
    count = 0
    with torch.no_grad():
        for module in root.modules():
            if isinstance(module, FiLMLayer):
                module.lang_proj.weight.zero_()
                module.lang_proj.bias.zero_()
                count += 1
    return count


class PairedConv2d(nn.Conv2d):
    """Stem convolution accepting a camera image channel-stacked with its goal image (goal-image early fusion).

    For a 6-channel `[current; goal]` input it computes `conv(current) + conv_goal(goal)`: one 6-channel
    convolution with weights `[W_obs, W_goal]` (BridgeData V2's channel stacking). `goal_weight` starts at zero, so
    at initialization the paired stem equals the plain stem and a zero goal is an exact "absent goal"; 3-channel
    inputs run the plain convolution. State-dict keys stay `weight`/`bias` plus `goal_weight`.
    """
    def __init__(self, conv):
        super().__init__(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
                         conv.dilation, conv.groups, conv.bias is not None, conv.padding_mode)
        with torch.no_grad():
            self.weight.copy_(conv.weight)
            if conv.bias is not None:
                self.bias.copy_(conv.bias)
        self.goal_weight = nn.Parameter(torch.zeros_like(self.weight))

    def forward(self, x):
        if x.shape[1] == self.in_channels:
            return super().forward(x)
        if x.shape[1] != 2 * self.in_channels:
            raise ValueError(f'PairedConv2d expects {self.in_channels} or {2 * self.in_channels} channels, got {x.shape[1]}')
        return self._conv_forward(x, torch.cat([self.weight, self.goal_weight.to(self.weight.dtype)], dim=1), self.bias)


class FiLMResidualBlock(nn.Module):
    def __init__(self, block, film=True, language_dim=LANGUAGE_DIM):
        super().__init__()
        self.block = block
        self.film = FiLMLayer(block.conv2.out_channels, language_dim) if film else None

    def forward(self, x, lang_emb=None):
        x = self.block(x)
        if self.film is None or lang_emb is None:
            return x  # no FiLM, or the explicit identity pass (gamma = beta = 0 leaves relu(x) = x)
        return self.film(x, lang_emb)


class ResNet18FiLM(nn.Module):
    """ResNet18 (GroupNorm by default) with optional per-block FiLM, optional paired stem, hybrid spatial-softmax head."""
    def __init__(self, group_norm=True, weights=None, spatial_shape=None, film=True, paired_stem=False):
        super().__init__()
        backbone = get_resnet('resnet18', weights=weights)
        if group_norm:
            replace_submodules(backbone, lambda module: isinstance(module, nn.BatchNorm2d),
                               lambda module: nn.GroupNorm(module.num_features // 16, module.num_features))
        conv1 = PairedConv2d(backbone.conv1) if paired_stem else backbone.conv1
        self.stem = nn.Sequential(conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.blocks = nn.ModuleList(FiLMResidualBlock(block, film) for layer in (
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4) for block in layer)
        self.film = film
        self.paired_stem = paired_stem
        # BatchNorm recomputation would update running statistics twice.
        self.checkpoint_blocks = group_norm and film
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(1)) if spatial_shape is None else nn.Sequential(
            SpatialSoftmax(spatial_shape), nn.Linear(64, 64), nn.ReLU())

    def forward(self, x, lang_emb=None):
        if self.film and lang_emb is not None and lang_emb.shape != (x.shape[0], LANGUAGE_DIM):
            raise ValueError('FiLM requires one 768-D lang_emb per image')
        if not self.film and lang_emb is not None:
            raise ValueError('This encoder has no FiLM layers; do not pass lang_emb')
        if x.shape[1] == 6 and not self.paired_stem:
            raise ValueError('6-channel [current; goal] input needs an encoder built with paired_stem=True')
        x = self.stem(x)
        for block in self.blocks:
            if self.checkpoint_blocks and lang_emb is not None and self.training and torch.is_grad_enabled():
                x = checkpoint(block, x, lang_emb, use_reentrant=False)
            else:
                x = block(x, lang_emb)
        return self.pool(x)


class SpatialSoftmax(nn.Module):
    """The hybrid recipe's 32-keypoint, fixed-temperature spatial pooling."""

    def __init__(self, image_shape):
        super().__init__()
        height, width = [math.ceil(size / 32) for size in image_shape]
        self.projection = nn.Conv2d(512, 32, 1)
        x, y = np.meshgrid(np.linspace(-1., 1., width), np.linspace(-1., 1., height))
        self.register_buffer('pos_x', torch.from_numpy(x.reshape(1, -1)).float())
        self.register_buffer('pos_y', torch.from_numpy(y.reshape(1, -1)).float())
        self.register_buffer('temperature', torch.ones(1))

    def forward(self, x):
        features = self.projection(x).reshape(-1, self.pos_x.numel())
        attention = (features / self.temperature).softmax(dim=-1)
        expected_x = torch.sum(self.pos_x * attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y * attention, dim=1, keepdim=True)
        keypoints = torch.cat([expected_x, expected_y], dim=1).reshape(-1, 32, 2)
        if self.training:
            # Preserve the upstream RNG draw even with zero keypoint noise.
            keypoints = keypoints + torch.randn_like(keypoints) * 0.0
        return keypoints.flatten(1)


class FiLMHybridObsEncoder(nn.Module):
    def __init__(self, shape_meta, crop_shape=None, group_norm=True, eval_fixed_crop=True, language=True,
                 goal_keys=(), goal_pairs=None, goal_fusion='none', goal_encoder='shared_base',
                 language_on_goal_encoder=False):
        """
        goal_keys: observation keys holding goal images ((B, 3, H, W), no history axis), excluded from the base
            feature layout; goal_pairs maps each goal key to the camera key it is the goal of.
        """
        super().__init__()
        if goal_fusion not in GOAL_FUSIONS or goal_encoder not in GOAL_ENCODERS:
            raise ValueError(f'goal_fusion must be one of {GOAL_FUSIONS} and goal_encoder one of {GOAL_ENCODERS}')
        self.goal_keys = tuple(goal_keys) if goal_fusion != 'none' else ()
        self.goal_pairs = dict(goal_pairs or {})
        if set(self.goal_pairs) != set(self.goal_keys) or len(set(self.goal_pairs.values())) != len(self.goal_keys):
            raise ValueError('goal_pairs must map every goal key to a distinct camera key')
        if goal_fusion == 'early' and goal_encoder != 'shared_base':
            raise ValueError('Early fusion pairs the goal with its camera stem; goal_encoder must be shared_base')
        self.goal_fusion = goal_fusion
        self.goal_encoder = goal_encoder
        self.language = language
        self.language_on_goal_encoder = bool(language_on_goal_encoder)
        self.shapes = {key: meta for key, meta in shape_meta['obs'].items() if key not in self.goal_keys}
        self.encoders = nn.ModuleDict()
        self.crops = nn.ModuleDict()
        self.eval_fixed_crop = eval_fixed_crop
        self.feature_dim = 0
        paired_cameras = set(self.goal_pairs.values()) if goal_fusion == 'early' else set()
        for key, meta in self.shapes.items():
            if meta.get('type', 'low_dim') == 'rgb':
                shape = meta['shape']
                self.encoders[key] = ResNet18FiLM(group_norm=group_norm, spatial_shape=crop_shape or shape[1:],
                                                  film=language, paired_stem=key in paired_cameras)
                if crop_shape is not None and tuple(crop_shape) != tuple(shape[1:]):
                    self.crops[key] = CropRandomizer(shape, *crop_shape)
                self.feature_dim += GOAL_FEATURE_DIM
            else:
                self.feature_dim += math.prod(meta['shape'])
        for camera in self.goal_pairs.values():
            if camera not in self.encoders:
                raise ValueError(f'Goal view pairs with camera {camera!r}, which this encoder does not observe')
        self.goal_encoders = nn.ModuleDict()
        if goal_fusion == 'late' and goal_encoder == 'separate_base':
            for goal_key, camera in self.goal_pairs.items():
                shape = self.shapes[camera]['shape']
                self.goal_encoders[goal_key] = ResNet18FiLM(group_norm=group_norm, spatial_shape=crop_shape or shape[1:],
                                                            film=language and self.language_on_goal_encoder)

    def output_shape(self):
        return [self.feature_dim]

    @property
    def goal_feature_dim(self):
        """Width of `encode_goals` (late fusion): one 64-D feature per goal view; 0 otherwise."""
        return GOAL_FEATURE_DIM * len(self.goal_keys) if self.goal_fusion == 'late' else 0

    def crop(self, key, value):
        if key not in self.crops:
            return value
        crop = self.crops[key]
        if not self.training and not self.eval_fixed_crop:
            value, _ = sample_random_image_crops(value, crop.crop_height, crop.crop_width, num_crops=1)
            return value.flatten(0, 1)
        return crop(value)

    def language_embedding(self, obs_dict):
        if not self.language:
            return None
        if LANGUAGE_KEY not in obs_dict:
            raise ValueError(f'Language-conditioned encoder needs {LANGUAGE_KEY} in the observation')
        return obs_dict[LANGUAGE_KEY]

    def forward(self, obs_dict):
        lang_emb = self.language_embedding(obs_dict)
        pairs = {camera: goal_key for goal_key, camera in self.goal_pairs.items()} if self.goal_fusion == 'early' else {}
        features = []
        for key in self.shapes:
            value = obs_dict[key]
            if key in self.encoders:
                if key in pairs:
                    goal = obs_dict[pairs[key]]
                    if goal.shape != value.shape:
                        raise ValueError(f'Early fusion pairs {pairs[key]} with {key} row by row; got {tuple(goal.shape)} '
                                         f'vs {tuple(value.shape)} (repeat the goal over the observation steps)')
                    value = torch.cat([value, goal], dim=1)  # jointly cropped 6-channel [current; goal]
                value = self.encoders[key](self.crop(key, value), lang_emb)
            features.append(value.flatten(1))
        return torch.cat(features, dim=-1)

    def encode_goals(self, obs_dict):
        """Late fusion: (B, 64 * views) goal features from `obs_dict[goal_key]` (B, 3, H, W).

        The goal is cropped with the offsets drawn for the paired current image `obs_dict[camera]` (B, 3, H, W)
        -- stacked along channels, cropped once, split -- so current/goal geometry stays matched; then it runs
        through the camera's encoder (shared) or its own copy, with FiLM at the identity unless
        language_on_goal_encoder.
        """
        if self.goal_fusion != 'late':
            raise ValueError('encode_goals is defined for late fusion only')
        lang_emb = self.language_embedding(obs_dict) if self.language_on_goal_encoder else None
        features = []
        for goal_key, camera in self.goal_pairs.items():
            goal, current = obs_dict[goal_key], obs_dict[camera]
            if goal.shape != current.shape:
                raise ValueError(f'{goal_key} must match {camera} row by row for the paired crop; got '
                                 f'{tuple(goal.shape)} vs {tuple(current.shape)}')
            paired = self.crop(camera, torch.cat([current, goal], dim=1))
            encoder = self.goal_encoders[goal_key] if goal_key in self.goal_encoders else self.encoders[camera]
            features.append(encoder(paired[:, 3:], lang_emb if encoder.film else None))
        return torch.cat(features, dim=-1)
