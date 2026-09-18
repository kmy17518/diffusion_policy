"""Explicit RoboCasa-style language FiLM for ResNet18 image encoders."""

import math

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from diffusion_policy.b1k.language import LANGUAGE_DIM, LANGUAGE_KEY
from diffusion_policy.common.pytorch_util import replace_submodules
from diffusion_policy.model.vision.crop_randomizer import CropRandomizer, sample_random_image_crops
from diffusion_policy.model.vision.model_getter import get_resnet


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


class FiLMResidualBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block
        self.film = FiLMLayer(block.conv2.out_channels)

    def forward(self, x, lang_emb):
        return self.film(self.block(x), lang_emb)


class ResNet18FiLM(nn.Module):
    def __init__(self, group_norm=True, weights=None, spatial_shape=None):
        super().__init__()
        backbone = get_resnet('resnet18', weights=weights)
        if group_norm:
            replace_submodules(backbone, lambda module: isinstance(module, nn.BatchNorm2d),
                               lambda module: nn.GroupNorm(module.num_features // 16, module.num_features))
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.blocks = nn.ModuleList(FiLMResidualBlock(block) for layer in (
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4) for block in layer)
        # BatchNorm recomputation would update running statistics twice.
        self.checkpoint_blocks = group_norm
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(1)) if spatial_shape is None else nn.Sequential(
            SpatialSoftmax(spatial_shape), nn.Linear(64, 64), nn.ReLU())

    def forward(self, x, lang_emb):
        if lang_emb.shape != (x.shape[0], LANGUAGE_DIM):
            raise ValueError('FiLM requires one 768-D lang_emb per image')
        x = self.stem(x)
        for block in self.blocks:
            if self.checkpoint_blocks and self.training and torch.is_grad_enabled():
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
    def __init__(self, shape_meta, crop_shape=None, group_norm=True, eval_fixed_crop=True):
        super().__init__()
        self.shapes = shape_meta['obs']
        self.encoders = nn.ModuleDict()
        self.crops = nn.ModuleDict()
        self.eval_fixed_crop = eval_fixed_crop
        self.feature_dim = 0
        for key, meta in self.shapes.items():
            if meta.get('type', 'low_dim') == 'rgb':
                shape = meta['shape']
                self.encoders[key] = ResNet18FiLM(group_norm=group_norm, spatial_shape=crop_shape or shape[1:])
                if crop_shape is not None and tuple(crop_shape) != tuple(shape[1:]):
                    self.crops[key] = CropRandomizer(shape, *crop_shape)
                self.feature_dim += 64
            else:
                self.feature_dim += math.prod(meta['shape'])

    def output_shape(self):
        return [self.feature_dim]

    def forward(self, obs_dict):
        lang_emb = obs_dict[LANGUAGE_KEY]
        features = []
        for key in self.shapes:
            value = obs_dict[key]
            if key in self.encoders:
                if key in self.crops:
                    crop = self.crops[key]
                    if not self.training and not self.eval_fixed_crop:
                        value, _ = sample_random_image_crops(value, crop.crop_height, crop.crop_width, num_crops=1)
                        value = value.flatten(0, 1)
                    else:
                        value = crop(value)
                value = self.encoders[key](value, lang_emb)
            features.append(value.flatten(1))
        return torch.cat(features, dim=-1)
