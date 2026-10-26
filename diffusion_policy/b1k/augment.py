"""openpi pi05_b1k's training image augmentation (`--image-augment pi05`) for Diffusion Policy batches, on the GPU.

A port of kmy17518/act's `b1k_augment.py` (commit 8da77eb), which replicates openpi 93f8a4e `preprocess_observation
(train=True)` with augmax 0.4.1 bit for bit given the same draws:

- non-wrist cameras (the head) and the goal images paired with them: RandomCrop(0.95 H, 0.95 W) -> Resize(H, W) ->
  Rotate(+-5 deg), fused into one bilinear resampling (`F.grid_sample`, zero padding, align_corners=True), then
  ColorJitter(brightness 0.3, contrast 0.4, hue 0.1, p=0.5) in HSV. Every non-wrist image of a sample -- all
  `n_obs_steps` frames of the head camera and its goal image -- shares the sample's geometric draw and color draw, so
  current/goal geometry and color stay aligned (openpi applies one transform per sample to base_0_rgb and goal_0_rgb);
- wrist cameras: ColorJitter only, with the sample's second draw shared by both wrists and all their `n_obs_steps` frames
  (openpi's left_wrist_0_rgb / right_wrist_0_rgb share one draw per sample).

Applied to training batches only, in float [0, 1] after `images_to_float` and before the policy's normalizer, never in
validation or serving. The draws come from a CPU generator seeded by (seed, step, rank) -- the same values on every
device and after a resume, and ACT's draws for the same (seed, step, rank). `augment_obs(..., compiled=True)` runs the
resampling and jitter through torch.compile (dynamic=False, fullgraph=True) when the run compiles its policy, eagerly
when compiling fails.
"""

import hashlib
import logging
import math

import torch
import torch.nn.functional as F

LOGGER = logging.getLogger(__name__)
IMAGE_AUGMENTS = ('none', 'pi05')
CROP_FRACTION = 0.95
ANGLE_DEGREES = 5.0
BRIGHTNESS, CONTRAST, HUE = 0.3, 0.4, 0.1
JITTER_PROBABILITY = 0.5
DRAWS = ('theta', 'cy', 'cx', 'apply', 'brightness', 'contrast', 'hue', 'wrist_apply', 'wrist_brightness',
         'wrist_contrast', 'wrist_hue')
WRIST_CAMERAS = ('left_wrist', 'right_wrist')


def augment_generator(seed, step, rank):
    """One CPU generator per (seed, step, rank): the same draws on every device and after a resume (ACT's)."""
    key = hashlib.sha256(f'pi05-augment/{int(seed)}/{int(step)}/{int(rank)}'.encode()).digest()
    return torch.Generator().manual_seed(int.from_bytes(key[:8], 'little') & ((1 << 63) - 1))


def sample_params(batch, size, generator, device='cpu'):
    """Per-sample draws (augmax's order: the rotation angle, the crop offsets, then each ColorJitter's apply /
    brightness / contrast / hue), moved to `device` once."""
    uniforms = torch.rand(batch, len(DRAWS), generator=generator)
    if torch.device(device).type == 'cuda':
        uniforms = uniforms.pin_memory().to(device, non_blocking=True)
    draws = dict(zip(DRAWS, uniforms.to(device).unbind(1)))
    height, width = size
    crop_h, crop_w = int(height * CROP_FRACTION), int(width * CROP_FRACTION)

    def span(name, limit):
        return (2 * draws[name] - 1) * limit

    def color(prefix):
        return {'apply': draws[prefix + 'apply'] < JITTER_PROBABILITY,
                'brightness': span(prefix + 'brightness', BRIGHTNESS),
                'contrast': span(prefix + 'contrast', CONTRAST),
                'hue': span(prefix + 'hue', HUE)}
    return {'theta': span('theta', math.radians(ANGLE_DEGREES)),
            'cy': span('cy', (height - crop_h) / 2),
            'cx': span('cx', (width - crop_w) / 2),
            'crop': (crop_h, crop_w), 'color': color(''), 'wrist_color': color('wrist_')}


def sampling_grid(params, size, device):
    """grid_sample coordinates (batch, H, W, 2) of RandomCrop -> Resize -> Rotate about the output center."""
    height, width = size
    crop_h, crop_w = params['crop']
    sy, sx = crop_h / height, crop_w / width
    y = torch.arange(height, device=device, dtype=torch.float32) - (height / 2 - 0.5)
    x = torch.arange(width, device=device, dtype=torch.float32) - (width / 2 - 0.5)
    y, x = torch.meshgrid(y, x, indexing='ij')
    cos = torch.cos(params['theta']).float().view(-1, 1, 1)
    sin = torch.sin(params['theta']).float().view(-1, 1, 1)
    y_in = sy * (cos * y + sin * x) + params['cy'].float().view(-1, 1, 1) + (height / 2 - 0.5)
    x_in = sx * (-sin * y + cos * x) + params['cx'].float().view(-1, 1, 1) + (width / 2 - 0.5)
    return torch.stack((2 * x_in / (width - 1) - 1, 2 * y_in / (height - 1) - 1), dim=-1)


def warp(images, grid):
    """(batch, n, 3, H, W) images resampled with each sample's grid (shared by its n images)."""
    batch, count = images.shape[:2]
    flat = images.reshape(batch * count, *images.shape[2:])
    flat_grid = grid.unsqueeze(1).expand(batch, count, *grid.shape[1:]).reshape(batch * count, *grid.shape[1:])
    out = F.grid_sample(flat, flat_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    return out.reshape(images.shape)


def rgb_to_hsv(rgb):
    value = rgb.max(dim=-3).values
    argmax = torch.argmax(rgb, dim=-3)
    spread = value - rgb.min(dim=-3).values
    second = torch.gather(rgb, -3, ((argmax + 1) % 3).unsqueeze(-3)).squeeze(-3)
    third = torch.gather(rgb, -3, ((argmax + 2) % 3).unsqueeze(-3)).squeeze(-3)
    safe = torch.where(spread == 0, torch.ones_like(spread), spread)
    hue = torch.where(spread == 0, torch.zeros_like(spread), (2 * argmax + (second - third) / safe) / 6)
    saturation = torch.where(value == 0, torch.zeros_like(value),
                             spread / torch.where(value == 0, torch.ones_like(value), value))
    return hue, saturation, value


def hsv_to_rgb(hue, saturation, value):
    n = torch.arange(5.0, 0.0, -2.0, device=hue.device).view(*([1] * (hue.dim() - 2)), 3, 1, 1)
    k = torch.remainder(n + hue.unsqueeze(-3) * 6, 6)
    return value.unsqueeze(-3) - value.unsqueeze(-3) * saturation.unsqueeze(-3) * torch.clamp(torch.minimum(k, 4 - k), 0, 1)


def adjust_brightness(value, brightness):
    return torch.where(brightness < 0, value * (1 + brightness), value * (1 - brightness) + brightness)


def adjust_contrast(value, contrast):
    slant = torch.tan((contrast + 1) * (math.pi / 4))
    p1 = (slant - slant ** 2) / (2 * (1 - slant ** 2))
    p2 = 1 - p1
    middle = slant * (value - 0.5) + 0.5
    return torch.where(value < p1, value / slant, torch.where(value > p2, value / slant + 1 - 1 / slant, middle))


def color_jitter(images, color):
    """augmax's ColorJitter on (batch, n, 3, H, W) images in [0, 1]: brightness and contrast on V, a hue shift, applied
    where the sample's draw says so (p = 0.5); saturation stays (a no-op in augmax 0.4.1)."""
    hue, saturation, value = rgb_to_hsv(images)
    view = (-1,) + (1,) * (hue.dim() - 1)
    value = adjust_brightness(value, color['brightness'].float().view(view))
    value = adjust_contrast(value, color['contrast'].float().view(view))
    hue = hue + color['hue'].float().view(view)
    jittered = hsv_to_rgb(hue, saturation, value)
    return torch.where(color['apply'].view(view + (1,)), jittered, images)


def augment_stacks(nonwrist, wrists, params):
    """The augmentation of a sample's stacked images: `nonwrist` (batch, n, 3, H, W) warped with the sample's grid then
    jittered with its color draw; `wrists` (batch, m, 3, H, W) jittered with the wrist draw (None: no wrist images)."""
    out_nonwrist = None
    if nonwrist is not None:
        grid = sampling_grid(params, tuple(nonwrist.shape[-2:]), nonwrist.device)
        out_nonwrist = color_jitter(warp(nonwrist.float(), grid), params['color'])
    out_wrists = None if wrists is None else color_jitter(wrists.float(), params['wrist_color'])
    return out_nonwrist, out_wrists


_COMPILED = {}


def compiled_augment_stacks():
    if 'augment' not in _COMPILED:
        _COMPILED['augment'] = torch.compile(augment_stacks, dynamic=False, fullgraph=True)
    return _COMPILED['augment']


def augment_obs(obs, cameras, goal_pairs, seed, step, rank, wrist_cameras=WRIST_CAMERAS, compiled=False):
    """`obs` (a training batch's observation dict, images float [0, 1]: cameras (B, To, 3, H, W), goals (B, 3, H, W))
    with every image augmented; other entries (state, lang_emb) pass through. `cameras` are the camera keys of the
    policy, `goal_pairs` maps each goal key to the camera it is the goal of (the goal shares that camera's draw type:
    a head goal is warped and jittered with the head's draw)."""
    cameras = tuple(cameras)
    goal_pairs = dict(goal_pairs or {})
    wrist = {camera for camera in cameras if camera in wrist_cameras}
    nonwrist_keys = [camera for camera in cameras if camera not in wrist]
    nonwrist_keys += [key for key, camera in goal_pairs.items() if camera not in wrist]
    wrist_keys = [camera for camera in cameras if camera in wrist]
    wrist_keys += [key for key, camera in goal_pairs.items() if camera in wrist]
    reference = obs[cameras[0]]
    batch, size = reference.shape[0], tuple(reference.shape[-2:])
    params = sample_params(batch, size, augment_generator(seed, step, rank), reference.device)

    def stack(keys):  # (B, sum of n_k, 3, H, W); a goal (B, 3, H, W) counts as one image
        parts = [obs[key] if obs[key].dim() == 5 else obs[key].unsqueeze(1) for key in keys]
        return torch.cat(parts, dim=1) if parts else None

    def split(stacked, keys):
        offset = 0
        for key in keys:
            count = obs[key].shape[1] if obs[key].dim() == 5 else 1
            piece = stacked[:, offset:offset + count]
            yield key, piece if obs[key].dim() == 5 else piece[:, 0]
            offset += count
    nonwrist, wrists = stack(nonwrist_keys), stack(wrist_keys)
    if compiled and 'failed' not in _COMPILED:
        try:
            out_nonwrist, out_wrists = compiled_augment_stacks()(nonwrist, wrists, params)
        except Exception as error:  # noqa: BLE001 -- compiled once per process; eager is exact too
            _COMPILED['failed'] = repr(error)
            LOGGER.warning('--image-augment pi05: torch.compile failed (%r); running eagerly', error)
            out_nonwrist, out_wrists = augment_stacks(nonwrist, wrists, params)
    else:
        out_nonwrist, out_wrists = augment_stacks(nonwrist, wrists, params)
    result = dict(obs)
    if out_nonwrist is not None:
        result.update(split(out_nonwrist, nonwrist_keys))
    if out_wrists is not None:
        result.update(split(out_wrists, wrist_keys))
    return result
