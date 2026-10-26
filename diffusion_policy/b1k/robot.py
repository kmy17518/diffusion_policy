"""R1Pro joint-command mapping shared by training and inference."""

import cv2
import numpy as np

STATE_INDICES = np.r_[0:3, 53:57, 3:10, 24:26, 28:35, 49:51]
# Image resize modes (ModelConfig.resize_mode): `dp` is this repository's aspect-preserving cv2 resize with centered zero
# padding (`resize_rgb`); `act` is kmy17518/act's `preprocess_image` -- a plain bilinear, antialiased resize of the whole
# frame to image_size x image_size (torch, align_corners=False) rounded to the nearest uint8 (`quantize_image`) -- the
# pixels of ACT's `act_b1k_frame_cache_v1` caches and of its serving path, so a DP policy trained from such a cache sees
# the same pixels at serving time and when decoding goal frames.
RESIZE_MODES = ('dp', 'act')
# Relative actions (ModelConfig.action_representation == 'relative', ACT's ACTION_DELTA with `--gripper-state sum`):
# the joint commands (torso 3:7, arms 7:14 and 15:22) are taken relative to the same joints of the current state, the
# gripper commands (14, 22) relative to the opening the summed finger state implies in command units
# (`20 * opening - 1`), the base velocities (0:3) stay absolute.
ACTION_DELTA = {'joint_action_dims': list(range(3, 14)) + list(range(15, 22)),
                'joint_state_dims': list(range(3, 14)) + list(range(15, 22)),
                'gripper_action_dims': [14, 22], 'gripper_state_dims': [14, 22],
                'gripper_command_from_state': [20.0, -1.0], 'absolute_action_dims': [0, 1, 2]}
ACTION_REPRESENTATIONS = ('absolute', 'relative')
GRIPPER_ACTION_DIMS = (14, 22)  # left and right gripper commands, in [-1, 1] (+1 = fully open)
DEFAULT_GRIPPER_SNAP = 0.9
# Gripper proprioception (ModelConfig.gripper_state): `sum` adds each gripper's two finger positions into one opening,
# so the state has one value per action dimension in the action's order (23 values, as in openpi); `fingers` keeps
# both positions (25 values), the layout of checkpoints that do not record `gripper_state`.
GRIPPER_STATES = ('sum', 'fingers')
CAMERAS = {
    'head': ('observation.rgb.zed_link_camera_0',
             'robot_r1::robot_r1:zed_link:Camera:0::rgb'),
    'left_wrist': ('observation.rgb.left_realsense_link_camera_0',
                   'robot_r1::robot_r1:left_realsense_link:Camera:0::rgb'),
    'right_wrist': ('observation.rgb.right_realsense_link_camera_0',
                    'robot_r1::robot_r1:right_realsense_link:Camera:0::rgb'),
}
PROPRIO_KEY = 'robot_r1::proprio'
# Goal images (goal-image conditioning): the dataset's dedicated goal streams and the wire keys of goal images in
# serving requests (`goal::` + the camera's observation key). `goal_key(camera)` names them in observation dicts.
GOAL_VIDEO_KEYS = {camera: keys[0].replace('observation.rgb.', 'observation.goal_rgb.') for camera, keys in CAMERAS.items()}
GOAL_OBS_KEYS = {camera: f'goal::{keys[1]}' for camera, keys in CAMERAS.items()}


def goal_key(camera):
    """Observation-dict key of a camera's goal image."""
    return f'goal_{camera}'


def extract_state(state, gripper_state='sum'):
    """Proprioception of R1Pro states (last axis of 61): the STATE_INDICES values (base velocity 0:3, torso 3:7,
    left arm 7:14, left fingers 14:16, right arm 16:23, right fingers 23:25), each finger pair summed for `sum`."""
    state = np.asarray(state, dtype=np.float32)
    if state.shape[-1] != 61 or not np.isfinite(state).all():
        raise ValueError('R1Pro proprio must be finite with last dimension 61')
    if gripper_state not in GRIPPER_STATES:
        raise ValueError(f'Unknown gripper_state {gripper_state!r}; expected one of {GRIPPER_STATES}')
    selected = state[..., STATE_INDICES]
    if gripper_state == 'fingers':
        return selected
    return np.concatenate([selected[..., :14], selected[..., 14:16].sum(-1, keepdims=True),
                           selected[..., 16:23], selected[..., 23:25].sum(-1, keepdims=True)], axis=-1)


def proprio_dim(gripper_state='sum'):
    """Number of proprioception values `extract_state` returns for a gripper layout."""
    return len(STATE_INDICES) - 2 if gripper_state == 'sum' else len(STATE_INDICES)


def resize_rgb(image, image_size, mode='dp'):
    """uint8 HWC RGB/RGBA to uint8 (image_size, image_size, 3) RGB with `mode` (RESIZE_MODES)."""
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] not in (3, 4):
        raise ValueError('Camera must be uint8 HWC RGB or RGBA')
    image = image[..., :3]
    height, width = image.shape[:2]
    if min(height, width) < 1:
        raise ValueError('Camera dimensions must be positive')
    if mode == 'act':
        return resize_rgb_act(image, image_size)
    if mode != 'dp':
        raise ValueError(f'Unknown resize mode {mode!r}; expected one of {RESIZE_MODES}')
    scale = min(image_size / height, image_size / width)
    h, w = max(1, int(height * scale)), max(1, int(width * scale))
    resized = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    result = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    top, left = (image_size - h) // 2, (image_size - w) // 2
    result[top:top + h, left:left + w] = resized
    return result


def resize_rgb_act(image, image_size):
    """ACT's `quantize_image(preprocess_image(image, (S, S)))`: the whole frame resized to S x S by torch's bilinear,
    antialiased interpolation (align_corners=False) in float [0, 1], then rounded to the nearest uint8; a frame that
    already has the target size is returned unchanged (as ACT's `preprocess_image` skips the resize then: the evaluator
    sends frames resized and quantized on its side)."""
    import torch
    import torch.nn.functional as F
    image = np.ascontiguousarray(np.asarray(image)[..., :3])
    if image.shape[:2] == (image_size, image_size):
        return image
    tensor = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float() / 255.0
    tensor = F.interpolate(tensor[None], size=(image_size, image_size), mode='bilinear', align_corners=False,
                           antialias=True)[0]
    return torch.round(tensor * 255).clamp_(0, 255).to(torch.uint8).permute(1, 2, 0).contiguous().numpy()


def action_reference(proprio, delta=ACTION_DELTA):
    """What relative actions are taken against (float64, (..., 23)), from `sum`-layout proprioception (..., 23): the
    state on the joint dims, the gripper openings in command units on the gripper dims, zero on the absolute dims
    (ACT's `b1k_dataset.action_reference`)."""
    proprio = np.asarray(proprio, dtype=np.float64)
    if proprio.shape[-1] != 23:
        raise ValueError(f'Relative actions need the 23-value `sum` gripper state, got {proprio.shape[-1]} values')
    reference = np.zeros(proprio.shape[:-1] + (23,))
    reference[..., delta['joint_action_dims']] = proprio[..., delta['joint_state_dims']]
    scale, offset = delta['gripper_command_from_state']
    reference[..., delta['gripper_action_dims']] = scale * proprio[..., delta['gripper_state_dims']] + offset
    return reference


def relative_actions(actions, proprio, delta=ACTION_DELTA):
    """Raw actions (..., T, 23) relative to the reference of `proprio` (..., 23), float32."""
    return (np.asarray(actions, dtype=np.float64) - action_reference(proprio, delta)[..., None, :]).astype(np.float32)


def absolute_actions(relative, proprio, delta=ACTION_DELTA):
    """Inverse of `relative_actions`: relative actions (..., T, 23) or single actions (..., 23) plus the reference of
    `proprio`, the state they were predicted from."""
    reference = action_reference(proprio, delta)
    relative = np.asarray(relative, dtype=np.float64)
    return relative + (reference[..., None, :] if relative.ndim == reference.ndim + 1 else reference)


def snap_grippers(actions, threshold=DEFAULT_GRIPPER_SNAP):
    """Gripper commands clamped to [-1, 1], those above `threshold` set to exactly +1 (None: untouched); ACT's
    `b1k_server.snap_grippers`: OmniGibson releases an assisted grasp only at a command of exactly +1."""
    if threshold is None:
        return actions
    actions = np.array(actions, copy=True)
    grippers = np.clip(actions[..., list(GRIPPER_ACTION_DIMS)], -1.0, 1.0)
    grippers[grippers > threshold] = 1.0
    actions[..., list(GRIPPER_ACTION_DIMS)] = grippers
    return actions


def task_family(task_name):
    """The task family of an ISG task name: the part before the first '-' (camera_relocalization, object_scaling, ...)."""
    return str(task_name).split('-')[0]


def condition_state(state, task_ids, task_map, task_onehot=True):
    """Validate task ids against the checkpoint task map; append their one-hot to the 25-D state unless
    `task_onehot` is False (the ids are still required and rejected when unknown)."""
    ids = np.asarray(task_ids)
    if ids.dtype.kind not in 'iu' or not np.isin(ids, list(task_map)).all():
        raise ValueError(f'Unknown or noninteger task_id; known ids: {sorted(task_map)}')
    if not task_onehot:
        return np.asarray(state, dtype=np.float32)
    positions = {task: i for i, task in enumerate(sorted(task_map))}
    onehot = np.eye(len(task_map), dtype=np.float32)[
        np.asarray([positions[int(task)] for task in ids.reshape(-1)]).reshape(ids.shape)]
    return np.concatenate((state, onehot), axis=-1).astype(np.float32)
