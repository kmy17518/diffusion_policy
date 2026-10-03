"""R1Pro joint-command mapping shared by training and inference."""

import cv2
import numpy as np

STATE_INDICES = np.r_[0:3, 53:57, 3:10, 24:26, 28:35, 49:51]
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


def resize_rgb(image, image_size):
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] not in (3, 4):
        raise ValueError('Camera must be uint8 HWC RGB or RGBA')
    image = image[..., :3]
    height, width = image.shape[:2]
    if min(height, width) < 1:
        raise ValueError('Camera dimensions must be positive')
    scale = min(image_size / height, image_size / width)
    h, w = max(1, int(height * scale)), max(1, int(width * scale))
    resized = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    result = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    top, left = (image_size - h) // 2, (image_size - w) // 2
    result[top:top + h, left:left + w] = resized
    return result


def condition_state(state, task_ids, task_map):
    ids = np.asarray(task_ids)
    if ids.dtype.kind not in 'iu' or not np.isin(ids, list(task_map)).all():
        raise ValueError(f'Unknown or noninteger task_id; known ids: {sorted(task_map)}')
    positions = {task: i for i, task in enumerate(sorted(task_map))}
    onehot = np.eye(len(task_map), dtype=np.float32)[
        np.asarray([positions[int(task)] for task in ids.reshape(-1)]).reshape(ids.shape)]
    return np.concatenate((state, onehot), axis=-1).astype(np.float32)
