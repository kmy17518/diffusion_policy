"""R1Pro joint-command mapping shared by training and inference."""

import cv2
import numpy as np

STATE_INDICES = np.r_[0:3, 53:57, 3:10, 24:26, 28:35, 49:51]
CAMERAS = {
    'head': ('observation.rgb.zed_link_camera_0',
             'robot_r1::robot_r1:zed_link:Camera:0::rgb'),
    'left_wrist': ('observation.rgb.left_realsense_link_camera_0',
                   'robot_r1::robot_r1:left_realsense_link:Camera:0::rgb'),
    'right_wrist': ('observation.rgb.right_realsense_link_camera_0',
                    'robot_r1::robot_r1:right_realsense_link:Camera:0::rgb'),
}
PROPRIO_KEY = 'robot_r1::proprio'


def extract_state(state):
    state = np.asarray(state, dtype=np.float32)
    if state.shape[-1] != 61 or not np.isfinite(state).all():
        raise ValueError('R1Pro proprio must be finite with last dimension 61')
    return state[..., STATE_INDICES]


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
