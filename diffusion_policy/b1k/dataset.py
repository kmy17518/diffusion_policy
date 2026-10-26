"""Read-only, bounded-memory LeRobot v3 parquet/video sequences."""

from collections import OrderedDict, defaultdict
from decimal import Decimal
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

import av
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch

from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.b1k.robot import (ACTION_DELTA, ACTION_REPRESENTATIONS, CAMERAS, GOAL_VIDEO_KEYS, GRIPPER_STATES,
                                        RESIZE_MODES, action_reference, condition_state, extract_state, goal_key,
                                        proprio_dim, relative_actions, resize_rgb)

NORMALIZATIONS = ('limits', 'zscore', 'quantile')
QUANTILE_EPS = 1e-6  # openpi's (ACT's QUANTILE_EPS)
# Dimensions whose q99 - q01 is narrower are normalized over this range around their center instead (a constant state
# or a near-constant relative torso command would otherwise blow noise up by 1 / (q99 - q01)); ACT's QUANTILE_MIN_RANGE.
QUANTILE_MIN_RANGE = 0.02
ZSCORE_STD_FLOOR = 0.01  # ACT's std floor
VAL_SPLIT_FORMAT = 'isg-val-split/v1'

GOAL_SOURCES = ('episode_last', 'goal_key')
EPISODE_SPLIT_FORMAT = 'isg-episode-split/v1'
DEFAULT_EPISODE_SPLIT = 'isg_meta/train_split.json'
TASK_GROUPS_FORMAT = 'isg-task-groups/v1'
DEFAULT_TASK_GROUPS = 'isg_meta/task_groups.json'
SETTLE_WINDOWS_FORMAT = 'isg-settle-windows/v1'
DEFAULT_SETTLE_WINDOWS = 'isg_meta/settle_windows.json'


class EpisodeSequenceIndex:
    """SequenceSampler's clipped edge padding without an O(frames) index array."""

    def __init__(self, lengths, horizon, pad_before=0, pad_after=0):
        if horizon < 1:
            raise ValueError('horizon must be positive')
        self.lengths = np.asarray(lengths, dtype=np.int64)
        self.horizon = horizon
        self.pad_before = int(np.clip(pad_before, 0, horizon - 1))
        self.pad_after = int(np.clip(pad_after, 0, horizon - 1))
        counts = np.maximum(0, self.lengths - horizon + self.pad_before + self.pad_after + 1)
        self.ends = np.cumsum(counts)

    def __len__(self):
        return int(self.ends[-1]) if len(self.ends) else 0

    def locate(self, index, with_valid=False):
        """(episode position, clipped frame indices) of sequence `index`; with `with_valid` also the mask of the steps
        that lie inside the episode (False = edge padding, the frame repeated)."""
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode = int(np.searchsorted(self.ends, index, side='right'))
        previous = int(self.ends[episode - 1]) if episode else 0
        start = index - previous - self.pad_before
        unclipped = np.arange(start, start + self.horizon)
        frames = np.clip(unclipped, 0, self.lengths[episode] - 1)
        if with_valid:
            return episode, frames, (unclipped >= 0) & (unclipped < self.lengths[episode])
        return episode, frames


def _matrix(column):
    column = column.combine_chunks()
    return column.flatten().to_numpy(zero_copy_only=False).reshape(len(column), -1).astype(np.float32)


def images_to_float(obs):
    """Convert uint8 image tensors (any device) to float32 in [0, 1]; other values pass through.

    Bit-identical to `image.astype(np.float32) / 255.`: dividing by a 0-dim tensor forces a true
    IEEE division kernel, whereas a Python scalar divisor becomes a multiply by the reciprocal on
    CUDA, which is off by one ulp for some of the 256 values.
    """
    if isinstance(obs, dict):
        return {key: images_to_float(value) for key, value in obs.items()}
    if torch.is_tensor(obs) and obs.dtype == torch.uint8:
        return torch.div(obs.to(torch.float32), torch.tensor(255., dtype=torch.float32, device=obs.device))
    return obs


class VideoReader:
    """Random access into packed videos; one open decoder per file, bounded LRU of both.

    `max_open` should cover the files a worker touches repeatedly: reopening a 200 MB packed
    MP4 costs ~5 ms (index parse + stream probe), more than decoding a short GOP.
    """

    def __init__(self, image_size, tolerance=0.008, max_open=32, cache_frames=64, resize_mode='dp'):
        self.image_size = image_size
        self.tolerance = tolerance
        self.max_open = max_open
        self.cache_frames = cache_frames
        self.resize_mode = resize_mode
        self.containers = OrderedDict()
        self.frames = OrderedDict()

    def close(self):
        for container in self.containers.values():
            container.close()
        self.containers.clear()
        self.frames.clear()

    def read(self, path, timestamps):
        path = str(path)
        timestamps = np.asarray(timestamps, dtype=np.float64)
        keys = [(path, round(float(t), 6)) for t in timestamps]
        missing = sorted({key[1] for key in keys if key not in self.frames})
        decoded = {}
        if missing:
            if path not in self.containers:
                self.containers[path] = av.open(path, mode='r')
                stream = self.containers[path].streams.video[0]
                stream.thread_type = 'SLICE'
                stream.codec_context.thread_count = 1
                while len(self.containers) > self.max_open:
                    self.containers.popitem(last=False)[1].close()
            self.containers.move_to_end(path)
            container = self.containers[path]
            stream = container.streams.video[0]
            # Backward seek lands on the last keyframe at or before the target. Aiming at
            # `t + tolerance` (rather than `t - tolerance`) cannot overshoot the wanted frame
            # as long as consecutive frames are more than 2 * tolerance apart, but it avoids
            # decoding a whole extra GOP whenever the wanted frame is itself a keyframe. The
            # decoded frames are identical either way.
            target = max(0, missing[0] - self.tolerance)
            rate = stream.average_rate
            if rate and float(rate) > 0 and 1 / float(rate) > 2 * self.tolerance:
                target = max(0, missing[0] + self.tolerance)
            container.seek(int(target / float(stream.time_base)), stream=stream, backward=True)
            pending = set(missing)
            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                timestamp = float(frame.pts * stream.time_base)
                matches = [t for t in pending if abs(timestamp - t) <= self.tolerance]
                if matches:
                    image = resize_rgb(frame.to_ndarray(format='rgb24'), self.image_size, self.resize_mode)
                    for t in matches:
                        decoded[(path, t)] = image
                        pending.remove(t)
                if not pending or timestamp > missing[-1] + self.tolerance:
                    break
            if pending:
                raise ValueError(f'Video timestamps not found within {self.tolerance}s in {path}: {sorted(pending)}')
        result = [decoded[key] if key in decoded else self.frames[key] for key in keys]
        self.frames.update(decoded)
        for key in keys:
            if key in self.frames:
                self.frames.move_to_end(key)
        while len(self.frames) > self.cache_frames:
            self.frames.popitem(last=False)
        return np.stack(result)


def resolve_episode_split(dataset_path, spec='auto', task_names=None):
    """Resolve `--episode-split`: auto (<root>/isg_meta/train_split.json if present), none, or a split file path.

    Returns None (no split) or the record saved with checkpoints: resolved file, sha256 of its bytes, format, name,
    subset 'train', the selected tasks (`task_names`, else every task of the split) and the sorted union of their
    `train` episodes; load it with `B1KLeRobotDataset(root, record['tasks'], episodes=record['episodes'])`.
    """
    if spec == 'none':
        return None
    path = (Path(dataset_path) / DEFAULT_EPISODE_SPLIT if spec == 'auto' else Path(spec)).resolve()
    if not path.is_file():
        if spec == 'auto':
            return None
        raise FileNotFoundError(f'Episode split file not found: {path}')
    content = path.read_bytes()
    split = json.loads(content)
    if not isinstance(split, dict) or split.get('format') != EPISODE_SPLIT_FORMAT or not isinstance(split.get('tasks'), dict):
        raise ValueError(f'{path} is not an {EPISODE_SPLIT_FORMAT} episode split')
    names = list(dict.fromkeys([task_names] if isinstance(task_names, str) else task_names or split['tasks']))
    missing = [name for name in names if name not in split['tasks']]
    if missing:
        raise ValueError(f'Selected task(s) {missing} have no entry in episode split {path}')
    episodes = set()
    for name in names:
        train = split['tasks'][name].get('train') if isinstance(split['tasks'][name], dict) else None
        if not isinstance(train, list) or any(type(index) is not int for index in train):
            raise ValueError(f'{path}: tasks[{name!r}].train must be a list of integer episode indices')
        episodes.update(train)
    return {'file': str(path), 'sha256': hashlib.sha256(content).hexdigest(), 'format': EPISODE_SPLIT_FORMAT,
            'name': split.get('name'), 'subset': 'train', 'tasks': names, 'episodes': sorted(episodes)}


def expand_task_groups(dataset_path, task_names):
    """Replace every task group of <root>/isg_meta/task_groups.json in `task_names` by the tasks it reaches.

    A group lists task names and/or other groups. None (every task) stays None and without a group file the names
    are returned unchanged; otherwise the tasks come back without duplicates in order of first appearance. Unknown
    names, a name that is both a task and a group, empty groups and cycles raise ValueError.
    """
    if task_names is None:
        return None
    names = [task_names] if isinstance(task_names, str) else list(task_names)
    path = Path(dataset_path).resolve() / DEFAULT_TASK_GROUPS
    if not path.is_file():
        return names
    content = json.loads(path.read_text())
    groups = content.get('groups') if isinstance(content, dict) else None
    if not isinstance(groups, dict) or content.get('format') != TASK_GROUPS_FORMAT:
        raise ValueError(f'{path} is not an {TASK_GROUPS_FORMAT} task group file')
    table = pq.read_table(Path(dataset_path) / 'meta/tasks.parquet').to_pydict()
    tasks = set(next((table[key] for key in ('task_name', 'task', '__index_level_0__') if key in table), []))

    def expand(name, parents):
        if name in tasks:
            if name in groups:
                raise ValueError(f'{name!r} is both a task and a task group in {path}')
            return [name]
        if name not in groups:
            if parents:
                raise ValueError(f'Task group {parents[-1]!r} in {path} lists unknown task or group {name!r}')
            raise ValueError(f'Unknown task or task group {name!r}; groups in {path}: {sorted(groups)}; '
                             f'tasks: {sorted(tasks)}')
        if name in parents:
            raise ValueError(f'Task group cycle {" -> ".join([*parents, name])} in {path}')
        members = groups[name]
        if not isinstance(members, list) or not members or not all(isinstance(member, str) for member in members):
            raise ValueError(f'Task group {name!r} in {path} must be a nonempty list of task or group names')
        return [task for member in members for task in expand(member, (*parents, name))]

    return list(dict.fromkeys(task for name in names for task in expand(name, ())))


def parse_settle_steps(spec):
    """Canonical `--settle-steps` value: `all`, a number of settle frames ('0', '10') or a decimal fraction of each
    episode's program length ('0.2'; '1.0' is the whole program length, '1' one frame). Raises ValueError otherwise."""
    text = str(spec).strip()
    if text == 'all':
        return text
    if re.fullmatch(r'[0-9]+', text):
        return str(int(text))
    if re.fullmatch(r'[0-9]*\.[0-9]+|[0-9]+\.', text):
        value = format(Decimal(text).normalize(), 'f')
        return value if '.' in value else f'{value}.0'
    raise ValueError(f'--settle-steps must be all, a number of frames (0, 10) or a decimal fraction of the program '
                     f'length (0.1, 0.2); got {spec!r}')


def settle_frames(spec, program_length):
    """Settle frames that a canonical value other than `all` requests after `program_length` program frames."""
    return math.ceil(Fraction(spec) * program_length) if '.' in spec else int(spec)


def apply_settle_steps(dataset_path, spec, episodes):
    """Cut episodes (metadata rows; `length` and `dataset_to_index` change in place) to `--settle-steps SPEC`.

    `all` keeps every recorded frame and returns None. Otherwise each episode keeps its program -- the first
    `program_length` frames listed by <root>/isg_meta/settle_windows.json -- followed by SPEC settle frames, capped
    at the recorded window; returns the record saved with checkpoints.
    """
    spec = parse_settle_steps(spec)
    if spec == 'all':
        return None
    path = Path(dataset_path).resolve() / DEFAULT_SETTLE_WINDOWS
    if not path.is_file():
        raise FileNotFoundError(f'--settle-steps {spec} needs {path}, the program length of every episode; only datasets '
                                'whose episodes end with a recorded settle window ship it')
    content = path.read_bytes()
    windows = json.loads(content)
    if not isinstance(windows, dict) or windows.get('format') != SETTLE_WINDOWS_FORMAT or \
            not isinstance(windows.get('episodes'), dict):
        raise ValueError(f'{path} is not an {SETTLE_WINDOWS_FORMAT} file')
    recorded = settle = capped = 0
    for row in episodes:
        entry = windows['episodes'].get(str(row['episode_index']))
        if not entry or entry.get('length') != row['length'] or not 0 < entry.get('program_length', 0) <= row['length']:
            raise ValueError(f'{path} has no settle window for episode {row["episode_index"]} of length {row["length"]} '
                             f'(entry {entry}); regenerate it for this dataset')
        program = entry['program_length']
        wanted = settle_frames(spec, program)
        kept = min(wanted, row['length'] - program)
        capped += wanted > kept
        recorded += row['length']
        settle += kept
        row['length'] = program + kept
        row['dataset_to_index'] = row['dataset_from_index'] + row['length']
    if capped:
        print(f'WARNING: --settle-steps {spec} exceeds the recorded settle window of {capped} of {len(episodes)} '
              'episodes; they keep their whole window', flush=True)
    return {'spec': spec, 'file': str(path), 'sha256': hashlib.sha256(content).hexdigest(),
            'format': SETTLE_WINDOWS_FORMAT, 'frames': sum(row['length'] for row in episodes), 'settle_frames': settle,
            'recorded_frames': recorded, 'capped_episodes': capped}


class B1KLeRobotDataset(BaseImageDataset):
    def __init__(self, dataset_path, task_names=None, horizon=16, n_obs_steps=2,
                 n_action_steps=8, cameras=tuple(CAMERAS), image_size=96,
                 pad_before=None, pad_after=None, episode_cache_size=8,
                 parquet_cache_mb=256, max_episodes=None, observation_mode='image',
                 obs_steps=None, imagenet_norm=False, language_conditioning='none',
                 prompt_source='task_name', task_onehot=True, image_dtype='float32', frame_cache=None,
                 video_max_open=32, goal_views=(), goal_source='episode_last', episodes=None, settle_steps='all',
                 gripper_state='sum', resize_mode='dp', action_representation='absolute', return_meta=False):
        """
        goal_views / goal_source (goal-image conditioning): for every camera in `goal_views` (a subset of `cameras`)
        each sample also carries `goal_<camera>`, one (3, S, S) image for the whole episode with no observation-
        history axis, looked up from a table built once at construction. `episode_last` decodes the episode's last
        recorded frame of that camera (any LeRobot v3 root; also when `settle_steps` cuts the episode earlier);
        `goal_key` decodes the dataset's static `observation.goal_rgb.<camera>_camera_0` clip.

        episodes (episode split, see `resolve_episode_split`): episode_index values to load, applied after the task
        filter and before `max_episodes`. Each must be present locally with all files it needs and belong to a
        selected task, otherwise construction fails, so the selection equals the request exactly. None loads every
        local episode of the selected tasks.

        settle_steps (see `apply_settle_steps`): cut every episode after its program plus that many settle frames
        without rewriting the dataset. Sequences (edge-padded at the cut), the normalizer statistics of
        `iter_lowdim` and the fingerprint cover the kept frames only.

        gripper_state (see robot.GRIPPER_STATES): proprioception layout of the state and of the normalizer.

        resize_mode (robot.RESIZE_MODES): how decoded frames are resized to image_size (`dp`: aspect-preserving cv2
        letterbox; `act`: ACT's whole-frame antialiased bilinear resize, the pixels of ACT's frame caches). A frame
        cache must hold pixels of the same mode.

        action_representation (robot.ACTION_REPRESENTATIONS): `relative` returns each sample's actions relative to the
        state of its current frame (observation step n_obs_steps - 1; `robot.relative_actions`, needs gripper_state
        `sum`); the state itself stays absolute. `absolute` returns the recorded commands.

        return_meta: samples also carry `action_valid` (horizon,) -- False at edge-padded steps -- and `meta`
        {'task': task_index, 'episode': position in self.episodes, 'frame': current frame} for validation.
        """
        if gripper_state not in GRIPPER_STATES:
            raise ValueError(f'Unknown gripper_state {gripper_state!r}; expected one of {GRIPPER_STATES}')
        if resize_mode not in RESIZE_MODES:
            raise ValueError(f'Unknown resize_mode {resize_mode!r}; expected one of {RESIZE_MODES}')
        if action_representation not in ACTION_REPRESENTATIONS:
            raise ValueError(f'Unknown action_representation {action_representation!r}; expected one of {ACTION_REPRESENTATIONS}')
        if action_representation == 'relative' and gripper_state != 'sum':
            raise ValueError('Relative actions need gripper_state sum (one state value per action dimension)')
        self.gripper_state = gripper_state
        self.resize_mode = resize_mode
        self.action_representation = action_representation
        self.return_meta = bool(return_meta)
        self.root = Path(dataset_path).resolve()
        self.goal_views = tuple(goal_views or ())
        if goal_source not in GOAL_SOURCES:
            raise ValueError(f'goal_source must be one of {GOAL_SOURCES}')
        self.goal_source = goal_source
        self.goal_table = None
        self.info = json.loads((self.root / 'meta/info.json').read_text())
        if not self.info.get('codebase_version', '').startswith('v3'):
            raise ValueError('B1K requires native LeRobot v3 metadata')
        if observation_mode not in ('image', 'lowdim'):
            raise ValueError('observation_mode must be image or lowdim')
        if language_conditioning not in ('none', 'clip_film'):
            raise ValueError('language_conditioning must be none or clip_film')
        if prompt_source not in ('task_name', 'task_description'):
            raise ValueError('prompt_source must be task_name or task_description')
        if language_conditioning != 'none' and observation_mode != 'image':
            raise ValueError('clip_film requires image observations')
        if image_dtype not in ('float32', 'uint8'):
            raise ValueError('image_dtype must be float32 (CHW in [0, 1]) or uint8 (CHW, convert with images_to_float)')
        if video_max_open < 1:
            raise ValueError('video_max_open must be positive')
        self.language_conditioning = language_conditioning
        self.prompt_source = prompt_source
        self.task_onehot = bool(task_onehot)
        self.language = None
        self.observation_mode = observation_mode
        self.imagenet_norm = imagenet_norm
        self.image_dtype = image_dtype
        self.frame_cache = None if frame_cache is None else Path(frame_cache).resolve()
        self.video_max_open = video_max_open
        self.obs_steps = n_obs_steps if obs_steps is None else obs_steps
        if not n_obs_steps <= self.obs_steps <= horizon:
            raise ValueError('obs_steps must be between n_obs_steps and horizon')
        self.cameras = () if observation_mode == 'lowdim' else tuple(cameras)
        if ((observation_mode == 'image' and not self.cameras) or
                len(set(self.cameras)) != len(self.cameras) or set(self.cameras) - set(CAMERAS)):
            raise ValueError(f'cameras must be unique names from {list(CAMERAS)}')
        if len(set(self.goal_views)) != len(self.goal_views) or set(self.goal_views) - set(self.cameras):
            raise ValueError(f'goal_views must be distinct cameras among {list(self.cameras)}')
        if not 1 <= n_obs_steps <= horizon or not 1 <= n_action_steps <= horizon - n_obs_steps + 1:
            raise ValueError('Need 1 <= n_obs_steps and n_action_steps <= horizon - n_obs_steps + 1')
        if image_size < 1 or episode_cache_size < 1 or parquet_cache_mb < 0:
            raise ValueError('Invalid image size or cache bounds')
        self.horizon, self.n_obs_steps, self.n_action_steps = horizon, n_obs_steps, n_action_steps
        self.image_size = image_size
        self.episode_cache_size = episode_cache_size
        self.parquet_cache_bytes = int(parquet_cache_mb * 1024 ** 2)
        tasks = pq.read_table(self.root / 'meta/tasks.parquet').to_pydict()
        name_column = next((key for key in ('task_name', 'task', '__index_level_0__') if key in tasks), None)
        if name_column is None or 'task_index' not in tasks:
            raise ValueError('meta/tasks.parquet must contain task names and task_index')
        ids, raw_names = tasks['task_index'], tasks[name_column]
        if (any(type(index) is not int for index in ids) or
                any(not isinstance(name, str) or not name.strip() for name in raw_names)):
            raise ValueError('meta/tasks.parquet requires integer task_index and nonempty task names')
        if len(set(ids)) != len(ids) or len(set(raw_names)) != len(raw_names):
            raise ValueError('Duplicate/conflicting tasks in meta/tasks.parquet')
        all_tasks = dict(zip(ids, raw_names))
        names = [task_names] if isinstance(task_names, str) else list(task_names or [])
        unknown = set(names) - set(all_tasks.values())
        if unknown:
            raise ValueError(f'Unknown task(s) {sorted(unknown)}; available: {sorted(all_tasks.values())}')
        selected = {i for i, name in all_tasks.items() if not names or name in names}
        requested = None if episodes is None else {int(index) for index in episodes}
        self.episodes = []
        metadata_paths = sorted((self.root / 'meta/episodes').glob('*/*.parquet'))
        if not metadata_paths:
            raise ValueError('No LeRobot v3 episode metadata found')
        for path in metadata_paths:
            file = pq.ParquetFile(path)
            columns = ['episode_index', 'length', 'data/chunk_index', 'data/file_index',
                       'dataset_from_index', 'dataset_to_index']
            schema = file.schema_arrow.names
            columns += ['task_index'] if 'task_index' in schema else ['tasks']
            for camera in self.cameras:
                key = CAMERAS[camera][0]
                columns += [f'videos/{key}/{field}' for field in
                            ('chunk_index', 'file_index', 'from_timestamp', 'to_timestamp')]
            if self.goal_source == 'goal_key':
                for camera in self.goal_views:
                    key = GOAL_VIDEO_KEYS[camera]
                    wanted = [f'videos/{key}/{field}' for field in ('chunk_index', 'file_index', 'from_timestamp', 'to_timestamp')]
                    if any(column not in schema for column in wanted):
                        raise ValueError(f'{path}: no metadata for goal stream {key}; goal_source=goal_key needs the '
                                         'dataset\'s observation.goal_rgb.* streams (use goal_source=episode_last otherwise)')
                    columns += wanted
            table = file.read(columns=columns)
            if requested is not None:
                table = table.filter(pc.is_in(table['episode_index'],
                                              value_set=pa.array(sorted(requested), table['episode_index'].type)))
            elif 'task_index' in schema:
                table = table.filter(pc.is_in(table['task_index'], value_set=pa.array(sorted(selected))))
            for row in table.to_pylist():
                if 'task_index' not in row:
                    ids = [i for i, name in all_tasks.items() if name in (row.get('tasks') or [])]
                    if len(ids) != 1:
                        raise ValueError(f'Episode {row["episode_index"]} needs exactly one categorical task')
                    row['task_index'] = ids[0]
                if row['task_index'] not in selected:
                    if requested is not None:
                        raise ValueError(f'Requested episode {row["episode_index"]} belongs to task '
                                         f'{all_tasks.get(row["task_index"], row["task_index"])!r}, which is not selected')
                    continue
                data_path = self.data_path(row)
                if not data_path.is_file():
                    if names or requested is not None:
                        raise FileNotFoundError(f'Selected episode {row["episode_index"]}: missing {data_path}')
                    continue
                if row['length'] <= 0 or row['dataset_to_index'] - row['dataset_from_index'] != row['length']:
                    raise ValueError(f'Invalid episode bounds: {row["episode_index"]}')
                for camera in self.cameras:
                    if not self.video_path(row, camera).is_file():
                        raise FileNotFoundError(f'Selected camera missing: {self.video_path(row, camera)}')
                if self.goal_source == 'goal_key':
                    for camera in self.goal_views:
                        if not self.goal_video_path(row, camera).is_file():
                            raise FileNotFoundError(f'Goal stream missing: {self.goal_video_path(row, camera)}')
                self.episodes.append(row)
        self.episodes.sort(key=lambda row: row['episode_index'])
        ids = [row['episode_index'] for row in self.episodes]
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate episode_index in metadata')
        if requested is not None and set(ids) != requested:
            raise ValueError(f'Requested episode(s) {sorted(requested - set(ids))} are not in the local episode metadata')
        present = {row['task_index'] for row in self.episodes}
        if names and selected - present:
            raise ValueError(f'No {"requested " if requested is not None else ""}episodes of task(s) '
                             f'{[all_tasks[i] for i in sorted(selected - present)]} on disk')
        if max_episodes is not None:
            if max_episodes < 1:
                raise ValueError('max_episodes must be positive')
            self.episodes = self.episodes[:max_episodes]
            if names and selected - {row['task_index'] for row in self.episodes}:
                raise ValueError('max_episodes would remove a requested task; increase it')
        if not self.episodes:
            raise ValueError('No episodes with local data found')
        present = {row['task_index'] for row in self.episodes}
        self.task_map = {i: all_tasks[i] for i in sorted(present)}
        self.recorded_lengths = {row['episode_index']: row['length'] for row in self.episodes}
        self.settle = apply_settle_steps(self.root, settle_steps, self.episodes)
        if self.settle:
            print(json.dumps({'settle_steps': self.settle}), flush=True)
        self.sampler = EpisodeSequenceIndex(
            [row['length'] for row in self.episodes], horizon,
            n_obs_steps - 1 if pad_before is None else pad_before,
            n_action_steps - 1 if pad_after is None else pad_after)
        if not len(self.sampler):
            raise ValueError('No sequences for this horizon and padding')
        self.frame_cache_formats = set()
        if self.frame_cache is not None:
            from diffusion_policy.b1k.frame_cache import FrameCacheReader
            self.frame_cache_formats = FrameCacheReader(
                self.frame_cache, self.root, self.image_size, resize_mode=self.resize_mode).validate(
                {self.video_path(row, camera) for row in self.episodes for camera in self.cameras})
        self._reset_cache()
        if self.goal_views:
            self._build_goal_table()

    def goal_video_path(self, episode, camera):
        key = GOAL_VIDEO_KEYS[camera]
        return self.root / self.info['video_path'].format(
            video_key=key, chunk_index=episode[f'videos/{key}/chunk_index'],
            file_index=episode[f'videos/{key}/file_index'])

    def _build_goal_table(self):
        """Decode every episode's goal image once (main process) into uint8 [episodes, views, S, S, 3].

        One stored goal per episode with a batch-time lookup; workers receive the table through pickling. The
        frames go through the same `resize_rgb` as the cameras (from the frame cache when one is attached and the
        goal is a camera frame), so the goal path matches the observation path exactly.
        """
        started = time.perf_counter()
        table = np.empty((len(self.episodes), len(self.goal_views), self.image_size, self.image_size, 3), dtype=np.uint8)
        reader = (VideoReader(self.image_size, max_open=self.video_max_open, resize_mode=self.resize_mode)
                  if self.goal_source == 'goal_key' else self._video)
        for position, episode in enumerate(self.episodes):
            for view, camera in enumerate(self.goal_views):
                if self.goal_source == 'episode_last':
                    key = CAMERAS[camera][0]
                    timestamp = float(self._read_episode(episode)['timestamp'][-1]) + episode[f'videos/{key}/from_timestamp']
                    table[position, view] = reader.read(self.video_path(episode, camera), [timestamp])[0]
                else:
                    key = GOAL_VIDEO_KEYS[camera]
                    table[position, view] = reader.read(self.goal_video_path(episode, camera), [episode[f'videos/{key}/from_timestamp']])[0]
        if reader is not self._video:
            reader.close()
        self._episodes.clear()
        self.goal_table = table
        print(json.dumps({'goal_table': {'source': self.goal_source, 'episodes': len(self.episodes),
                                         'views': list(self.goal_views), 'image_size': self.image_size,
                                         'seconds': round(time.perf_counter() - started, 1)}}), flush=True)

    def prepare_language(self, checkpoint_language=None):
        from diffusion_policy.b1k.language import prepare_language
        cached = self.language if checkpoint_language is None else checkpoint_language
        self.language = prepare_language(self.root, self.task_map, self.language_conditioning,
                                         self.prompt_source, cached=cached)
        self._language_rows = {index: row for row, index in enumerate(sorted(self.task_map))}
        return self.language

    def _reset_cache(self):
        self._pid = os.getpid()
        self._episodes = OrderedDict()
        self._row_groups = OrderedDict()
        self._row_group_bytes = 0
        if self.frame_cache is not None:
            from diffusion_policy.b1k.frame_cache import FrameCacheReader
            self._video = FrameCacheReader(self.frame_cache, self.root, self.image_size, resize_mode=self.resize_mode)
        else:
            self._video = VideoReader(self.image_size, max_open=self.video_max_open, resize_mode=self.resize_mode)

    def __getstate__(self):
        state = self.__dict__.copy()
        for key in ('_episodes', '_row_groups', '_video'):
            state.pop(key, None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._reset_cache()

    def close(self):
        self._video.close()
        self._episodes.clear()
        self._row_groups.clear()
        self._row_group_bytes = 0

    def data_path(self, episode):
        return self.root / self.info['data_path'].format(
            chunk_index=episode['data/chunk_index'], file_index=episode['data/file_index'])

    def video_path(self, episode, camera):
        key = CAMERAS[camera][0]
        return self.root / self.info['video_path'].format(
            video_key=key, chunk_index=episode[f'videos/{key}/chunk_index'],
            file_index=episode[f'videos/{key}/file_index'])

    def _read_episode(self, episode):
        if self._pid != os.getpid():
            self.close()
            self._reset_cache()
        index = episode['episode_index']
        if index in self._episodes:
            self._episodes.move_to_end(index)
            return self._episodes[index]
        path = self.data_path(episode)
        file = pq.ParquetFile(path)
        columns = ['observation.state', 'action', 'timestamp', 'frame_index', 'episode_index', 'task_index']
        ep_column = file.schema.names.index('episode_index')
        parts = []
        for group in range(file.num_row_groups):
            stats = file.metadata.row_group(group).column(ep_column).statistics
            if stats and stats.has_min_max and not stats.min <= index <= stats.max:
                continue
            key = (str(path), group)
            table = self._row_groups.get(key)
            if table is None:
                table = file.read_row_group(group, columns=columns)
                if table.nbytes <= self.parquet_cache_bytes:
                    self._row_groups[key] = table
                    self._row_group_bytes += table.nbytes
                    while self._row_group_bytes > self.parquet_cache_bytes:
                        self._row_group_bytes -= self._row_groups.popitem(last=False)[1].nbytes
            else:
                self._row_groups.move_to_end(key)
            parts.append(table.filter(pc.equal(table['episode_index'], index)))
        if not parts:
            raise ValueError(f'Episode {index} absent from {path}')
        table = pa.concat_tables(parts).sort_by('frame_index')
        recorded = self.recorded_lengths[index]  # whole episodes: the goal is the last recorded frame
        if len(table) != recorded or not np.array_equal(table['frame_index'].to_numpy(), np.arange(recorded)):
            raise ValueError(f'Episode {index} length/frame indices disagree with metadata')
        if not np.all(table['task_index'].to_numpy() == episode['task_index']):
            raise ValueError(f'Episode {index} task_index disagrees with metadata')
        state = extract_state(_matrix(table['observation.state']), self.gripper_state)
        actions = _matrix(table['action'])
        if actions.shape != (len(table), 23) or not np.isfinite(actions).all():
            raise ValueError(f'Episode {index} requires finite 23-D actions')
        timestamps = table['timestamp'].to_numpy().astype(np.float64)
        if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
            raise ValueError(f'Invalid timestamps in episode {index}')
        value = {'state': state, 'action': actions, 'timestamp': timestamps}
        self._episodes[index] = value
        while len(self._episodes) > self.episode_cache_size:
            self._episodes.popitem(last=False)
        return value

    def __len__(self):
        return len(self.sampler)

    def __getitems__(self, indices):
        grouped = defaultdict(list)
        for offset, index in enumerate(indices):
            position, frames = self.sampler.locate(index)
            grouped[position].append((int(frames[0]), offset, index))
        result = [None] * len(indices)
        for entries in grouped.values():
            for _, offset, index in sorted(entries):
                result[offset] = self[index]
        return result

    def __getitem__(self, index):
        position, frames, valid = self.sampler.locate(index, with_valid=True)
        episode = self.episodes[position]
        data = self._read_episode(episode)
        obs_frames = frames[:self.obs_steps]
        current = int(frames[self.n_obs_steps - 1])  # the current frame: the last observation step
        actions = data['action'][frames]
        if self.action_representation == 'relative':
            actions = relative_actions(actions, data['state'][current])
        state = condition_state(data['state'][obs_frames],
                                np.full(len(obs_frames), episode['task_index']), self.task_map, self.task_onehot)
        obs = {'state': torch.from_numpy(state)}
        if self.goal_table is not None:
            for view, camera in enumerate(self.goal_views):
                # (3, S, S), no history axis; same dtype convention as the cameras (uint8 -> images_to_float on device)
                goal = np.ascontiguousarray(np.moveaxis(self.goal_table[position, view], -1, 0))
                obs[goal_key(camera)] = torch.from_numpy(goal) if self.image_dtype == 'uint8' else \
                    torch.from_numpy(goal.astype(np.float32) / 255.)
        if self.language_conditioning == 'clip_film':
            if self.language is None:
                raise RuntimeError('Call dataset.prepare_language() before loading clip_film samples')
            embedding = self.language['embeddings'][self._language_rows[episode['task_index']]]
            obs['lang_emb'] = embedding.expand(len(obs_frames), -1).clone()
        for camera in self.cameras:
            key = CAMERAS[camera][0]
            timestamps = data['timestamp'][obs_frames] + episode[f'videos/{key}/from_timestamp']
            end = episode[f'videos/{key}/to_timestamp']
            if np.any(timestamps >= end + self._video.tolerance):
                raise ValueError(f'Camera timestamps exceed episode boundary: {camera}')
            images = self._video.read(self.video_path(episode, camera), timestamps)
            images = np.moveaxis(images, -1, 1)
            if self.image_dtype == 'uint8':
                # 4x less loader IPC / pinning / host-to-device traffic; images_to_float on the
                # device reproduces exactly the float32 values of the branch below.
                obs[camera] = torch.from_numpy(np.ascontiguousarray(images))
            else:
                obs[camera] = torch.from_numpy(images.astype(np.float32) / 255.)
        sample = {'obs': obs['state'] if self.observation_mode == 'lowdim' else obs,
                  'action': torch.from_numpy(np.ascontiguousarray(actions, dtype=np.float32))}
        if self.return_meta:
            sample['action_valid'] = torch.from_numpy(valid.copy())
            sample['meta'] = {'task': int(episode['task_index']), 'episode': int(position), 'frame': current}
        return sample

    def iter_lowdim(self, batch_size=65536):
        """Stream each selected packed file once; no image decoding or frame index allocation."""
        by_file = defaultdict(list)
        for episode in self.episodes:
            by_file[self.data_path(episode)].append(episode)
        columns = ['episode_index', 'task_index', 'observation.state', 'action'] + (['frame_index'] if self.settle else [])
        for path, episodes in by_file.items():
            ids = np.array([episode['episode_index'] for episode in episodes], dtype=np.int64)  # ascending
            lengths = np.array([episode['length'] for episode in episodes], dtype=np.int64)
            for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=columns):
                table = pa.Table.from_batches([batch])
                table = table.filter(pc.is_in(table['episode_index'], value_set=pa.array(ids)))
                if self.settle:
                    kept = lengths[np.searchsorted(ids, table['episode_index'].to_numpy())]
                    table = table.filter(pa.array(table['frame_index'].to_numpy() < kept))
                if len(table):
                    yield {
                        'state': condition_state(extract_state(_matrix(table['observation.state']), self.gripper_state),
                                                 table['task_index'].to_numpy(), self.task_map, self.task_onehot),
                        'action': _matrix(table['action']),
                    }

    def fingerprint(self):
        paths = {self.data_path(row) for row in self.episodes}
        paths.update(self.video_path(row, camera) for row in self.episodes for camera in self.cameras)
        files = [(str(path.relative_to(self.root)), path.stat().st_size, path.stat().st_mtime_ns)
                 for path in sorted(paths)]
        payload = {'info': self.info, 'episodes': self.episodes, 'tasks': self.task_map, 'files': files}
        if self.gripper_state != 'fingers':  # `fingers` keeps the fingerprints of runs from before the option
            payload['gripper_state'] = self.gripper_state
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def iter_episode_lowdim(self):
        """Every selected episode's kept frames in frame order: (episode row, state (L, D) `extract_state` layout,
        action (L, 23)) -- the per-episode view the chunk statistics of `compute_action_stats` need."""
        by_file = defaultdict(list)
        for episode in self.episodes:
            by_file[self.data_path(episode)].append(episode)
        columns = ['episode_index', 'frame_index', 'observation.state', 'action']
        for path, episodes in by_file.items():
            ids = np.array([episode['episode_index'] for episode in episodes], dtype=np.int64)
            table = pq.read_table(path, columns=columns)
            table = table.filter(pc.is_in(table['episode_index'], value_set=pa.array(ids)))
            index = table['episode_index'].to_numpy()
            frame = table['frame_index'].to_numpy()
            order = np.lexsort((frame, index))
            index, frame = index[order], frame[order]
            state = extract_state(_matrix(table['observation.state'])[order], self.gripper_state)
            action = _matrix(table['action'])[order]
            bounds = np.searchsorted(index, ids, side='left'), np.searchsorted(index, ids, side='right')
            for episode, lo, hi in zip(episodes, *bounds):
                kept = episode['length']
                if hi - lo < kept or not np.array_equal(frame[lo:lo + kept], np.arange(kept)):
                    raise ValueError(f'Episode {episode["episode_index"]} frames in {path} disagree with its metadata')
                yield episode, state[lo:lo + kept], action[lo:lo + kept]

    def compute_action_stats(self, normalization, log=print):
        """Per-dimension statistics of the state (every kept frame) and of the actions as the trainer sees them -- every
        valid step k of every frame's horizon-long chunk, `action[t + k]` (k < horizon, t + k < kept length), taken
        relative to the state of frame t when the representation is relative (ACT's `compute_normalization_stats`
        with chunk_size = horizon; edge-padded steps are left out) -- for `zscore` (mean, sample std floored at
        ZSCORE_STD_FLOOR) or `quantile` (exact linear q01 / q99, dims narrower than QUANTILE_MIN_RANGE widened around
        their center as ACT does); min / max / mean / std always. Streams each packed file once; the chunk values of all
        23 action dimensions are held in memory (float32, 23 x valid steps)."""
        if normalization not in ('zscore', 'quantile'):
            raise ValueError(f'compute_action_stats serves zscore or quantile, not {normalization!r}')
        relative = self.action_representation == 'relative'
        horizon = self.horizon
        lengths = np.array([row['length'] for row in self.episodes], dtype=np.int64)
        total_frames = int(lengths.sum())
        total_steps = int(sum(min(horizon, length - t) for length in lengths.tolist() for t in range(length)))
        state_dim = proprio_dim(self.gripper_state)
        states = np.empty((state_dim, total_frames), dtype=np.float32)
        values = np.empty((23, total_steps), dtype=np.float32)
        f = s = 0
        started = time.perf_counter()
        for episode, state, action in self.iter_episode_lowdim():
            length = len(action)
            states[:, f:f + length] = state.T
            f += length
            reference = action_reference(state) if relative else None
            for k in range(min(horizon, length)):
                chunk = action[k:].astype(np.float64)
                if relative:
                    chunk = chunk - reference[:length - k]
                values[:, s:s + length - k] = chunk.T.astype(np.float32)
                s += length - k
        if f != total_frames or s != total_steps:
            raise ValueError(f'Statistics covered {f} frames / {s} chunk steps, metadata declares {total_frames} / {total_steps}')
        if total_frames < 2 or total_steps < 2:
            raise ValueError('At least two frames are required for normalization statistics')
        if not (np.isfinite(states).all() and np.isfinite(values).all()):
            raise ValueError('Non-finite state/action statistics')

        def moments(matrix):
            matrix64 = matrix.astype(np.float64)
            return {'min': matrix64.min(axis=1), 'max': matrix64.max(axis=1), 'mean': matrix64.mean(axis=1),
                    'std': np.maximum(matrix64.std(axis=1, ddof=1), ZSCORE_STD_FLOOR)}
        stats = {'method': normalization, 'action_representation': self.action_representation, 'horizon': horizon,
                 'count': total_frames, 'action_count': total_steps, 'state': moments(states), 'action': moments(values)}
        if normalization == 'quantile':
            stats.update(quantile_eps=QUANTILE_EPS, min_range=QUANTILE_MIN_RANGE, guarded={})
            for kind, matrix in (('state', states), ('action', values)):
                q01 = np.array([linear_quantile(matrix[dim], 0.01) for dim in range(len(matrix))])
                q99 = np.array([linear_quantile(matrix[dim], 0.99) for dim in range(len(matrix))])
                g01, g99, guarded = guard_quantiles(q01, q99, QUANTILE_MIN_RANGE)
                stats[kind].update(q01=g01, q99=g99, q01_raw=q01, q99_raw=q99)
                stats['guarded'][kind] = guarded
                if guarded:
                    log(f'WARNING: quantile statistics: {kind} dims {guarded} have q99 - q01 < {QUANTILE_MIN_RANGE}; '
                        f'normalized over {QUANTILE_MIN_RANGE} around their centers')
        if self.task_onehot:  # the one-hot task suffix of the state keeps its literal values (identity statistics)
            n = len(self.task_map)
            identity = {'min': 0.0, 'max': 1.0, 'mean': 0.0, 'std': 1.0, 'q01': 0.0, 'q99': 1.0, 'q01_raw': 0.0, 'q99_raw': 1.0}
            for key, value in list(stats['state'].items()):
                stats['state'][key] = np.concatenate([np.asarray(value, dtype=np.float64), np.full(n, identity[key])])
        if relative:
            stats['action_delta'] = json.loads(json.dumps(ACTION_DELTA))
        log(f'{normalization} statistics of {self.action_representation} actions: {total_frames} frames, {total_steps} '
            f'valid chunk steps in {time.perf_counter() - started:.1f}s')
        return stats

    def get_normalizer(self, normalization='limits', **kwargs):
        from diffusion_policy.b1k.normalization import fit_normalizer, normalizer_from_stats
        from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
        if normalization not in NORMALIZATIONS:
            raise ValueError(f'Unknown normalization {normalization!r}; expected one of {NORMALIZATIONS}')
        if normalization == 'limits':
            if self.action_representation != 'absolute':
                raise ValueError('Relative actions need --normalization zscore or quantile (their statistics are '
                                 'computed over chunk steps relative to the current state)')
            normalizer = fit_normalizer(self.iter_lowdim(), self.cameras, proprio_dim(self.gripper_state))
        else:
            normalizer = normalizer_from_stats(self.compute_action_stats(normalization), self.cameras,
                                               proprio_dim(self.gripper_state), self.task_onehot)
        if self.observation_mode == 'lowdim':
            result = LinearNormalizer()
            result['obs'], result['action'] = normalizer['state'], normalizer['action']
            return result
        if self.language_conditioning == 'clip_film':
            normalizer['lang_emb'] = SingleFieldLinearNormalizer.create_identity()
        for camera in self.goal_views:
            normalizer[goal_key(camera)] = normalizer[camera]  # the same image range mapping as its camera
        if self.imagenet_norm:
            for camera in self.cameras:
                normalizer[camera] = SingleFieldLinearNormalizer.create_identity()
        return normalizer


def linear_quantile(values, q):
    """numpy's default (linear) quantile of a 1-D array, from a partial sort of a copy (ACT's `linear_quantiles`)."""
    values = np.array(values, dtype=np.float64, copy=True)
    n = len(values)
    position = q * (n - 1)
    low, high = int(position), min(int(position) + 1, n - 1)
    values.partition(sorted({low, high}))
    return float(values[low] + (position - low) * (values[high] - values[low]))


def guard_quantiles(q01, q99, min_range):
    """(q01, q99, guarded dims): dims with q99 - q01 < min_range get [center - min_range / 2, center + min_range / 2]."""
    q01, q99 = np.asarray(q01, dtype=np.float64), np.asarray(q99, dtype=np.float64)
    narrow = q99 - q01 < min_range
    center = (q01 + q99) / 2
    return (np.where(narrow, center - min_range / 2, q01), np.where(narrow, center + min_range / 2, q99),
            np.flatnonzero(narrow).tolist())


def resolve_val_split(dataset_path, task_names, episode_split, val_fraction, seed):
    """Per-task train/val episode split (isg-val-split/v1) when `val_fraction > 0`, else None -- ACT's
    `b1k_training.resolve_val_split`, so DP, ACT and openpi hold out the same episodes.

    Each task's candidate episodes (from `episode_split` when given, else every episode of the task with local data)
    are partitioned deterministically with SeedSequence([seed, sha256(task_name)]) into round(N * val_fraction)
    validation episodes (at least 1, at most N - 1) and the remaining training episodes.
    """
    if not val_fraction:
        return None
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f'--val-fraction must be in [0, 1); got {val_fraction}')
    catalog = B1KLeRobotDataset(dataset_path, task_names, observation_mode='lowdim',
                                episodes=episode_split['episodes'] if episode_split else None)
    try:
        by_task = {}
        for task_id, task_name in sorted(catalog.task_map.items()):
            episodes = sorted(int(ep['episode_index']) for ep in catalog.episodes if int(ep['task_index']) == task_id)
            if len(episodes) < 2:
                raise ValueError(f'Task {task_name!r} has {len(episodes)} episode(s); at least 2 are required for '
                                 f'--val-fraction {val_fraction}')
            n_val = min(len(episodes) - 1, max(1, round(len(episodes) * val_fraction)))
            task_seed = int.from_bytes(hashlib.sha256(task_name.encode()).digest()[:8], 'little')
            rng = np.random.default_rng(np.random.SeedSequence([int(seed), task_seed]))
            perm = [episodes[i] for i in rng.permutation(len(episodes))]
            by_task[task_name] = {'train': sorted(perm[n_val:]), 'val': sorted(perm[:n_val])}
    finally:
        catalog.close()
    train_episodes = sorted(ep for entry in by_task.values() for ep in entry['train'])
    val_episodes = sorted(ep for entry in by_task.values() for ep in entry['val'])
    return {'format': VAL_SPLIT_FORMAT, 'val_fraction': float(val_fraction), 'seed': int(seed),
            'tasks': by_task, 'train_episodes': train_episodes, 'val_episodes': val_episodes}
