"""Read-only, bounded-memory LeRobot v3 parquet/video sequences."""

from collections import OrderedDict, defaultdict
import hashlib
import json
import os
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch

from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.b1k.robot import CAMERAS, condition_state, extract_state, resize_rgb


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

    def locate(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode = int(np.searchsorted(self.ends, index, side='right'))
        previous = int(self.ends[episode - 1]) if episode else 0
        start = index - previous - self.pad_before
        frames = np.clip(np.arange(start, start + self.horizon), 0, self.lengths[episode] - 1)
        return episode, frames


def _matrix(column):
    column = column.combine_chunks()
    return column.flatten().to_numpy(zero_copy_only=False).reshape(len(column), -1).astype(np.float32)


class VideoReader:
    def __init__(self, image_size, tolerance=0.008, max_open=3, cache_frames=64):
        self.image_size = image_size
        self.tolerance = tolerance
        self.max_open = max_open
        self.cache_frames = cache_frames
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
            target = max(0, missing[0] - self.tolerance)
            container.seek(int(target / float(stream.time_base)), stream=stream, backward=True)
            pending = set(missing)
            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                timestamp = float(frame.pts * stream.time_base)
                matches = [t for t in pending if abs(timestamp - t) <= self.tolerance]
                if matches:
                    image = resize_rgb(frame.to_ndarray(format='rgb24'), self.image_size)
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


class B1KLeRobotDataset(BaseImageDataset):
    def __init__(self, dataset_path, task_names=None, horizon=16, n_obs_steps=2,
                 n_action_steps=8, cameras=tuple(CAMERAS), image_size=96,
                 pad_before=None, pad_after=None, episode_cache_size=8,
                 parquet_cache_mb=256, max_episodes=None, observation_mode='image',
                 obs_steps=None, imagenet_norm=False, language_conditioning='none',
                 prompt_source='task_name'):
        self.root = Path(dataset_path).resolve()
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
        self.language_conditioning = language_conditioning
        self.prompt_source = prompt_source
        self.language = None
        self.observation_mode = observation_mode
        self.imagenet_norm = imagenet_norm
        self.obs_steps = n_obs_steps if obs_steps is None else obs_steps
        if not n_obs_steps <= self.obs_steps <= horizon:
            raise ValueError('obs_steps must be between n_obs_steps and horizon')
        self.cameras = () if observation_mode == 'lowdim' else tuple(cameras)
        if ((observation_mode == 'image' and not self.cameras) or
                len(set(self.cameras)) != len(self.cameras) or set(self.cameras) - set(CAMERAS)):
            raise ValueError(f'cameras must be unique names from {list(CAMERAS)}')
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
            table = file.read(columns=columns)
            if 'task_index' in schema:
                table = table.filter(pc.is_in(table['task_index'], value_set=pa.array(sorted(selected))))
            for row in table.to_pylist():
                if 'task_index' not in row:
                    ids = [i for i, name in all_tasks.items() if name in (row.get('tasks') or [])]
                    if len(ids) != 1:
                        raise ValueError(f'Episode {row["episode_index"]} needs exactly one categorical task')
                    row['task_index'] = ids[0]
                if row['task_index'] not in selected:
                    continue
                data_path = self.data_path(row)
                if not data_path.is_file():
                    if names:
                        raise FileNotFoundError(f'Selected episode {row["episode_index"]}: missing {data_path}')
                    continue
                if row['length'] <= 0 or row['dataset_to_index'] - row['dataset_from_index'] != row['length']:
                    raise ValueError(f'Invalid episode bounds: {row["episode_index"]}')
                for camera in self.cameras:
                    if not self.video_path(row, camera).is_file():
                        raise FileNotFoundError(f'Selected camera missing: {self.video_path(row, camera)}')
                self.episodes.append(row)
        self.episodes.sort(key=lambda row: row['episode_index'])
        ids = [row['episode_index'] for row in self.episodes]
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate episode_index in metadata')
        present = {row['task_index'] for row in self.episodes}
        if names and selected - present:
            raise ValueError(f'No episodes of task(s) {[all_tasks[i] for i in sorted(selected - present)]} on disk')
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
        self.sampler = EpisodeSequenceIndex(
            [row['length'] for row in self.episodes], horizon,
            n_obs_steps - 1 if pad_before is None else pad_before,
            n_action_steps - 1 if pad_after is None else pad_after)
        if not len(self.sampler):
            raise ValueError('No sequences for this horizon and padding')
        self._reset_cache()

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
        self._video = VideoReader(self.image_size)

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
        if len(table) != episode['length'] or not np.array_equal(
                table['frame_index'].to_numpy(), np.arange(episode['length'])):
            raise ValueError(f'Episode {index} length/frame indices disagree with metadata')
        if not np.all(table['task_index'].to_numpy() == episode['task_index']):
            raise ValueError(f'Episode {index} task_index disagrees with metadata')
        state = extract_state(_matrix(table['observation.state']))
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
        position, frames = self.sampler.locate(index)
        episode = self.episodes[position]
        data = self._read_episode(episode)
        obs_frames = frames[:self.obs_steps]
        state = condition_state(data['state'][obs_frames],
                                np.full(len(obs_frames), episode['task_index']), self.task_map)
        obs = {'state': torch.from_numpy(state)}
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
            obs[camera] = torch.from_numpy(np.moveaxis(images, -1, 1).astype(np.float32) / 255.)
        return {'obs': obs['state'] if self.observation_mode == 'lowdim' else obs,
                'action': torch.from_numpy(data['action'][frames].copy())}

    def iter_lowdim(self, batch_size=65536):
        """Stream each selected packed file once; no image decoding or frame index allocation."""
        by_file = defaultdict(list)
        for episode in self.episodes:
            by_file[self.data_path(episode)].append(episode['episode_index'])
        for path, ids in by_file.items():
            for batch in pq.ParquetFile(path).iter_batches(
                    batch_size=batch_size,
                    columns=['episode_index', 'task_index', 'observation.state', 'action']):
                table = pa.Table.from_batches([batch])
                table = table.filter(pc.is_in(table['episode_index'], value_set=pa.array(ids)))
                if len(table):
                    yield {
                        'state': condition_state(extract_state(_matrix(table['observation.state'])),
                                                 table['task_index'].to_numpy(), self.task_map),
                        'action': _matrix(table['action']),
                    }

    def fingerprint(self):
        paths = {self.data_path(row) for row in self.episodes}
        paths.update(self.video_path(row, camera) for row in self.episodes for camera in self.cameras)
        files = [(str(path.relative_to(self.root)), path.stat().st_size, path.stat().st_mtime_ns)
                 for path in sorted(paths)]
        payload = {'info': self.info, 'episodes': self.episodes, 'tasks': self.task_map, 'files': files}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def get_normalizer(self, **kwargs):
        from diffusion_policy.b1k.normalization import fit_normalizer
        from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
        normalizer = fit_normalizer(self.iter_lowdim(), self.cameras)
        if self.observation_mode == 'lowdim':
            result = LinearNormalizer()
            result['obs'], result['action'] = normalizer['state'], normalizer['action']
            return result
        if self.language_conditioning == 'clip_film':
            normalizer['lang_emb'] = SingleFieldLinearNormalizer.create_identity()
        if self.imagenet_norm:
            for camera in self.cameras:
                normalizer[camera] = SingleFieldLinearNormalizer.create_identity()
        return normalizer
