"""Read-only mmap tables beside (never inside) the frame cache and source dataset."""
import fcntl
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from .b1k_cache_v2 import atomic_json
from .robot import CAMERAS, extract_state, proprio_dim


def open_sidecar(dataset):
    from .dataset import _matrix
    d = dataset
    directory = Path(str(d.frame_cache) + '-dp-tables') / d.fingerprint()
    directory.mkdir(parents=True, exist_ok=True)
    if getattr(d, '_worker_fingerprint', None) is not None:
        from .b1k_selection import MappedLookup
        maps = {name:np.load(directory/(name+'.npy'),mmap_mode='r') for name in ('state','action','timestamp','starts')}
        maps['positions'] = MappedLookup(directory/'episode_ids.npy', directory/'episode_positions.npy')
        return maps
    lengths = np.array([d.recorded_lengths[e['episode_index']] for e in d.episodes], dtype=np.int64)
    starts = np.r_[0, np.cumsum(lengths)]
    count = int(starts[-1])
    shapes = {'state':(count,proprio_dim(d.gripper_state)), 'action':(count,23),
              'timestamp':(count,), 'starts':starts.shape}
    dtypes = {'state':np.float32,'action':np.float32,'timestamp':np.float64,'starts':np.int64}
    with (directory/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        complete = directory/'complete.json'
        if not complete.is_file():
            maps = {name:np.lib.format.open_memmap(directory/(name+'.npy'), mode='w+', dtype=dtypes[name], shape=shape)
                    for name,shape in shapes.items()}
            maps['starts'][:] = starts
            by_file = {}
            for position, ep in enumerate(d.episodes):
                by_file.setdefault(d.data_path(ep), []).append((position,ep))
            for path, episodes in by_file.items():
                table = pq.read_table(path, columns=['episode_index','frame_index','timestamp','observation.state','action'])
                ids, frames = table['episode_index'].to_numpy(), table['frame_index'].to_numpy()
                order = np.lexsort((frames,ids)); sorted_ids = ids[order]
                state = extract_state(_matrix(table['observation.state']), d.gripper_state)
                action = _matrix(table['action']); timestamps = table['timestamp'].to_numpy()
                for position, ep in episodes:
                    lo,hi = np.searchsorted(sorted_ids, ep['episode_index'], side='left'), np.searchsorted(sorted_ids, ep['episode_index'], side='right')
                    rows = order[lo:hi]; length = int(lengths[position])
                    if len(rows) != length or not np.array_equal(frames[rows], np.arange(length)):
                        raise ValueError('sidecar episode bounds disagree with metadata')
                    if not np.isfinite(state[rows]).all() or not np.isfinite(action[rows]).all() or not np.isfinite(timestamps[rows]).all() or np.any(np.diff(timestamps[rows]) <= 0):
                        raise ValueError('invalid sidecar state/action/timestamp values')
                    target = slice(starts[position],starts[position+1])
                    maps['state'][target],maps['action'][target],maps['timestamp'][target] = state[rows],action[rows],timestamps[rows]
            for mmap in maps.values(): mmap.flush()
            del maps
            atomic_json(complete, {'version':2,'fingerprint':d.fingerprint(),'frames':count})
        if not (directory/'episode_ids.npy').is_file():
            np.save(directory/'episode_ids.npy',np.asarray([ep['episode_index'] for ep in d.episodes],dtype=np.int64))
            np.save(directory/'episode_positions.npy',np.arange(len(d.episodes),dtype=np.int64))
        manifest = json.loads(complete.read_text())
        if manifest.get('fingerprint') != d.fingerprint() or manifest.get('frames') != count:
            raise ValueError('sidecar manifest mismatch')
        maps = {name:np.load(directory/(name+'.npy'),mmap_mode='r') for name in shapes}
        if any(maps[name].shape != shape or maps[name].dtype != dtypes[name] for name,shape in shapes.items()):
            raise ValueError('sidecar array shape/dtype mismatch')
    from .b1k_selection import MappedLookup
    maps['positions'] = MappedLookup(directory/'episode_ids.npy', directory/'episode_positions.npy')
    return maps


class CachedGoals:
    def __init__(self, dataset):
        self.dataset = dataset
        self.shape = (len(dataset.episodes),len(dataset.goal_views),dataset.image_size,dataset.image_size,3)

    def __getitem__(self, key):
        position, view = key if isinstance(key,tuple) else (key,None)
        d=self.dataset
        if not np.isscalar(position):
            return np.stack([self[int(i),view] for i in np.arange(len(d.episodes))[position]])
        ep=d.episodes[int(position)]
        timestamp=float(d._read_episode(ep)['timestamp'][-1])
        def read(i):
            camera=d.goal_views[i]; video_key=CAMERAS[camera][0]
            return d._video.read(d.video_path(ep,camera), [timestamp+ep[f'videos/{video_key}/from_timestamp']])[0]
        return np.stack([read(i) for i in range(len(d.goal_views))]) if view is None else read(int(view))
