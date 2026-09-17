"""Pixel-exact cache of resized RGB frames for random-access training.

Random access into the packed HEVC videos (GOP 8) costs ~13 ms of single-core decoding per
training sample (3 cameras x 2 frames, ~5.6 decoded frames per read), which caps a 30-core
loader at roughly 1.5k samples/s. Decoding every selected video once, sequentially, and storing
the *already resized* frames as uint8 makes a sample a handful of 27 KB memcpy's instead.

The cache stores exactly what `VideoReader.read` would return: the same PyAV decoder output,
converted with the same `rgb24` path and resized by `resize_rgb`. With no B-frames in the source
streams, decoding from the start and decoding from a keyframe yield identical pictures, which
`verify_frame_cache` checks against the native reader on random samples.

Layout, mirroring the dataset tree under the cache root (one entry per packed video):

    <cache_root>/<video_key>/chunk-XXX/file-YYY.frames.npy   uint8 [N, S, S, 3] RGB (resized)
    <cache_root>/<video_key>/chunk-XXX/file-YYY.pts.npy      float64 [N] presentation seconds
    <cache_root>/<video_key>/chunk-XXX/file-YYY.json         manifest (source size/mtime, S, N)
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
import os
from pathlib import Path
import time

import av
import numpy as np

from diffusion_policy.b1k.robot import CAMERAS, resize_rgb

FORMAT = 'diffusion_policy_b1k_frame_cache_v1'


def cache_stem(cache_root, dataset_root, video_path):
    relative = Path(video_path).resolve().relative_to(Path(dataset_root).resolve())
    return Path(cache_root) / relative.parent / relative.stem


def _paths(stem):
    stem = Path(stem)
    return (stem.with_name(stem.name + '.frames.npy'), stem.with_name(stem.name + '.pts.npy'),
            stem.with_name(stem.name + '.json'))


def _manifest_matches(manifest, source, image_size):
    stat = Path(source).stat()
    return (manifest.get('format') == FORMAT and manifest.get('image_size') == image_size
            and manifest.get('source_size') == stat.st_size
            and manifest.get('source_mtime_ns') == stat.st_mtime_ns)


def entry_is_valid(stem, source, image_size):
    frames_path, pts_path, manifest_path = _paths(stem)
    if not (frames_path.is_file() and pts_path.is_file() and manifest_path.is_file()):
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError):
        return False
    return _manifest_matches(manifest, source, image_size)


def build_entry(source, stem, image_size, dataset_root):
    """Decode one packed video sequentially into <stem>.frames.npy / .pts.npy / .json."""
    source, stem = Path(source), Path(stem)
    frames_path, pts_path, manifest_path = _paths(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with av.open(str(source), mode='r') as container:
        stream = container.streams.video[0]
        # Same decoder configuration as VideoReader; threading never changes decoded pixels.
        stream.thread_type = 'SLICE'
        stream.codec_context.thread_count = 1
        time_base = float(stream.time_base)
        expected = int(stream.frames or 0)
        frames = np.empty((expected, image_size, image_size, 3), dtype=np.uint8)
        overflow, pts, count = [], [], 0
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            image = resize_rgb(frame.to_ndarray(format='rgb24'), image_size)
            if count < expected:
                frames[count] = image
            else:
                overflow.append(image)
            pts.append(frame.pts * time_base)
            count += 1
    if overflow:
        frames = np.concatenate([frames, np.stack(overflow)])
    frames = frames[:count]
    pts = np.asarray(pts, dtype=np.float64)
    if count == 0:
        raise ValueError(f'No decodable frames in {source}')
    if np.any(np.diff(pts) <= 0):
        # The reader relies on presentation order == decode order (no B-frames).
        raise ValueError(f'Non-monotonic presentation timestamps in {source}; cannot cache safely')
    stat = source.stat()
    manifest = {
        'format': FORMAT, 'image_size': image_size, 'frames': count,
        'source': str(source.resolve().relative_to(Path(dataset_root).resolve())),
        'source_size': stat.st_size, 'source_mtime_ns': stat.st_mtime_ns,
        'resize': 'resize_rgb: cv2.INTER_LINEAR, aspect-preserving, centered zero padding',
        'decoder': f'PyAV {av.__version__}, libavcodec {av.library_versions.get("libavcodec")}',
        'build_seconds': round(time.monotonic() - started, 3),
    }
    tmp_frames, tmp_pts, tmp_manifest = (p.with_name('.' + p.name + '.tmp') for p in (frames_path, pts_path, manifest_path))
    for temporary, array in ((tmp_frames, frames), (tmp_pts, pts)):
        with temporary.open('wb') as stream:  # np.save(path) would append .npy to the temp name
            np.save(stream, array)
            stream.flush()
            os.fsync(stream.fileno())
    tmp_manifest.write_text(json.dumps(manifest, indent=2))
    for temporary, path in ((tmp_frames, frames_path), (tmp_pts, pts_path), (tmp_manifest, manifest_path)):
        os.replace(temporary, path)
    return manifest


class FrameCacheReader:
    """Drop-in for VideoReader.read over a built cache (lazily memory-mapped per process)."""

    def __init__(self, cache_root, dataset_root, image_size, tolerance=0.008):
        self.cache_root = Path(cache_root).resolve()
        self.dataset_root = Path(dataset_root).resolve()
        self.image_size = image_size
        self.tolerance = tolerance
        self.entries = {}

    def close(self):
        self.entries.clear()

    def _open(self, path):
        entry = self.entries.get(path)
        if entry is None:
            stem = cache_stem(self.cache_root, self.dataset_root, path)
            frames_path, pts_path, manifest_path = _paths(stem)
            if not manifest_path.is_file():
                raise FileNotFoundError(f'No frame cache entry for {path}; build it with scripts/b1k/build_frame_cache.py')
            manifest = json.loads(manifest_path.read_text())
            if not _manifest_matches(manifest, path, self.image_size):
                raise ValueError(f'Stale or mismatched frame cache entry {manifest_path}; rebuild it')
            frames = np.load(frames_path, mmap_mode='r')
            pts = np.load(pts_path)
            if frames.shape != (manifest['frames'], self.image_size, self.image_size, 3) or frames.dtype != np.uint8 \
                    or pts.shape != (manifest['frames'],):
                raise ValueError(f'Corrupt frame cache entry {stem}; rebuild it')
            entry = self.entries[path] = (frames, pts)
        return entry

    def validate(self, video_paths):
        for path in sorted(str(p) for p in video_paths):
            self._open(path)
        self.close()

    def read(self, path, timestamps):
        frames, pts = self._open(str(path))
        # Same key rounding and tolerance semantics as VideoReader: the first frame in
        # presentation order within `tolerance` of each requested time.
        timestamps = np.round(np.asarray(timestamps, dtype=np.float64), 6)
        positions = np.searchsorted(pts, timestamps - self.tolerance, side='left')
        clipped = np.minimum(positions, len(pts) - 1)
        found = (positions < len(pts)) & (np.abs(pts[clipped] - timestamps) <= self.tolerance)
        if not found.all():
            missing = sorted(set(timestamps[~found].tolist()))
            raise ValueError(f'Video timestamps not found within {self.tolerance}s in {path}: {missing}')
        return np.ascontiguousarray(frames[clipped])


def selected_videos(dataset):
    """Video files (sorted) that the dataset's selected episodes and cameras touch."""
    return sorted({dataset.video_path(row, camera) for row in dataset.episodes for camera in dataset.cameras})


def build_frame_cache(dataset, cache_root, workers=None, log=print):
    cache_root = Path(cache_root).resolve()
    if cache_root.is_relative_to(dataset.root):
        raise ValueError('The frame cache must not live inside the read-only dataset tree')
    videos = selected_videos(dataset)
    pending = [path for path in videos
               if not entry_is_valid(cache_stem(cache_root, dataset.root, path), path, dataset.image_size)]
    log(f'{len(videos)} selected videos, {len(pending)} to build, image_size={dataset.image_size}, cache={cache_root}')
    if pending:
        cache_root.mkdir(parents=True, exist_ok=True)
        # Longest files first so the pool tail is short.
        pending.sort(key=lambda path: path.stat().st_size, reverse=True)
        workers = max(1, min(workers or os.cpu_count() or 1, len(pending)))
        started = time.monotonic()
        with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn'),
                                 initializer=_worker_init) as pool:
            futures = {pool.submit(build_entry, path, cache_stem(cache_root, dataset.root, path),
                                   dataset.image_size, dataset.root): path for path in pending}
            for done, future in enumerate(as_completed(futures), 1):
                manifest = future.result()
                log(f'[{done}/{len(pending)}] {manifest["source"]}: {manifest["frames"]} frames '
                    f'in {manifest["build_seconds"]}s ({time.monotonic() - started:.0f}s elapsed)')
    return videos


def _worker_init():
    import cv2
    cv2.setNumThreads(1)
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass


def verify_frame_cache(dataset, cache_root, samples=256, seed=0, log=print):
    """Compare random native decodes against the cache byte for byte; raise on any mismatch."""
    from diffusion_policy.b1k.dataset import VideoReader
    native = VideoReader(dataset.image_size)
    cached = FrameCacheReader(cache_root, dataset.root, dataset.image_size)
    rng = np.random.default_rng(seed)
    compared = 0
    started = time.monotonic()
    try:
        for index in rng.integers(len(dataset), size=samples):
            position, frames = dataset.sampler.locate(int(index))
            episode = dataset.episodes[position]
            data = dataset._read_episode(episode)
            obs_frames = frames[:dataset.obs_steps]
            for camera in dataset.cameras:
                key = CAMERAS[camera][0]
                timestamps = data['timestamp'][obs_frames] + episode[f'videos/{key}/from_timestamp']
                path = dataset.video_path(episode, camera)
                expected = native.read(path, timestamps)
                actual = cached.read(path, timestamps)
                if expected.shape != actual.shape or not np.array_equal(expected, actual):
                    raise ValueError(f'Frame cache mismatch: episode {episode["episode_index"]} {camera} {timestamps}')
                compared += len(timestamps)
    finally:
        native.close()
        cached.close()
    log(f'verified {compared} cached frames from {samples} samples against native decoding in '
        f'{time.monotonic() - started:.1f}s: identical')
    return compared


def parser():
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    result.add_argument('--dataset-path', '--dataset-root', dest='dataset_path', required=True)
    result.add_argument('--cache-dir', required=True)
    result.add_argument('--task-names', nargs='+')
    result.add_argument('--cameras', choices=list(CAMERAS), nargs='+', default=list(CAMERAS))
    result.add_argument('--image-size', type=int, default=96)
    result.add_argument('--max-episodes', type=int)
    result.add_argument('--workers', type=int, default=max(1, min(28, os.cpu_count() or 1)))
    result.add_argument('--verify', type=int, default=256, help='Random samples to compare against native decoding; 0 skips')
    result.add_argument('--seed', type=int, default=0)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    from diffusion_policy.b1k.dataset import B1KLeRobotDataset
    dataset = B1KLeRobotDataset(args.dataset_path, args.task_names, cameras=tuple(args.cameras),
                                image_size=args.image_size, max_episodes=args.max_episodes)
    try:
        build_frame_cache(dataset, args.cache_dir, workers=args.workers)
        if args.verify:
            verify_frame_cache(dataset, args.cache_dir, samples=args.verify, seed=args.seed)
    finally:
        dataset.close()


if __name__ == '__main__':
    main()
