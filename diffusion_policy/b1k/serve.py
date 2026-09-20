"""BEHAVIOR MessagePack websocket adapter with connection-local temporal state."""

import argparse
import asyncio
from collections import deque
import functools
import http
import logging

import msgpack
import numpy as np
import torch
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

from diffusion_policy.b1k.model import ModelConfig, load_policy
from diffusion_policy.b1k.robot import CAMERAS, PROPRIO_KEY, condition_state, extract_state, resize_rgb

LOGGER = logging.getLogger(__name__)


def pack_array(obj):
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in 'VOc':
        raise ValueError(f'Unsupported dtype: {obj.dtype}')
    if isinstance(obj, np.ndarray):
        return {b'__ndarray__': True, b'data': obj.tobytes(), b'dtype': obj.dtype.str, b'shape': obj.shape}
    if isinstance(obj, np.generic):
        return {b'__npgeneric__': True, b'data': obj.item(), b'dtype': obj.dtype.str}
    raise TypeError(f'Cannot pack {type(obj)}')


def unpack_array(obj):
    if b'__ndarray__' in obj or b'__npgeneric__' in obj:
        dtype = np.dtype(obj[b'dtype'])
        if dtype.kind in 'VOc':
            raise ValueError(f'Unsupported dtype: {dtype}')
        if b'__ndarray__' in obj:
            return np.ndarray(buffer=obj[b'data'], dtype=dtype, shape=obj[b'shape'])
        return dtype.type(obj[b'data'])
    return obj


packb = functools.partial(msgpack.packb, default=pack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array, strict_map_key=False)


class B1KPolicySession:
    def __init__(self, policy, config, task_map, action_horizon=None, task_name=None, language=None):
        self.policy = policy
        self.model_config = ModelConfig(**config)
        self.config = self.model_config.to_dict()
        self.task_map = task_map
        self.language = None
        if self.model_config.language_conditioning == 'clip_film':
            from diffusion_policy.b1k.language import validate_language_cache
            self.language = validate_language_cache(
                language if language is not None else getattr(policy, 'language', None),
                task_map, self.model_config.prompt_source)
        self.language_rows = {index: row for row, index in enumerate(sorted(task_map))}
        self.action_horizon = config['n_action_steps'] if action_horizon is None else action_horizon
        if not 1 <= self.action_horizon <= config['n_action_steps']:
            raise ValueError('--action-horizon must be between 1 and checkpoint n_action_steps')
        self.default_task = next(iter(task_map)) if len(task_map) == 1 else None
        if task_name is not None:
            matches = [key for key, name in task_map.items() if name == task_name]
            if not matches:
                raise ValueError(f'Unknown --task-name {task_name!r}; known: {list(task_map.values())}')
            self.default_task = matches[0]
        self.reset()

    def reset(self):
        self.histories = []
        self.actions = []
        self.task_ids = []

    def _tasks(self, obs, batch_size):
        value = obs.get('task_id', self.default_task)
        if value is None:
            raise ValueError('Multitask checkpoint requires task_id or --task-name')
        ids = np.asarray(value).reshape(-1)
        if ids.size == 1:
            ids = np.repeat(ids, batch_size)
        if ids.size != batch_size or ids.dtype.kind not in 'iu':
            raise ValueError('task_id must be an integer scalar or one integer per batch element')
        if not np.isin(ids, list(self.task_map)).all():
            raise ValueError(f'Unknown task_id; known ids: {sorted(self.task_map)}')
        return ids.tolist()

    @torch.inference_mode()
    def act(self, obs):
        state = np.asarray(obs[PROPRIO_KEY])
        if state.ndim == 1:
            state = state[None]
        if state.ndim != 2 or not len(state):
            raise ValueError('proprio must have shape (61,) or (B,61)')
        batch_size = len(state)
        ids = self._tasks(obs, batch_size)
        state = condition_state(extract_state(state), np.asarray(ids), self.task_map, self.model_config.task_onehot)
        current = {'state': state}
        if self.language is not None:
            current['lang_emb'] = self.language['embeddings'][[self.language_rows[index] for index in ids]].numpy()
        for camera in (() if self.model_config.lowdim else self.config['cameras']):
            images = np.asarray(obs[CAMERAS[camera][1]])
            if images.ndim == 3:
                images = images[None]
            if images.ndim != 4 or len(images) != batch_size:
                raise ValueError(f'Camera {camera} batch does not match proprio')
            images = np.stack([resize_rgb(image, self.config['image_size']) for image in images])
            current[camera] = np.moveaxis(images, -1, 1).astype(np.float32) / 255.
        if len(self.histories) != batch_size:
            self.reset()
            self.histories = [deque(maxlen=self.config['n_obs_steps']) for _ in ids]
            self.actions = [deque() for _ in ids]
            self.task_ids = [None for _ in ids]
        for slot, task in enumerate(ids):
            if self.task_ids[slot] != task:
                self.histories[slot].clear()
                self.actions[slot].clear()
            self.task_ids[slot] = task
            frame = {key: value[slot].copy() for key, value in current.items()}
            if not self.histories[slot]:
                for _ in range(self.config['n_obs_steps'] - 1):
                    self.histories[slot].append(frame)
            self.histories[slot].append(frame)
        needs_plan = [slot for slot in range(batch_size) if not self.actions[slot]]
        if needs_plan:
            inputs = {key: torch.from_numpy(np.stack([
                np.stack([frame[key] for frame in self.histories[slot]]) for slot in needs_plan
            ])).to(self.policy.device) for key in current}
            if self.model_config.lowdim:
                inputs = {'obs': inputs['state']}
            predicted = self.policy.predict_action(inputs)['action'].detach().cpu().numpy()
            if predicted.shape != (len(needs_plan), self.config['n_action_steps'], 23):
                raise ValueError(f'Unexpected policy action shape {predicted.shape}')
            if not np.isfinite(predicted).all():
                raise ValueError('Policy returned non-finite action')
            for slot, plan in zip(needs_plan, predicted):
                self.actions[slot].extend(plan[:self.action_horizon].astype(np.float32))
        return np.stack([queue.popleft() for queue in self.actions]).astype(np.float32)


def health_check(connection, request):
    if request.path == '/healthz':
        return connection.respond(http.HTTPStatus.OK, 'OK\n')
    return None


class WebsocketPolicyServer:
    def __init__(self, policy, checkpoint, host='0.0.0.0', port=8000,
                 action_horizon=None, task_name=None):
        self.policy, self.checkpoint = policy, checkpoint
        self.host, self.port = host, port
        self.action_horizon, self.task_name = action_horizon, task_name
        self.new_session()

    def new_session(self):
        return B1KPolicySession(self.policy, self.checkpoint['config'], self.checkpoint['task_map'],
                                self.action_horizon, self.task_name, self.checkpoint.get('language'))

    async def handler(self, websocket):
        session = self.new_session()
        await websocket.send(packb({
            'policy': type(self.policy).__name__, 'variant': session.model_config.variant, 'action_dim': 23,
            'action_horizon': session.action_horizon,
            'n_obs_steps': session.config['n_obs_steps'],
            'language_conditioning': session.model_config.language_conditioning,
            'prompt_source': session.model_config.prompt_source,
            'task_map': {str(key): value for key, value in session.task_map.items()},
        }))
        try:
            async for payload in websocket:
                obs = unpackb(payload)
                if not isinstance(obs, dict):
                    raise ValueError('Observation must be a map')
                if 'reset' in obs:
                    session.reset()
                    continue
                # The scheduler mutates timesteps; serialize access to the shared model.
                async with self.inference_lock:
                    action = await asyncio.to_thread(session.act, obs)
                await websocket.send(packb({'action': action}))
        except ConnectionClosed:
            pass
        except Exception as error:
            LOGGER.exception('B1K connection failed')
            await websocket.close(code=1011, reason=str(error).encode('utf-8')[:120].decode('utf-8', errors='ignore'))

    async def run(self):
        self.inference_lock = asyncio.Lock()
        async with serve(self.handler, self.host, self.port, compression=None,
                         max_size=128 * 1024 ** 2, process_request=health_check) as server:
            LOGGER.info('B1K websocket listening on %s:%d', self.host, self.port)
            await server.serve_forever()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--action-horizon', type=int)
    parser.add_argument('--task-name')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--cpu-threads', type=int, default=4)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    torch.set_num_threads(args.cpu_threads)
    policy, checkpoint = load_policy(args.model_path, args.device)
    server = WebsocketPolicyServer(policy, checkpoint, args.host, args.port,
                                   args.action_horizon, args.task_name)
    asyncio.run(server.run())


if __name__ == '__main__':
    main()
