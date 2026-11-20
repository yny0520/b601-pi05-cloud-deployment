"""Offline Pi0.5 client. No CAN, camera acquisition, or motor commands."""
import argparse
import json
import time
from pathlib import Path

import msgpack
import numpy as np
import websocket


def pack_array(value):
    if isinstance(value, np.ndarray):
        if value.dtype.kind in 'VOc':
            raise ValueError('Unsupported array dtype')
        return {b'__ndarray__': True, b'data': value.tobytes(),
                b'dtype': value.dtype.str, b'shape': value.shape}
    if isinstance(value, np.generic):
        return value.item()
    return value


def unpack_array(value):
    if b'__ndarray__' in value:
        return np.ndarray(buffer=value[b'data'], dtype=np.dtype(value[b'dtype']),
                          shape=value[b'shape'])
    if b'__npgeneric__' in value:
        return np.dtype(value[b'dtype']).type(value[b'data'])
    return value


class RemotePolicy:
    def __init__(self, url='ws://127.0.0.1:18015', timeout=3):
        self.socket = websocket.create_connection(url, timeout=timeout,
                                                 http_no_proxy=['127.0.0.1', 'localhost'])
        self.metadata = self._decode(self.socket.recv())
        if self.metadata.get('model_id') != 'pi05_b601_lora_6000':
            self.close()
            raise ValueError('Unexpected server model')

    @staticmethod
    def _decode(payload):
        if isinstance(payload, str):
            raise RuntimeError(payload)
        return msgpack.unpackb(payload, object_hook=unpack_array)

    def infer(self, observation, max_age_seconds=3):
        started = time.monotonic()
        payload = msgpack.packb(observation, default=pack_array)
        try:
            self.socket.send_binary(payload)
            result = self._decode(self.socket.recv())
            actions = np.asarray(result['actions'], dtype=np.float32)
            elapsed = time.monotonic() - started
            if result.get('model_id') != self.metadata['model_id']:
                raise ValueError('Response model mismatch')
            if result.get('action_units') != 'radians':
                raise ValueError('Expected absolute radians')
            if actions.shape != (16, 7) or not np.isfinite(actions).all():
                raise ValueError('Invalid action sequence')
            if elapsed > max_age_seconds:
                raise TimeoutError('Response too old; discard actions')
            return result, elapsed, len(payload)
        except BaseException:
            self.close()  # Prevent consuming a late response as a later request.
            raise

    def close(self):
        self.socket.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('observation_npz', type=Path)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    with np.load(args.observation_npz, allow_pickle=False) as sample:
        observation = {'images': {k: sample[k] for k in ['camera1', 'camera2']},
                       'state': sample['state'], 'prompt': str(sample['prompt'])}
    client = RemotePolicy()
    results = []
    try:
        for _ in range(args.rounds):
            result, elapsed, size = client.infer(observation)
            results.append({'total_seconds': elapsed, 'payload_bytes': size,
                            'server_timing': result.get('server_timing'),
                            'action_shape': list(result['actions'].shape)})
            print(json.dumps(results[-1]), flush=True)
    finally:
        client.close()
    report = {'offline_only': True, 'robot_connected': False,
              'metadata': client.metadata, 'rounds': results}
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
