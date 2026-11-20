"""JSON-lines Pi0.5 SSH inference worker; no motor or camera device access."""
import json
import os
from pathlib import Path
import sys
import time

# This worker only packs two small RGB images for cloud inference.
# Keep local numerical/image workers from contending with CAN refresh.
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'

import cv2
cv2.setNumThreads(1)
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments/pi05_b601'))
from remote_inference_client import RemotePolicy
from b601_vla_model import load_model_spec, state_for_model, actions_for_controller


def main():
    policy = None
    for line in sys.stdin:
        request = json.loads(line)
        request_id = request.get('request_id')
        started = time.monotonic()
        spec = load_model_spec()
        if spec.get('policy_type') != 'pi05_remote' or request.get('model_id') != spec['id']:
            raise ValueError('云端模型选择已改变，拒绝推理')
        if policy is None:
            policy = RemotePolicy(spec['endpoint'])
        if policy.metadata.get('model_id') != spec['id']:
            raise ValueError('服务器模型不匹配')
        if request.get('warmup_only'):
            result = {'prepared': True, 'device': 'remote:A100', 'model_id': spec['id'],
                      'action_horizon': 16, 'server_metadata': policy.metadata}
        else:
            images = {}
            for key, path in [('camera1', request['top_path']), ('camera2', request['hand_path'])]:
                image = cv2.imread(path, cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError('本轮图像无法读取')
                images[key] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            observation = {'images': images,
                           'state': np.asarray(state_for_model(request['state_deg'], spec), dtype=np.float32),
                           'prompt': spec['task']}
            response, elapsed, size = policy.infer(observation)
            actions = actions_for_controller(response['actions'].tolist(), spec)
            result = {'action_deg': actions[0], 'action_chunk_deg': actions,
                      'device': 'remote:A100', 'model_id': spec['id'], 'model_reused': True,
                      'inference_seconds': response.get('server_timing', {}).get('infer_ms', 0)/1000,
                      'network_roundtrip_seconds': elapsed, 'payload_bytes': size,
                      'total_seconds': time.monotonic()-started}
        print('VLA_RESULT '+json.dumps({'request_id': request_id, **result}, allow_nan=False), flush=True)
    if policy is not None:
        policy.close()


if __name__ == '__main__':
    main()
