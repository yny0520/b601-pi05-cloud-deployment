"""Local deployment contract shared by the worker, web UI and controller.

Selecting a model never enables motors. Change selection only with control stopped.
All motor-facing actions remain absolute degrees; model units are converted once.
"""
import json
import math
from pathlib import Path

SELECTION = Path('/home/jetson/Desktop/b601-rs/calibration/vla_model_config.json')
LEGACY_PACKAGE = Path('/home/jetson/Desktop/b601-rs/datasets/NX部署包')
LEGACY = dict(id='insertion_legacy', label='旧网线插接模型', collection_profile='insertion',
              model_dir=str(LEGACY_PACKAGE/'best_model'), backbone=str(LEGACY_PACKAGE/'vlm_assets'),
              units='degrees', task='网线插接', fps=20,
              motion_profile='/home/jetson/Desktop/b601-rs/calibration/vla_motion_config.json')


def load_model_spec(path=SELECTION):
    spec = json.loads(Path(path).read_text()) if Path(path).exists() else dict(LEGACY)
    if (not isinstance(spec, dict) or not set(LEGACY) <= set(spec)
            or spec['units'] not in ('degrees', 'radians')
            or spec.get('policy_type', 'smolvla') not in ('smolvla', 'act', 'pi05_remote')
            or spec['collection_profile'] not in ('pick_place', 'insertion')
            or type(spec['fps']) not in (int, float) or not 1 <= spec['fps'] <= 30
            or any(not isinstance(spec[k], str) or not spec[k] for k in LEGACY if k != 'fps')
            or any(not Path(spec[k]).is_absolute() for k in ('model_dir', 'backbone', 'motion_profile'))):
        raise ValueError('模型部署配置无效；未启用运动')
    if spec.get('policy_type') == 'pi05_remote' and (
            spec.get('endpoint') != 'ws://127.0.0.1:18015' or spec.get('action_horizon') != 16
            or spec['units'] != 'radians' or spec['fps'] != 14):
        raise ValueError('Pi0.5云端模型接口或SSH端点无效')
    return spec


def check_collection(spec, profile):
    if profile != spec['collection_profile']:
        raise ValueError('当前加载的是'+spec['label']+'，请切换到对应任务后测试')


def validate_policy_config(spec, config):
    kind = spec.get('policy_type', 'smolvla')
    if config.get('type') != kind:
        raise ValueError('模型类型与部署配置不一致')
    if kind == 'act':
        expected = {'observation.state': [7],
                    'observation.images.camera1': [3, 480, 640],
                    'observation.images.camera2': [3, 360, 640]}
        features = config.get('input_features', {})
        if (spec['units'] != 'radians' or spec['fps'] != 14 or
                {k: v.get('shape') for k, v in features.items()} != expected or
                config.get('output_features', {}).get('action', {}).get('shape') != [7] or
                config.get('n_obs_steps') != 1 or config.get('chunk_size') != 50 or
                config.get('n_action_steps') != 5 or config.get('temporal_ensemble_coeff') is not None):
            raise ValueError('ACT输入、单位或时序与已核验部署约定不符')


def state_for_model(state_deg, spec):
    if (not isinstance(state_deg, (list, tuple)) or len(state_deg) != 7
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in state_deg)):
        raise ValueError('输入须为J1–J7七维有限角度')
    return [math.radians(v) for v in state_deg] if spec['units'] == 'radians' else list(state_deg)


def actions_for_controller(actions, spec):
    if (not isinstance(actions, list) or not 1 <= len(actions) <= 50
            or any(not isinstance(a, list) or len(a) != 7 or
                   any(type(v) not in (int, float) or not math.isfinite(v) for v in a) for a in actions)):
        raise ValueError('模型输出须为1–50个J1–J7七维有限动作')
    return ([[math.degrees(v) for v in a] for a in actions]
            if spec['units'] == 'radians' else actions)
