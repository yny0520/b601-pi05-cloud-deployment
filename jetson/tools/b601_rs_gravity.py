"""Bounded gravity feedforward using measured RS encoder angles, no device IO."""
import importlib.util
import math
from pathlib import Path
import sys
import time

import numpy as np

SOURCE = Path('/home/jetson/Desktop/b601-rs/B601-RS-lerobot/lerobot_robot_seeed_b601/gravity.py')
URDF = SOURCE.parent / 'assets/00-arm-rs_asm-v3.urdf'


def validate_gravity(value):
    if value is None:
        return None
    keys = {'enabled', 'autonomous_only', 'scales', 'max_torque_nm', 'slew_nm_s'}
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError('重力补偿配置字段无效')
    if type(value['enabled']) is not bool or type(value['autonomous_only']) is not bool:
        raise ValueError('重力补偿开关必须为布尔值')
    for key, maximum in [('scales', 1.0), ('max_torque_nm', 8.0)]:
        seq = value[key]
        if (not isinstance(seq, list) or len(seq) != 6 or
                any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= maximum for v in seq)):
            raise ValueError('重力补偿比例/力矩上限无效')
    slew = value['slew_nm_s']
    if type(slew) not in (int, float) or not math.isfinite(slew) or not 0 < slew <= 2:
        raise ValueError('重力补偿变化率无效')
    return dict(value)


class GravityFeedforward:
    def __init__(self, config, autonomous=False, clock=time.monotonic):
        self.config = validate_gravity(config)
        self.clock = clock
        self.last_time = None
        self.output = np.zeros(6)
        self.enabled = bool(config and config['enabled'] and
                            (autonomous or not config['autonomous_only']))
        if not self.enabled:
            return
        name = '_b601_rs_official_gravity'
        module = sys.modules.get(name)
        if module is None:
            spec = importlib.util.spec_from_file_location(name, SOURCE)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        self.compute = module.compute_generalized_gravity
        self.model = module.load_dynamics_model(str(URDF))
        arm = [(j.name, j.q_idx) for j in self.model.joints if j.q_idx >= 0]
        if arm[:6] != [(f'joint{i+1}', i) for i in range(6)]:
            raise ValueError('重力模型关节顺序不符，禁止使能')
        # The two URDF fingers are not encoder J7. Keep their model coordinates
        # at zero; only explicitly selected arm joints receive feedforward.
        self.scales = np.asarray(config['scales'], dtype=float)
        self.limits = np.asarray(config['max_torque_nm'], dtype=float)

    def reset(self):
        self.output[:] = 0
        self.last_time = self.clock()

    def update(self, actual_deg):
        if not self.enabled:
            return [0.0]*6
        actual = np.asarray(actual_deg, dtype=float)
        if actual.shape != (7,) or not np.isfinite(actual).all():
            raise ValueError('重力补偿缺少有限的七轴实测角度')
        q = np.zeros(self.model.nq)
        q[:6] = np.deg2rad(actual[:6])
        raw = np.asarray(self.compute(model=self.model, q=q), dtype=float)
        if raw.shape != (self.model.nq,) or not np.isfinite(raw).all():
            raise ValueError('重力补偿计算无效')
        target = np.clip(raw[:6]*self.scales, -self.limits, self.limits)
        now = self.clock()
        dt = 0 if self.last_time is None else min(.1, max(0, now-self.last_time))
        self.last_time = now
        step = self.config['slew_nm_s']*dt
        self.output += np.clip(target-self.output, -step, step)
        return self.output.tolist()
