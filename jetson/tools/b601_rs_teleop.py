#!/usr/bin/env python3
"""Star Arm FashionStar UART -> B601-RS CAN, preview by default.

No hardware origins are changed. Each R captures a session-relative arm alignment;
an optional gripper endpoint mapping is independent of the session alignment.
Only --live followed by R may configure MIT mode and enable the follower.
"""
from __future__ import annotations

import argparse
import builtins
from collections import deque
from contextlib import contextmanager, nullcontext
from concurrent.futures import ThreadPoolExecutor
import fcntl
import errno
import json
import math
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import termios
import time
import traceback
import uuid
import tty
from datetime import datetime
from zoneinfo import ZoneInfo
# This process handles only small control matrices. Keep numerical worker pools
# from competing with its CAN refresh loop; cloud/model workers are separate.
for _thread_env in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_thread_env] = '1'
from b601_telemetry import TelemetryPublisher
from b601_rs_feedback import BatchFeedback
from b601_rs_gravity import GravityFeedforward, validate_gravity
from b601_control_io import runtime_path, atomic_runtime_json, BufferedJournal
from b601_control_timing import ControlTiming
from b601_vla_motion import load_motion_profile, validate_profile, VlaPlan, validate_plan_replacement

from b601_rs_calibrate import (MOTOR_IDS, MOTOR_MODELS, JOINT_NAMES, HOST_ID,
                               RESPONDER_IDS, open_motors, close_all,
                               assert_channel_is_explicitly_safe)

ROOT = Path('/home/jetson/Desktop/b601-rs')
DEFAULT_CONFIG = ROOT / 'calibration/star_arm_teleop_config.json'
TERMINAL_LOST = False


def print(*args, **kwargs):
    # Losing a PTY/output reader must not unwind an enabled CAN controller.
    try:
        builtins.print(*args, **kwargs)
    except OSError as exc:
        if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
            return  # A full diagnostic PTY must not block motor refresh.
        if exc.errno not in (errno.EIO, errno.EPIPE, errno.EBADF):
            raise
        terminal_lost()


def terminal_lost(*_):
    # Closing a terminal is not an explicit request to remove motor torque.
    global TERMINAL_LOST
    TERMINAL_LOST = True


def stamp():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def finite(x):
    x = float(x)
    if not math.isfinite(x):
        raise RuntimeError('角度或参数不是有限数值')
    return x


def bounded_vla_targets(actual, predicted, expected, bounds, max_step_deg=0.5,
                        previous_command=None, session_origin=None, profile=None):
    """Validate one human-approved tiny VLA step; J7 remains under the gripper controller."""
    if not all(isinstance(values, (list, tuple)) and len(values) == 7
               for values in (actual, predicted, expected, bounds)):
        raise ValueError('VLA单步必须包含七关节状态、预测和软限位')
    result = list(actual)
    if not math.isfinite(max_step_deg) or not 0 < max_step_deg <= 0.5:
        raise ValueError('单步限幅必须在0..0.5°内')
    for vector in (previous_command, session_origin):
        if vector is not None and (len(vector) != 7 or not all(math.isfinite(float(v)) for v in vector)):
            raise ValueError('VLA参考姿态无效')
    settings = validate_profile(profile) if profile is not None else None
    for i in range(7):
        if len(bounds[i]) != 2 or bounds[i][0] > bounds[i][1]:
            raise ValueError('VLA软限位无效')
        values = (actual[i], predicted[i], expected[i], *bounds[i])
        if not all(math.isfinite(float(v)) for v in values):
            raise ValueError(f'J{i+1}的VLA单步数据含非有限数值')
        gripper_active = i == 6 and settings is not None and settings['gripper_enabled']
        drift_limit = 2.0 if gripper_active else .5
        if abs(float(actual[i])-float(expected[i])) > drift_limit:
            raise ValueError(f'推理后J{i+1}位置变化超过{drift_limit:g}°，请重新推理')
        if i < 6 or gripper_active:
            # Keep feedback noise away from the independent tracking
            # interlock. At exactly that boundary a stationary joint could pass
            # settling and be rejected on the very next encoder sample.
            step = settings['command_step_deg'] if settings else max_step_deg * 0.9
            if gripper_active:
                step = settings['gripper_step_deg']
            lo, hi = float(actual[i])-step, float(actual[i])+step
            if previous_command is not None:
                abort_error = settings['abort_error_deg'] if settings else 1.0
                if gripper_active:
                    abort_error = settings['gripper_abort_error_deg']
                if abs(previous_command[i]-actual[i]) > abort_error:
                    raise ValueError(f'J{i+1}目标与实测相差超过{abort_error:g}°，停止继续推进，请检查负载/姿态')
                # Advance from the last COMMAND so steady load/friction error
                # does not reset every new goal to the same measured angle.
                # Independently bound lead, command increments and session travel.
                lead = settings['lead_deg'] if settings else .9
                if gripper_active:
                    lead = settings['gripper_lead_deg']
                lo = max(float(actual[i])-lead, previous_command[i]-step)
                hi = min(float(actual[i])+lead, previous_command[i]+step)
            if session_origin is not None:
                span = settings['session_span_deg'] if settings else 5
                if gripper_active:
                    span = settings['gripper_span_deg']
                lo, hi = max(lo, session_origin[i]-span), min(hi, session_origin[i]+span)
            # A prediction outside the physical range must saturate at the
            # boundary, rather than reject every near-zero starting pose.
            lo, hi = max(lo, float(bounds[i][0])), min(hi, float(bounds[i][1]))
            if settings and settings['task_bounds_deg'] is not None:
                lo=max(lo,settings['task_bounds_deg'][i][0])
                hi=min(hi,settings['task_bounds_deg'][i][1])
            if lo > hi:
                raise ValueError(f'J{i+1}没有满足小步、软限位和累计范围的可执行目标')
            goal = clamp(float(predicted[i]), lo, hi)
            if not float(bounds[i][0]) <= goal <= float(bounds[i][1]):
                raise ValueError(f'J{i+1}单步目标超出软限位')
            result[i] = goal
        elif previous_command is not None:
            result[i] = previous_command[i]
    return result


def atomic_json(path, value):
    atomic_runtime_json(path, value)


def fixed_io_action(action):
    helper = Path('/usr/local/sbin/b601-io')
    if not helper.is_file():
        return False
    info = helper.stat()
    if info.st_uid != 0 or info.st_mode & 0o022:
        raise RuntimeError('B601 I/O助手权限不正确')
    result = subprocess.run(['sudo','-n',str(helper),action],text=True,capture_output=True)
    if result.returncode:
        raise RuntimeError('I/O准备失败：'+(result.stderr.strip() or result.stdout.strip()))
    return True


def peak_usb_channels():
    status = Path('/proc/pcan')
    if not status.exists():
        raise RuntimeError('找不到PEAK /proc/pcan状态；请检查XCAN-USB驱动')
    return sorted({row[2] for line in status.read_text().splitlines()
                   if len(row := line.split()) >= 3 and row[1] == 'usb'
                   and row[2].startswith('can') and row[2][3:].isdigit()})


def resolve_can_channel(channel):
    if channel != 'auto':
        return channel
    channels = peak_usb_channels()
    if len(channels) != 1:
        raise RuntimeError(f'需要唯一的PEAK USB-CAN，当前识别到{channels}；请检查连接或显式指定--channel')
    return channels[0]


def ensure_peak_can_ready(channel):
    """Bring up only a PEAK USB SocketCAN interface at the robot bitrate."""
    pcan_status = Path('/proc/pcan')
    if not pcan_status.exists():
        raise RuntimeError('找不到PEAK /proc/pcan状态；拒绝自动配置CAN，请检查XCAN-USB驱动')
    rows = [line.split() for line in pcan_status.read_text().splitlines()]
    if not any(len(row) >= 3 and row[1] == 'usb' and row[2] == channel for row in rows):
        raise RuntimeError(f'{channel}未在PEAK USB-CAN列表中，拒绝自动配置；请确认XCAN-USB接口')
    def link_state():
        detail = subprocess.run(['ip', '-json', '-details', 'link', 'show', channel],
                                text=True, capture_output=True)
        if detail.returncode:
            raise RuntimeError(f'找不到CAN接口{channel}: {detail.stderr.strip()}')
        return json.loads(detail.stdout)[0]
    def require_ready(link):
        info = link.get('linkinfo', {}).get('info_data', {})
        if ('UP' not in link.get('flags', []) or info.get('state') != 'ERROR-ACTIVE'
                or info.get('bittiming', {}).get('bitrate') != 1000000):
            raise RuntimeError(f'{channel}状态或波特率不符；请检查：ip -details link show {channel}')
    link = link_state()
    if 'UP' in link.get('flags', []):
        # Never bring a running bus down to change its bitrate.
        require_ready(link)
        return
    print(f'{channel}是已识别的PEAK USB-CAN，正在配置1 Mbps并启动接口。', flush=True)
    if peak_usb_channels() == [channel] and fixed_io_action('can-config'):
        fixed_io_action('can-up')
        require_ready(link_state())
        return
    subprocess.run(['sudo', '-v'], check=True)
    setup = subprocess.run(['sudo', '-n', 'ip', 'link', 'set', 'dev', channel, 'type', 'can',
                            'bitrate', '1000000', 'restart-ms', '100', 'loopback', 'off'],
                           text=True, capture_output=True)
    if setup.returncode:
        raise RuntimeError(f'配置{channel}失败：{setup.stderr.strip() or setup.stdout.strip()}')
    up = subprocess.run(['sudo', '-n', 'ip', 'link', 'set', 'dev', channel, 'up'],
                        text=True, capture_output=True)
    if up.returncode:
        raise RuntimeError(f'启动{channel}失败：{up.stderr.strip() or up.stdout.strip()}')
    require_ready(link_state())


def load_config(path):
    c = json.loads(path.read_text())
    validate_gravity(c.get('gravity_feedforward'))
    if c['schema'] != 'b601-star-rs-teleop-v1':
        raise ValueError('配置版本不匹配')
    for key in ('directions', 'scales', 'kp', 'kd'):
        if len(c[key]) != 7:
            raise ValueError(f'{key} 必须包含七个数值')
        c[key] = [finite(v) for v in c[key]]
    if any(v not in (-1, 1) for v in c['directions']):
        raise ValueError('directions 只能是1或-1')
    # The B601-RS gripper uses a larger mechanical angle than the Star Arm
    # hand.  Values above 6 are valid when the configured follower soft limit
    # and the per-joint lead guard still bound the command.
    if any(not 0 < v <= 8 for v in c['scales']):
        raise ValueError('scales必须在0..8之间')
    if any(not 0 <= v <= 200 for v in c['kp']):
        raise ValueError('Kp必须在0..200之间')
    if any(not 0 <= v <= 10 for v in c['kd']):
        raise ValueError('Kd必须在0..10之间')
    ratios = c.get('velocity_feedforward_ratio', [0.0]*7)
    if len(ratios) != 7:
        raise ValueError('velocity_feedforward_ratio必须包含七个数值')
    c['velocity_feedforward_ratio'] = [finite(v) for v in ratios]
    if any(not 0 <= v <= .5 for v in c['velocity_feedforward_ratio']):
        raise ValueError('速度前馈比例必须在0..0.5之间')
    if c['velocity_feedforward_ratio'][6] != 0:
        raise ValueError('J7使用独立转矩控制，速度前馈比例必须为0')
    if not 0 < finite(c['gripper_torque_nm']) <= 2:
        raise ValueError('夹爪转矩限制必须在0..2Nm之间')
    if len(c['max_command_lead_deg']) != 7 or any(
            not 0 < finite(v) <= 45 for v in c['max_command_lead_deg']):
        raise ValueError('max_command_lead_deg需为七个0..45度的值')
    travel = json.loads(Path(c['travel_calibration']).read_text())
    c['limits_deg'] = []
    for j in MOTOR_IDS:
        joint = travel['joints'][str(j)]
        if joint['model'] != MOTOR_MODELS[j]:
            raise ValueError(f'J{j}电机型号不符')
        lo, hi = map(finite, joint['commissioning_limits_deg'])
        if lo >= hi:
            raise ValueError('软件限位顺序错误')
        c['limits_deg'].append([lo, hi])
    gripper = c.get('gripper_mapping')
    if gripper is not None:
        if gripper.get('mode') != 'endpoints':
            raise ValueError('gripper_mapping.mode必须是endpoints')
        for key in ('leader_closed_deg', 'leader_open_deg',
                    'follower_closed_deg', 'follower_open_deg'):
            gripper[key] = finite(gripper[key])
        leader_span = gripper['leader_open_deg']-gripper['leader_closed_deg']
        follower_span = gripper['follower_open_deg']-gripper['follower_closed_deg']
        if abs(leader_span) < 5 or follower_span == 0:
            raise ValueError('夹爪主臂端点至少相差5度，从臂端点不能相同')
        gain = follower_span/leader_span
        if not 0 < abs(gain) <= 8 or math.copysign(1, gain) != c['directions'][6]:
            raise ValueError('夹爪端点比例超出0..8或方向与directions不符')
        if any(not c['limits_deg'][6][0] <= gripper[key] <= c['limits_deg'][6][1]
               for key in ('follower_closed_deg', 'follower_open_deg')):
            raise ValueError('夹爪端点必须位于J7软件限位内')
        c['scales'][6] = abs(gain)
    return c


class RelativeMapper:
    def __init__(self, config, span, selected):
        self.c, self.span, self.selected = config, span, selected
        self.leader0 = self.follower0 = None

    def rebase(self, leader, follower):
        for i, v in enumerate(follower):
            lo, hi = self.c['limits_deg'][i]
            if not lo - 3 <= v <= hi + 3:
                raise RuntimeError(f'J{i+1}当前{v:.1f}°超出调试范围{lo}..{hi}°，请退出后手动调整')
        self.leader0, self.follower0 = list(leader), list(follower)

    def envelope(self, i):
        # Allow only the measured initial <=3 degree zero-side tolerance, so
        # holding a slightly negative calibrated zero does not cause a jump.
        lo, hi = self.c['limits_deg'][i]
        start = self.follower0[i]
        lo, hi = min(lo, start), max(hi, start)
        if self.span is not None:
            lo, hi = max(lo, start-self.span), min(hi, start+self.span)
        return lo, hi

    def target(self, leader):
        result = []
        for i in range(7):
            delta = leader[i] - self.leader0[i] if i in self.selected else 0.0
            # A counted encoder reset/wrap must stop control, not become a
            # nearest-turn heuristic while the follower is enabled.
            if abs(delta) > 300:
                raise RuntimeError(f'主臂J{i+1}相对起点变化超过300°，需停止并检查计圈')
            v = self.follower0[i] + self.c['directions'][i]*self.c['scales'][i]*delta
            gripper = self.c.get('gripper_mapping') if i == 6 and i in self.selected else None
            if gripper is not None:
                # Relative alignment can preserve a partly open follower pose
                # as "closed" forever. Use measured leader endpoints for J7;
                # Trajectory still starts at actual and bounds every command.
                fraction = clamp((leader[i]-gripper['leader_closed_deg']) /
                                 (gripper['leader_open_deg']-gripper['leader_closed_deg']), 0, 1)
                v = gripper['follower_closed_deg'] + fraction * (
                    gripper['follower_open_deg']-gripper['follower_closed_deg'])
            result.append(clamp(v, *self.envelope(i)))
        return result


class Trajectory:
    def __init__(self, start, vmax, amax):
        self.position, self.velocity = list(start), [0.0]*7
        self.command_velocity = [0.0]*7
        self.vmax = [float(vmax)]*7 if not isinstance(vmax, (list, tuple)) else list(map(float, vmax))
        self.amax = [float(amax)]*7 if not isinstance(amax, (list, tuple)) else list(map(float, amax))
        if len(self.vmax) != 7 or len(self.amax) != 7:
            raise ValueError('轨迹速度/加速度必须是一个数或七个数')

    def step(self, goal, dt, bounds):
        if not math.isfinite(dt) or not 0 < dt <= .15:
            raise ValueError('轨迹周期必须在0..150ms以内')
        # Small mathematical integration steps make braking insensitive to
        # the 60-100ms I/O cadence. They do not add hardware samples/commands.
        substeps = math.ceil(dt/.005)
        h = dt/substeps
        for i in range(7):
            before = self.position[i]
            target = clamp(goal[i], *bounds[i])
            pos, vel = before, self.velocity[i]
            for _ in range(substeps):
                error = target-pos
                # Account for the next integration step when braking. Near
                # the goal, a 60ms approach avoids snapping velocity to zero.
                ah = self.amax[i]*h
                brake_speed = math.sqrt(ah*ah+2*self.amax[i]*abs(error))-ah
                desired = math.copysign(min(self.vmax[i], brake_speed, abs(error)/.06), error)
                vel += clamp(desired-vel, -ah, ah)
                pos += vel*h
                bounded = clamp(pos, *bounds[i])
                if bounded != pos:
                    # Physical software bounds take precedence over ramps.
                    pos, vel = bounded, 0.0
            self.command_velocity[i] = clamp((pos-before)/dt, -self.vmax[i], self.vmax[i])
            self.position[i], self.velocity[i] = pos, vel
        return list(self.position)

    def limit_lead(self, actual, limits, bounds):
        """Bound command lead and reset its integrator; never disable on lag.

        This regulates the command relative to current feedback rather than
        accumulating an unreachable target. Position limits take precedence.
        """
        limited = []
        for i in range(7):
            lo = max(bounds[i][0], actual[i]-limits[i])
            hi = min(bounds[i][1], actual[i]+limits[i])
            if lo>hi:
                raise RuntimeError(f'J{i+1}反馈已离开软件范围，无法继续发送范围内目标')
            bounded = clamp(self.position[i], lo, hi)
            if bounded != self.position[i]:
                self.position[i], self.velocity[i] = bounded, 0.0
                # No velocity assistance into a stalled/lead-limited joint.
                self.command_velocity[i] = 0.0
                limited.append(i+1)
        return list(self.position), limited


class Hardware:
    def __init__(self, args, config):
        self.args, self.c = args, config
        self.controller = self.leader = None
        self.motors, self.saved = {}, {}
        self.touched = False
        self.gripper_velocity_rad_s = 0.0
        self.gripper_torque_command_nm = 0.0
        self.last_read_s = 0.0
        self.read_window_ns = None
        self.joint_read_windows_ns = None
        self.last_mit_commands = None
        self.motor_feedback = []
        self.mark_phase = lambda name: None
        self.read_phase_s = {}
        self.send_phase_s = {}
        self.leader_reader_pool = None
        self.leader_error = ''
        self.pending_leader = None
        self.leader_recovery = None
        self.feedback_reader = None
        self.gravity = GravityFeedforward(config.get('gravity_feedforward'),
                                          autonomous=getattr(args, 'autonomous', False))
        self.gravity_torque_nm = [0.0]*6

    def open(self):
        from star_arm_readonly import FreshStarReader
        assert_channel_is_explicitly_safe(self.args.channel, False)
        ensure_peak_can_ready(self.args.channel)
        self.controller, self.motors = open_motors(self.args.channel, MOTOR_IDS)
        self.feedback_reader = BatchFeedback(self.args.channel)
        if not getattr(self.args, 'autonomous', False):
            self.leader = FreshStarReader(self.args.port)
        for j, m in self.motors.items():
            try:
                device, responder = m.robstride_ping_host_id(HOST_ID, 200)
            except Exception as exc:
                if 'timed out' not in str(exc).lower():
                    raise
                raise RuntimeError(
                    f'从臂J{j}未回复，尚未使能电机。请检查从臂独立电源、急停状态以及'
                    f'XCAN-USB到从臂的CAN接线；USB接口在线不表示从臂已供电。原始错误：{exc}'
                ) from exc
            if device != j or responder not in RESPONDER_IDS:
                raise RuntimeError(f'从臂J{j}身份不符')
        for i in range(7) if self.leader is not None else ():
            if not self.leader.ping(i):
                raise RuntimeError(f'主臂ID{i}无响应')
        self.check_faults()

    def check_faults(self):
        for j, m in self.motors.items():
            fault, warning = m.robstride_get_fault_report()
            if fault or warning:
                raise RuntimeError(f'J{j}故障/警告：{fault}/{warning}；不自动清故障')

    def read(self, allow_leader_fault=False, skip_leader=False):
        started = time.monotonic()
        read_start_ns = time.monotonic_ns()
        # UART and CAN are separate interfaces. Overlap the fresh UART batch
        # with the CAN reads, without sharing the CAN controller across threads
        # or accepting a cached/partial leader batch.
        if self.leader is not None and not skip_leader and self.leader_reader_pool is None:
            self.leader_reader_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='star-read')
        def read_leader():
            start_ns = time.monotonic_ns()
            values = [finite(v) for v in self.leader.read_positions()]
            return values, [start_ns, time.monotonic_ns()]
        pending = self.leader_reader_pool.submit(read_leader) if self.leader is not None and not skip_leader else None
        if pending is not None:
            self.pending_leader = pending
        follower = []
        joint_windows = []
        try:
            if self.feedback_reader is not None:
                self.mark_phase('can_batch_read')
                batch_start = time.monotonic()
                follower, self.gripper_velocity_rad_s, joint_windows = self.feedback_reader.read()
                self.read_phase_s = {'batch': time.monotonic()-batch_start, 'sdk': []}
            else:
                # Unopened hardware in offline lifecycle tests only. open()
                # always requires the batch reader; runtime errors never fall back.
                for j in MOTOR_IDS:
                    joint_start_ns = time.monotonic_ns()
                    follower.append(math.degrees(finite(self.motors[j].robstride_get_param_f32(0x7019, 80))))
                    joint_windows.append([joint_start_ns, time.monotonic_ns()])
                self.gripper_velocity_rad_s = finite(self.motors[7].robstride_get_param_f32(0x701B, 80))
        except BaseException:
            # Never delay a CAN failure/explicit shutdown behind a UART worker.
            # close() disables first, then joins before serial cleanup/reuse.
            raise
        # Internal zero vector is used only by the dormant relative mapper.
        # Autonomous telemetry explicitly reports no leader measurement.
        self.leader_error = ''
        try:
            # The deadline includes the overlapping CAN work: a slow UART must
            # not add another220ms and accidentally trip the350ms fatal guard.
            leader, leader_window = pending.result(timeout=max(0., .22-(time.monotonic()-started))) if pending is not None else (
                (None, None) if skip_leader else ([0.0]*7, None))
            if leader is not None and (len(leader) != 7 or not all(math.isfinite(v) for v in leader)):
                raise RuntimeError('主臂缺少七个有效的新鲜角度')
        except Exception as exc:
            if not allow_leader_fault:
                raise
            # All seven follower positions and velocity above were queried on
            # CAN this cycle. No leader cache or substitute zero is permitted.
            self.leader_error = f'主臂读取失败：{type(exc).__name__}: {exc}'
            leader, leader_window = None, None
        # MIT replies update this SDK cache without extra CAN requests. Keep
        # torque/velocity alongside the independently queried encoder position.
        self.motor_feedback = []
        for j in MOTOR_IDS:
            self.mark_phase(f'sdk_get_state_J{j}')
            cache_start = time.monotonic()
            state = self.motors[j].get_state()
            if 'sdk' in self.read_phase_s:
                self.read_phase_s['sdk'].append(time.monotonic()-cache_start)
            self.motor_feedback.append(None if state is None else {
                'source': 'sdk_cached_mit_reply', 'position_rad': state.pos,
                'velocity_rad_s': state.vel, 'torque_nm': state.torq,
                'status_code': state.status_code, 'temperature_mos_c': state.t_mos})
        self.joint_read_windows_ns = {'leader': leader_window, 'follower': joint_windows}
        self.last_read_s = time.monotonic()-started
        self.read_window_ns = [read_start_ns, time.monotonic_ns()]
        if self.last_read_s > 0.35:
            raise RuntimeError('两臂读取超过350ms；停止后检查通信')
        return leader, follower

    def request_leader_recovery(self):
        """Explicit R only: reopen UART in its worker while CAN hold keeps running."""
        if self.leader_recovery is not None and not self.leader_recovery.done():
            return
        if self.pending_leader is not None and not self.pending_leader.done():
            raise RuntimeError('主臂旧读取尚未结束，继续保持；请稍后再点恢复')
        if self.leader_reader_pool is None:
            self.leader_reader_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='star-read')
        def reconnect():
            from star_arm_readonly import FreshStarReader
            if self.leader is not None:
                try:
                    self.leader.close()
                except Exception:
                    pass
            self.leader = None
            reader = FreshStarReader(self.args.port)
            try:
                values = [finite(v) for v in reader.read_positions()]
                if len(values) != 7:
                    raise RuntimeError('重连主臂缺少七关节新鲜角度')
            except Exception:
                reader.close()
                raise
            self.leader = reader
            return values, time.monotonic_ns()
        self.leader_recovery = self.leader_reader_pool.submit(reconnect)

    def take_leader_recovery(self):
        if self.leader_recovery is None or not self.leader_recovery.done():
            return None
        pending, self.leader_recovery = self.leader_recovery, None
        return pending.result()

    def prepare(self, on_backup=None):
        from motorbridge import Mode
        self.check_faults()
        # Back up runtime settings before any write; never persist to flash.
        for j, m in self.motors.items():
            self.saved[j] = {'mode': m.robstride_get_param_i8(0x7005, 200),
                             'timeout': m.robstride_get_param_u32(0x7028, 200)}
        if on_backup:
            on_backup(self.saved)
        self.touched = True
        for m in self.motors.values():
            m.disable()
            m.ensure_mode(Mode.MIT, 500)
            # RobStride canTimeout units: 20000 = 1 second. Keep this active
            # on abnormal termination; ordinary exit restores the old value.
            m.robstride_write_param_u32(0x7028, 20000)
            if m.robstride_get_param_u32(0x7028, 200) != 20000:
                raise RuntimeError('CAN超时参数回读不符，未允许使能')

    def enable_at(self, follower):
        self.gravity.reset()
        for j, m in self.motors.items():
            i = j-1
            # Seed the measured pose, not zero or an old leader target.
            m.send_mit(math.radians(follower[i]), 0, self.c['kp'][i] if i<6 else 0,
                       self.c['kd'][i] if i<6 else 1.5, 0)
            m.enable()

    def send(self, command, actual, command_velocity=None):
        command_velocity = command_velocity if command_velocity is not None else [0.0]*7
        ratios = self.c.get('velocity_feedforward_ratio', [0.0]*7)
        sent = []
        self.mark_phase('gravity')
        gravity_start = time.monotonic()
        self.gravity_torque_nm = self.gravity.update(actual)
        self.send_phase_s = {'gravity': time.monotonic()-gravity_start, 'sdk': []}
        for j, m in self.motors.items():
            i = j-1
            self.mark_phase(f'sdk_send_mit_J{j}')
            motor_send_start = time.monotonic()
            if i < 6:
                # Partial trajectory velocity reduces damping drag during
                # motion. Holding still retains the full existing damping.
                values = [math.radians(command[i]), math.radians(command_velocity[i]*ratios[i]),
                          self.c['kp'][i], self.c['kd'][i], self.gravity_torque_nm[i]]
                m.send_mit(*values)
            else:
                # Measured motor velocity avoids noisy finite differences of
                # slow host position samples; bound the host torque command.
                velocity = self.gripper_velocity_rad_s
                torque = self.c['kp'][i]*math.radians(command[i]-actual[i])-self.c['kd'][i]*velocity
                torque = clamp(torque, -self.c['gripper_torque_nm'], self.c['gripper_torque_nm'])
                self.gripper_torque_command_nm = torque
                # Use the motor-side velocity damping as well as the host-side
                # measured-velocity term.  The RS reference gripper profile
                # uses kd=1.5; kd=0 here lets a fast gripper ring between
                # successive 90 ms feedback samples.
                values = [0, 0, 0, 1.5, torque]
                m.send_mit(*values)
            self.send_phase_s['sdk'].append(time.monotonic()-motor_send_start)
            sent.append(values)
        self.last_mit_commands = sent

    def close(self):
        errors = []
        disabled = set()
        if self.touched:
            # Retry stop frames; do not home. A dead CAN path cannot guarantee
            # delivery, which is why the verified motor watchdog is required.
            for _ in range(3):
                for j, m in self.motors.items():
                    try:
                        m.disable()
                        disabled.add(j)
                    except Exception as exc:
                        errors.append(f'J{j}失能发送失败:{exc}')
            for j, old in self.saved.items():
                if j not in disabled:
                    errors.append(f'J{j}未成功发送失能，保留1秒CAN超时设置')
                    continue
                try:
                    m = self.motors[j]
                    m.robstride_write_param_i8(0x7005, old['mode'])
                    m.robstride_write_param_u32(0x7028, old['timeout'])
                except Exception as exc:
                    errors.append(f'J{j}运行参数恢复失败:{exc}')
        if self.leader_reader_pool is not None:
            self.leader_reader_pool.shutdown(wait=True)
            self.leader_reader_pool = None
        if self.leader is not None:
            try:
                self.leader.close()
            except Exception as exc:
                errors.append(f'主臂串口关闭失败:{exc}')
        if self.controller is not None:
            close_all(self.controller, self.motors)
        if self.feedback_reader is not None:
            self.feedback_reader.close()
            self.feedback_reader = None
        if errors:
            print('\n'.join(errors), file=sys.stderr)
        return errors


class Keyboard:
    def __enter__(self):
        self.old = termios.tcgetattr(sys.stdin) if sys.stdin.isatty() else None
        if self.old:
            tty.setcbreak(sys.stdin.fileno())
        return self

    def read(self):
        if select.select([sys.stdin], [], [], 0)[0]:
            key = os.read(sys.stdin.fileno(), 32).decode(errors='ignore').lower()
            if not key:
                raise OSError('控制终端已关闭')
            return key
        return ''

    def __exit__(self, *_):
        if self.old:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old)
            except termios.error:
                pass  # Closed PTY; hardware cleanup is owned by the outer scope.


def stop_process(args):
    # Atomic stop file is observed before reads, before enable and before each
    # batch of motor commands. SIGKILL is deliberately not used.
    stop = Path(args.stop_file)
    stop.parent.mkdir(parents=True, exist_ok=True)
    stop.touch()
    print('已发送停止请求；运行窗口将退出并向从臂发送失能命令。')


def run(args):
    config = load_config(args.config)
    args.channel = resolve_can_channel(args.channel)
    print(f'从臂CAN接口：{args.channel}（按PEAK USB设备身份核验）', flush=True)
    if args.live and not sys.stdin.isatty():
        raise RuntimeError('带运动模式请在前台终端运行，按R开始')
    stop = Path(args.stop_file)
    lockpath = ROOT / 'teleop.lock'
    lock = lockpath.open('a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('另一个遥操作程序正在运行')
    stop.unlink(missing_ok=True)
    vla_step_path = runtime_path(os.getpid(), 'json')
    vla_status_path = runtime_path(os.getpid(), 'status.json')
    vla_abort_path = runtime_path(os.getpid(), 'abort')
    vla_step_path.unlink(missing_ok=True)
    vla_status_path.unlink(missing_ok=True)
    vla_abort_path.unlink(missing_ok=True)
    hardware = Hardware(args, config)
    selected = set(range(7)) if args.joint == 'all' else {int(args.joint)-1}
    span = None if args.full_range else args.span_deg
    mapper = RelativeMapper(config, span, selected)
    trajectory = None
    vla_goal = None
    vla_path = None
    vla_path_started = None
    vla_step_state = {'state': 'idle', 'message': '', 'id': None}
    default_vmax = default_amax = None
    tracking, paused = False, False
    fault_hold_reason = ''
    recovery_requested = False
    keyboard_failed = ''
    duration_expired = False
    previous_leader = None
    last_actual = None
    intervals = deque(maxlen=30)
    last_lag_notice = 0.0
    started = time.monotonic()
    last_tick, last_print, last_fault = started, 0., 0.
    path = ROOT/'teleop-logs'/datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    path.mkdir(parents=True)
    journal = BufferedJournal(path/'events.jsonl')
    telemetry = TelemetryPublisher(str(path), config, vars(args))
    alignment = None
    def log(event, **data):
        journal.write(json.dumps({'time': stamp(), 'event': event, **data}, ensure_ascii=False, default=str)+'\n')
        if event == 'runtime_backup':
            journal.flush()
            os.fsync(journal.fileno())
    timing = ControlTiming(log)
    hardware.mark_phase = timing.mark
    # Model selection/configuration is immutable for a running controller. Load
    # before enabling so a filesystem read cannot interrupt command refresh.
    controller_vla_profile = None
    def check_stop():
        if stop.exists():
            raise KeyboardInterrupt
    def enter_fault_hold(reason, actual):
        nonlocal paused, fault_hold_reason, vla_goal, vla_path, vla_step_state, recovery_requested
        paused = True
        fault_hold_reason = reason
        recovery_requested = False
        # Freeze at this cycle's actual CAN feedback, not the interrupted target.
        trajectory.position[:] = actual[:]
        trajectory.velocity[:] = [0.0]*7
        trajectory.command_velocity[:] = [0.0]*7
        trajectory.vmax, trajectory.amax = list(default_vmax), list(default_amax)
        vla_goal = vla_path = None
        vla_step_path.unlink(missing_ok=True)
        vla_step_state = {'state': 'aborted', 'message': '故障保持：'+reason, 'id': vla_step_state.get('id')}
        atomic_json(vla_status_path, vla_step_state)
        log('fault_hold', reason=reason, follower_deg=actual, recovery='explicit_R_only')
        print('\n故障保持：'+reason+'；从臂CAN正常，持续发送当前位置保持；不会自动恢复跟随。'
              '检查连接后按R尝试重连，Q/停止脚本仍会失能。', flush=True)
    print('运动模式（尚未使能）' if args.live else '只读预览（不会使能电机）', flush=True)
    print('R 开始/重新对齐；P或空格 暂停并保持；Q/Ctrl+C 退出并失能。' if args.live else
          '只读预览：R开始计算，P或空格暂停计算，Q/Ctrl+C退出；本程序不使能、不发送运动命令。', flush=True)
    print('方向预设:', config['directions'], '夹爪比例:', config['scales'][6], flush=True)
    gripper = config.get('gripper_mapping')
    if gripper:
        print(f"J7固定开合映射：主臂{gripper['leader_closed_deg']}°闭合/"
              f"{gripper['leader_open_deg']}°全开 → 从臂{gripper['follower_closed_deg']}°/"
              f"{gripper['follower_open_deg']}°；按R后从实测位置限速靠近，不写零点。", flush=True)
    print('参与跟随：'+', '.join(f'J{i+1}' for i in sorted(selected))+
          ('；其余关节保持起点。' if len(selected)<7 else '（全部七关节）。'), flush=True)
    print('Kp:', config['kp'], 'Kd:', config['kd'], flush=True)
    print('轨迹速度前馈比例:', config['velocity_feedforward_ratio'], flush=True)
    print(f'J1-J6限速{args.speed}°/s、加速度{args.accel}°/s²；'
          f'J7限速{args.j7_speed}°/s、加速度{args.j7_accel}°/s²；范围：'+
          ('配置软限位' if args.full_range else f'起点±{args.span_deg}°与配置软限位的交集'), flush=True)
    print('按Q退出前托住从臂；失能后关节可能因重力下落。日志：' if args.live else
          '此窗口只计算预览目标；从臂不会由本程序驱动。日志：', path, flush=True)
    try:
        if args.live and sys.stdout.isatty():
            os.set_blocking(sys.stdout.fileno(), False)
        controller_vla_profile = load_motion_profile()
        hardware.open()
        log('open', config=config, live=args.live, arguments=vars(args))
        with Keyboard() as keyboard:
            print('遥操作硬件就绪，等待开始。', flush=True)
            while True:
                iteration_started = time.monotonic()
                timing.mark('stop_and_keyboard')
                check_stop()
                timed_out = args.seconds and not duration_expired and time.monotonic()-started >= args.seconds
                if timed_out and not tracking:
                    break
                try:
                    if TERMINAL_LOST:
                        raise OSError('控制终端挂断(SIGHUP)')
                    keys = keyboard.read() if not keyboard_failed else ''
                except OSError as exc:
                    if not args.live or not tracking:
                        raise
                    keyboard_failed = str(exc)
                    keys = ''
                if args.autonomous and 'r' in keys:
                    print('自主控制不跟随主臂；请用VLA启动或停止按钮。', flush=True)
                    keys = keys.replace('r', '')
                if 'q' in keys or '\x03' in keys:
                    break
                leader, actual = hardware.read(allow_leader_fault=bool(args.live and tracking),
                                               skip_leader=bool(fault_hold_reason))
                timing.mark('control_logic')
                check_stop()
                if 'p' in keys or ' ' in keys:
                    recovery_requested = False
                    keys = keys.replace('r', '').replace('h', '')
                if fault_hold_reason:
                    if 'r' in keys:
                        try:
                            hardware.request_leader_recovery()
                            recovery_requested = True
                            print('正在后台重连主臂；从臂继续保持，不释放使能。', flush=True)
                        except Exception as exc:
                            print('主臂重连未完成，继续故障保持：'+str(exc), flush=True)
                        keys = keys.replace('r', '')
                    try:
                        recovered = hardware.take_leader_recovery()
                        if recovered is not None and recovery_requested:
                            values, received_ns = recovered
                            if not 0 <= time.monotonic_ns()-received_ns < 300_000_000:
                                raise RuntimeError('重连角度已过期，请再次按R')
                            mapper.rebase(values, actual)  # validate before lifting latch
                            leader = values
                            previous_leader = None
                            fault_hold_reason = ''
                            recovery_requested = False
                            keys += 'r'
                            log('fault_hold_recovered', follower_deg=actual)
                    except Exception as exc:
                        recovery_requested = False
                        print('主臂重连失败，继续故障保持：'+str(exc), flush=True)
                if not fault_hold_reason and leader is None:
                    if not args.live or not tracking:
                        raise RuntimeError(hardware.leader_error or '主臂无新鲜角度')
                    enter_fault_hold(hardware.leader_error or '主臂无新鲜角度', actual)
                    keys = keys.replace('r', '').replace('h', '')
                if not fault_hold_reason and previous_leader is not None and any(abs(a-b)>45 for a,b in zip(leader, previous_leader)):
                    if not args.live or not tracking:
                        raise RuntimeError('主臂相邻读数跳变超过45°，停止检查计圈/电源')
                    enter_fault_hold('主臂相邻读数跳变超过45°，请检查计圈/电源', actual)
                    leader = None
                    keys = keys.replace('r', '').replace('h', '')
                if timed_out or (keyboard_failed and not fault_hold_reason):
                    if args.live and tracking:
                        enter_fault_hold('定时运行结束' if timed_out else '控制终端丢失：'+keyboard_failed, actual)
                        keys = keys.replace('r', '').replace('h', '')
                    if timed_out:
                        duration_expired = True
                if leader is not None:
                    previous_leader = list(leader)
                now = time.monotonic()
                cycle_dt = now-last_tick
                if 0 < cycle_dt < .4:
                    intervals.append(cycle_dt)
                dt = min(now-last_tick, 0.15)
                if tracking and now-last_tick > 0.4:
                    if not fault_hold_reason:
                        enter_fault_hold('控制循环超过400ms，取消运动并保持新鲜实测姿态', actual)
                    keys = keys.replace('r', '').replace('h', '')
                dt = max(dt, 0.001)
                last_tick = now
                if last_actual is None:
                    last_actual = list(actual)
                if 'r' in keys and tracking and paused and (
                        vla_step_state['state'] in ('pending', 'executing') or vla_step_path.exists()):
                    vla_step_path.unlink(missing_ok=True)
                    vla_goal = None
                    vla_path = None
                    if default_vmax is not None:
                        trajectory.vmax, trajectory.amax = list(default_vmax), list(default_amax)
                    vla_step_state = {'state': 'aborted', 'message': '重新开始遥操作；VLA单步已取消',
                                      'id': vla_step_state['id']}
                    atomic_json(vla_status_path, vla_step_state)
                    log('vla_step_aborted', **vla_step_state)
                if not fault_hold_reason and ('r' in keys or 'h' in keys) and (not tracking or paused):
                    mapper.rebase(leader, actual)  # validation before writes
                    if args.live and not hardware.touched:
                        check_stop()
                        hardware.prepare(lambda saved: log('runtime_backup', settings=saved))
                        leader, actual = hardware.read()
                        mapper.rebase(leader, actual)
                        check_stop()
                        hardware.enable_at(actual)
                    check_stop()
                    tracking, paused = True, 'h' in keys
                    # Keep the six arm joints smooth for insertion; give J7 a
                    # separate faster profile so gripper motion does not slow
                    # the rest of the arm or vice versa.
                    vmax = [args.speed]*6 + [args.j7_speed]
                    amax = [args.accel]*6 + [args.j7_accel]
                    vmax[3], amax[3] = args.j4_speed, args.j4_accel
                    trajectory = Trajectory(actual, vmax, amax)
                    default_vmax, default_amax = list(trajectory.vmax), list(trajectory.amax)
                    vla_goal = None
                    vla_path = None
                    last_actual = list(actual)
                    last_tick = time.monotonic()
                    log('align', leader_reference_deg=None if args.autonomous else leader, follower_reference_deg=actual,
                        previous_runtime_settings=hardware.saved,
                        effective_limits_deg=[mapper.envelope(i) for i in range(7)],
                        following_joints=[i+1 for i in sorted(selected)])
                    alignment = {'leader_reference_deg': None if args.autonomous else list(leader),
                                 'follower_reference_deg': list(actual),
                                 'mono_ns': time.monotonic_ns(),
                                 'effective_limits_deg': [mapper.envelope(i) for i in range(7)]}
                    print('本次目标角度范围：'+', '.join(
                        f'J{i+1} {mapper.envelope(i)[0]:.1f}..{mapper.envelope(i)[1]:.1f}°'
                        for i in sorted(selected)), flush=True)
                    print('\n已开始；以此刻两臂姿态为对应起点，不追赶旧目标。' if args.live else
                          '\n已开始预览计算；以当前两臂姿态为对应起点，本程序不发送运动命令。', flush=True)
                    if gripper and 6 in selected:
                        print('J7按固定开合端点跟随，不把当前夹爪开口记成闭合零位。', flush=True)
                    if paused:
                        print('已暂停，真机测试从实测姿态保持；不跟随主臂。', flush=True)
                abort_vla = vla_abort_path.exists() or 'p' in keys or ' ' in keys
                if abort_vla and tracking:
                    vla_abort_path.unlink(missing_ok=True)
                    vla_step_path.unlink(missing_ok=True)
                    paused = True
                    # Freeze arm feedback even after plan completion or during
                    # manual following. Preserve J7's existing grip target so
                    # pause does not silently release a held object.
                    trajectory.position[:6] = actual[:6]
                    trajectory.velocity = [0.0]*7
                    trajectory.command_velocity = [0.0]*7
                    vla_goal = None
                    vla_path = None
                    trajectory.vmax, trajectory.amax = list(default_vmax), list(default_amax)
                    if vla_step_state['state'] == 'executing':
                        vla_step_state = {'state': 'aborted', 'message': 'VLA单步已中止，臂部保持当前实测位置，夹爪保持原开合目标',
                                          'id': vla_step_state['id']}
                        atomic_json(vla_status_path, vla_step_state)
                        log('vla_step_aborted', **vla_step_state)
                    log('pause')
                    print('\n已暂停，臂部保持实测位置，夹爪保持原目标；R重新对齐后继续，Q退出失能。' if args.live else
                          '\n预览计算已暂停；R重新对齐后继续，Q退出。', flush=True)
                command = list(actual)
                lead_limited = []
                send_start_ns = send_end_ns = None
                if tracking and paused and args.live and not fault_hold_reason and not abort_vla and vla_step_path.exists():
                    request = {}
                    try:
                        request_text = vla_step_path.read_text()
                        vla_step_path.unlink(missing_ok=True)
                        request = json.loads(request_text)
                        if (time.monotonic_ns()-int(request.get('created_mono_ns', 0)) > 2_000_000_000
                                or not isinstance(request.get('id'), str)):
                            raise ValueError('单步确认请求已过期，请重新推理和确认')
                        bounds = [mapper.envelope(i) for i in range(7)]
                        timing.mark('plan_validation')
                        vla_profile = controller_vla_profile
                        replacing = validate_plan_replacement(request, vla_step_state, vla_profile)
                        vla_path = None
                        if request.get('trajectory_deg') is not None:
                            if not vla_profile['continuous_plan']:
                                raise ValueError('当前配置未启用连续模型轨迹')
                            expected=request.get('expected_state_deg')
                            expected_age = (time.monotonic_ns()-request.get('expected_mono_ns', 0))/1e9
                            if replacing and not 0 <= expected_age <= .3:
                                raise ValueError('滚动接续姿态超过300ms，已保持')
                            # Account for measured motion after a fresh snapshot;
                            # the legacy stationary path keeps its original tolerance.
                            allowances = [(2 if i==6 else .5) +
                                ((vla_profile['gripper_speed_deg_s'] if i==6 else vla_profile['speed_deg_s'])*expected_age
                                 if replacing else 0) for i in range(7)]
                            if (not isinstance(expected,list) or len(expected)!=7 or
                                    any(not isinstance(x,(int,float)) or not math.isfinite(x) for x in expected) or
                                    any(abs(actual[i]-expected[i]) > allowances[i] for i in range(7))):
                                raise ValueError('连续轨迹提交前实测姿态已改变')
                            vla_path=VlaPlan(request['trajectory_deg'],actual,trajectory.position,
                                             request.get('session_origin_deg'),bounds,vla_profile)
                            vla_path_started=time.monotonic()
                            vla_goal=list(vla_path.actions[0])
                        else:
                            vla_goal = bounded_vla_targets(actual, request.get('action_deg'),
                                                       request.get('expected_state_deg'), bounds,
                                                       previous_command=trajectory.position,
                                                       session_origin=request.get('session_origin_deg'), profile=vla_profile)
                        if not replacing:
                            trajectory.velocity = [0.0]*7
                        trajectory.vmax = [vla_profile['speed_deg_s']]*6 + [vla_profile['gripper_speed_deg_s'] if vla_profile['gripper_enabled'] else 0.0]
                        trajectory.amax = [vla_profile['accel_deg_s2']]*6 + [vla_profile['gripper_accel_deg_s2'] if vla_profile['gripper_enabled'] else 0.0]
                        vla_step_state = {'state': 'executing', 'message': '执行限幅关节目标；J7沿用转矩/阻尼控制',
                                          'id': request['id'], 'target_deg': vla_goal,
                                          'motion_profile': vla_profile}
                        if request.get('replace_id'):
                            vla_step_state['replaces'] = request['replace_id']
                        if vla_path is not None:
                            vla_step_state.update(continuous_plan=True,points=len(vla_path.actions),
                                                  point_index=0,target_deg=vla_path.actions[-1],
                                                  action_filter=vla_path.filter_info)
                            if vla_path.filter_info['enabled']:
                                log('vla_plan_filtered', id=request['id'],
                                    filter_info=vla_path.filter_info, trajectory_deg=vla_path.actions)
                        atomic_json(vla_status_path, vla_step_state)
                        log('vla_step_begin', **vla_step_state)
                    except Exception as exc:
                        vla_goal = None
                        vla_path = None
                        trajectory.position[:] = actual[:]
                        trajectory.velocity[:] = [0.0]*7
                        trajectory.vmax, trajectory.amax = list(default_vmax), list(default_amax)
                        vla_step_state = {'state': 'rejected', 'message': str(exc),
                                          'id': request.get('id') if isinstance(request, dict) else None}
                        atomic_json(vla_status_path, vla_step_state)
                        log('vla_step_rejected', message=str(exc))
                if tracking:
                    if vla_path is not None and vla_step_state['state']=='executing':
                        if vla_path.stalled() or time.monotonic()-vla_path_started>vla_profile['plan_timeout_s']:
                            reason = ('连续轨迹4秒无推进' if vla_path.stalled() else
                                      f"连续轨迹{vla_profile['plan_timeout_s']:g}秒内未完成")
                            vla_step_state.update(state='aborted',message=reason+'，已保持；检查关节跟踪或夹爪接触')
                            atomic_json(vla_status_path,vla_step_state)
                            log('vla_plan_timeout',**vla_step_state)
                            trajectory.position[:]=actual[:]
                            trajectory.velocity[:]=[0.]*7
                            vla_path=vla_goal=None
                        else:
                            vla_goal=vla_path.goal(trajectory.position)
                            vla_step_state['point_index']=vla_path.index
                    target = (vla_goal if vla_step_state['state'] == 'executing' and vla_goal is not None
                              else trajectory.position) if paused else mapper.target(leader)
                    bounds = [mapper.envelope(i) for i in range(7)]
                    command = trajectory.step(target, dt, bounds)
                    for i in range(7):
                        if not mapper.envelope(i)[0]-3 <= actual[i] <= mapper.envelope(i)[1]+3:
                            raise RuntimeError(f'J{i+1}实际位置超出软件范围')
                    leads=config['max_command_lead_deg']
                    if vla_step_state['state']=='executing':
                        leads=[min(leads[i],vla_profile['gripper_lead_deg'] if i==6 else vla_profile['lead_deg']) for i in range(7)]
                    command, lead_limited = trajectory.limit_lead(actual, leads, bounds)
                    if vla_step_state['state'] == 'executing' and vla_goal is not None:
                        complete=(vla_path.final_reached(command,trajectory.velocity) if vla_path is not None else
                                  all(abs(command[i]-vla_goal[i]) < 0.03 and
                                      abs(trajectory.velocity[i]) < 0.05 for i in range(7 if vla_profile['gripper_enabled'] else 6)))
                        if complete:
                            final_goal=vla_path.actions[-1] if vla_path is not None else vla_goal
                            points=len(vla_path.actions) if vla_path is not None else 1
                            vla_step_state = {'state': 'complete', 'message': '单步完成，保持暂停',
                                              'id': vla_step_state['id'], 'target_deg': final_goal,'points':points}
                            vla_goal = None
                            vla_path = None
                            trajectory.velocity = [0.0]*7
                            trajectory.vmax, trajectory.amax = list(default_vmax), list(default_amax)
                            atomic_json(vla_status_path, vla_step_state)
                            log('vla_step_complete', **vla_step_state)
                    if lead_limited and now-last_lag_notice >= 1:
                        print('提示：'+','.join(f'J{j}' for j in lead_limited)+
                              '目标超前量已限制，保持使能并继续跟随。', flush=True)
                        log('lead_limited', joints=lead_limited)
                        last_lag_notice = now
                    if args.live:
                        check_stop()
                        send_start_ns = time.monotonic_ns()
                        hardware.send(command, actual, trajectory.command_velocity)
                        send_end_ns = time.monotonic_ns()
                timing.mark('telemetry_publish')
                telemetry.publish(
                    read_window_ns=hardware.read_window_ns, send_window_ns=[send_start_ns, send_end_ns],
                    joint_read_windows_ns=hardware.joint_read_windows_ns,
                    leader_deg=None if args.autonomous or fault_hold_reason else leader, follower_deg=actual, command_deg=command,
                    leader_read=bool(not args.autonomous and not fault_hold_reason and leader is not None),
                    fault_hold=bool(fault_hold_reason), fault_hold_reason=fault_hold_reason,
                    tracking=tracking, paused=paused, live=args.live,
                    motor_commands_sent=bool(args.live and tracking), alignment=alignment,
                    mit_commands=hardware.last_mit_commands if args.live and tracking else None,
                    mit_fields=['position_rad', 'velocity_rad_s', 'kp', 'kd', 'torque_nm'],
                    motor_feedback=getattr(hardware, 'motor_feedback', None),
                    gravity_torque_nm=getattr(hardware, 'gravity_torque_nm', [0.0]*6),
                    gripper_torque_command_nm=hardware.gripper_torque_command_nm,
                    gripper_velocity_rad_s=hardware.gripper_velocity_rad_s,
                    lead_limited_joints=lead_limited, read_s=hardware.last_read_s,
                    actual_hz=len(intervals)/sum(intervals) if intervals and sum(intervals)>0 else 0,
                    feedback_backend='socketcan_batch_mechPos' if getattr(hardware, 'feedback_reader', None) is not None else 'sdk_sequential',
                    diagnostic_dropped_lines=journal.dropped, diagnostic_error=journal.error,
                    read_phase_s=getattr(hardware, 'read_phase_s', {}), send_phase_s=getattr(hardware, 'send_phase_s', {}),
                    vla_step=vla_step_state)
                if now-last_fault > 1:
                    timing.mark('sdk_fault_check')
                    hardware.check_faults()
                    last_fault = now
                timing.mark('diagnostic_output')
                last_actual = list(actual)
                if now-last_print >= .5:
                    state = ('故障保持' if fault_hold_reason else ('保持中' if paused else ('跟随中' if tracking else '等待R'))) if args.live else (
                        '只读预览：暂停计算' if paused else ('只读预览：计算中，不下发' if tracking else '只读预览：等待R'))
                    def fmt(v): return ' '.join(f'{x:7.2f}' for x in v)
                    target_label = '目标 °' if args.live else '预览目标 °'
                    print(f'\n[{state}]        J1      J2      J3      J4      J5      J6      J7\n'
                          f'主臂 °     {"主臂反馈不可用，不跟随" if fault_hold_reason else "自主模式，未读取主臂" if args.autonomous else fmt(leader)}\n从臂 °     {fmt(actual)}\n{target_label}     {fmt(command)}', flush=True)
                    measured_hz = len(intervals)/sum(intervals) if intervals and sum(intervals)>0 else 0
                    print(f'实际循环约{measured_hz:.1f}Hz；本轮读取{hardware.last_read_s*1000:.0f}ms', flush=True)
                    if tracking and gripper and 6 in selected:
                        print(f'J7开合目标{target[6]:.2f}°；轨迹指令{command[6]:.2f}°；'
                              f'主机转矩指令{hardware.gripper_torque_command_nm:.2f} Nm'
                              + ('（只读预览未下发）' if not args.live else ''), flush=True)
                    log('sample', leader_deg=None if args.autonomous or fault_hold_reason else leader, follower_deg=actual, command_deg=command,
                        fault_hold=bool(fault_hold_reason), fault_hold_reason=fault_hold_reason,
                        tracking=tracking, paused=paused, live=args.live,
                        motor_commands_sent=bool(args.live and tracking),
                        gravity_torque_nm=getattr(hardware, 'gravity_torque_nm', [0.0]*6),
                        lead_limited_joints=lead_limited, actual_hz=measured_hz,
                        mapped_target_deg=list(target) if tracking else None,
                        velocity_target_deg_s=[v*r for v,r in zip(
                            trajectory.command_velocity, config['velocity_feedforward_ratio'])]
                            if tracking else [0.0]*7,
                        gripper_torque_command_nm=hardware.gripper_torque_command_nm,
                        gripper_velocity_rad_s=hardware.gripper_velocity_rad_s,
                        read_s=hardware.last_read_s)
                    last_print = now
                work_s = time.monotonic()-iteration_started
                log('control_timing', work_s=work_s, cycle_dt_s=cycle_dt,
                    read_phase_s=getattr(hardware, 'read_phase_s', {}), send_phase_s=getattr(hardware, 'send_phase_s', {}))
                timing.mark('idle')
                remaining = 1/args.hz-(time.monotonic()-iteration_started)
                if remaining > 0:
                    time.sleep(min(remaining, .1))
    except KeyboardInterrupt:
        print('\n收到停止请求。', flush=True)
        log('stop_requested')
    except BaseException as exc:
        log('error', message=str(exc), exception_type=type(exc).__name__, traceback=traceback.format_exc())
        raise
    finally:
        timing.mark('idle')
        errors = hardware.close()
        log('closed', stop_send_errors=errors)
        timing.close()
        journal.close()
        telemetry.close()
        lock.close()
        print('已退出；未执行自动回零。' if hardware.touched else '只读程序已退出，未使能/未写电机。', flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', nargs='?', choices=['run', 'stop'], default='run')
    p.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    p.add_argument('--port', default='/dev/ttyUSB0')
    p.add_argument('--channel', default='auto', help='默认自动识别唯一PEAK USB-CAN，不依赖can编号')
    p.add_argument('--live', action='store_true', help='允许按R后使能从臂；默认只读')
    p.add_argument('--autonomous', action='store_true', help='模型专用：只读取从臂，H保持，不读取或跟随主臂')
    p.add_argument('--joint', choices=['all','1','2','3','4','5','6','7'], default='all',
                   help='只让选中关节跟随，其余关节保持起始位置')
    ranges = p.add_mutually_exclusive_group()
    ranges.add_argument('--span-deg', type=float, default=None,
                        help='可选调试范围：起点±指定角度；默认使用配置软限位')
    ranges.add_argument('--full-range', action='store_true',
                        help='使用配置软限位（现为默认行为，保留旧命令兼容）')
    p.add_argument('--speed', type=float, default=120)
    p.add_argument('--accel', type=float, default=240)
    p.add_argument('--j4-speed', type=float, default=150)
    p.add_argument('--j4-accel', type=float, default=300)
    p.add_argument('--j7-speed', type=float, default=180)
    p.add_argument('--j7-accel', type=float, default=450)
    p.add_argument('--hz', type=float, default=30)
    p.add_argument('--seconds', type=float, default=0, help='0表示持续运行')
    p.add_argument('--stop-file', default=str(ROOT/'TELEOP_STOP'))
    a = p.parse_args(argv)
    a.full_range = a.full_range or a.span_deg is None
    if not (0 < a.speed <= 120 and 0 < a.accel <= 360 and
            0 < a.j4_speed <= 150 and 0 < a.j4_accel <= 500 and
            0 < a.j7_speed <= 180 and 0 < a.j7_accel <= 500 and
            0 < a.hz <= 30
            and (a.span_deg is None or 0 < a.span_deg <= 30) and a.seconds >= 0):
        p.error('speed需在0..120，accel在0..360，j4-speed在0..150，j4-accel在0..500，j7-speed在0..180，j7-accel在0..500')
    return a


@contextmanager
def serial_service():
    active = subprocess.run(['systemctl', 'is-active', '--quiet', 'uxrce-agent.service']).returncode == 0
    stopped = False
    try:
        if active:
            print('正在准备主臂串口；退出后恢复uxrce-agent服务。', flush=True)
            if not fixed_io_action('serial-stop'):
                subprocess.run(['sudo', '-v'], check=True)
                subprocess.run(['sudo', '-n', 'systemctl', 'stop', 'uxrce-agent.service'], check=True)
            stopped = True
        yield
    finally:
        if stopped:
            try:
                if not fixed_io_action('serial-start'):
                    subprocess.run(['sudo', '-n', 'systemctl', 'start', 'uxrce-agent.service'], check=True)
            except Exception:
                print('串口服务恢复失败，请执行 sudo systemctl start uxrce-agent.service', file=sys.stderr)
            else:
                print('已恢复uxrce-agent服务。', flush=True)


def main():
    args = parse_args()
    if args.command == 'stop':
        stop_process(args)
        return 0
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    signal.signal(signal.SIGHUP, terminal_lost)
    try:
        with nullcontext() if args.autonomous else serial_service():
            run(args)
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f'错误：{exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
