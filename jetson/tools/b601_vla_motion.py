"""Shared, bounded motion settings; separate from teleoperation gains/zeroes."""
import json
import math
import time
from pathlib import Path
from b601_vla_model import load_model_spec
from b601_vla_filter import filter_action_path

PROFILE_PATH = Path('/home/jetson/Desktop/b601-rs/calibration/vla_motion_config.json')
DEFAULT_PROFILE = dict(command_step_deg=.45, lead_deg=.9, abort_error_deg=1.,
                       session_span_deg=5., speed_deg_s=5., accel_deg_s2=10.,
                       settle_error_deg=.95, max_steps=10, max_seconds=180.,
                       gripper_enabled=False, gripper_step_deg=10., gripper_lead_deg=30.,
                       gripper_span_deg=90., gripper_speed_deg_s=30., gripper_accel_deg_s2=60.,
                       gripper_abort_error_deg=40., gripper_settle_error_deg=20.,
                       task_bounds_deg=None, action_steps=1, continuous_plan=False,
                       rolling_replan=False, prediction_max_age_s=3.,
                       join_error_deg=8., gripper_join_error_deg=45., replan_min_progress=.65,
                       complete_chunk=False, plan_timeout_s=15., progress_timeout_s=4.,
                       grasp_gate=False, action_filter_enabled=False, action_filter_max_delta_deg=2.)


def validate_profile(value):
    # Older saved profiles predate the task envelope/action chunk fields.
    legacy = set(DEFAULT_PROFILE) - {'task_bounds_deg', 'action_steps', 'continuous_plan',
                                   'rolling_replan', 'prediction_max_age_s', 'join_error_deg',
                                   'gripper_join_error_deg', 'replan_min_progress',
                                   'complete_chunk', 'plan_timeout_s', 'progress_timeout_s', 'grasp_gate',
                                   'action_filter_enabled', 'action_filter_max_delta_deg'}
    if not isinstance(value, dict) or not legacy <= set(value) <= set(DEFAULT_PROFILE):
        raise ValueError('VLA运动配置字段不完整')
    result = {**DEFAULT_PROFILE, **value}
    maxima = dict(command_step_deg=10, lead_deg=8, abort_error_deg=10,
                  session_span_deg=180, speed_deg_s=25, accel_deg_s2=60,
                  settle_error_deg=6, max_steps=2000, max_seconds=900, action_steps=50,
                  gripper_step_deg=20, gripper_lead_deg=45, gripper_span_deg=340,
                  gripper_speed_deg_s=60, gripper_accel_deg_s2=120,
                  gripper_abort_error_deg=50, gripper_settle_error_deg=40,
                  prediction_max_age_s=5, join_error_deg=8, gripper_join_error_deg=45,
                  replan_min_progress=1, plan_timeout_s=35, progress_timeout_s=4,
                  action_filter_max_delta_deg=3)
    if not isinstance(result['gripper_enabled'], bool):
        raise ValueError('gripper_enabled必须是布尔值')
    if not isinstance(result['continuous_plan'], bool):
        raise ValueError('continuous_plan必须是布尔值')
    if (not isinstance(result['action_filter_enabled'], bool) or
            (result['action_filter_enabled'] and not result['continuous_plan'])):
        raise ValueError('动作滤波要求连续轨迹配置和布尔开关')
    if not isinstance(result['rolling_replan'], bool) or (result['rolling_replan'] and not result['continuous_plan']):
        raise ValueError('滚动预测要求连续轨迹配置')
    if not isinstance(result['complete_chunk'], bool):
        raise ValueError('complete_chunk必须是布尔值')
    if not isinstance(result['grasp_gate'], bool) or (result['grasp_gate'] and
            (not result['complete_chunk'] or not result['gripper_enabled'])):
        raise ValueError('抓取确认要求连续动作块及夹爪控制')
    if result['complete_chunk'] and (not result['continuous_plan'] or result['rolling_replan'] or result['action_steps'] not in (5, 16, 25, 50)):
        raise ValueError('顺序执行窗口必须是5、16、25或50点，禁止滚动替换')
    if not result['complete_chunk'] and result['plan_timeout_s'] > 15:
        raise ValueError('只有完整动作块对照模式允许单段超过15秒')
    bounds=result['task_bounds_deg']
    if bounds is not None:
        if not isinstance(bounds,list) or len(bounds)!=7:
            raise ValueError('任务范围必须有七对角度')
        for pair in bounds:
            if (not isinstance(pair,list) or len(pair)!=2 or
                    not all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in pair)
                    or pair[0]>=pair[1] or pair[0]<-180 or pair[1]>340):
                raise ValueError('任务关节范围无效')
    for key, maximum in maxima.items():
        v = result[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 < v <= maximum:
            raise ValueError(f'VLA运动配置{key}无效')
    if not isinstance(result['max_steps'], int) or not isinstance(result['action_steps'],int):
        raise ValueError('max_steps和action_steps必须是整数')
    if not result['lead_deg'] < result['abort_error_deg'] or result['settle_error_deg'] > result['abort_error_deg']:
        raise ValueError('VLA跟踪误差配置次序无效')
    if not result['gripper_lead_deg'] < result['gripper_abort_error_deg'] or result['gripper_settle_error_deg'] > result['gripper_abort_error_deg']:
        raise ValueError('VLA夹爪跟踪误差配置次序无效')
    return result


def load_motion_profile():
    path = Path(load_model_spec()['motion_profile'])
    if not path.exists():
        if path != PROFILE_PATH:
            raise ValueError('选定模型的运动配置不存在；拒绝使用其他任务范围')
        return dict(DEFAULT_PROFILE)
    return validate_profile(json.loads(path.read_text()))


def describe_profile(p):
    return (('闭爪/回撤候选前暂停对位，人工确认夹住后继续，' if p.get('grasp_gate') else
             f"顺序{p['action_steps']}点后重观察，4秒无推进停止，" if p.get('complete_chunk') else
             '运动中滚动重观察，' if p.get('rolling_replan') else '')+
            (f"每段最多{p['action_steps']}步预测，" if p['continuous_plan'] else f"每步指令≤{p['command_step_deg']:g}°，")+
            ('J1–J6短窗滤波已开（J7原样），' if p.get('action_filter_enabled') else '')+
            f"目标超前≤{p['lead_deg']:g}°，"
            f"速度{p['speed_deg_s']:g}°/s，"+
            ('按示范任务范围执行；' if p['task_bounds_deg'] is not None else f"起点±{p['session_span_deg']:g}°；")+
            f"最多{p['max_steps']}轮/{p['max_seconds']:g}秒；" +
            (("J7连续开合、" if p['continuous_plan'] else f"J7每步≤{p['gripper_step_deg']:g}°、")+
             f"速度{p['gripper_speed_deg_s']:g}°/s，沿用转矩控制"
             if p['gripper_enabled'] else 'J7保持'))


class PlanJoinError(ValueError):
    pass


def grasp_checkpoint_index(actions, current):
    """Conservative JOINT-space phase hints, never a grasp/Cartesian detector.

    Stop before a sustained closing candidate or a substantial J2 return
    after extension in this screwdriver task. False positives require human
    inspection; an absent hint is handled by the runner's approach budget.
    """
    peak_gripper, peak_j2 = current[6], current[1]
    for index, action in enumerate(actions):
        peak_gripper = max(peak_gripper, action[6])
        peak_j2 = max(peak_j2, action[1])
        future = actions[index:index+4]
        closing = peak_gripper >= 40 and len(future) >= 3 and sum(
            peak_gripper-a[6] >= 20 for a in future) >= 3
        returning = peak_j2 >= 80 and len(future) >= 3 and sum(
            peak_j2-a[1] >= 8 for a in future) >= 3
        if closing or returning:
            return index, 'closing_candidate' if closing else 'joint_return_candidate'
    return None, None


def replacement_ready(step, telemetry_step, request_id, profile):
    """Do not repeatedly replace the first few points before extension occurs."""
    if step.get('id') != request_id:
        return False
    if step.get('state') == 'complete':
        return True
    if step.get('state') != 'executing' or telemetry_step.get('id') != request_id:
        return False
    points = telemetry_step.get('points', 0)
    index = telemetry_step.get('point_index', -1)
    if type(points) is not int or not 1 <= points <= 50 or type(index) is not int:
        return False
    return index >= min(points-1, math.ceil(points*profile['replan_min_progress']))


def rolling_actions(actions, observed_command, current_command, profile, moving):
    """Geometric prefix alignment; no invented training FPS or Cartesian correction.

    Search only the consumed prefix, and include the observed command as its
    start. When stationary, never skip model points. Validate the FULL output
    before selecting a suffix so invalid unused points cannot be hidden.
    """
    if (not isinstance(actions, list) or not 1 <= len(actions) <= 50 or
            any(not isinstance(a, list) or len(a) != 7 or
                any(type(v) not in (int, float) or not math.isfinite(v) for v in a) for a in actions)):
        raise ValueError('模型输出不是有效的七维动作序列')
    prefix = actions[:profile['action_steps']]
    if not moving:
        return prefix, 0
    candidates = [observed_command] + prefix[:-1]
    dimensions = 7 if profile['gripper_enabled'] else 6
    def error(p):
        return max(abs(p[i]-current_command[i]) /
                   (profile['gripper_join_error_deg'] if i == 6 else profile['join_error_deg'])
                   for i in range(dimensions))
    index = min(range(len(candidates)), key=lambda n: error(candidates[n]))
    if error(candidates[index]) > 1:
        raise PlanJoinError('新预测与当前轨迹无法衔接，已丢弃并重新观察')
    return prefix[index:], index


def validate_plan_replacement(request, active, profile):
    """Only an explicit ID-matched rolling request may replace a path."""
    previous = request.get('replace_id')
    if previous is not None:
        if (not profile['rolling_replan'] or request.get('trajectory_deg') is None or
                not isinstance(previous, str) or previous != active.get('id') or
                active.get('state') not in ('executing', 'complete')):
            raise ValueError('滚动轨迹对应的旧计划已改变，拒绝接续')
    elif active.get('state') == 'executing':
        raise ValueError('已有模型轨迹执行中，缺少接续确认')
    return previous is not None and active.get('state') == 'executing'


class VlaPlan:
    """Advance a finite model path in the hardware loop without stopping at each point.

    The original Trajectory enforces speed, acceleration and measured command
    lead on every real I/O cycle. Progress is tied to that bounded command,
    so a slow or blocked joint does not silently consume the rest of the path.
    """
    def __init__(self, actions, actual, command, origin, bounds, profile):
        self.profile = validate_profile(profile)
        profile=self.profile
        if (not isinstance(actions, list) or not 1 <= len(actions) <= 50 or
                any(not isinstance(a, list) or len(a) != 7 or
                    any(not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(x) for x in a)
                    for a in actions)):
            raise ValueError('连续预测必须有1至50个有限的七维动作')
        if profile['complete_chunk'] and (len(actions) > profile['action_steps'] or
                (not profile['grasp_gate'] and len(actions) != profile['action_steps'])):
            raise ValueError('顺序轨迹长度与当前模型执行窗口不符')
        for v in (actual, command, origin):
            if not isinstance(v, (list, tuple)) or len(v) != 7 or not all(math.isfinite(x) for x in v):
                raise ValueError('连续轨迹参考姿态无效')
        if len(bounds) != 7:
            raise ValueError('连续轨迹软限位无效')
        self.bounds=[]
        for i, pair in enumerate(bounds):
            if len(pair)!=2 or not all(math.isfinite(x) for x in pair) or pair[0]>pair[1]:
                raise ValueError('连续轨迹软限位无效')
            gripper=i==6
            threshold=profile['gripper_abort_error_deg'] if gripper else profile['abort_error_deg']
            if abs(command[i]-actual[i])>threshold:
                raise ValueError(f'J{i+1}跟踪误差过大，拒绝连续轨迹')
            span=profile['gripper_span_deg'] if gripper else profile['session_span_deg']
            lo,hi=max(pair[0],origin[i]-span),min(pair[1],origin[i]+span)
            if profile['task_bounds_deg'] is not None:
                lo=max(lo,profile['task_bounds_deg'][i][0]);hi=min(hi,profile['task_bounds_deg'][i][1])
            if lo>hi:
                raise ValueError(f'J{i+1}连续轨迹范围无交集')
            self.bounds.append([lo,hi])
        self.actions=[[max(self.bounds[i][0],min(self.bounds[i][1],a[i]))
                       if i<6 or profile['gripper_enabled'] else command[i] for i in range(7)] for a in actions]
        bounded_actions = self.actions
        if profile['action_filter_enabled']:
            self.actions = filter_action_path(bounded_actions, profile['action_filter_max_delta_deg'])
        self.filter_info = dict(enabled=profile['action_filter_enabled'],
                                method='median3_triangle3_bounded', points=len(actions),
                                max_correction_deg=max(abs(a[j]-b[j])
                                    for a,b in zip(bounded_actions,self.actions) for j in range(6)),
                                endpoints_preserved=True, gripper_filtered=False)
        self.index=0
        self.last_progress=time.monotonic()

    def goal(self, command):
        # Advance at most once per real cycle; never add synthetic I/O samples.
        near=all(abs(command[i]-self.actions[self.index][i]) <= (8. if i==6 else 1.5)
                 for i in range(7 if self.profile['gripper_enabled'] else 6))
        if near and self.index<len(self.actions)-1:
            self.index+=1
            self.last_progress=time.monotonic()
        return list(self.actions[self.index])

    def stalled(self):
        return (self.profile['complete_chunk'] and
                time.monotonic()-self.last_progress >= self.profile['progress_timeout_s'])

    def final_reached(self, command, velocity):
        return self.index==len(self.actions)-1 and all(
            abs(command[i]-self.actions[-1][i])<.03 and abs(velocity[i])<.05
            for i in range(7 if self.profile['gripper_enabled'] else 6))
