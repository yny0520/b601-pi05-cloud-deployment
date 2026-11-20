"""Cancellable, slow, bounded live-model experiments using the original controller.

No model or CAN calls occur on the HTTP thread. Stopping cancels the inference
subprocess and signals the control loop independently of inference completion.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import tempfile
import threading
import time
import uuid

import cv2
from b601_vla_inference_client import InferenceClient, Cancelled, latest_worker_progress
from b601_vla_model import load_model_spec, check_collection
from b601_vla_motion import (DEFAULT_PROFILE, load_motion_profile, validate_profile, describe_profile,
                            rolling_actions, PlanJoinError, replacement_ready, grasp_checkpoint_index)

PYTHON = Path('/home/jetson/miniforge3/envs/lerobot/bin/python')
DIAGNOSTICS = Path('/home/jetson/Desktop/b601-rs/diagnostics/vla-inference')



class VlaRunner:
    def __init__(self, teleop, recorder, frames, predictor=None, motion_profile=None):
        self.teleop, self.recorder, self.frames = teleop, recorder, frames
        self.predictor = predictor or self._predict
        self.injected_predictor = predictor is not None
        self.configured_profile = motion_profile
        self.motion_profile = dict(DEFAULT_PROFILE)
        self.lock = threading.RLock()
        self.cancel = threading.Event()
        self.thread = None
        self.inference_client = InferenceClient()
        self.run_dir = None
        self.recent_plans = []
        self.retry_request = None
        self.retry_generation = 0
        self.grasp_pending = None
        self.model_spec = load_model_spec()
        self.state = {'state': 'idle', 'active': False, 'message': '尚未启动真机测试', 'steps': 0}

    def status(self):
        with self.lock:
            return copy.deepcopy(self.state)

    def _set(self, **values):
        with self.lock:
            if self.cancel.is_set():
                raise Cancelled()
            self.state.update(values)

    def _check(self):
        if self.cancel.is_set():
            raise Cancelled()

    def _sensors(self, require_hold=True):
        self._check()
        if not self.injected_predictor and load_model_spec() != self.model_spec:
            raise RuntimeError('模型部署配置已改变，停止并保持后重新加载')
        status = self.teleop.status()
        with self.recorder.lock:
            if self.recorder.active is not None or self.recorder.finalizing is not None:
                raise RuntimeError('先结束本集数据采集，再启动VLA测试')
            packet = copy.deepcopy(self.recorder.latest_telemetry or {})
        snapshots = {k: self.frames.snapshot(k, 300) for k in ('top', 'hand')}
        if any(v is None for v in snapshots.values()):
            missing = [k for k, v in snapshots.items() if v is None]
            details = {}
            # Preserve a diagnostic age only; stale frames never become inputs.
            for name in missing:
                old = self.frames.snapshot(name, 60_000)
                details[name] = None if old is None else round((time.monotonic_ns()-old[1])/1e6, 1)
            self._record('camera_stale', missing=missing, age_ms=details)
            names = '、'.join({'top': '顶置彩色', 'hand': '腕部彩色'}[k] for k in missing)
            raise RuntimeError(names+'相机没有300ms内的新画面，已停止并保持；恢复后需重新启动')
        if not require_hold:
            return status, packet, snapshots
        if not status['running'] or status['state'] != 'paused':
            raise RuntimeError('从臂不再处于暂停保持；本次VLA测试已终止')
        age = time.monotonic_ns()-packet.get('publish_mono_ns', 0)
        if (packet.get('context', {}).get('process_id') != status['pid'] or not 0 <= age <= 300_000_000
                or not all(packet.get(k) for k in ('live', 'tracking', 'paused'))):
            raise RuntimeError('暂停保持的从臂真实反馈陈旧或状态不匹配')
        for field in ('follower_deg', 'command_deg'):
            v = packet.get(field)
            if not isinstance(v, list) or len(v) != 7 or not all(isinstance(a, (int, float)) and math.isfinite(a) for a in v):
                raise RuntimeError('从臂七关节反馈或指令无效')
        return status, packet, snapshots

    def start(self, task='网线插接', action_steps=None):
        with self.lock:
            if self.thread is not None and self.thread.is_alive():
                raise RuntimeError('VLA测试或停止清理尚未结束')
            self.cancel.clear()
            self.model_spec = load_model_spec()
            remote = self.model_spec.get('policy_type') == 'pi05_remote'
            if not self.injected_predictor and remote:
                self.inference_client.close()
                self.inference_client = InferenceClient(
                    command=['/usr/bin/python3', str(Path(__file__).with_name('b601_pi05_remote_worker.py'))])
            if not self.injected_predictor:
                check_collection(self.model_spec, self.recorder.collection_profile['id'])
                task = self.model_spec['task']
            self.motion_profile = (validate_profile(self.configured_profile)
                                   if self.configured_profile is not None else load_motion_profile())
            if action_steps is not None:
                choices = (5, 10, 15, 25, 50) + ((16,) if remote and not self.injected_predictor else ())
                if type(action_steps) is not int or action_steps not in choices:
                    raise ValueError('重新观察步数请选择5、10、15、25或50')
                if (not self.injected_predictor and self.model_spec.get('policy_type') == 'act' and
                        action_steps != self.motion_profile['action_steps']):
                    raise ValueError('ACT必须按当前配置的完整观察窗口执行')
                if not self.motion_profile['continuous_plan']:
                    raise ValueError('当前控制配置未启用连续轨迹')
                self.motion_profile = validate_profile({**self.motion_profile, 'action_steps': action_steps})
            if (not self.injected_predictor and self.model_spec.get('policy_type') == 'act' and
                    (self.motion_profile['action_steps'] not in (5, 25, 50) or
                     not self.motion_profile['complete_chunk'])):
                raise ValueError('ACT须使用固定顺序执行观察窗口')
            self._sensors(require_hold=False)
            if not self.injected_predictor and remote and (self.motion_profile['action_steps'] != 16 or not self.motion_profile['complete_chunk']):
                raise ValueError('Pi0.5必须按完整16点窗口执行')
            if not self.injected_predictor and not remote and (not PYTHON.exists() or not (Path(self.model_spec['model_dir'])/'model.safetensors').is_file()):
                raise RuntimeError('本机模型权重或LeRobot环境不存在')
            ts = self.teleop.status()
            if ts['running'] and ts['state'] != 'paused':
                raise RuntimeError('已有遥操作在运行，请先暂停或停止，再启动模型测试')
            if ts['running'] and ts.get('autonomous') is False:
                raise RuntimeError('当前连接为人工遥操作；先托稳从臂并停止，再点模型启动以连接自主控制器')
            self.state = {'id': uuid.uuid4().hex, 'active': True, 'state': 'connecting',
                          'steps': 0, 'message': '正在连接；将保持实测姿态，不跟随主臂',
                          'limits': describe_profile(self.motion_profile),
                          'motion_profile': dict(self.motion_profile)}
            self.state['model'] = dict(self.model_spec)
            self.run_dir = None if self.injected_predictor else DIAGNOSTICS.parent/'vla-live-runs'/self.state['id']
            self.recent_plans = []
            self.retry_request = None
            self.retry_generation = 0
            self.grasp_pending = None
            if self.run_dir is not None:
                self.run_dir.mkdir(parents=True, exist_ok=False)
                self.state['record_path'] = str(self.run_dir)
                self._record('start', task=task, profile=self.motion_profile)
            self.thread = threading.Thread(target=self._run, args=(str(task)[:300],), daemon=True)
            self.thread.start()
            return self.status()

    def grasp_command(self, body):
        """Queue one explicit human action for the current stationary checkpoint."""
        with self.lock:
            self._check()
            if (not self.state.get('active') or self.state.get('state') != 'awaiting_grasp' or
                    not self.state.get('checkpoint_ready') or self.grasp_pending is not None or
                    body.get('run_id') != self.state.get('id') or
                    body.get('checkpoint_id') != self.state.get('checkpoint_id')):
                raise ValueError('当前没有可操作的抓取检查点；请刷新状态')
            op = body.get('operation')
            if op not in ('nudge', 'close', 'open', 'confirm', 'reobserve'):
                raise ValueError('未知抓取操作')
            if op == 'confirm' and not self.state.get('grasp_close_done'):
                raise ValueError('请先对位并收紧夹爪，人工核实夹住后再继续')
            if op == 'nudge' and (type(body.get('joint')) is not int or not 1 <= body['joint'] <= 6 or
                    type(body.get('delta_deg')) not in (int, float) or
                    not math.isfinite(body['delta_deg']) or not 0 < abs(body['delta_deg']) <= 1):
                raise ValueError('微调仅允许J1–J6每次最多1°')
            self._sensors()
            self.grasp_pending = dict(body)
            self.state.update(checkpoint_ready=False, message='正在处理抓取操作；可立即停止')
            return self.status()

    def retry_grasp(self):
        """Human-triggered hold/reobserve. Keep weights, never add a blind reach."""
        with self.lock:
            if not self.state.get('active') or not self.motion_profile['rolling_replan']:
                raise RuntimeError('先启动滚动模型执行，再使用抓空重观察')
            status = self.teleop.status()
            if not status['running'] or status['state'] != 'paused':
                raise RuntimeError('自主控制器尚未就绪')
            self.teleop.abort_vla_step()
            status = self.teleop.status()
            self.retry_request = {'epoch': status['control_epoch'], 'pid': status['pid']}
            self.retry_generation += 1
            self.state.update(message='已请求原地保持，丢弃旧预测；保留模型后重新观察尝试', state='retrying')
            self._record('human_retry', control_epoch=status['control_epoch'])
            return self.status()

    def report_grasp_miss(self):
        """Cancel first, then save human-labelled evidence; never move or label success."""
        with self.lock:
            if self.run_dir is None:
                raise RuntimeError('尚无本次模型诊断记录')
            self.stop('已标记抓空并请求保持；正在保存现场')
            with self.recorder.lock:
                packet = copy.deepcopy(self.recorder.latest_telemetry or {})
            now = time.monotonic_ns()
            folder = self.run_dir / ('grasp-miss-' + uuid.uuid4().hex[:12])
            try:
                folder.mkdir()
                status = self.teleop.status()
                evidence = {'label': 'human_reported_grasp_miss', 'mono_ns': now,
                            'telemetry': packet,
                            'telemetry_age_ms': (now-packet['publish_mono_ns'])/1e6
                                if packet.get('publish_mono_ns') else None,
                            'controller_running': status['running'], 'controller_pid': status['pid'],
                            'telemetry_matches_controller': bool(status['running'] and
                                packet.get('context', {}).get('process_id') == status['pid']),
                            'images': {}, 'note': '人工标记；停止请求先于拍照。不是成功示范或精确空间偏差。'}
                for name in ('top', 'hand'):
                    frame = self.frames.snapshot(name, 300)
                    if frame is None:
                        evidence['images'][name] = {'available': False}
                    else:
                        filename = name + '.jpg'
                        if not cv2.imwrite(str(folder/filename), frame[0]):
                            raise OSError('无法保存' + name + '现场画面')
                        evidence['images'][name] = {'available': True, 'file': filename,
                                                   'received_mono_ns': frame[1]}
                (folder/'evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2))
                self._record('grasp_miss', path=str(folder), evidence=evidence)
            except Exception as exc:
                self.state['message'] = '抓空停止请求已发送，但现场保存失败：' + str(exc)
                raise
            self.state.update(grasp_miss_path=str(folder),
                              message='已标记抓空并请求保持；现场已保存：' + str(folder))
            return self.status()

    def _record_plan_motion(self, samples):
        """Warn about recurring joint poses without claiming visual grasp progress."""
        self.recent_plans.append(samples)
        self.recent_plans = self.recent_plans[-4:]
        if len(self.recent_plans) < 4:
            return
        ends = [s[-1] for s in self.recent_plans]
        positions = [p for s in self.recent_plans for p in s]
        endpoint_spread = max(max(p[i] for p in ends)-min(p[i] for p in ends) for i in range(6))
        excursion = max(max(p[i] for p in positions)-min(p[i] for p in positions) for i in range(6))
        warning = ('最近4段反复回到相近关节姿态，请看画面核对是否抓空；可点“抓空暂停并保存现场”。'
                   if endpoint_spread < 6 and excursion > 12 else '')
        self._set(repetition_warning=warning)
        self._record('motion_pattern', endpoint_spread_deg=endpoint_spread,
                     joint_excursion_deg=excursion, warning=warning)

    def stop(self, message='已请求停止；取消推理和后续动作，从臂保持'):
        with self.lock:
            self.cancel.set()
            self.state.update(active=False, state='stopping' if self.thread and self.thread.is_alive() else 'stopped',
                              message=message)
            status = self.teleop.status()
            if status['running']:
                if status['state'] in ('starting', 'waiting', 'resuming'):
                    self.teleop.stop()  # No late enable after a cancelled startup.
                else:
                    self.teleop.abort_vla_step()
            return self.status()

    def _wait(self, predicate, seconds):
        deadline = time.monotonic()+seconds
        while time.monotonic() < deadline:
            self._check()
            result = predicate()
            if result:
                return result
            self.cancel.wait(.05)
        raise RuntimeError('等待控制程序确认超时；已停止VLA测试')

    def _wait_abort_hold(self):
        # The abort file may be acknowledged before the manager receives the
        # new paused telemetry. Wait for both before asking _sensors to sample.
        self._wait(lambda: self.teleop.status()['state'] == 'paused' and
                   self.teleop.vla_step_status().get('state') not in ('pending', 'executing'), 2)

    def _settle_step(self, target, start, origin):
        deadline = time.monotonic() + 4
        while True:
            _, feedback, _ = self._sensors()
            actual = feedback['follower_deg']
            errors = [abs(actual[i]-target[i]) for i in range(6)]
            self._set(actual_deg=list(actual),
                      actual_delta_deg=[actual[i]-start[i] for i in range(6)],
                      net_motion_deg=[actual[i]-origin[i] for i in range(6)],
                      tracking_error_deg=errors)
            gripper_error = abs(actual[6]-target[6])
            self._set(gripper_motion_deg=actual[6]-origin[6], gripper_tracking_error_deg=gripper_error)
            if (max(errors) <= self.motion_profile['settle_error_deg'] and
                    (not self.motion_profile['gripper_enabled'] or gripper_error <= self.motion_profile['gripper_settle_error_deg'])):
                return
            if time.monotonic() >= deadline:
                if self.motion_profile['gripper_enabled'] and gripper_error > self.motion_profile['gripper_settle_error_deg']:
                    raise RuntimeError(f'J7指令已发完但实测未到位：目标{target[6]:.2f}°，'
                                       f'实测{actual[6]:.2f}°，误差{gripper_error:.2f}°；已保持')
                joint = max(range(6), key=errors.__getitem__)
                raise RuntimeError(f'J{joint+1}指令已发完但实测未到位：'
                                   f'目标{target[joint]:.2f}°，实测{actual[joint]:.2f}°，'
                                   f'误差{errors[joint]:.2f}°；已停止后续动作并保持')
            self.cancel.wait(.05)

    def _wait_stationary(self):
        """Allow gravity/load settling after H before taking the scene/state."""
        previous, since = None, time.monotonic()
        def stable():
            nonlocal previous, since
            _, packet, _ = self._sensors()
            actual = packet['follower_deg']
            if previous is None or max(abs(a-b) for a,b in zip(actual[:6],previous[:6])) > .15:
                previous, since = list(actual), time.monotonic()
            if self.motion_profile['gripper_enabled'] and abs(actual[6]-previous[6]) > .5:
                previous, since = list(actual), time.monotonic()
            return time.monotonic()-since >= .6
        self._wait(stable, 8)

    def _execute_plan(self, actions, current, origin, epoch, pid, deadline):
        actions=actions[:self.motion_profile['action_steps']]
        execution_started = time.monotonic()
        nominal_seconds = len(actions)/self.model_spec['fps']
        samples=[list(current['follower_deg'])]
        with self.lock:
            self._check()
            self._sensors()
            request=self.teleop.submit_vla_step(actions[0],current['follower_deg'],origin,
                                               trajectory_deg=actions)
            self.state.update(state='executing',message=f'连续执行{len(actions)}步预测，伸臂和夹爪共同推进；可立即停止',
                              step_id=request['id'],plan_index=0,plan_points=len(actions))
            self._record('submitted_plan', round=self.state.get('rounds'), step_id=request['id'],
                         points=len(actions), skipped_points=0, execution='sequential', target_deg=actions[-1],
                         model_fps=self.model_spec['fps'], nominal_seconds=nominal_seconds)
        last_recorded=-1
        last_progress=time.monotonic()
        def completed():
            nonlocal last_recorded, last_progress
            status,feedback,_=self._sensors()
            if status['control_epoch']!=epoch or status['pid']!=pid:
                raise RuntimeError('连续执行期间控制会话改变')
            if time.monotonic()>=deadline:
                raise RuntimeError('达到本次执行时间上限，停止并保持')
            actual=feedback['follower_deg']
            active=feedback.get('vla_step',{})
            self._set(actual_deg=list(actual),net_motion_deg=[actual[i]-origin[i] for i in range(6)],
                      actual_delta_deg=[actual[i]-current['follower_deg'][i] for i in range(6)],
                      gripper_motion_deg=actual[6]-origin[6],
                      tracking_error_deg=[abs(actual[i]-feedback['command_deg'][i]) for i in range(6)],
                      gripper_tracking_error_deg=abs(actual[6]-feedback['command_deg'][6]))
            if active.get('id')==request['id']:
                self._set(plan_index=active.get('point_index',0))
                if active.get('point_index',0)!=last_recorded:
                    last_recorded=active.get('point_index',0)
                    last_progress=time.monotonic()
                    samples.append(list(actual))
                    self._record('plan_progress',step_id=request['id'],point_index=last_recorded,actual_deg=actual,command_deg=feedback['command_deg'])
            step=self.teleop.vla_step_status()
            if step.get('id')!=request['id']:
                return False
            if step.get('state') in ('rejected','aborted'):
                raise RuntimeError(step.get('message') or '连续轨迹已停止')
            if (self.motion_profile['complete_chunk'] and step.get('state') != 'complete' and
                    time.monotonic()-last_progress >= self.motion_profile['progress_timeout_s']):
                raise RuntimeError('连续轨迹无推进，停止并保持；请检查夹爪接触或关节跟踪')
            return step if step.get('state')=='complete' else False
        step=self._wait(completed,self.motion_profile['plan_timeout_s']+3)
        self._settle_step(step['target_deg'],current['follower_deg'],origin)
        self._set(steps=self.status()['steps']+len(actions),plan_index=len(actions))
        samples.append(list(self.status()['actual_deg']))
        self._record_plan_motion(samples)
        actual_seconds = time.monotonic()-execution_started
        self._set(last_plan_timing={'nominal_seconds': nominal_seconds, 'actual_seconds': actual_seconds,
                                   'dilation': actual_seconds/nominal_seconds})
        self._record('executed_plan',points=len(actions),target_deg=step['target_deg'],actual_deg=self.status().get('actual_deg'),
                     step_id=request['id'], nominal_seconds=nominal_seconds, actual_seconds=actual_seconds,
                     dilation=actual_seconds/nominal_seconds)

    def _predict_held(self, packet, snapshots, task, epoch, pid, deadline):
        """Keep observing health while inference runs; never submit from this thread."""
        finished = threading.Event()
        result = {}
        def predict():
            try:
                result['prediction'] = self.predictor(packet, snapshots, task, self.cancel)
            except BaseException as exc:
                result['error'] = exc
            finally:
                finished.set()
        threading.Thread(target=predict, daemon=True).start()
        while not finished.wait(.05):
            status, _, _ = self._sensors()
            if status['control_epoch'] != epoch or status['pid'] != pid:
                raise RuntimeError('推理期间控制会话改变，丢弃预测')
            if time.monotonic() >= deadline:
                raise RuntimeError('达到本次执行时间上限，停止并保持')
        self._check()
        if 'error' in result:
            raise result['error']
        return result['prediction']

    def _grasp_checkpoint(self, origin, epoch, pid, deadline, reason):
        """Discard the unused policy tail; hold until a human checks the grasp.

        No uncalibrated XYZ offsets or torque-based success labels. Joint
        nudges are explicit, bounded, and sent through the same CAN trajectory.
        """
        _, packet, snapshots = self._sensors()
        anchor = list(packet['command_deg'])
        checkpoint_id = uuid.uuid4().hex
        checkpoint_messages = {
            'closing_candidate': '模型预测到闭爪候选，已在执行闭爪前暂停并保持。先核对夹爪是否围住螺丝刀；重新观察不会执行夹爪动作，若新预测仍要闭爪会再次暂停。',
            'joint_return_candidate': '模型预测到手臂回撤候选，已暂停并保持。先核对是否确实夹住物体；重新观察不会自动确认抓取。',
            'approach_inspection_budget': '已达到接近阶段检查点，暂停并保持。先核对夹爪与物体位置，再决定是否微调、收紧或重新观察。',
        }
        self._set(state='awaiting_grasp', checkpoint_id=checkpoint_id, checkpoint_ready=True,
                  grasp_close_done=False, grasp_confirmed=False,
                  checkpoint_anchor_deg=anchor, checkpoint_command_deg=anchor,
                  message=checkpoint_messages.get(reason, '抓取检查点：保持当前位置。先核对夹爪与物体；确认夹住后才继续搬运。'))
        self._record('grasp_checkpoint', reason=reason, checkpoint_id=checkpoint_id,
                     actual_deg=packet['follower_deg'], command_deg=anchor)
        if self.run_dir is not None:
            for name, frame in snapshots.items():
                if not cv2.imwrite(str(self.run_dir/f'grasp-{checkpoint_id}-{name}.jpg'), frame[0]):
                    raise RuntimeError('抓取检查点现场保存失败，保持并停止')
        while True:
            self._check()
            status, packet, snapshots = self._sensors()
            if status['control_epoch'] != epoch or status['pid'] != pid:
                raise RuntimeError('抓取检查期间控制会话改变，已停止')
            if time.monotonic() >= deadline:
                raise RuntimeError(f"抓取等待达到本次{self.motion_profile['max_seconds']:g}秒上限，已保持；未自动继续搬运")
            with self.lock:
                command, self.grasp_pending = self.grasp_pending, None
            if command is None:
                self.cancel.wait(.05)
                continue
            op = command['operation']
            self._record('human_grasp_operation', operation=op, checkpoint_id=checkpoint_id,
                         joint=command.get('joint'), delta_deg=command.get('delta_deg'))
            if op in ('confirm', 'reobserve'):
                if op == 'confirm':
                    self._record('human_confirmed_grasp', checkpoint_id=checkpoint_id,
                                 actual_deg=packet['follower_deg'], command_deg=packet['command_deg'],
                                 source='explicit_human_confirmation_not_automatic_detection')
                    if self.run_dir is not None:
                        for name, frame in snapshots.items():
                            if not cv2.imwrite(str(self.run_dir/f'confirmed-{checkpoint_id}-{name}.jpg'), frame[0]):
                                raise RuntimeError('人工抓持确认画面保存失败，保持并停止')
                self._set(checkpoint_ready=False, checkpoint_id=None,
                          grasp_confirmed=op == 'confirm',
                          message='已记录人工确认夹住，丢弃旧尾段并重新拍照' if op == 'confirm'
                          else '尚未确认抓住；丢弃旧尾段，重新观察接近')
                return op == 'confirm'
            target = list(packet['command_deg'])
            if op == 'nudge':
                i = command['joint']-1
                target[i] += command['delta_deg']
                if abs(target[i]-anchor[i]) > 10.00001:
                    raise RuntimeError('对位微调超出检查点±10°；已保持，请重新检查场景')
            else:
                target[6] += -15 if op == 'close' else 15
            bounds = self.motion_profile['task_bounds_deg']
            if bounds is not None:
                target = [max(bounds[i][0], min(bounds[i][1], target[i])) for i in range(7)]
            self._execute_plan([target], packet, origin, epoch, pid, deadline)
            self._set(state='awaiting_grasp', checkpoint_ready=True,
                      checkpoint_command_deg=target, grasp_close_done=op == 'close',
                      message='已完成微调/开合，保持当前位置。请看实物；确认夹住后再点继续。')

    def _run_rolling(self, task, origin, epoch, pid, deadline):
        """One prediction in flight while the bounded CAN loop runs independently.

        Fresh results replace only the named preceding path. No guessed FPS,
        no extension offset, no predicted grasp-success label.
        """
        request = None
        progress = 0
        failures = 0
        last_capture = None

        def sample():
            nonlocal progress
            while True:
                apply_retry()  # Acknowledgement wait must not hold the HTTP stop lock.
                with self.lock:
                    if self.retry_request is not None:
                        continue
                    status, packet, snapshots = self._sensors()
                    if status['pid'] != pid or status['control_epoch'] != epoch:
                        raise RuntimeError('滚动执行期间控制会话改变，已取消后续预测')
                    packet['_retry_generation'] = self.retry_generation
                    break
            if time.monotonic() >= deadline:
                raise RuntimeError('达到本次执行时间上限，已保持')
            step = self.teleop.vla_step_status()
            if request and step.get('id') == request['id'] and step.get('state') in ('rejected', 'aborted'):
                raise RuntimeError(step.get('message') or '滚动轨迹已中止')
            active = packet.get('vla_step', {})
            point = (active.get('point_index', 0) if active.get('id') == (request or {}).get('id') else progress)
            if request and step.get('id') == request['id'] and step.get('state') == 'complete':
                point = step.get('points', progress)
            if point > progress:
                self._set(steps=self.status()['steps']+point-progress)
                progress = point
                self._record('plan_progress', step_id=request['id'], point_index=point,
                             actual_deg=packet['follower_deg'], command_deg=packet['command_deg'])
            actual = packet['follower_deg']
            self._set(actual_deg=actual, net_motion_deg=[actual[i]-origin[i] for i in range(6)],
                      gripper_motion_deg=actual[6]-origin[6], plan_index=point,
                      tracking_error_deg=[abs(actual[i]-packet['command_deg'][i]) for i in range(6)],
                      gripper_tracking_error_deg=abs(actual[6]-packet['command_deg'][6]))
            return status, packet, snapshots

        def apply_retry():
            nonlocal epoch, request, progress
            with self.lock:
                retry = self.retry_request
                if retry is None:
                    return False
                self._check()
                status = self.teleop.status()
                if status['pid'] != pid or retry['pid'] != pid or status['control_epoch'] != retry['epoch']:
                    raise RuntimeError('重观察期间控制会话改变')
                epoch = retry['epoch']
                self.retry_request = None
                request, progress = None, 0
            # Wait for the control loop to acknowledge P, not just the HTTP request.
            self._wait_abort_hold()
            self._set(state='retrying', message='已原地保持；正在用新画面重新预测，模型无需重载')
            return True

        for index in range(self.motion_profile['max_steps']):
            if time.monotonic() >= deadline:
                break
            apply_retry()
            _, packet, snapshots = sample()
            prediction_generation = packet['_retry_generation']
            captured = time.monotonic()
            interval = None if last_capture is None else captured-last_capture
            last_capture = captured
            self._set(state='predicting', rounds=index+1, worker_progress=None,
                      observation_interval_s=interval, captured_mono_ns=time.monotonic_ns(),
                      message=f'第{index+1}轮滚动观察与预测；上一段由独立控制循环继续限速执行')
            result, errors, done = [], [], threading.Event()
            def predict():
                try:
                    result.append(self.predictor(packet, snapshots, task, self.cancel))
                except BaseException as exc:
                    errors.append(exc)
                finally:
                    done.set()
            worker = threading.Thread(target=predict, daemon=True)
            worker.start()
            retry_seen = False
            while not done.wait(.04):
                self._check()
                retry_seen = apply_retry() or retry_seen
                sample()  # Camera, feedback, plan and stop monitoring during inference.
            self._check()
            retry_seen = apply_retry() or retry_seen
            if errors:
                raise errors[0]
            if retry_seen or prediction_generation != self.retry_generation:
                self._record('discarded_prediction', reason='human_retry', round=index+1)
                continue
            _, current, _ = sample()
            prediction = result[0]
            age = time.monotonic()-captured
            self._set(prediction=prediction, prediction_age_s=age)
            self._record('prediction', round=index+1, input_state_deg=packet['follower_deg'],
                         input_command_deg=packet['command_deg'], age_s=age,
                         observation_interval_s=interval, prediction=prediction)
            if age > self.motion_profile['prediction_max_age_s']:
                self._record('discarded_prediction', reason='stale', age_s=age)
                if request:
                    raise RuntimeError(f'预测耗时{age:.1f}秒，旧画面结果已丢弃；停止并保持')
                failures += 1
                if failures >= 3:
                    raise RuntimeError('连续3次预测超过新鲜度上限，未提交运动；检查GPU推理')
                self._set(message='首次推理预热结果已丢弃；重新拍照，不执行旧画面动作')
                continue
            actions = prediction.get('action_chunk_deg', [prediction.get('action_deg')])
            # Validate even a deferred full output; no malformed suffix is hidden.
            rolling_actions(actions, packet['command_deg'], current['command_deg'],
                            self.motion_profile, moving=False)
            if request and not replacement_ready(self.teleop.vla_step_status(), current.get('vla_step', {}),
                                                  request['id'], self.motion_profile):
                self._set(deferred_updates=self.status().get('deferred_updates', 0)+1,
                          message='原轨迹仍在伸臂，暂缓替换；继续取新画面，避免反复截断开头')
                self._record('deferred_prediction', round=index+1, reason='minimum_commit',
                             step_id=request['id'], point_index=progress)
                continue
            try:
                path, skipped = rolling_actions(actions, packet['command_deg'], current['command_deg'],
                                                self.motion_profile, moving=request is not None)
            except PlanJoinError:
                # Invalid model arrays are errors; only geometric join failures retry.
                if request is None:
                    raise
                failures += 1
                if failures >= 3:
                    raise RuntimeError('连续3轮新预测无法衔接，已保持；需要核对模型/现场偏差')
                with self.lock:
                    self._check()
                    self.teleop.abort_vla_step()
                    epoch = self.teleop.status()['control_epoch']
                self._wait_abort_hold()
                request, progress = None, 0
                self._record('discarded_prediction', reason='cannot_join', round=index+1)
                continue
            failures = 0
            with self.lock:
                self._check()
                # Retry/stop cannot interleave between this check and publication.
                if self.retry_request is not None:
                    continue
                _, current, _ = sample()
                kwargs = {'replace_id': request['id'], 'expected_mono_ns': current['publish_mono_ns']} if request else {}
                request = self.teleop.submit_vla_step(path[0], current['follower_deg'], origin,
                                                     trajectory_deg=path, **kwargs)
                progress = 0
                self.state.update(state='executing', step_id=request['id'], plan_index=0,
                                  plan_points=len(path), skipped_points=skipped,
                                  plans_submitted=self.state.get('plans_submitted', 0)+1,
                                  message='滚动轨迹已提交，下一轮观察与运动并行；可立即停止')
                self._record('submitted_plan', round=index+1, step_id=request['id'],
                             points=len(path), skipped_points=skipped, target_deg=path[-1])
            # Do not race a still-pending file with the next replan request.
            self._wait(lambda: self.teleop.vla_step_status().get('state') != 'pending', 2)

        # No unmonitored tail after budget expiration. Hold does not disable.
        with self.lock:
            self._check()
            self.teleop.abort_vla_step()
        self._set(active=False, state='complete', message='本次滚动执行达到时间或轮数上限，已请求保持；未自动判定抓取或插接成功')

    def _run(self, task):
        try:
            # Serialize enable requests against stop; the HTTP stop always wins
            # before a later inference result can be submitted.
            with self.lock:
                self._check()
                if not self.teleop.status()['running']:
                    self.teleop.start(auto_follow=False, autonomous=True)
            self._wait(lambda: self.teleop.status()['ready'] or
                       self._raise_if_exited(), 25)
            with self.lock:
                self._check()
                if self.teleop.status()['state'] != 'paused':
                    self.teleop.send_key('h')
            def held_ready():
                status = self.teleop.status()
                with self.recorder.lock:
                    packet = self.recorder.latest_telemetry or {}
                age = time.monotonic_ns()-packet.get('publish_mono_ns', 0)
                return (status['state'] == 'paused' and packet.get('paused') and packet.get('tracking')
                        and packet.get('context', {}).get('process_id') == status['pid']
                        and 0 <= age <= 300_000_000)
            self._wait(held_ready, 8)
            if not self.injected_predictor:
                self._set(state='loading', message='先加载模型；加载完成后重新拍照和读取关节，不执行旧画面预测')
                prepared = self.inference_client.request({'warmup_only': True, 'model_id': self.model_spec['id'], 'fallback_cpu': False}, self.cancel,
                    progress=lambda p: self._set(worker_progress=p, message=p.get('message', '模型加载中')))
                if prepared.get('prepared') is not True:
                    raise RuntimeError('模型预加载未确认，未提交动作')
                self._set(model_preparation=prepared)
                self._wait_stationary()
            first_status, first, _ = self._sensors()
            origin = list(first['follower_deg'])
            epoch = first_status['control_epoch']
            deadline = time.monotonic()+self.motion_profile['max_seconds']
            grasp_confirmed = False
            approach_points = 0
            if self.motion_profile['rolling_replan']:
                self._run_rolling(task, origin, epoch, first_status['pid'], deadline)
                return
            for index in range(self.motion_profile['max_steps']):
                if time.monotonic() >= deadline:
                    break
                captured = time.monotonic()
                before, packet, snapshots = self._sensors()
                if before['control_epoch'] != epoch or before['pid'] != first_status['pid']:
                    raise RuntimeError('控制会话已改变，本次模型测试结束')
                self._set(state='predicting', message=f'第{index+1}轮正在准备模型；实际推理设备尚未确认',
                          worker_progress=None,
                          captured_mono_ns=time.monotonic_ns())
                prediction = self._predict_held(packet, snapshots, task, epoch, before['pid'], deadline)
                self._check()
                if time.monotonic() >= deadline:
                    break  # An inference finishing late must not move the arm.
                after, current, _ = self._sensors()
                if after['control_epoch'] != epoch or after['pid'] != before['pid']:
                    raise RuntimeError('推理期间控制会话改变，丢弃预测')
                age = time.monotonic()-captured
                self._set(prediction=prediction, rounds=index+1, prediction_age_s=age)
                self._record('prediction', round=index+1, age_s=age,
                             input_state_deg=packet['follower_deg'], input_command_deg=packet['command_deg'],
                             prediction=prediction)
                if age > self.motion_profile['prediction_max_age_s']:
                    self._set(discarded_predictions=self.status().get('discarded_predictions', 0)+1,
                              message='预测超过时效，保持并重新拍照；未提交动作')
                    self._record('discarded_prediction', round=index+1, reason='age', age_s=age)
                    continue
                actions = prediction.get('action_chunk_deg', [prediction.get('action_deg')])
                if (not isinstance(actions, list) or not 1<=len(actions)<=50 or
                        any(not isinstance(a, list) or len(a) != 7 or
                            not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in a)
                            for a in actions)):
                    raise RuntimeError('模型输出不是有效的七维动作序列')
                expected_horizon = self.model_spec.get('action_horizon', 50) if not self.injected_predictor else 50
                if self.motion_profile['complete_chunk'] and len(actions) != expected_horizon:
                    raise RuntimeError(f'完整动作块模式要求模型输出{expected_horizon}点，拒绝不完整动作块')
                if (max(abs(a-b) for a,b in zip(current['follower_deg'][:6],packet['follower_deg'][:6])) > .5
                        or abs(current['follower_deg'][6]-packet['follower_deg'][6]) > (2 if self.motion_profile['gripper_enabled'] else .5)):
                    raise RuntimeError('推理期间从臂移动超过允许范围，丢弃预测')
                if self.motion_profile['continuous_plan']:
                    # Validate all 50 predictions above, then apply the model's
                    # execution horizon before checking phase/closing hints.
                    full_actions = actions
                    actions = actions[:self.motion_profile['action_steps']]
                    if self.motion_profile['grasp_gate'] and not grasp_confirmed:
                        cut, reason = grasp_checkpoint_index(full_actions, current['command_deg'])
                        if cut is not None and cut >= len(actions):
                            cut, reason = None, None
                        # A model with no detectable phase change must not run
                        # forever without someone inspecting the approach.
                        if cut is None and approach_points >= 100:
                            cut, reason = 0, 'approach_inspection_budget'
                        if cut is not None:
                            self._record('guarded_prediction', round=index+1, cut_index=cut,
                                         reason=reason, unused_points=len(actions)-cut)
                            if cut:
                                self._execute_plan(actions[:cut], current, origin, epoch, first_status['pid'], deadline)
                            grasp_confirmed = self._grasp_checkpoint(origin, epoch, first_status['pid'], deadline, reason)
                            approach_points = 0
                            continue  # Never resume the now stale unused tail.
                    self._execute_plan(actions,current,origin,epoch,first_status['pid'],deadline)
                    approach_points += len(actions)
                    if not self.injected_predictor:
                        self._wait_stationary()
                    continue
                chunk_started = time.monotonic()
                for action_index, action in enumerate(actions[:self.motion_profile['action_steps']]):
                    self._check()
                    if time.monotonic() >= deadline or time.monotonic()-chunk_started > 12:
                        break  # Reobserve rather than execute an old chunk indefinitely.
                    before_step, current, _ = self._sensors()
                    if before_step['control_epoch'] != epoch or before_step['pid'] != first_status['pid']:
                        raise RuntimeError('动作块执行期间控制会话改变')
                    with self.lock:
                        self._check()
                        self._sensors()
                        request = self.teleop.submit_vla_step(action, current['follower_deg'], origin)
                        self.state.update(state='executing', message=f'第{index+1}轮，动作{action_index+1}低速执行中；可立即停止',
                                          step_id=request['id'])
                    def completed():
                        self._sensors()
                        if self.teleop.status()['control_epoch'] != epoch:
                            raise RuntimeError('执行期间控制会话改变')
                        step = self.teleop.vla_step_status()
                        if step.get('id') != request['id']:
                            return False
                        if step.get('state') in ('rejected', 'aborted'):
                            raise RuntimeError(step.get('message') or '小步被中止')
                        return step if step.get('state') == 'complete' else False
                    step = self._wait(completed, 6)
                    self._settle_step(step['target_deg'], current['follower_deg'], origin)
                    self._set(steps=self.status()['steps']+1)
                    self._record('executed', round=index+1, action_index=action_index,
                                 target_deg=step['target_deg'], actual_deg=self.status().get('actual_deg'))
                if not self.injected_predictor:
                    self._wait_stationary()
                if (self.motion_profile['task_bounds_deg'] is None and
                        any(abs(step['target_deg'][i]-origin[i]) >= self.motion_profile['session_span_deg']-.01 for i in range(6))):
                    break
            motion = max(abs(v) for v in self.status().get('net_motion_deg', [0]))
            self._set(active=False, state='complete', message=
                      f'本次执行达到时间、轮数或局部行程上限，已保持；六关节最大净位移{motion:.2f}°。'
                      '尚未自动判定任务成功，请人工核对；可重新启动继续观察执行')
        except Cancelled:
            with self.lock:
                self.state.update(active=False, state='stopped', message='模型任务已取消，不再提交动作；请查看控制器保持反馈')
        except Exception as exc:
            with self.lock:
                # stop() sets cancellation before changing controller state.
                # A sensor check already in flight may observe that transition
                # and raise RuntimeError; preserve explicit cancellation while
                # retaining its diagnostic evidence. Real faults still fail.
                cancelled = self.cancel.is_set()
                self.state.update(active=False, state='stopped' if cancelled else 'error',
                                  message='模型任务已取消，不再提交动作；请查看控制器保持反馈'
                                  if cancelled else str(exc))
                self.cancel.set()
                if cancelled:
                    self._record('cancelled_during_validation', message=str(exc))
                status = self.teleop.status()
                if not cancelled and status['running']:
                    if status['state'] in ('starting', 'waiting', 'resuming'):
                        self.teleop.stop()
                    else:
                        self.teleop.abort_vla_step()
        finally:
            # Free resident weights after completion, cancel or failure. A later
            # start gets a clean model process and a fresh request namespace.
            try:
                self._record('end', status=self.status())
            finally:
                self.inference_client.close()

    def _record(self, event, **data):
        with self.lock:
            if self.run_dir is not None:
                with (self.run_dir/'events.jsonl').open('a') as stream:
                    stream.write(json.dumps({'event': event, 'mono_ns': time.monotonic_ns(), **data},
                                            ensure_ascii=False, allow_nan=False)+'\n')

    def _raise_if_exited(self):
        status = self.teleop.status()
        if not status['running']:
            raise RuntimeError(status.get('error') or '机械臂连接程序已退出')
        return False

    def _predict(self, packet, snapshots, task, cancel):
        with tempfile.TemporaryDirectory(prefix='b601-vla-run-') as tmp:
            paths = {k: str(Path(tmp)/(k+'.jpg')) for k in ('top', 'hand')}
            for k in paths:
                if not cv2.imwrite(paths[k], snapshots[k][0]):
                    raise RuntimeError('无法保存本轮相机快照')
            if self.run_dir is not None:
                number = self.status().get('rounds', 0) + (0 if self.motion_profile['rolling_replan'] else 1)
                for k in paths:
                    if not cv2.imwrite(str(self.run_dir/f'{number:04d}-{k}.jpg'), snapshots[k][0]):
                        raise RuntimeError('无法记录本轮诊断画面')
            request = {'state_deg': packet['follower_deg'], 'task': task,
                       'model_id': self.model_spec['id'], 'fallback_cpu': False,
                       'top_path': paths['top'], 'hand_path': paths['hand']}
            return self.inference_client.request(request, cancel,
                progress=lambda p: self._set(worker_progress=p, message=p.get('message', '模型推理中')))

    def close(self):
        self.stop()
        if self.thread:
            self.thread.join(timeout=4)
        if not self.thread or not self.thread.is_alive():
            self.inference_client.close()
