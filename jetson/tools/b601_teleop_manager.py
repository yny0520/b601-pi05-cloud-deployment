"""Launch the existing teleop in a private PTY; send only its R/P/Q keys."""
from __future__ import annotations

import codecs
from datetime import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import pty
import select
import signal
import subprocess
import threading
import time
import uuid
from b601_control_io import runtime_path, atomic_runtime_json

ROOT = Path('/home/jetson/Desktop/b601-rs')
WORKSPACE = Path('/home/jetson/Documents/ChatGPT/b601-rs')
LAUNCHER = ROOT/'启动完整遥操作.sh'
READY_MESSAGE = '遥操作硬件就绪，等待开始。'


class TeleopManager:
    def __init__(self, command=None, log_dir=None, telemetry_getter=None, stop_file=None):
        self.command = command or ['/bin/bash', str(LAUNCHER)]
        self.log_dir = Path(log_dir or ROOT/'teleop-logs')
        self.telemetry_getter = telemetry_getter
        self.stop_file = Path(stop_file or ROOT/'TELEOP_STOP')
        self.last_key_ns = 0
        self.control_epoch = 0
        self.lock = threading.RLock()
        self.proc = None
        self.fd = None
        self.reader = None
        self.ready = threading.Event()
        self.cancel = threading.Event()
        self.state = 'stopped'
        self.output_tail = ''
        self.log_path = None
        self.last_error = ''
        self.returncode = None
        self.autonomous = False
        self._log = None

    def _read_loop(self, proc, fd):
        decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
        pending = ''
        try:
            while True:
                if not select.select([fd], [], [], .2)[0]:
                    if proc.poll() is not None:
                        break
                    continue
                try:
                    data = os.read(fd, 4096)
                except BlockingIOError:
                    continue
                except OSError:
                    break
                if not data:
                    break
                text = decoder.decode(data)
                with self.lock:
                    self.output_tail = (self.output_tail+text)[-6000:]
                    if self._log:
                        self._log.write(text)
                        self._log.flush()
                    pending += text
                    while '\n' in pending:
                        line, pending = pending.split('\n', 1)
                        if READY_MESSAGE in line:
                            self.ready.set()
                            self.state = 'waiting'
                        elif '已开始；' in line:
                            self.state = 'following'
                            self.last_error = ''
                        elif '已暂停，' in line:
                            self.state = 'paused'
                        elif '故障保持：' in line:
                            self.state = 'fault_hold'
                            self.last_error = line.strip()
                        elif '错误：' in line:
                            self.last_error = line.strip()
                if len(pending)>8192:
                    pending = pending[-1024:]
        finally:
            proc.wait()
            with self.lock:
                if self.proc is proc:
                    self.returncode = proc.returncode
                    self.state = 'error' if proc.returncode and not self.cancel.is_set() else 'stopped'
                    if self.state=='error' and not self.last_error:
                        self.last_error = '遥操作异常退出，查看下方日志'
                    if self._log:
                        self._log.close()
                        self._log = None
                    try:
                        os.close(fd)
                    except OSError:
                        pass
                    self.fd = None

    def _auto_follow(self, proc, key='r'):
        deadline = time.monotonic()+20
        while not self.ready.wait(.1):
            if self.cancel.is_set() or proc.poll() is not None or time.monotonic()>deadline:
                return
        with self.lock:
            if self.proc is not proc or self.cancel.is_set() or proc.poll() is not None:
                return
            try:
                self.send_key(key)
            except RuntimeError as exc:
                self.last_error = str(exc)

    def start(self, auto_follow=True, autonomous=False):
        with self.lock:
            if autonomous and auto_follow:
                raise ValueError('自主模式不能自动跟随主臂')
            if self.proc is not None and self.proc.poll() is None:
                raise RuntimeError('遥操作已在运行，请使用暂停/继续')
            if self.reader is not None and self.reader.is_alive():
                raise RuntimeError('正在等待上一遥操作进程退出，请稍后重试')
            # Do not disturb the UART service if a CLI controller owns the arms.
            with (ROOT/'teleop.lock').open('a+') as probe:
                try:
                    fcntl.flock(probe, fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:
                    raise RuntimeError('已有终端遥操作在运行，请先在该终端按Q退出，再用网页启动')
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.log_path = self.log_dir/('web-launch-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f')+'.log')
            self._log = self.log_path.open('a', buffering=1)
            self.ready.clear()
            self.cancel.clear()
            self.output_tail = self.last_error = ''
            self.returncode = None
            self.state = 'starting'
            master, slave = pty.openpty()
            try:
                self.autonomous = autonomous
                command = self.command + (['--autonomous'] if autonomous else [])
                if autonomous:
                    from b601_vla_model import load_model_spec
                    command += ['--hz', str(load_model_spec()['fps'])]
                proc = subprocess.Popen(command, cwd=WORKSPACE, stdin=slave,
                                        stdout=slave, stderr=slave, start_new_session=True,
                                        close_fds=True)
            except Exception:
                os.close(master)
                self._log.close()
                self._log = None
                self.state = 'error'
                raise
            finally:
                os.close(slave)
            self.proc, self.fd = proc, master
            os.set_blocking(master, False)
            self.reader = threading.Thread(target=self._read_loop,args=(proc,master),daemon=True)
            self.reader.start()
            if auto_follow:
                threading.Thread(target=self._auto_follow,args=(proc,),daemon=True).start()
            return self.status()

    def send_key(self, key):
        if key not in ('r','p','q','h'):
            raise ValueError('只允许R/P/Q/H控制')
        with self.lock:
            if key == 'r' and self.autonomous:
                raise RuntimeError('当前为自主模型控制；要遥操作请先停止并重新连接')
            if self.proc is None or self.proc.poll() is not None or self.fd is None:
                raise RuntimeError('遥操作未运行，先点击启动')
            if key!='q' and not self.ready.is_set():
                raise RuntimeError('正在连接机械臂，请等待检查完成')
            if self.cancel.is_set() and key!='q':
                raise RuntimeError('遥操作正在停止')
            try:
                self.last_key_ns = time.monotonic_ns()
                self.control_epoch += 1
                os.write(self.fd, key.encode())
            except OSError as exc:
                raise RuntimeError(f'发送控制键失败：{exc}') from exc
            if key=='p':
                self.state = 'pausing'
            elif key in ('r', 'h'):
                self.state = 'resuming'
            return self.status()

    def start_hold(self, autonomous=False):
        """Connect in measured-pose hold for supervised motion diagnostics."""
        with self.lock:
            self.start(auto_follow=False, autonomous=autonomous)
            threading.Thread(target=self._auto_follow, args=(self.proc, 'h'), daemon=True).start()
            return self.status()

    def toggle(self):
        with self.lock:
            state = self.status()['state']
            if state in ('starting','pausing','resuming','stopping'):
                raise RuntimeError('控制请求尚未完成，请稍候')
            if state == 'paused' and self.vla_step_status().get('state') in ('pending', 'executing'):
                raise RuntimeError('VLA单步尚未结束；请先等待完成或点击“立即中止小步并保持”')
            return self.send_key('r' if state in ('paused','waiting','fault_hold') else 'p')

    def pause(self):
        with self.lock:
            state = self.status()['state']
            if state == 'fault_hold':
                return self.send_key('p')  # Also cancels any in-flight recovery request.
            if state == 'paused':
                if self.vla_step_status().get('state') in ('pending', 'executing'):
                    return self.abort_vla_step()
                return self.status()
            if state != 'following':
                raise RuntimeError('从臂当前不在跟随状态，无法发送暂停键')
            return self.send_key('p')

    def resume(self):
        with self.lock:
            state = self.status()['state']
            if state == 'following':
                return self.status()
            if state not in ('paused','fault_hold'):
                raise RuntimeError('从臂当前不是暂停保持状态，无法继续跟随')
            if self.vla_step_status().get('state') in ('pending', 'executing'):
                raise RuntimeError('VLA单步尚未结束；请先等待完成或中止')
            return self.send_key('r')

    def submit_vla_step(self, action_deg, expected_state_deg, session_origin_deg=None, trajectory_deg=None,
                        replace_id=None, expected_mono_ns=None):
        with self.lock:
            status = self.status()
            if not status['running'] or status['state'] != 'paused' or self.cancel.is_set():
                raise RuntimeError('真机VLA单步要求原遥操作正在暂停保持')
            active = self.vla_step_status()
            if replace_id is not None and (active.get('id') != replace_id or active.get('state') not in ('executing', 'complete')):
                raise RuntimeError('旧轨迹已改变，拒绝滚动接续')
            if replace_id is None and active.get('state') in ('pending', 'executing'):
                raise RuntimeError('已有VLA单步正在等待或执行')
            packet = self.telemetry_getter() if self.telemetry_getter else None
            if not packet or packet.get('context', {}).get('process_id') != self.proc.pid:
                raise RuntimeError('找不到当前遥操作进程的真实关节遥测')
            if time.monotonic_ns()-packet.get('publish_mono_ns', 0) > 300_000_000:
                raise RuntimeError('从臂状态超过300ms未更新，请恢复通信后重试')
            if not packet.get('live') or not packet.get('tracking') or not packet.get('paused'):
                raise RuntimeError('遥操作遥测未确认处于暂停保持')
            if not isinstance(action_deg, list) or len(action_deg) != 7 or not isinstance(expected_state_deg, list) or len(expected_state_deg) != 7:
                raise ValueError('VLA输出和确认姿态必须各有7个关节')
            if any(not isinstance(v, (int, float)) or not math.isfinite(v)
                   for v in (*action_deg, *expected_state_deg)):
                raise ValueError('VLA单步含无效角度')
            pid = self.proc.pid
            request_path = runtime_path(pid, 'json')
            if request_path.exists():
                raise RuntimeError('已有VLA单步正在等待控制循环处理')
            request = {'id': uuid.uuid4().hex, 'created_mono_ns': time.monotonic_ns(),
                       'action_deg': action_deg, 'expected_state_deg': expected_state_deg,
                       'session_origin_deg': session_origin_deg}
            if replace_id is not None:
                if trajectory_deg is None or type(expected_mono_ns) is not int or not 0 <= time.monotonic_ns()-expected_mono_ns <= 300_000_000:
                    raise ValueError('滚动接续需要新鲜姿态时间和完整轨迹')
                request.update(replace_id=replace_id, expected_mono_ns=expected_mono_ns)
            if trajectory_deg is not None:
                if (not isinstance(trajectory_deg,list) or not 1<=len(trajectory_deg)<=50 or
                        any(not isinstance(a,list) or len(a)!=7 or
                            any(not isinstance(x,(int,float)) or isinstance(x,bool) or not math.isfinite(x) for x in a)
                            for a in trajectory_deg)):
                    raise ValueError('连续轨迹必须为1至50个七维动作')
                request['trajectory_deg']=trajectory_deg
            atomic_runtime_json(request_path, request)
            return {'id': request['id'], 'state': 'pending',
                    'message': '已提交给原控制循环，按当前配置限制速度、实测超前和关节范围'}

    def vla_step_status(self):
        with self.lock:
            if self.proc is None:
                return {'state': 'idle', 'message': '遥操作未运行'}
            request = runtime_path(self.proc.pid, 'json')
            if request.exists():
                return {'state': 'pending', 'message': '等待控制循环接收动作'}
            path = runtime_path(self.proc.pid, 'status.json')
            try:
                return json.loads(path.read_text())
            except (OSError, ValueError):
                request = runtime_path(self.proc.pid, 'json')
                return {'state': 'pending' if request.exists() else 'idle',
                        'message': '等待控制循环' if request.exists() else ''}

    def abort_vla_step(self):
        with self.lock:
            if self.proc is None or self.proc.poll() is not None:
                raise RuntimeError('遥操作未运行')
            pid = self.proc.pid
            status = self.vla_step_status()
            runtime_path(pid, 'abort').touch()
            request = runtime_path(pid, 'json')
            request.unlink(missing_ok=True)
            if status.get('state') == 'pending':
                status_path = runtime_path(pid, 'status.json')
                atomic_runtime_json(status_path, {'state': 'aborted', 'message': '等待中的VLA单步已取消'})
            return self.send_key('p')

    def stop(self):
        with self.lock:
            if self.proc is None or self.proc.poll() is not None:
                return self.status()
            self.cancel.set()
            self.control_epoch += 1
            self.state = 'stopping'
            proc = self.proc
            try:
                os.write(self.fd,b'q')
            except OSError:
                pass
            # Same stop request as the existing CLI stop script, checked before enable.
            self.stop_file.touch()
            threading.Thread(target=self._stop_worker,args=(proc,),daemon=True).start()
            return self.status()

    def _stop_worker(self, proc):
        try:
            proc.wait(timeout=4)
        except subprocess.TimeoutExpired:
            # SIGTERM is handled by the original teleop's cleanup, never SIGKILL.
            try:
                proc.send_signal(signal.SIGTERM)
                proc.wait(timeout=6)
            except subprocess.TimeoutExpired:
                with self.lock:
                    self.last_error = '遥操作尚未退出，失能未确认；检查连接及原日志'
            except ProcessLookupError:
                pass

    def status(self):
        with self.lock:
            running = self.proc is not None and self.proc.poll() is None
            state = self.state
            if running and self.telemetry_getter and not self.cancel.is_set():
                t = self.telemetry_getter() or {}
                if (t.get('context',{}).get('process_id')==self.proc.pid
                        and self.last_key_ns <= t.get('publish_mono_ns',0)
                        and 0 <= time.monotonic_ns()-t.get('publish_mono_ns',0)<800_000_000):
                    if t.get('tracking'):
                        state = 'fault_hold' if t.get('fault_hold') else ('paused' if t.get('paused') else 'following')
            if not running and state in ('starting','waiting','following','paused','fault_hold','resuming','pausing','stopping'):
                state = 'error' if self.proc and self.proc.returncode else 'stopped'
            return {'running':running,'pid':self.proc.pid if running else None,
                    'state':state,'paused':state in ('paused','fault_hold'),'fault_hold':state=='fault_hold','ready':self.ready.is_set(),
                    'control_epoch': self.control_epoch,
                    'autonomous': self.autonomous,
                    'log':str(self.log_path) if self.log_path else None,
                    'output_tail':self.output_tail[-3000:], 'error':self.last_error,
                    'returncode':self.proc.returncode if self.proc else None}

    def close(self):
        with self.lock:
            proc = self.proc
        self.stop()
        if proc:
            try:
                proc.wait(timeout=11)
            except subprocess.TimeoutExpired:
                pass
        if self.reader:
            self.reader.join(timeout=1)
