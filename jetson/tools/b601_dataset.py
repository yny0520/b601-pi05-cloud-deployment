"""Timestamped raw dataset recording; no robot-control or device access."""
from __future__ import annotations

from bisect import bisect_right
from collections import deque
from datetime import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import secrets
import shutil
import socket
import threading
import time

import cv2
import numpy as np

from b601_telemetry import SOCKET_PATH

DEFAULT_ROOT = Path('/home/jetson/Desktop/b601-rs/datasets/b601-insertion')
SENSORS = ('hand', 'top', 'depth')


def write_json(path, obj):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def timestamp():
    return datetime.now().astimezone().isoformat()


def finite_seven(values):
    return isinstance(values, list) and len(values) == 7 and all(
        isinstance(v, (int, float)) and math.isfinite(v) for v in values)


class DatasetRecorder:
    """Asynchronous image writer and episode controls, independent of teleop."""

    def __init__(self, root=DEFAULT_ROOT, socket_path=SOCKET_PATH, camera_hz=15,
                 enable_socket=True, min_free_bytes=2 * 1024**3, hand_camera_hz=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.min_free_bytes = min_free_bytes
        self.socket_path = socket_path
        self.period_ns = int(1e9 / camera_hz)
        hand_camera_hz = camera_hz if hand_camera_hz is None else hand_camera_hz
        if not 1 <= camera_hz <= 30 or not 1 <= hand_camera_hz <= 30:
            raise ValueError('相机保存上限必须在1..30Hz')
        self.periods_ns = {name: int(1e9 / (hand_camera_hz if name=='hand' else camera_hz)) for name in SENSORS}
        self.lock = threading.RLock()
        self.controls = threading.Lock()
        self.queue = queue.Queue(maxsize=192)
        self.stop = threading.Event()
        self.active = self.finalizing = None
        self.latest = {}
        self.sensor_info = {}
        self.latest_telemetry = None
        self.last_result = None
        self.collection_profile = None
        self.error = ''
        self.invalid_packets = 0
        self.token = secrets.token_urlsafe(24)
        self.sock = self.socket_lock = None
        self.receiver = None
        if enable_socket:
            self.socket_lock = open(socket_path + '.lock', 'a')
            try:
                fcntl.flock(self.socket_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                self.socket_lock.close()
                raise RuntimeError('采集服务已在运行，请使用现有页面')
            path = Path(socket_path)
            if path.exists():
                if not path.is_socket():
                    raise RuntimeError('遥测路径已被非socket文件占用')
                path.unlink()
            self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1_048_576)
            self.sock.bind(socket_path)
            os.chmod(socket_path, 0o600)
            self.sock.settimeout(.3)
            self.receiver = threading.Thread(target=self._receive, daemon=True)
            self.receiver.start()
        self.writer = threading.Thread(target=self._write_loop, daemon=True)
        self.writer.start()

    def _receive(self):
        while not self.stop.is_set():
            try:
                self.on_telemetry(json.loads(self.sock.recv(131072)))
            except socket.timeout:
                continue
            except (ValueError, OSError, TypeError, KeyError):
                if not self.stop.is_set():
                    self.invalid_packets += 1

    def _offer(self, kind, ep, data):
        if ep['io_failed']:
            return
        try:
            self.queue.put_nowait((kind, ep, data))
        except queue.Full:
            ep['dropped'][kind] = ep['dropped'].get(kind, 0) + 1

    def _update_source(self, name, received_ns):
        source = self.latest.setdefault(name, {'seq': 0, 'times': deque(maxlen=90)})
        source['seq'] += 1
        source['mono_ns'] = received_ns
        source['times'].append(received_ns)
        return source['seq']

    def on_frame(self, name, array, ros_stamp_ns=None, frame_id='', encoding='',
                 received_ns=None):
        received_ns = received_ns or time.monotonic_ns()
        with self.lock:
            seq = self._update_source(name, received_ns)
            source = self.latest[name]
            source.update(shape=list(array.shape), encoding=encoding)
            ep = self.active
            if ep is None or received_ns < ep['start_ns'] or ep['io_failed']:
                return
            period = self.periods_ns[name]
            # Follow the nominal timeline rather than restarting the interval
            # after each frame: 33.2ms vs 33.3ms jitter must not halve 30Hz.
            due = ep['next_frame_due'].get(name, received_ns)
            if received_ns < due - period // 6:
                return
            ep['next_frame_due'][name] = (due + period if received_ns < due + period
                                           else received_ns + period)
            ep['last_offered'][name] = received_ns
            self._offer(name, ep, {
                'array': array.copy(), 'source_seq': seq,
                'received_mono_ns': received_ns, 'received_wall_ns': time.time_ns(),
                'ros_stamp_ns': ros_stamp_ns, 'frame_id': frame_id, 'encoding': encoding,
            })

    def on_telemetry(self, packet):
        now = time.monotonic_ns()
        autonomous = (packet.get('leader_read') is False
                      and packet.get('leader_deg') is None
                      and packet.get('context', {}).get('arguments', {}).get('autonomous') is True)
        fault_hold = (packet.get('fault_hold') is True and packet.get('paused') is True
                      and packet.get('leader_read') is False and packet.get('leader_deg') is None)
        if (packet.get('schema') != 'b601-telemetry-v1'
                or not all(finite_seven(packet.get(k)) for k in
                           ('follower_deg', 'command_deg'))
                or (not autonomous and not fault_hold and not finite_seven(packet.get('leader_deg')))
                or not isinstance(packet.get('session'), str)
                or not isinstance(packet.get('seq'), int)
                or len(packet.get('read_window_ns') or []) != 2):
            raise ValueError('Invalid telemetry packet')
        start, end = packet['read_window_ns']
        if not all(isinstance(v, int) for v in (start, end)) or not 0 <= now-end < 2_000_000_000 or start > end:
            raise ValueError('Stale/invalid telemetry timing')
        with self.lock:
            previous = self.latest_telemetry
            if previous and previous['session'] == packet['session'] and previous['seq'] >= packet['seq']:
                return
            self.latest_telemetry = packet
            self._update_source('teleop', now)
            ep = self.active
            if ep and end >= ep['start_ns']:
                self._offer('teleop', ep, {**packet, 'collector_received_mono_ns': now})

    def set_info(self, name, data):
        with self.lock:
            self.sensor_info[name] = data

    def _health(self):
        now = time.monotonic_ns()
        result = {}
        for name in (*SENSORS, 'teleop'):
            s = self.latest.get(name)
            times = s['times'] if s else []
            result[name] = {
                'age_ms': (now - s['mono_ns']) / 1e6 if s else None,
                'hz': (len(times)-1)*1e9/(times[-1]-times[0]) if len(times)>1 and times[-1]>times[0] else 0,
                'shape': s.get('shape') if s else None,
                'encoding': s.get('encoding') if s else None,
            }
        return result

    def status(self):
        with self.lock:
            ep = self.active or self.finalizing
            t = self.latest_telemetry or {}
            return {
                'dataset_root': str(self.root),
                'camera_hz_caps': {name: 1e9/period for name,period in self.periods_ns.items()},
                'collection': dict(self.collection_profile) if self.collection_profile else None,
                'recording': self.active is not None,
                'finalizing': self.finalizing is not None,
                'episode': ep['folder'].name if ep else None,
                'folder': str(ep['folder']) if ep else None,
                'seconds': (time.monotonic_ns()-ep['start_ns'])/1e9 if ep else 0,
                'counts': dict(ep['counts']) if ep else {},
                'dropped': dict(ep['dropped']) if ep else {},
                'io_failed': ep['io_failed'] if ep else False,
                'error': self.error,
                'sensors': self._health(),
                'teleop': {k: t.get(k) for k in ('live','tracking','paused','leader_deg','follower_deg','command_deg',
                                                'leader_read','fault_hold','fault_hold_reason',
                                                'motor_feedback','mit_commands','gravity_torque_nm','publish_mono_ns','seq',
                                                'read_window_ns','send_window_ns','context','vla_step')},
                'free_gib': round(shutil.disk_usage(self.root).free / 1024**3, 1),
                'last_result': self.last_result,
                'invalid_packets': self.invalid_packets,
            }

    def set_collection_profile(self, profile, settings_path=None):
        """Switch only between episodes; never move or delete existing recordings."""
        with self.controls, self.lock:
            if self.active or self.finalizing:
                raise ValueError('记录或保存期间不能切换数据集，请先结束本集')
            root = Path(profile['root'])
            root.mkdir(parents=True, exist_ok=True)
            if settings_path is not None:
                write_json(settings_path, {'profile': profile['id']})
            self.root = root
            self.collection_profile = dict(profile)
            self.last_result = None
            return self.status()

    def start_episode(self, task=None, setup='', success_criterion=None):
        with self.controls, self.lock:
            profile = self.collection_profile or {}
            task = task or profile.get('task', '网线插接')
            success_criterion = success_criterion or profile.get('success_criterion', '人工确认插接到位')
            if self.active or self.finalizing:
                raise ValueError('已有一集正在记录或保存；请先结束当前集')
            t = self.latest_telemetry or {}
            if t.get('fault_hold'):
                raise ValueError('从臂处于故障保持，不能开始示范；检查主臂连接并手动恢复后再采集')
            if t.get('leader_read') is False or t.get('context', {}).get('arguments', {}).get('autonomous'):
                raise ValueError('自主模型执行不能记录为人工遥操作示范；请先停止并切换到遥操作')
            health = self._health()
            missing = [k for k, s in health.items() if s['age_ms'] is None or s['age_ms'] > 1000]
            if missing:
                raise ValueError('等待新数据：' + ', '.join(missing) + '。先启动遥操作并按R；确认相机在线')
            if not t.get('live') or not t.get('tracking') or t.get('paused') or not t.get('motor_commands_sent'):
                raise ValueError('遥操作还未处于跟随状态，请在遥操作终端按R；采集页面不使能电机')
            if shutil.disk_usage(self.root).free < self.min_free_bytes:
                raise ValueError('磁盘可用空间不足2GiB，请先另行安排存储；本程序不会删除文件')
            self.error = ''
            folder = self.root / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
            folder.mkdir()
            for name in SENSORS:
                (folder / name).mkdir()
            start_ns = time.monotonic_ns()
            metadata = {
                'schema': 'b601-raw-episode-v1', 'status': 'recording',
                'started_at': timestamp(), 'started_wall_ns': time.time_ns(),
                'started_mono_ns': start_ns,
                'task': str(task)[:500], 'setup': str(setup)[:1500],
                'dataset': {'id': profile.get('id'), 'root': str(self.root)},
                'success_criterion': str(success_criterion)[:1000],
                'outcome': 'unlabeled', 'human_review_required': True,
                'camera_hz_cap': 1e9/self.period_ns,
                'camera_hz_caps': {name: 1e9/period for name,period in self.periods_ns.items()},
                'telemetry_hz': 'one record per received real control iteration, no resampling',
                'timing': 'host CLOCK_MONOTONIC receive times; original ROS stamps also saved. No hardware sync. Arm feedback is read sequentially over read_window_ns.',
                'image_join': 'latest received frame at/before read_window_ns[1]; 200 ms quality threshold; reused frames explicitly flagged',
                'depth': {'16UC1_unit': 'millimeter', '32FC1_unit': 'meter',
                          'saved': 'lossless original depth, no 0-2m clipping or colorization',
                          'registered_to_color': False},
                'hand_intrinsics': 'not calibrated; device/settings recorded, no wrist camera_info available',
                'action': 'MIT arguments submitted by teleop before protocol quantization; no actuator acknowledgement. J7 command_deg is a position reference, while its actual MIT action is torque plus damping.',
                'joint_order': [f'J{i}' for i in range(1,8)],
                'format': 'raw archive, review/convert before LeRobot training',
                'camera_info': dict(self.sensor_info),
            }
            ep = {'folder': folder, 'meta': metadata, 'start_ns': start_ns,
                  'last_offered': {}, 'next_frame_due': {}, 'counts': {}, 'dropped': {}, 'io_failed': False,
                  'errors': [], 'contexts': {}, 'last_checkpoint': 0,
                  'files': {name: (folder / f'{name}.jsonl').open('w', buffering=1)
                            for name in (*SENSORS, 'teleop', 'events')}}
            write_json(folder / 'metadata.json', metadata)
            self.active = ep
            self._offer('events', ep, {'type': 'start', 'mono_ns': start_ns, 'wall_ns': time.time_ns()})
            return {'episode': folder.name, 'folder': str(folder)}

    def mark(self, phase, note=''):
        allowed = ('接近', '抓取', '抬起', '搬运', '放置', '松爪', '对孔', '插接', '退回纠错', '布线', '验证', '备注')
        if phase not in allowed:
            raise ValueError('未知阶段')
        with self.lock:
            if self.active is None:
                raise ValueError('当前未记录')
            self._offer('events', self.active, {'type': 'phase', 'phase': phase,
                        'note': str(note)[:1500], 'mono_ns': time.monotonic_ns(), 'wall_ns': time.time_ns()})
        return {'marked': phase}

    def finish_episode(self, outcome, reason=''):
        if outcome not in ('success', 'failure', 'unlabeled', 'interrupted'):
            raise ValueError('未知结果')
        with self.controls:
            with self.lock:
                if self.active is None:
                    raise ValueError('当前没有正在记录的一集')
                ep = self.active
                self.active = None
                self.finalizing = ep
                ep['meta'].update(outcome=outcome, reason=str(reason)[:1500],
                                  ended_mono_ns=time.monotonic_ns(), ended_at=timestamp())
                ep['meta']['camera_info'] = dict(self.sensor_info)
            done = threading.Event()
            self.queue.put(('finish', ep, done))
            done.wait()
            with self.lock:
                self.finalizing = None
                self.last_result = {'episode': ep['folder'].name, 'folder': str(ep['folder']),
                                    'outcome': outcome, 'counts': dict(ep['counts']),
                                    'quality': ep.get('quality'), 'errors': ep['errors']}
                if ep['io_failed']:
                    raise OSError('保存遇到错误，文件已尽量保留：' + '; '.join(ep['errors']))
                return self.last_result

    def _write_item(self, kind, ep, data):
        if kind in SENSORS:
            arr = data.pop('array')
            n = ep['counts'].get(kind, 0)
            filename = f'{kind}/{n:07d}.png' if kind == 'depth' and arr.dtype == np.uint16 else f'{kind}/{n:07d}.jpg'
            if kind == 'depth' and arr.dtype != np.uint16:
                filename = f'{kind}/{n:07d}.npy'
                np.save(ep['folder']/filename, arr, allow_pickle=False)
            else:
                params = [cv2.IMWRITE_PNG_COMPRESSION, 1] if kind=='depth' else [cv2.IMWRITE_JPEG_QUALITY, 90]
                if not cv2.imwrite(str(ep['folder']/filename), arr, params):
                    raise OSError(f'图像保存失败 {filename}')
            data.update(file=filename, shape=list(arr.shape), dtype=str(arr.dtype))
        elif kind == 'teleop':
            session = data['session']
            context = data.pop('context', {})
            if session not in ep['contexts']:
                filename = f'teleop-context-{len(ep["contexts"])+1}.json'
                snapshot = {'session': session, **context}
                calibration = context.get('config', {}).get('travel_calibration')
                if calibration and Path(calibration).is_file():
                    raw = Path(calibration).read_bytes()
                    snapshot['travel_calibration_snapshot'] = json.loads(raw)
                    snapshot['travel_calibration_sha256'] = hashlib.sha256(raw).hexdigest()
                ep['contexts'][session] = filename
                write_json(ep['folder']/filename, snapshot)
            data['context_file'] = ep['contexts'][session]
        ep['files'][kind].write(json.dumps(data, ensure_ascii=False, allow_nan=False)+'\n')
        with self.lock:
            ep['counts'][kind] = ep['counts'].get(kind, 0)+1

    def _checkpoint(self, ep):
        meta = {**ep['meta'], 'counts': dict(ep['counts']), 'queue_dropped': dict(ep['dropped']),
                'write_errors': ep['errors']}
        write_json(ep['folder']/'metadata.json', meta)

    def _write_loop(self):
        while not self.stop.is_set() or not self.queue.empty():
            try:
                kind, ep, data = self.queue.get(timeout=.3)
            except queue.Empty:
                continue
            try:
                if kind == 'finish':
                    for f in ep['files'].values():
                        f.flush()
                        os.fsync(f.fileno())
                        f.close()
                    ep['quality'] = self._build_samples(ep)
                    ep['meta']['quality'] = ep['quality']
                    ep['meta']['status'] = 'saved_with_errors' if ep['io_failed'] else 'saved'
                    self._checkpoint(ep)
                elif not ep['io_failed']:
                    self._write_item(kind, ep, data)
                    if time.monotonic()-ep['last_checkpoint'] > 2:
                        if shutil.disk_usage(self.root).free < self.min_free_bytes//2:
                            raise OSError('剩余空间不足1GiB，采集写入已暂停；请结束本集并安排存储')
                        self._checkpoint(ep)
                        ep['last_checkpoint'] = time.monotonic()
            except Exception as exc:
                with self.lock:
                    ep['io_failed'] = True
                    message = f'{type(exc).__name__}: {exc}'
                    ep['errors'].append(message)
                    self.error = message
                if kind == 'finish':
                    for f in ep['files'].values():
                        if not f.closed:
                            f.close()
                    ep['meta']['status'] = 'save_error'
                    try:
                        self._checkpoint(ep)
                    except OSError:
                        pass
            finally:
                if kind == 'finish':
                    data.set()
                self.queue.task_done()

    def _build_samples(self, ep):
        def rows(name):
            with (ep['folder']/f'{name}.jsonl').open() as f:
                return [json.loads(line) for line in f if line.strip()]
        streams = {name: rows(name) for name in SENSORS}
        teleop = rows('teleop')
        times = {name: [f['received_mono_ns'] for f in frames] for name, frames in streams.items()}
        missing = {name: 0 for name in SENSORS}
        reused = {name: 0 for name in SENSORS}
        previous = {}
        gaps = 0
        last_packet = None
        action_count = 0
        with (ep['folder']/'samples.jsonl').open('w') as output:
            for i, t in enumerate(teleop):
                anchor = t['read_window_ns'][1]
                images = {}
                for name in SENSORS:
                    index = bisect_right(times[name], anchor)-1
                    frame = streams[name][index] if index >= 0 else None
                    age = (anchor-frame['received_mono_ns'])/1e6 if frame else None
                    ok = age is not None and 0 <= age <= 200
                    if not ok:
                        missing[name] += 1
                    repeat = frame is not None and previous.get(name) == frame['source_seq']
                    reused[name] += int(repeat)
                    images[name] = {'file': frame['file'] if frame else None,
                                    'source_seq': frame['source_seq'] if frame else None,
                                    'age_ms': age, 'fresh': ok, 'reused': repeat}
                    if frame:
                        previous[name] = frame['source_seq']
                if last_packet and t['session']==last_packet['session']:
                    gaps += max(0, t['seq']-last_packet['seq']-1)
                last_packet = t
                demonstration = bool(t.get('live') and t.get('tracking') and not t.get('paused') and not t.get('fault_hold')
                                     and t.get('motor_commands_sent'))
                action_count += int(demonstration)
                output.write(json.dumps({
                    'index': i, 'teleop_seq': t['seq'], 'teleop_session': t['session'],
                    'read_window_ns': t['read_window_ns'], 'send_window_ns': t['send_window_ns'],
                    'state_deg': t['follower_deg'], 'leader_deg': t['leader_deg'],
                    'action_position_reference_deg': t['command_deg'],
                    'mit_commands': t.get('mit_commands'), 'alignment': t.get('alignment'),
                    'images': images, 'demonstration_segment': demonstration,
                    'image_timing_ok': all(v['fresh'] for v in images.values()),
                    'paused': t.get('paused'), 'context_file': t['context_file'],
                    'fault_hold': t.get('fault_hold', False), 'fault_hold_reason': t.get('fault_hold_reason', ''),
                }, ensure_ascii=False, allow_nan=False)+'\n')
        quality = {
            'telemetry_records': len(teleop), 'demonstration_records': action_count,
            'telemetry_sequence_gaps': gaps, 'teleop_sessions': len(ep['contexts']),
            'missing_or_stale_image_matches': missing, 'reused_image_matches': reused,
            'queue_dropped': dict(ep['dropped']), 'write_errors': list(ep['errors']),
            'sensors': {}, 'human_review_required': True,
            'tail_without_telemetry_ms': (ep['meta']['ended_mono_ns']-teleop[-1]['read_window_ns'][1])/1e6 if teleop else None,
            'max_control_gap_ms': max((b['read_window_ns'][1]-a['read_window_ns'][1]
                                       for a,b in zip(teleop,teleop[1:])), default=0)/1e6,
        }
        for name, frames in streams.items():
            ns = times[name]
            quality['sensors'][name] = {
                'frames': len(frames),
                'saved_hz': (len(ns)-1)*1e9/(ns[-1]-ns[0]) if len(ns)>1 else 0,
                'max_gap_ms': max((b-a for a,b in zip(ns,ns[1:])), default=0)/1e6,
            }
        write_json(ep['folder']/'quality.json', quality)
        return quality

    def close(self):
        try:
            if self.active:
                self.finish_episode('interrupted', '采集服务退出')
        finally:
            self.stop.set()
            self.writer.join(timeout=3)
            if self.sock:
                self.sock.close()
                if self.receiver:
                    self.receiver.join(timeout=1)
                Path(self.socket_path).unlink(missing_ok=True)
            if self.socket_lock:
                self.socket_lock.close()
