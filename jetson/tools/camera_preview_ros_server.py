#!/usr/bin/env python3
"""Live page: DECXIN via V4L2 and Gemini 336L color via Orbbec ROS2."""
from __future__ import annotations
import argparse, threading, time, glob, json
from pathlib import Path
from urllib.parse import urlsplit
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from rclpy.qos import QoSProfile, DurabilityPolicy
from tf2_msgs.msg import TFMessage
from rosidl_runtime_py.convert import message_to_ordereddict
from b601_dataset import DatasetRecorder
from b601_dataset_web import get_dataset_response
from b601_teleop_manager import TeleopManager
from b601_vla_runner import VlaRunner
from b601_vla_motion import load_motion_profile, describe_profile
from b601_vla_model import load_model_spec, check_collection
from b601_collection_profiles import get_profile, load_profile_name, render_panel, PROFILES, SETTINGS
from b601_hand_camera import open_hand_camera
from b601_grasp_observer import GraspObserver

VLA_REPORT = Path('/home/jetson/Desktop/b601-rs/datasets/b601-insertion/20261103-145506-689538/annotations/smolvla_offline_eval/report.json')


def render_depth(depth, encoding):
    """Display ROS depth in a fixed 0–2 m range; never change the source depth."""
    if encoding == '16UC1':
        meters = np.asarray(depth, dtype=np.float32) * 0.001
    elif encoding == '32FC1':
        meters = np.asarray(depth, dtype=np.float32)
    else:
        raise ValueError(f'Unsupported depth encoding: {encoding}')
    if meters.ndim != 2:
        raise ValueError('Expected a single-channel depth image')
    valid = np.isfinite(meters) & (meters > 0)
    far = valid & (meters > 2.0)
    # Avoid casting NaN/Inf, and reserve black for missing depth in both views.
    scaled = np.rint(np.clip(np.where(valid, meters, 0), 0, 2) * 127.5).astype(np.uint8)
    gray = cv2.cvtColor(np.maximum(scaled, 1), cv2.COLOR_GRAY2BGR)
    heat = cv2.applyColorMap(scaled, cv2.COLORMAP_JET)
    for view in (gray, heat):
        view[~valid] = (0, 0, 0)
        view[far] = (70, 70, 70)
    return gray, heat


def add_depth_legend(view, heatmap=False):
    height, width = view.shape[:2]
    result = np.zeros((height + 66, width, 3), dtype=np.uint8)
    result[:height] = view
    left, right = 28, width - 28
    ramp = np.tile(np.linspace(0, 255, right-left).astype(np.uint8), (12, 1))
    bar = cv2.applyColorMap(ramp, cv2.COLORMAP_JET) if heatmap else cv2.cvtColor(ramp, cv2.COLOR_GRAY2BGR)
    result[height+9:height+21, left:right] = bar
    for fraction in (0, .25, .5, .75, 1):
        x = int(left + fraction * (right-left-1))
        cv2.putText(result, f'{fraction*2:g}m', (x-13, height+40), cv2.FONT_HERSHEY_SIMPLEX, .42, (220,220,220), 1, cv2.LINE_AA)
    cv2.putText(result, 'Black: no depth   |   Gray: beyond 2m', (left, height+59), cv2.FONT_HERSHEY_SIMPLEX, .43, (180,180,180), 1, cv2.LINE_AA)
    return result


class Frames:
    def __init__(self):
        self.lock = threading.Lock(); self.hand = None; self.top = None; self.depth = None; self.heatmap = None
        self.raw = {}
    def put(self, name, frame):
        ok, enc = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            with self.lock:
                setattr(self, name, enc.tobytes())
                self.raw[name] = (frame.copy(), time.monotonic_ns())
    def get(self, name):
        with self.lock: return getattr(self, name)
    def snapshot(self, name, max_age_ms=300):
        with self.lock:
            item = self.raw.get(name)
            if item is None or time.monotonic_ns()-item[1] > max_age_ms*1e6:
                return None
            return item[0].copy(), item[1]


DASHBOARD = Path(__file__).with_name('b601_preview_dashboard.html')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hand', default='auto')
    ap.add_argument('--port', type=int, default=8766)
    ap.add_argument('--camera-hz', type=float, default=15)
    ap.add_argument('--hand-camera-hz', type=float, default=30)
    ap.add_argument('--dataset-root', type=Path, default=None)
    ap.add_argument('--collection-profile', choices=PROFILES, default=None)
    args = ap.parse_args()
    if not 1 <= args.camera_hz <= 30:
        ap.error('--camera-hz必须在1..30之间')
    if not 1 <= args.hand_camera_hz <= 30:
        ap.error('--hand-camera-hz必须在1..30之间')
    profile = get_profile(args.collection_profile or load_profile_name())
    if args.dataset_root is not None:
        profile.update(id='custom', label='自定义数据集', root=str(args.dataset_root.resolve()))
    recorder = DatasetRecorder(Path(profile['root']), camera_hz=args.camera_hz,
                               hand_camera_hz=args.hand_camera_hz)
    recorder.set_collection_profile(profile)
    teleop = TeleopManager(telemetry_getter=lambda: recorder.latest_telemetry)
    frames = Frames(); stop = threading.Event()
    vla = VlaRunner(teleop, recorder, frames)
    observer = GraspObserver(frames)
    threading.Thread(target=observer.run, args=(stop,), daemon=True).start()
    cap = None
    def find_hand():
        # Never probe arbitrary video nodes: they may belong to the ROS-owned depth camera.
        candidates = glob.glob('/dev/v4l/by-id/usb-DECXIN*video-index0') if args.hand == 'auto' else [args.hand]
        for path in candidates:
            c, info = open_hand_camera(path)
            if c is not None:
                recorder.set_info('hand_device', info)
                print('腕部相机配置：'+json.dumps(info,ensure_ascii=False),flush=True)
                return c
        return None
    def hand_loop():
        nonlocal cap
        while not stop.is_set():
            if cap is None:
                try:
                    cap = find_hand()
                except Exception as exc:
                    print('腕部相机连接错误:', exc, flush=True)
                    cap = None
                if cap is None:
                    time.sleep(1); continue
            ok, frame = cap.read()
            if ok:
                recorder.on_frame('hand', frame, encoding='bgr8')
                frames.put('hand', frame)
            else:
                cap.release(); cap = None; time.sleep(.2)
    threading.Thread(target=hand_loop, daemon=True).start()
    rclpy.init(); node = rclpy.create_node('b601_camera_preview_server'); bridge = CvBridge()
    def top_cb(msg):
        received_ns = time.monotonic_ns()
        try:
            frame = bridge.imgmsg_to_cv2(msg, 'bgr8')
            recorder.on_frame('top', frame, msg.header.stamp.sec*10**9+msg.header.stamp.nanosec,
                              msg.header.frame_id, 'bgr8', received_ns)
            frames.put('top', frame)
        except Exception as exc: print('彩色取帧错误:', exc, flush=True)
    def depth_cb(msg):
        received_ns = time.monotonic_ns()
        try:
            depth = bridge.imgmsg_to_cv2(msg, 'passthrough')
            recorder.on_frame('depth', depth, msg.header.stamp.sec*10**9+msg.header.stamp.nanosec,
                              msg.header.frame_id, msg.encoding, received_ns)
            gray, heat = render_depth(depth, msg.encoding)
            frames.put('depth', add_depth_legend(gray))
            frames.put('heatmap', add_depth_legend(heat, heatmap=True))
        except Exception as exc: print('深度取帧错误:', exc, flush=True)
    node.create_subscription(Image, '/camera_preview/color/image_raw', top_cb, 10)
    node.create_subscription(Image, '/camera_preview/depth/image_raw', depth_cb, 10)
    for name in ('color', 'depth'):
        node.create_subscription(CameraInfo, f'/camera_preview/{name}/camera_info',
            lambda msg, name=name: recorder.set_info(name+'_camera_info', message_to_ordereddict(msg)), 10)
    node.create_subscription(TFMessage, '/tf_static',
        lambda msg: recorder.set_info('tf_static', message_to_ordereddict(msg)),
        QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL))
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_): pass
        def reply(self, status, mime, body):
            self.send_response(status)
            self.send_header('Content-Type', mime)
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def json_response(self, status, payload):
            self.reply(status, 'application/json; charset=utf-8', json.dumps(payload, ensure_ascii=False).encode())
        def do_GET(self):
            if self.path in ('/', '/index.html'):
                with recorder.lock:
                    panel = render_panel(recorder.token, recorder.collection_profile)
                page = DASHBOARD.read_text().replace('__COLLECTION_PANEL__', panel)
                self.reply(200, 'text/html; charset=utf-8', page.encode()); return
            if self.path == '/api/status':
                self.json_response(200, recorder.status()); return
            if self.path == '/api/teleop/status':
                self.json_response(200, teleop.status()); return
            if self.path == '/api/grasp-observation':
                self.json_response(200, observer.status()); return
            if urlsplit(self.path).path == '/snapshot/grasp.jpg':
                data = observer.image()
                if data is None:
                    self.send_error(503, 'No fresh observation'); return
                self.reply(200, 'image/jpeg', data); return
            if self.path == '/api/vla/status':
                model = load_model_spec()
                status = teleop.status()
                run_status = vla.status()
                profile = run_status.get('motion_profile') or load_motion_profile()
                with recorder.lock:
                    packet = recorder.latest_telemetry or {}
                    camera_health = recorder._health()
                    telemetry_age_ms = ((time.monotonic_ns()-packet['publish_mono_ns'])/1e6
                                         if packet.get('publish_mono_ns') else None)
                    joint_state = packet.get('follower_deg')
                    feedback_fresh = bool(status['running'] and
                        packet.get('context', {}).get('process_id') == status['pid'] and
                        telemetry_age_ms is not None and 0 <= telemetry_age_ms <= 300)
                    if not feedback_fresh:
                        joint_state = None
                try:
                    report = json.loads(VLA_REPORT.read_text()) if model['id'] == 'insertion_legacy' else None
                except (OSError, ValueError):
                    report = None
                self.json_response(200, {
                    'mode': 'bounded_live_test',
                    'model': model,
                    'model_matches_collection': model['collection_profile'] == recorder.collection_profile['id'],
                    'motor_output_enabled': bool(status['running'] and packet.get('motor_commands_sent')
                        and packet.get('context', {}).get('process_id') == status['pid']
                        and telemetry_age_ms is not None and 0 <= telemetry_age_ms <= 300),
                    'teleop_state': status['state'], 'teleop_running': status['running'],
                    'telemetry_age_ms': telemetry_age_ms, 'follower_deg': joint_state,
                    'feedback_fresh': feedback_fresh,
                    'control_io': ({'actual_hz': packet.get('actual_hz'),
                                    'target_hz': packet.get('context', {}).get('arguments', {}).get('hz'),
                                    'read_ms': packet.get('read_s', 0)*1000,
                                    'backend': packet.get('feedback_backend')} if feedback_fresh else None),
                    'cameras': {name: camera_health.get(name) for name in ('hand','top','depth')},
                    'cpu_inference_median_s': report.get('inference_seconds_median') if report else None,
                    'gpu_inference_sample_s': 5.742 if model['id'] == 'insertion_legacy' else None,
                    'gpu_inference_mode': ('Pi0.5云端A100推理；每轮16点，Jetson只执行本地控制' if model.get('policy_type') == 'pi05_remote' else
                                           'ACT FP32 CUDA推理；按配置窗口执行后重新观察' if model.get('policy_type') == 'act' else
                                           'BF16加载，释放CPU临时缓冲后搬入CUDA；一次测试复用同一模型，结束时释放'),
                    'live_vla_inference_running': run_status['state'] == 'predicting',
                    'run': run_status,
                    'teleop': status,
                    'offline_mae_deg': report.get('mean_action_reference_mae_deg') if report else None,
                    'hold_baseline_mae_deg': report.get('mean_hold_current_state_mae_deg') if report else None,
                    'j7_mae_deg': report.get('action_reference_mae_deg_by_joint', {}).get('J7') if report else None,
                    'step': teleop.vla_step_status(),
                    'motion_profile': profile,
                    'motion_limits': describe_profile(profile),
                    'reason': describe_profile(profile)+'。'+model['label']+'；实机完整任务成功率尚未测量。',
                }); return
            response = get_dataset_response(self.path, recorder.root)
            if response:
                self.reply(*response); return
            name = {'/stream/hand':'hand','/stream/top':'top','/stream/depth':'depth','/stream/heatmap':'heatmap'}.get(self.path)
            if not name: self.send_error(404); return
            self.send_response(200); self.send_header('Cache-Control','no-cache'); self.send_header('Content-Type','multipart/x-mixed-replace; boundary=frame'); self.end_headers()
            try:
                previous = None
                while not stop.is_set():
                    data=frames.get(name)
                    if data and data is not previous:
                        self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '+str(len(data)).encode()+b'\r\n\r\n'+data+b'\r\n'); self.wfile.flush()
                        previous = data
                    time.sleep(.005)
            except (BrokenPipeError, ConnectionResetError): pass
        def do_POST(self):
            if self.headers.get('X-Record-Token') != recorder.token:
                self.json_response(403, {'error':'页面已过期，请刷新后操作'}); return
            origin = self.headers.get('Origin')
            if origin and urlsplit(origin).netloc != self.headers.get('Host'):
                self.json_response(403, {'error':'请求来源不匹配'}); return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 <= length <= 16384:
                    raise ValueError('请求内容过长')
                body = json.loads(self.rfile.read(length) or b'{}')
                if self.path == '/api/collection/profile':
                    with vla.lock:
                        if vla.status()['active']:
                            raise ValueError('请先停止模型执行，再切换采集任务')
                        result = recorder.set_collection_profile(get_profile(body.get('profile')), SETTINGS)
                elif self.path == '/api/start':
                    with vla.lock:
                        if vla.status()['active']:
                            raise ValueError('请先停止VLA测试，再开始遥操作示范采集')
                        result = recorder.start_episode(body.get('task'), body.get('setup', ''),
                                                        body.get('success_criterion'))
                elif self.path == '/api/finish':
                    result = recorder.finish_episode(body.get('outcome'), body.get('reason', ''))
                elif self.path == '/api/mark':
                    result = recorder.mark(body.get('phase'), body.get('note', ''))
                elif self.path == '/api/teleop/start':
                    with vla.lock:
                        if vla.status()['active']:
                            raise ValueError('VLA测试运行中，请先停止模型执行')
                        result = teleop.start()
                elif self.path == '/api/teleop/hold':
                    with vla.lock:
                        if vla.status()['active']:
                            raise ValueError('先停止VLA测试再进行运动诊断')
                        if type(body.get('autonomous', False)) is not bool:
                            raise ValueError('autonomous必须为布尔值')
                        result = teleop.start_hold(autonomous=body.get('autonomous', False))
                elif self.path == '/api/teleop/toggle':
                    with vla.lock:
                        if vla.status()['active']:
                            raise ValueError('VLA测试运行中，请先停止模型执行')
                        result = teleop.toggle()
                elif self.path == '/api/teleop/pause':
                    result = vla.stop() if vla.status()['active'] else teleop.pause()
                elif self.path == '/api/teleop/resume':
                    with vla.lock:
                        if vla.status()['active']:
                            raise ValueError('VLA测试运行中，请先停止模型执行')
                        result = teleop.resume()
                elif self.path == '/api/teleop/vla-abort':
                    result = vla.stop()
                elif self.path == '/api/teleop/stop':
                    with vla.lock:
                        if vla.status()['active']:
                            vla.stop()
                        result = teleop.stop()
                elif self.path == '/api/vla/start':
                    model = load_model_spec()
                    check_collection(model, recorder.collection_profile['id'])
                    result = vla.start(model['task'], body.get('action_steps'))
                elif self.path == '/api/vla/grasp-miss':
                    result = vla.report_grasp_miss()
                elif self.path == '/api/vla/retry':
                    result = vla.retry_grasp()
                elif self.path == '/api/vla/grasp-command':
                    result = vla.grasp_command(body)
                elif self.path == '/api/vla/stop':
                    result = vla.stop()
                else:
                    self.json_response(404, {'error':'未知接口'}); return
                self.json_response(200, result)
            except Exception as exc:
                self.json_response(400, {'error':str(exc)})
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler); print(f'相机预览与采集: http://127.0.0.1:{args.port}/',flush=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        while True: rclpy.spin_once(node, timeout_sec=.05); server.handle_request() if False else None
    except KeyboardInterrupt: pass
    finally:
        stop.set();
        vla.close()
        teleop.close()
        recorder.close()
        if cap is not None: cap.release()
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()
        server.server_close()


if __name__ == '__main__': main()
