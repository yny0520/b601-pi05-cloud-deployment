"""Verified DECXIN capture settings; advertised FPS differs from observed FPS."""
import json
from pathlib import Path
import subprocess

import cv2

CONFIG_PATH = Path('/home/jetson/Desktop/b601-rs/calibration/hand_camera_config.json')
DEFAULT_CONFIG = {'fourcc': 'MJPG', 'width': 640, 'height': 360,
                  'device_fps': 60, 'target_received_hz': 30,
                  'auto_exposure': 3, 'exposure_dynamic_framerate': 0}


def open_hand_camera(path):
    config = {**DEFAULT_CONFIG}
    if CONFIG_PATH.exists():
        config.update(json.loads(CONFIG_PATH.read_text()))
    if config['fourcc'] not in ('MJPG', 'YUYV') or config['device_fps'] not in (30, 60, 80, 120):
        raise ValueError('腕部相机格式或设备帧率配置无效')
    if (config['width'], config['height']) != (640, 360):
        raise ValueError('当前腕部图像接口要求640×360；更改前需同步采集和模型配置')
    if config['auto_exposure'] not in (1, 3) or config['exposure_dynamic_framerate'] not in (0, 1):
        raise ValueError('腕部相机曝光配置无效')
    focus = config.get('focus_absolute')
    if focus is not None and (type(focus) is not int or not 0 <= focus <= 1023):
        raise ValueError('腕部相机固定焦点必须在0..1023之间')
    cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        return None, None
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*config['fourcc']))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config['width'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config['height'])
        cap.set(cv2.CAP_PROP_FPS, config['device_fps'])
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        controls = subprocess.run(['v4l2-ctl', '--device='+path,
            '--set-ctrl=auto_exposure='+str(config['auto_exposure'])+
            ',exposure_dynamic_framerate='+str(config['exposure_dynamic_framerate'])],
            capture_output=True, text=True, timeout=4, check=True)
        if focus is not None:
            subprocess.run(['v4l2-ctl', '--device='+path,
                '--set-ctrl=focus_automatic_continuous=0'],
                capture_output=True, text=True, timeout=4, check=True)
            subprocess.run(['v4l2-ctl', '--device='+path,
                '--set-ctrl=focus_absolute='+str(focus)],
                capture_output=True, text=True, timeout=4, check=True)
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        info = {'path': path, 'width': cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                'height': cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                'fourcc': ''.join(chr((fourcc >> (8*i)) & 255) for i in range(4)),
                'requested_device_fps': config['device_fps'],
                'reported_device_fps': cap.get(cv2.CAP_PROP_FPS),
                'target_received_hz': config['target_received_hz'],
                'auto_exposure': config['auto_exposure'],
                'exposure_dynamic_framerate': config['exposure_dynamic_framerate'],
                'focus_absolute': focus,
                'focus_mode': 'locked' if focus is not None else 'device current setting',
                'timestamp': 'host receive after V4L read; exposure timestamp unavailable',
                'fps_note': 'Device-reported FPS is not measured receive or saved FPS'}
        if info['fourcc'] != config['fourcc'] or (info['width'],info['height']) != (640,360):
            raise RuntimeError('腕部相机未接受指定格式或分辨率')
        return cap, info
    except Exception:
        cap.release()
        raise
