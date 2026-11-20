"""Passive, task-specific red handle / dark finger candidate observation.

This module has no motor interface. Image overlap cannot verify height,
contact, grip force, or that an object will follow the gripper when lifted.
"""
import threading
import time

import cv2
import numpy as np


def observe_hand(frame):
    h, w = frame.shape[:2]
    marked = frame.copy()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red = (((hsv[:, :, 0] < 12) | (hsv[:, :, 0] > 170)) &
           (hsv[:, :, 1] > 105) & (hsv[:, :, 2] > 60)).astype(np.uint8) * 255
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    targets = sorted((c for c in contours if cv2.contourArea(c) > h*w*.0015),
                     key=cv2.contourArea, reverse=True)
    result = dict(mode='passive_candidates', target=None, fingers=None,
                  lateral_state='unknown', alignment_verified=False, grasp_verified=False,
                  message='未找到唯一红色手柄候选；无法判断对位',
                  limitations='仅图像候选；高度、接触和是否夹住均未验证，不自动闭爪或放行搬运。')
    if targets and (len(targets) == 1 or cv2.contourArea(targets[1]) < .35*cv2.contourArea(targets[0])):
        contour = targets[0]
        mask = np.zeros((h, w), np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        ys, xs = np.nonzero(mask)
        x, y, bw, bh = cv2.boundingRect(contour)
        result['target'] = dict(kind='red_handle_candidate', bbox=[x, y, bw, bh],
                                center_px=[float(xs.mean()), float(ys.mean())],
                                clipped=bool(x == 0 or y == 0 or x+bw >= w or y+bh >= h))
        cv2.drawContours(marked, [contour], -1, (0, 200, 255), 2)

    # Both actual dark finger silhouettes must attach to the lower border.
    # Never replace a missing finger with the image centre or a fixed ROI.
    dark = (hsv[:, :, 2] < 55).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    candidates = []
    for label in range(1, count):
        x, y, bw, bh, area = map(int, stats[label])
        if y+bh >= h-2 and area > h*w*.025 and bw > w*.07 and .15*h < y < .8*h:
            yy, xx = np.nonzero(labels == label)
            tip_band = xx[yy <= y+max(3, h*.02)]
            candidates.append((label, x, y, bw, tip_band))
    candidates.sort(key=lambda c: c[1])
    if len(candidates) == 2:
        left, right = candidates
        lx, rx = int(left[4].max()), int(right[4].min())
        ly, ry = left[2], right[2]
        if lx < rx and .025*w < rx-lx < .7*w and abs(ly-ry) < .2*h:
            gap = np.zeros((h, w), np.uint8)
            borders = []
            for row in range(max(ly, ry), h):
                a = np.flatnonzero(labels[row] == left[0])
                b = np.flatnonzero(labels[row] == right[0])
                if len(a) and len(b) and a[-1]+3 < b[0]:
                    gap[row, a[-1]+2:b[0]-1] = 255
                    borders.append((row, int(a[-1]), int(b[0])))
            if len(borders) >= h*.12:
                result['fingers'] = dict(kind='dark_finger_candidates', tips_px=[[lx, ly], [rx, ry]])
                for px, py in ((lx, ly), (rx, ry)):
                    cv2.circle(marked, (px, py), 6, (255, 220, 0), 2)
                gap_contours, _ = cv2.findContours(gap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(marked, gap_contours, -1, (255, 220, 0), 1)
                if result['target']:
                    overlap = float(np.count_nonzero((mask > 0) & (gap > 0))/np.count_nonzero(mask))
                    cx, cy = result['target']['center_px']
                    row = min(borders, key=lambda r: abs(r[0]-cy))
                    result['gap_overlap_fraction'] = overlap
                    result['lateral_error_px'] = float(cx-(row[1]+row[2])/2)
                    # A truncated/occluded handle is never a positive alignment.
                    if overlap >= .8 and not result['target']['clipped']:
                        result['lateral_state'] = 'inside_gap_2d'
                        result['message'] = '手柄候选大部分在两指间的图像区域；高度与夹持仍未知'
                    elif overlap < .2:
                        result['lateral_state'] = 'outside_gap_2d'
                        result['message'] = '手柄候选主要在两指间图像区域之外，不能认定对准'
                    else:
                        result['message'] = '手柄与指间区域部分重叠或画面截断，不能认定对准'
    if result['target'] and result['fingers'] is None:
        result['message'] = '已找到红色手柄候选，双指轮廓不明确；无法判断对位'
    cv2.putText(marked, 'PASSIVE / ' + result['lateral_state'], (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 220, 255), 1, cv2.LINE_AA)
    return result, marked


class GraspObserver:
    def __init__(self, frames):
        self.frames = frames
        self.lock = threading.Lock()
        self.result = None
        self.jpeg = None
        self.stamp_ns = 0

    def run(self, stop):
        while not stop.is_set():
            snapshot = self.frames.snapshot('hand', max_age_ms=300)
            if snapshot is not None and snapshot[1] != self.stamp_ns:
                frame, stamp_ns = snapshot
                try:
                    result, marked = observe_hand(frame)
                    ok, encoded = cv2.imencode('.jpg', marked)
                    with self.lock:
                        self.result = result
                        self.jpeg = encoded.tobytes() if ok else None
                        self.stamp_ns = stamp_ns
                except Exception:
                    with self.lock:
                        self.result = None
                        self.jpeg = None
                        self.stamp_ns = 0
            stop.wait(.2)

    def status(self):
        with self.lock:
            age_ms = (time.monotonic_ns()-self.stamp_ns)/1e6 if self.stamp_ns else None
            if age_ms is None or age_ms > 600:
                return dict(fresh=False, age_ms=age_ms, lateral_state='unknown',
                            alignment_verified=False, grasp_verified=False,
                            message='腕部图像或检测结果过期，无法判断对位')
            return dict(self.result, fresh=True, age_ms=age_ms)

    def image(self):
        with self.lock:
            return self.jpeg if self.stamp_ns and time.monotonic_ns()-self.stamp_ns <= 600e6 else None
