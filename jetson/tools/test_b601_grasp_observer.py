import threading
import time
import unittest
from unittest.mock import Mock

import cv2
import numpy as np

from b601_grasp_observer import observe_hand, GraspObserver


class ObservationTests(unittest.TestCase):
    def scene(self, target_x=300):
        im = np.full((360, 640, 3), 190, np.uint8)
        cv2.fillPoly(im, [np.array([[0,359],[240,359],[240,160],[200,180],[120,300]])], (10,10,10))
        cv2.fillPoly(im, [np.array([[400,160],[400,359],[639,359],[510,300],[440,180]])], (10,10,10))
        cv2.rectangle(im, (target_x,200), (target_x+30,270), (0,0,200), -1)
        return im

    def test_actual_finger_gap_used_instead_of_image_centre(self):
        inside,_ = observe_hand(self.scene())
        self.assertEqual(inside['lateral_state'], 'inside_gap_2d')
        outside,_ = observe_hand(self.scene(520))
        self.assertEqual(outside['lateral_state'], 'outside_gap_2d')
        for result in (inside,outside):
            self.assertFalse(result['alignment_verified'])
            self.assertFalse(result['grasp_verified'])

    def test_blank_missing_fingers_multiple_red_objects_and_clipping_are_unknown(self):
        images = [np.full((360,640,3),190,np.uint8)]
        red_only = images[0].copy()
        cv2.rectangle(red_only,(300,200),(330,270),(0,0,200),-1)
        images.append(red_only)
        multiple = self.scene()
        cv2.rectangle(multiple,(345,200),(375,270),(0,0,200),-1)
        images.append(multiple)
        for im in images:
            result,_ = observe_hand(im)
            self.assertEqual(result['lateral_state'],'unknown')
            self.assertFalse(result['alignment_verified'])

    def test_stale_image_cannot_retain_positive_observation(self):
        observer = GraspObserver(Mock())
        observer.result,_ = observe_hand(self.scene())
        observer.stamp_ns = time.monotonic_ns()-1_000_000_000
        observer.jpeg = b'old'
        self.assertFalse(observer.status()['fresh'])
        self.assertEqual(observer.status()['lateral_state'],'unknown')
        self.assertIsNone(observer.image())

    def test_processing_failure_does_not_kill_observer_or_preserve_old_result(self):
        frames = Mock()
        frames.snapshot.return_value=(np.zeros((2,2),np.uint8),time.monotonic_ns())
        observer = GraspObserver(frames)
        stop=threading.Event()
        frames.snapshot.side_effect=lambda *a,**k: (stop.set() or (np.zeros((2,2),np.uint8),time.monotonic_ns()))
        observer.run(stop)
        self.assertFalse(observer.status()['fresh'])


if __name__=='__main__': unittest.main()
