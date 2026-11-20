"""No hardware: feedforward isolation, bounded startup and URDF consistency."""
import copy
import math
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from b601_rs_gravity import GravityFeedforward, validate_gravity
from b601_rs_teleop import Hardware, load_config, DEFAULT_CONFIG

CFG = dict(enabled=True, autonomous_only=True, scales=[0, .6, 0, 0, 0, 0],
           max_torque_nm=[0, 8, 0, 0, 0, 0], slew_nm_s=1.5)


class GravityTests(unittest.TestCase):
    def make(self):
        self.now = 0
        g = GravityFeedforward(CFG, autonomous=True, clock=lambda: self.now)
        g.reset()
        return g

    def test_manual_mode_and_absent_config_keep_zero_feedforward(self):
        for g in (GravityFeedforward(CFG), GravityFeedforward(None, True)):
            self.assertEqual(g.update([0]*7), [0]*6)

    def test_startup_and_long_gap_cannot_jump_to_full_torque(self):
        g = self.make()
        g.compute = lambda **kw: np.ones(g.model.nq)*100
        self.assertEqual(g.update([0]*7), [0]*6)
        self.now = 30
        output = g.update([0]*7)
        self.assertAlmostEqual(output[1], .15)
        self.assertEqual([v for i, v in enumerate(output) if i != 1], [0]*5)
        for _ in range(100):
            self.now += .1
            output = g.update([0]*7)
        self.assertAlmostEqual(output[1], 8)
        g.reset()
        self.assertEqual(g.update([0]*7), [0]*6)

    def test_real_urdf_direction_and_gripper_encoder_is_not_model_finger(self):
        g = self.make()
        pose = [-19.89,117.98,83.10,-31.62,10.02,45.19,121.42]
        for _ in range(100):
            self.now += .1
            output = g.update(pose)
        self.assertTrue(-8 < output[1] < -5)
        pose[6] = -3
        self.now += .1
        self.assertAlmostEqual(g.update(pose)[1], output[1])

    def test_invalid_input_and_unsafe_config_rejected(self):
        g = self.make()
        for pose in ([0]*6, [0]*6+[float('nan')]):
            with self.assertRaises(ValueError): g.update(pose)
        for changes in ({'scales':[2]*6}, {'max_torque_nm':[9]*6},
                        {'slew_nm_s':3}, {'enabled':1}):
            with self.assertRaises(ValueError):validate_gravity({**CFG, **changes})

    def test_only_selected_motor_gets_torque_and_gripper_remains_independent(self):
        config=load_config(DEFAULT_CONFIG)
        config['gravity_feedforward']=copy.deepcopy(CFG)
        h=Hardware(SimpleNamespace(autonomous=True), config)
        h.motors={j:Mock() for j in range(1,8)}
        h.gravity.update=Mock(return_value=[0,-2,0,0,0,0])
        h.send([0]*7,[0]*7)
        self.assertEqual(h.motors[2].send_mit.call_args.args[4], -2)
        for j in (1,3,4,5,6):self.assertEqual(h.motors[j].send_mit.call_args.args[4],0)
        self.assertEqual(h.motors[7].send_mit.call_args.args,(0,0,0,1.5,0))


if __name__ == '__main__':
    unittest.main()
