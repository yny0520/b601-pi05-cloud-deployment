from b601_control_io import runtime_path
import sys
import copy
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, '/home/jetson/Documents/ChatGPT/b601-rs/tools')
import b601_rs_teleop as control
from b601_rs_teleop import bounded_vla_targets
from b601_vla_motion import DEFAULT_PROFILE


class BoundedVlaStepTests(unittest.TestCase):
    def setUp(self):
        self.actual = [10.0, 20.0, 30.0, 0.0, 5.0, -5.0, 12.0]
        self.bounds = [[-140, 140], [0, 165], [0, 175], [-75, 85],
                       [-85, 85], [-85, 85], [-3, 340]]

    def test_clamps_six_arm_joints_and_holds_gripper(self):
        predicted = [100.0, -30.0, 30.2, 4.0, 4.8, -4.5, 300.0]
        result = bounded_vla_targets(self.actual, predicted, self.actual, self.bounds)
        self.assertEqual(result, [10.45, 19.55, 30.2, 0.45, 4.8, -4.55, 12.0])

    def test_prediction_beyond_soft_stop_saturates_instead_of_rejecting(self):
        actual = list(self.actual)
        actual[1], actual[2] = .1, .12
        predicted = list(actual)
        predicted[1], predicted[2] = -20., -10.
        result = bounded_vla_targets(actual, predicted, actual, self.bounds,
                                     previous_command=actual, session_origin=actual)
        self.assertEqual(result[1:3], [0., 0.])

    def test_stationary_joint_and_encoder_noise_do_not_reject_next_step(self):
        previous = list(self.actual)
        for n in range(10):
            measured = [v + (.01 if n % 2 else -.01) for v in self.actual]
            previous = bounded_vla_targets(measured, [100.]*7, measured, self.bounds,
                                            previous_command=previous, session_origin=self.actual)
            self.assertLessEqual(abs(previous[2]-measured[2]), .9000001)

    def test_command_advances_through_steady_load_error_with_bounded_lead(self):
        first = bounded_vla_targets(self.actual, [100.]*7, self.actual, self.bounds,
                                    previous_command=self.actual, session_origin=self.actual)
        second = bounded_vla_targets(self.actual, [100.]*7, self.actual, self.bounds,
                                     previous_command=first, session_origin=self.actual)
        third = bounded_vla_targets(self.actual, [100.]*7, self.actual, self.bounds,
                                    previous_command=second, session_origin=self.actual)
        self.assertAlmostEqual(first[2]-self.actual[2], .45)
        self.assertAlmostEqual(second[2]-first[2], .45)
        self.assertAlmostEqual(third[2]-self.actual[2], .9)

    def test_excessive_feedback_error_blocks_further_command_buildup(self):
        previous = list(self.actual)
        previous[2] += 1.01
        with self.assertRaisesRegex(ValueError, '相差超过1°'):
            bounded_vla_targets(self.actual, [100.]*7, self.actual, self.bounds,
                                previous_command=previous, session_origin=self.actual)

    def test_rejects_state_drift_after_inference(self):
        expected = list(self.actual)
        expected[2] += .51
        with self.assertRaisesRegex(ValueError, '变化超过0.5°'):
            bounded_vla_targets(self.actual, self.actual, expected, self.bounds)

    def test_rejects_non_finite_predictions(self):
        predicted = list(self.actual)
        predicted[0] = float('nan')
        with self.assertRaisesRegex(ValueError, '非有限数值'):
            bounded_vla_targets(self.actual, predicted, self.actual, self.bounds)

    def test_previous_command_and_session_envelope_bound_the_goal(self):
        previous = list(self.actual)
        previous[0] -= .4
        origin = list(self.actual)
        origin[1] -= 4.9
        result = bounded_vla_targets(self.actual, [100.]*7, self.actual, self.bounds,
                                     previous_command=previous, session_origin=origin)
        self.assertAlmostEqual(result[0], 10.05)
        self.assertAlmostEqual(result[1], 20.1)
        self.assertEqual(result[6], previous[6])


class ControlLoopTests(unittest.TestCase):
    def exercise(self, keys, malformed=False, autonomous=False, continuous=False, replace=None, completed_pause=False):
        pose = [10., 20., 30., 0., 5., -5., 12.]
        samples, sends = [], []
        clock = [100.]
        request_path = runtime_path(os.getpid(), 'json')
        status_path = runtime_path(os.getpid(), 'status.json')
        abort_path = runtime_path(os.getpid(), 'abort')
        self.addCleanup(request_path.unlink, missing_ok=True)
        self.addCleanup(status_path.unlink, missing_ok=True)
        self.addCleanup(abort_path.unlink, missing_ok=True)

        class Hardware:
            touched = False
            saved = {}
            read_window_ns = [0, 1]
            joint_read_windows_ns = []
            last_read_s = .001
            last_mit_commands = []
            gripper_torque_command_nm = gripper_velocity_rad_s = 0.

            def __init__(self, *a): pass
            def open(self): pass
            def read(self, allow_leader_fault=False, skip_leader=False):
                clock[0] += .01
                if completed_pause and len(samples)==2:
                    pose[1] += 2.
                    pose[6] += 3.
                return None if skip_leader else [0.]*7, list(pose)
            def prepare(self, callback): self.touched = True
            def enable_at(self, actual): self.enabled = list(actual)
            def check_faults(self): pass
            def close(self): return []
            def send(self, command, actual, velocity): sends.append(list(command))

        iterator = iter(keys)
        class Keyboard:
            def __enter__(self): return self
            def __exit__(self, *a): pass
            def read(self): return next(iterator)

        class Telemetry:
            def __init__(self, *a): pass
            def close(self): pass
            def publish(self, **sample):
                samples.append(copy.deepcopy(sample))
                if len(samples) == 1:
                    request_path.write_text('{bad' if malformed else json.dumps({
                        'id': 'test', 'created_mono_ns': time.monotonic_ns(),
                        'action_deg': list(pose) if completed_pause else [100.]*7, 'expected_state_deg': pose,
                        'session_origin_deg': pose, **({'trajectory_deg':[[100.]*7 for _ in range(50)]} if continuous else {})}))
                if replace is not None and len(samples) == 3:
                    request_path.write_text(json.dumps({
                        'id':'next', 'replace_id':replace, 'created_mono_ns':time.monotonic_ns(),
                        'expected_mono_ns':time.monotonic_ns(), 'action_deg':[100.]*7,
                        'expected_state_deg':pose, 'session_origin_deg':pose,
                        'trajectory_deg':[[100.]*7 for _ in range(15)]}))

        with tempfile.TemporaryDirectory() as tmp:
            args = control.parse_args(['--live', '--full-range', '--stop-file', str(Path(tmp)/'stop')]+
                                      (['--autonomous'] if autonomous else []))
            with patch.object(control, 'ROOT', Path(tmp)), patch.object(control, 'Hardware', Hardware), \
                 patch.object(control, 'Keyboard', Keyboard), patch.object(control, 'TelemetryPublisher', Telemetry), \
                 patch.object(control, 'resolve_can_channel', return_value='fake'), \
                 patch.object(control, 'load_motion_profile', return_value={**DEFAULT_PROFILE,'continuous_plan':continuous,
                              'rolling_replan':replace is not None}), \
                 patch.object(control.sys.stdin, 'isatty', return_value=True), \
                 patch('builtins.print'), patch.object(control.time, 'sleep'), \
                 patch.object(control.time, 'monotonic', side_effect=lambda: clock[0]):
                control.run(args)
        return samples, sends, request_path

    def test_pause_after_completed_plan_resets_arm_target_and_zeroes_feedforward(self):
        samples,sends,_=self.exercise(['h','','p','','q'],autonomous=True,completed_pause=True)
        self.assertEqual(samples[1]['vla_step']['state'],'complete')
        self.assertEqual(samples[2]['command_deg'][:6],samples[2]['follower_deg'][:6])
        self.assertEqual(samples[2]['command_deg'][6],samples[1]['command_deg'][6])
        self.assertEqual(samples[3]['command_deg'],samples[2]['command_deg'])

    def test_real_loop_accepts_named_replacement_without_restarting_velocity(self):
        samples,sends,_=self.exercise(['h','','','','','p','q'],autonomous=True,continuous=True,replace='test')
        self.assertEqual(samples[3]['vla_step']['id'],'next')
        self.assertEqual(samples[3]['vla_step']['replaces'],'test')
        # Constant positive goal: a velocity reset would restart the tiny first
        # ramp. Equal minimum dt means the next displacement must grow instead.
        self.assertGreater(sends[3][0]-sends[2][0], sends[1][0]-sends[0][0])
        self.assertEqual(samples[-1]['vla_step']['state'],'aborted')
        self.assertEqual(sends[-1],sends[0])

    def test_real_loop_rejects_wrong_replacement_and_holds_feedback(self):
        samples,sends,_=self.exercise(['h','','','','','q'],autonomous=True,continuous=True,replace='wrong')
        self.assertEqual(samples[3]['vla_step']['state'],'rejected')
        self.assertEqual(sends[3],sends[0])

    def test_pause_wins_over_replacement_in_same_cycle(self):
        samples,sends,path=self.exercise(['h','','','p','q'],autonomous=True,continuous=True,replace='test')
        self.assertFalse(path.exists())
        self.assertFalse(any(s['vla_step']['id']=='next' for s in samples))
        self.assertEqual(sends[-1],sends[0])

    def test_continuous_request_enters_control_loop_and_pause_aborts_whole_path(self):
        samples,sends,_=self.exercise(['h','','','p','q'],autonomous=True,continuous=True)
        self.assertTrue(samples[1]['vla_step']['continuous_plan'])
        self.assertEqual(samples[1]['vla_step']['points'],50)
        self.assertEqual(samples[-1]['vla_step']['state'],'aborted')
        self.assertEqual(sends[-1],sends[0])

    def test_pause_wins_over_pending_full_path(self):
        samples,sends,path=self.exercise(['h','p','q'],autonomous=True,continuous=True)
        self.assertFalse(path.exists())
        self.assertFalse(any(s['vla_step']['state']=='executing' for s in samples))

    def test_autonomous_hold_has_no_fabricated_leader_and_r_does_not_follow(self):
        samples, sends, _ = self.exercise(['h', 'r', 'p', 'q'], autonomous=True)
        self.assertTrue(sends)
        for sample in samples:
            self.assertIsNone(sample['leader_deg'])
            self.assertFalse(sample['leader_read'])
            self.assertIsNone(sample['alignment']['leader_reference_deg'])
            self.assertTrue(sample['paused'])

    def test_hold_starts_without_leader_or_gripper_chase_and_abort_holds_feedback(self):
        samples, sends, _ = self.exercise(['h', '', '', 'p', 'q'])
        self.assertTrue(samples[0]['paused'])
        self.assertEqual(sends[0], [10., 20., 30., 0., 5., -5., 12.])
        self.assertEqual(samples[1]['vla_step']['state'], 'executing')
        self.assertEqual(samples[-1]['vla_step']['state'], 'aborted')
        self.assertEqual(sends[-1], sends[0])
        for command in sends:
            self.assertTrue(all(abs(a-b) <= .5 for a, b in zip(command, sends[0])))
            self.assertEqual(command[6], 12.)

    def test_pause_wins_over_pending_request_in_same_control_cycle(self):
        samples, sends, path = self.exercise(['h', 'p', 'q'])
        self.assertFalse(path.exists())
        self.assertFalse(any(s['vla_step']['state'] == 'executing' for s in samples))
        self.assertEqual(sends[0], sends[-1])

    def test_corrupt_request_is_consumed_once_without_motion(self):
        samples, sends, path = self.exercise(['h', '', '', 'q'], malformed=True)
        self.assertFalse(path.exists())
        self.assertEqual(samples[1]['vla_step']['state'], 'rejected')
        self.assertTrue(all(c == sends[0] for c in sends))


if __name__ == '__main__':
    unittest.main()
