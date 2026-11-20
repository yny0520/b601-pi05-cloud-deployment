"""Action smoothing behavior and bounded controller integration, no hardware."""
import copy
import math
import unittest

from b601_vla_filter import filter_action_path
from b601_vla_motion import DEFAULT_PROFILE, VlaPlan, validate_profile
from b601_rs_teleop import Trajectory


PROFILE = {**DEFAULT_PROFILE, 'continuous_plan':True, 'action_filter_enabled':True,
           'action_steps':50, 'session_span_deg':180., 'speed_deg_s':20.,
           'accel_deg_s2':40., 'lead_deg':8., 'abort_error_deg':10.,
           'gripper_enabled':True, 'gripper_span_deg':340., 'gripper_lead_deg':45.,
           'gripper_abort_error_deg':50., 'gripper_speed_deg_s':60., 'gripper_accel_deg_s2':120.}


def roughness(path):
    return sum(abs(path[t-1][j]-2*path[t][j]+path[t+1][j])
               for t in range(1,len(path)-1) for j in range(6))


class FilterTests(unittest.TestCase):
    def test_alternating_noise_is_reduced_without_changing_grasp_endpoints(self):
        actions = [[10.+(-1)**t]*6+[100. if t<25 else 10.] for t in range(50)]
        original = copy.deepcopy(actions)
        filtered = filter_action_path(actions)
        self.assertLess(roughness(filtered), .2*roughness(actions))
        self.assertEqual(filtered[0], actions[0])
        self.assertEqual(filtered[-1], actions[-1])
        self.assertEqual([a[6] for a in filtered], [a[6] for a in actions])
        self.assertEqual(actions, original)

    def test_single_point_gripper_closure_is_not_deleted_or_delayed(self):
        actions = [[float(t)]*6+[150.] for t in range(50)]
        actions[35][6] = -3.
        filtered = filter_action_path(actions)
        self.assertEqual([a[6] for a in filtered], [a[6] for a in actions])

    def test_linear_motion_and_explicit_short_jogs_are_exact(self):
        actions = [[t*.5+j for j in range(7)] for t in range(50)]
        self.assertEqual(filter_action_path(actions), actions)
        self.assertEqual(filter_action_path(actions[:1]), actions[:1])
        self.assertEqual(filter_action_path(actions[:2]), actions[:2])

    def test_large_spike_correction_is_capped_and_no_new_extrema_created(self):
        actions = [[0.]*7 for _ in range(50)]
        actions[20][:6] = [50.]*6
        filtered = filter_action_path(actions, 2.)
        self.assertEqual(filtered[20][:6], [48.]*6)
        for raw, result in zip(actions, filtered):
            for j in range(6):
                self.assertLessEqual(abs(raw[j]-result[j]), 2.000001)
                self.assertTrue(0 <= result[j] <= 50)

    def test_new_chunk_has_no_memory_of_previous_pose(self):
        first = filter_action_path([[1.+(-1)**t]*7 for t in range(50)])
        second = filter_action_path([[100.]*7 for _ in range(50)])
        self.assertEqual(second, [[100.]*7 for _ in range(50)])
        self.assertNotEqual(first, second)

    def test_invalid_points_cannot_be_hidden_by_filtering(self):
        for invalid in (math.nan, math.inf, True, '0'):
            path = [[0.]*7 for _ in range(50)]
            path[-1][2] = invalid
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                filter_action_path(path)
        for cap in (0,4,True,math.nan):
            with self.subTest(cap=cap), self.assertRaises(ValueError):
                filter_action_path([[0.]*7],cap)

    def test_profile_opt_in_and_legacy_limits_unchanged(self):
        old = {k:v for k,v in DEFAULT_PROFILE.items() if not k.startswith('action_filter_')}
        loaded = validate_profile(old)
        self.assertFalse(loaded['action_filter_enabled'])
        self.assertEqual(loaded['speed_deg_s'], DEFAULT_PROFILE['speed_deg_s'])
        with self.assertRaises(ValueError):
            validate_profile({**DEFAULT_PROFILE, 'action_filter_enabled':True})
        with self.assertRaises(ValueError):
            validate_profile({**PROFILE, 'action_filter_max_delta_deg':4.})

    def test_cut_prefix_end_and_every_soft_limit_preserved(self):
        bounds = [[-20.,20.]]*6+[[-3.,180.]]
        actions = [[19.+(-1)**t]*6+[150. if t<16 else 10.] for t in range(50)]
        # A guarded prefix ends at the exact raw stopping pose. Later samples
        # cannot bleed across that checkpoint.
        plan = VlaPlan(actions[:16],[0.]*7,[0.]*7,[0.]*7,bounds,PROFILE)
        self.assertEqual(plan.actions[-1], actions[15])
        self.assertEqual([a[6] for a in plan.actions],[a[6] for a in actions[:16]])
        self.assertTrue(all(bounds[j][0] <= a[j] <= bounds[j][1]
                            for a in plan.actions for j in range(7)))

    def test_filtered_plan_remains_speed_limited_and_blocked_joint_stops_progress(self):
        bounds = [[-20.,20.]]*6+[[-3.,180.]]
        actions = [[15.+(-1)**t]*6+[20.] for t in range(50)]
        actual = [0.]*7
        plan = VlaPlan(actions,actual,actual,actual,bounds,PROFILE)
        trajectory = Trajectory(actual,[20.]*6+[60.],[40.]*6+[120.])
        for _ in range(140):
            trajectory.step(plan.goal(trajectory.position),1/14,bounds)
            command,_ = trajectory.limit_lead(actual,[8.]*6+[45.],bounds)
            for j in range(7):
                self.assertLessEqual(abs(trajectory.command_velocity[j]),20.00001 if j<6 else 60.00001)
                if j!=2:
                    actual[j] += .75*(command[j]-actual[j])
            self.assertLessEqual(command[2],8.00001)
        self.assertEqual(plan.index,0)
        self.assertFalse(plan.final_reached(trajectory.position,trajectory.velocity))


if __name__ == '__main__':
    unittest.main()
