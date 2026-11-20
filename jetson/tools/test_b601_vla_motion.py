import math
import unittest
from b601_vla_motion import DEFAULT_PROFILE, validate_profile, describe_profile, VlaPlan
from b601_rs_teleop import bounded_vla_targets, Trajectory


class MotionProfileTests(unittest.TestCase):
    def test_profile_without_arm_velocity_cap_is_described(self):
        profile = validate_profile({**self.profile(), 'speed_deg_s':None,
                                    'gripper_speed_deg_s':60.})
        self.assertIsNone(profile['speed_deg_s'])
        self.assertIn('J1–J6不设独立速度上限', describe_profile(profile))
        self.assertEqual(profile['accel_deg_s2'], 10.)
        self.assertEqual(profile['gripper_speed_deg_s'], 60.)

    def test_no_arm_velocity_cap_preserves_acceleration_gripper_and_lead_limits(self):
        start = [0.]*7
        bounds = [[-180.,340.]]*7
        trajectory = Trajectory(start, [None]*6+[60.], [10.]*6+[120.])
        previous_velocity = list(trajectory.velocity)
        for _ in range(30):
            trajectory.step([90.]*6+[300.], .1, bounds)
            for i in range(7):
                self.assertTrue(math.isfinite(trajectory.position[i]))
                self.assertLessEqual(abs(trajectory.velocity[i]-previous_velocity[i]),
                                     trajectory.amax[i]*.1+1e-8)
            self.assertLessEqual(abs(trajectory.command_velocity[6]), 60.+1e-8)
            previous_velocity = list(trajectory.velocity)
        self.assertTrue(all(abs(v)>5. for v in trajectory.command_velocity[:6]))
        command, limited = trajectory.limit_lead(start, [8.]*6+[45.], bounds)
        self.assertIn(2, limited)
        self.assertLessEqual(abs(command[1]), 8.)
        self.assertEqual(trajectory.command_velocity[1], 0.)

    def test_only_arm_velocity_cap_accepts_null(self):
        self.assertIsNone(validate_profile({**self.profile(), 'speed_deg_s':None})['speed_deg_s'])
        for key in ('accel_deg_s2', 'gripper_speed_deg_s', 'lead_deg'):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_profile({**self.profile(), key:None})

    def test_legacy_profile_is_upgraded_without_losing_limits(self):
        legacy={k:v for k,v in DEFAULT_PROFILE.items() if k not in ('task_bounds_deg','action_steps')}
        self.assertEqual(validate_profile(legacy), DEFAULT_PROFILE)

    def test_task_envelope_allows_travel_beyond_old_session_boundary(self):
        p={**self.profile(), 'session_span_deg':180,
           'task_bounds_deg':[[-21,46],[0,80],[0,84],[-53,65],[-18,48],[-21,45],[-3,179]]}
        actual=[0.,40.,40.,0.,0.,0.,0.]
        origin=[0.]*7
        goal=bounded_vla_targets(actual,[100.]*7,actual,[[-140,175]]*7,
                                 previous_command=actual,session_origin=origin,profile=p)
        self.assertEqual(goal[1:3],[42.,42.])
        actual[1]=79.
        goal=bounded_vla_targets(actual,[100.]*7,actual,[[-140,175]]*7,
                                 previous_command=actual,session_origin=origin,profile=p)
        self.assertEqual(goal[1],80.)
    def profile(self):
        return {**DEFAULT_PROFILE, 'command_step_deg': 2., 'lead_deg': 4.,
                'abort_error_deg': 4.5, 'session_span_deg': 8., 'settle_error_deg': 3.5}

    def test_known_steady_error_allows_bounded_target_progress(self):
        actual = [10.,20.,30.,0.,0.,0.,0.]
        previous = list(actual)
        previous[2] += 2.3
        result = bounded_vla_targets(actual, [100.]*7, actual, [[-140,175]]*7,
                                    previous_command=previous, session_origin=actual,
                                    profile=self.profile())
        self.assertAlmostEqual(result[2], 34.)
        self.assertLessEqual(result[2]-previous[2], 2.)
        self.assertEqual(result[6], previous[6])

    def test_profile_preserves_session_and_soft_bounds(self):
        actual=[0.,164.,174.,0.,0.,0.,10.]
        origin=list(actual); origin[0]=-7.5
        limits=[[-140,140],[0,165],[0,175],[-75,85],[-85,85],[-85,85],[-3,340]]
        result=bounded_vla_targets(actual,[200.]*7,actual,limits,
                                   previous_command=actual,session_origin=origin,profile=self.profile())
        self.assertEqual(result[:3],[.5,165.,175.])
        self.assertEqual(result[6],10.)

    def test_oversized_and_nonfinite_settings_rejected(self):
        for key,value in [('lead_deg',100),('command_step_deg',float('nan')),
                          ('speed_deg_s',100),('max_steps',10.5),('max_steps',True)]:
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                validate_profile({**self.profile(),key:value})

    def test_blocked_joint_cannot_keep_accumulating_target(self):
        actual=[0.]*7; previous=[0.]*7;previous[2]=4.6
        with self.assertRaisesRegex(ValueError,'超过4.5°'):
            bounded_vla_targets(actual,[100.]*7,actual,[[-140,175]]*7,
                                previous_command=previous,session_origin=actual,profile=self.profile())

    def test_gripper_requires_enabled_profile_and_uses_separate_bounds(self):
        p={**self.profile(),'gripper_enabled':True}
        actual=[0.]*7
        actual[6]=20.
        result=bounded_vla_targets(actual,[100.]*7,actual,[[-140,340]]*7,
                                   previous_command=actual,session_origin=actual,profile=p)
        self.assertEqual(result[6],30.)
        previous=list(actual);previous[6]=48.
        result=bounded_vla_targets(actual,[100.]*7,actual,[[-140,340]]*7,
                                   previous_command=previous,session_origin=actual,profile=p)
        self.assertEqual(result[6],50.)
        previous[6]=61.
        with self.assertRaisesRegex(ValueError,'J7.*超过40°'):
            bounded_vla_targets(actual,[100.]*7,actual,[[-140,340]]*7,
                                previous_command=previous,session_origin=actual,profile=p)


class ContinuousPlanTests(unittest.TestCase):
    def setUp(self):
        self.p={**DEFAULT_PROFILE,'continuous_plan':True,'action_steps':50,
                'speed_deg_s':20.,'accel_deg_s2':40.,'lead_deg':8.,'abort_error_deg':10.,
                'session_span_deg':180.,'gripper_enabled':True,'gripper_span_deg':340.,
                'gripper_speed_deg_s':60.,'gripper_accel_deg_s2':120.}
        self.bounds=[[-140,140],[0,165],[0,175],[-75,85],[-85,85],[-85,85],[-3,340]]
        self.actions=[[0.,55.*i/49,38.*i/49,0.,0.,0.,120.*i/49] for i in range(50)]

    def simulate(self, blocked=False):
        actual=[0.]*7
        t=Trajectory(actual,[20.]*6+[60.],[40.]*6+[120.])
        plan=VlaPlan(self.actions,actual,actual,actual,self.bounds,self.p)
        dt=1/14.
        rows=[]
        for n in range(210):
            goal=plan.goal(t.position)
            cmd=t.step(goal,dt,self.bounds)
            cmd,_=t.limit_lead(actual,[8.]*6+[30.],self.bounds)
            for i in range(7):
                if blocked and i==1:continue
                actual[i]+=.75*(cmd[i]-actual[i])
            rows.append((list(cmd),list(actual),list(t.command_velocity),plan.index))
            if plan.final_reached(cmd,t.velocity):break
        return plan,t,rows

    def test_complete_path_reaches_extension_and_gripper_with_bounded_speed(self):
        plan,t,rows=self.simulate()
        self.assertTrue(plan.final_reached(t.position,t.velocity))
        self.assertLess(len(rows)/14.,15.)
        self.assertGreater(rows[-1][1][1],54.)
        self.assertGreater(rows[-1][1][6],119.)
        for cmd,actual,velocity,index in rows:
            self.assertLessEqual(max(abs(v) for v in velocity[:6]),20.00001)
            self.assertLessEqual(abs(velocity[6]),60.00001)
            self.assertLessEqual(abs(cmd[1]-actual[1]),8.)
        # Middle waypoints do not require returning to zero velocity.
        self.assertGreater(sum(r[2][1]>5 for r in rows),20)

    def test_blocked_joint_does_not_consume_the_rest_of_the_plan(self):
        plan,t,rows=self.simulate(blocked=True)
        self.assertFalse(plan.final_reached(t.position,t.velocity))
        self.assertLess(plan.index,20)
        self.assertLessEqual(max(x[0][1] for x in rows),8.00001)
        self.assertEqual(rows[-1][1][1],0.)

    def test_invalid_last_point_rejected_before_any_motion(self):
        self.actions[-1][2]=float('nan')
        with self.assertRaisesRegex(ValueError,'有限'):
            VlaPlan(self.actions,[0.]*7,[0.]*7,[0.]*7,self.bounds,self.p)

    def test_task_soft_limits_apply_to_every_point_and_gripper_can_be_held(self):
        p={**self.p,'task_bounds_deg':[[-20,20],[0,40],[0,30],[-30,30],[-30,30],[-30,30],[-3,90]],'gripper_enabled':False}
        plan=VlaPlan(self.actions,[0.]*7,[0.]*7,[0.]*7,self.bounds,p)
        self.assertEqual(plan.actions[-1][1:3],[40,30])
        self.assertTrue(all(a[6]==0 for a in plan.actions))
