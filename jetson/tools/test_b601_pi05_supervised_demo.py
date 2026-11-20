import unittest
from b601_pi05_supervised_demo import can_return, return_trajectory


class RecoveryOwnershipTests(unittest.TestCase):
    def state(self, run_state='complete', **changes):
        v = {'teleop': {'running': True, 'pid': 10, 'control_epoch': 3,
                        'state': 'paused', 'fault_hold': False},
             'feedback_fresh': True, 'run': {'active': False, 'state': run_state}}
        v['teleop'].update(changes)
        return v

    def test_user_model_stop_does_not_start_return(self):
        self.assertFalse(can_return(self.state('stopped'), 10, 3))

    def test_own_thermal_stop_can_return(self):
        self.assertTrue(can_return(self.state('stopped'), 10, 3, own_stop=True))

    def test_changed_epoch_or_process_revokes_recovery(self):
        self.assertFalse(can_return(self.state(control_epoch=4), 10, 3, True))
        self.assertFalse(can_return(self.state(pid=11), 10, 3, True))

    def test_fault_or_lost_feedback_revokes_recovery(self):
        self.assertFalse(can_return(self.state(fault_hold=True), 10, 3, True))
        v = self.state()
        v['feedback_fresh'] = False
        self.assertFalse(can_return(v, 10, 3, True))

    def test_return_preserves_current_gripper_without_reopening(self):
        goal, actions = return_trajectory([20]*6+[85], [0]*6+[300])
        self.assertEqual(goal, [0]*6+[85])
        self.assertTrue(all(a[6] == 85 for a in actions))


if __name__ == '__main__':
    unittest.main()
