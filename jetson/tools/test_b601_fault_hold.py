"""Motor-free fault injection through the actual teleop control loop."""
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import b601_rs_teleop as control


class LoopFaultHoldTests(unittest.TestCase):
    def run_loop(self, failure='uart', keys=None, recover=False):
        folder=tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        root=Path(folder.name)
        actual=[1.,20.,30.,-10.,5.,6.,1.]
        reads=[];sends=[];packets=[];close_count=[]
        hardware=SimpleNamespace(touched=False,saved={},read_window_ns=[1,2],
            joint_read_windows_ns={},last_read_s=.01,leader_error='',last_mit_commands=None,
            gripper_torque_command_nm=0.,gripper_velocity_rad_s=0.,motor_feedback=[])
        hardware.open=Mock();hardware.check_faults=Mock()
        def prepare(callback):
            hardware.touched=True
        hardware.prepare=prepare
        hardware.enable_at=Mock()
        def read(**opts):
            reads.append(dict(opts))
            if len(reads)==3 and failure=='pause':
                actual[1] += 3.
                actual[6] += 4.
            if len(reads)==3 and failure=='can':
                raise RuntimeError('CAN lost; cannot guarantee hold')
            if len(reads)==3 and failure=='uart':
                hardware.leader_error="主臂读取失败：SerialException: (5, 'Input/output error')"
                return None,list(actual)
            if opts.get('skip_leader'):
                return None,list(actual)
            return [0.]*7,list(actual)
        hardware.read=read
        hardware.send=lambda command,state,velocity:sends.append((list(command),list(state),list(velocity)))
        hardware.close=lambda:close_count.append(True) or []
        hardware.request_leader_recovery=Mock()
        def recovered():
            if recover and hardware.request_leader_recovery.called:
                if recover=='failure':
                    raise OSError('UART still disconnected')
                # Deliberately far from old leader state. Explicit R rebases,
                # never chases the previous mapping or disconnected movement.
                return [40.]*7,control.time.monotonic_ns()
            return None
        hardware.take_leader_recovery=recovered
        keys=iter(keys or ['r','','','q'])
        keyboard=Mock()
        keyboard.__enter__=Mock(return_value=keyboard)
        keyboard.__exit__=Mock(return_value=False)
        def read_keys():
            k=next(keys)
            if k=='input-error':
                raise OSError('PTY Input/output error')
            if k=='check-hold-before-q':
                self.assertFalse(close_count)
                self.assertGreaterEqual(len(sends),3)
                return 'q'
            return k
        keyboard.read=read_keys
        telemetry=Mock()
        def publish(**p):
            packets.append(p)
            if failure=='input' and len(packets)>=3:
                # Once stdin is broken, exit through the separate explicit
                # stop-file channel, never through the unavailable keyboard.
                (root/'STOP').touch()
        telemetry.publish.side_effect=publish
        args=control.parse_args(['--live','--joint','all','--full-range','--stop-file',str(root/'STOP')])
        with patch.object(control,'ROOT',root),patch.object(control,'Hardware',return_value=hardware),patch.object(
                control,'Keyboard',return_value=keyboard),patch.object(control,'TelemetryPublisher',return_value=telemetry),patch.object(
                control,'resolve_can_channel',return_value='can-test'),patch.object(control.sys.stdin,'isatty',return_value=True),patch.object(
                control,'TERMINAL_LOST',False),contextlib.redirect_stdout(io.StringIO()):
            if failure=='can':
                with self.assertRaisesRegex(RuntimeError,'CAN lost'):
                    control.run(args)
            else:
                control.run(args)
        events=[]
        for p in (root/'teleop-logs').glob('*/events.jsonl'):
            events.extend(json.loads(s) for s in p.read_text().splitlines())
        return hardware,reads,sends,packets,events

    def test_uart_disconnect_keeps_can_hold_until_explicit_quit(self):
        h,reads,sends,packets,events=self.run_loop(keys=['r','','','check-hold-before-q'])
        holds=[p for p in packets if p.get('fault_hold')]
        self.assertGreaterEqual(len(holds),2)
        self.assertTrue(reads[-1]['skip_leader'])
        h.request_leader_recovery.assert_not_called()
        for p in holds:
            self.assertIsNone(p['leader_deg'])
            self.assertFalse(p['leader_read'])
            self.assertTrue(p['paused'])
            self.assertTrue(p['motor_commands_sent'])
            self.assertEqual(p['vla_step']['state'],'aborted')
            self.assertEqual(p['command_deg'],p['follower_deg'])
        for command,actual,velocity in sends[1:]:
            self.assertEqual(command,actual)
            self.assertEqual(velocity,[0.]*7)
        self.assertTrue(any(e['event']=='fault_hold' for e in events))

    def test_recovery_requires_explicit_r_and_rebases_to_measured_pose(self):
        h,reads,sends,packets,events=self.run_loop(keys=['r','','','r','','q'],recover=True)
        h.request_leader_recovery.assert_called_once()
        recovered=[e for e in events if e['event']=='fault_hold_recovered']
        self.assertEqual(len(recovered),1)
        self.assertFalse(packets[-1]['fault_hold'])
        # First resumed arm commands hold measured pose even though the
        # leader moved40degrees while disconnected. Gripper remains independent.
        first=next(p for p in packets[2:] if not p['fault_hold'])
        self.assertEqual(first['command_deg'][:6],first['follower_deg'][:6])

    def test_terminal_io_error_holds_instead_of_disabling(self):
        h,reads,sends,packets,events=self.run_loop(failure='input',keys=['r','input-error','check-hold-before-q'])
        self.assertTrue(any(p.get('fault_hold') for p in packets))
        self.assertIn('终端',next(e['reason'] for e in events if e['event']=='fault_hold'))

    def test_failed_reconnect_stays_latched_without_automatic_resume(self):
        h,_,_,packets,events=self.run_loop(keys=['r','','','r','','q'],recover='failure')
        h.request_leader_recovery.assert_called_once()
        self.assertTrue(packets[-1]['fault_hold'])
        self.assertFalse(any(e['event']=='fault_hold_recovered' for e in events))

    def test_pause_cancels_recovery_even_when_result_arrives(self):
        h,_,_,packets,events=self.run_loop(keys=['r','','','rp','','q'],recover=True)
        h.request_leader_recovery.assert_not_called()
        self.assertTrue(packets[-1]['fault_hold'])
        self.assertFalse(any(e['event']=='fault_hold_recovered' for e in events))

    def test_broken_output_requests_hold_and_other_output_errors_propagate(self):
        with patch.object(control,'TERMINAL_LOST',False),patch('builtins.print',side_effect=OSError(5,'PTY closed')):
            control.print('test')
            self.assertTrue(control.TERMINAL_LOST)
        with patch('builtins.print',side_effect=OSError(28,'disk full')):
            with self.assertRaises(OSError):control.print('test')

    def test_can_failure_does_not_claim_successful_hold(self):
        _,_,_,packets,events=self.run_loop(failure='can')
        self.assertFalse(any(p.get('fault_hold') for p in packets))
        error=next(e for e in events if e['event']=='error')
        self.assertEqual(error['exception_type'],'RuntimeError')
        self.assertIn('CAN lost',error['traceback'])

    def test_manual_pause_freezes_measured_arm_and_preserves_gripper_target(self):
        _,_,_,packets,_=self.run_loop(failure='pause',keys=['r','','p','','q'])
        before, paused = packets[1], packets[2]
        self.assertTrue(paused['paused'])
        self.assertEqual(paused['command_deg'][:6],paused['follower_deg'][:6])
        self.assertEqual(paused['command_deg'][6],before['command_deg'][6])
        self.assertNotEqual(paused['command_deg'][6],paused['follower_deg'][6])


if __name__=='__main__':unittest.main()
