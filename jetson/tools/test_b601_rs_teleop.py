"""Offline regressions: protocol validity and failure-safe control plumbing."""
import copy
import math
import json
import struct
import threading
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from b601_rs_teleop import (Hardware, RelativeMapper, Trajectory, load_config,
                            DEFAULT_CONFIG, parse_args, ensure_peak_can_ready,
                            resolve_can_channel)
from star_arm_readonly import FreshStarReader, query_packet, take_packet, sync_monitor_packet


def reply(code, params):
    data = b'\x05\x1c'+bytes((code,len(params)))+params
    return data+bytes((sum(data)&255,))


class ProtocolTests(unittest.TestCase):
    @patch('b601_rs_teleop.Path.exists', return_value=True)
    @patch('b601_rs_teleop.Path.read_text')
    def test_auto_can_tracks_usb_identity_across_number_changes(self, read_text, exists):
        for channel in ('can0', 'can1', 'can12'):
            read_text.return_value = f'0 mttcan can99\n32 usb {channel} ffffffff'
            self.assertEqual(resolve_can_channel('auto'), channel)
        for listing in ('0 mttcan can0', '32 usb can0\n33 usb can1'):
            read_text.return_value = listing
            with self.assertRaisesRegex(RuntimeError, '唯一'):
                resolve_can_channel('auto')
        self.assertEqual(resolve_can_channel('can1'), 'can1')

    @patch('b601_rs_teleop.fixed_io_action', return_value=True)
    @patch('b601_rs_teleop.subprocess.run')
    @patch('b601_rs_teleop.Path.read_text', return_value='32 usb can1 ffffffff')
    @patch('b601_rs_teleop.Path.exists', return_value=True)
    def test_renumbered_usb_uses_fixed_helper_without_password_prompt(self, exists, read_text, run, io_action):
        ready = json.dumps([{'flags':['UP'], 'linkinfo':{'info_data':
            {'state':'ERROR-ACTIVE', 'bittiming':{'bitrate':1000000}}}}])
        run.side_effect = [SimpleNamespace(returncode=0, stdout='[{"flags":[]}]'),
                           SimpleNamespace(returncode=0, stdout=ready)]
        ensure_peak_can_ready('can1')
        self.assertEqual([c.args for c in io_action.call_args_list], [('can-config',), ('can-up',)])
        self.assertEqual(run.call_count, 2)

    @patch('b601_rs_teleop.fixed_io_action', return_value=False)
    @patch('b601_rs_teleop.subprocess.run')
    @patch('b601_rs_teleop.Path.read_text', return_value='*header*\n32 usb can0 ffffffff')
    @patch('b601_rs_teleop.Path.exists', return_value=True)
    def test_down_peak_interface_is_configured_and_started(self, exists, read_text, run, io_action):
        ready=json.dumps([{'flags':['UP'], 'operstate':'UNKNOWN', 'linkinfo':{'info_data':
                          {'state':'ERROR-ACTIVE','bittiming':{'bitrate':1000000}}}}])
        run.side_effect=[SimpleNamespace(returncode=0,stdout='[{"flags":[]}]',stderr=''),
                         SimpleNamespace(returncode=0),
                         SimpleNamespace(returncode=0,stdout='',stderr=''),
                         SimpleNamespace(returncode=0,stdout='',stderr=''),
                         SimpleNamespace(returncode=0,stdout=ready,stderr='')]
        ensure_peak_can_ready('can0')
        self.assertEqual(run.call_args_list[2].args[0][0:8],
                         ['sudo','-n','ip','link','set','dev','can0','type'])
        self.assertEqual(run.call_args_list[3].args[0],
                         ['sudo','-n','ip','link','set','dev','can0','up'])

    @patch('b601_rs_teleop.subprocess.run')
    @patch('b601_rs_teleop.Path.read_text', return_value='32 usb can0 ffffffff')
    @patch('b601_rs_teleop.Path.exists', return_value=True)
    def test_running_bus_is_not_reconfigured(self, exists, read_text, run):
        for bitrate in (1000000,500000):
            run.reset_mock()
            run.return_value=SimpleNamespace(returncode=0,stderr='',stdout=json.dumps([
                {'flags':['UP'],'linkinfo':{'info_data':{'state':'ERROR-ACTIVE',
                 'bittiming':{'bitrate':bitrate}}}}]))
            if bitrate == 1000000:
                ensure_peak_can_ready('can0')
            else:
                with self.assertRaisesRegex(RuntimeError,'波特率不符'):
                    ensure_peak_can_ready('can0')
            run.assert_called_once()

    @patch('b601_rs_teleop.subprocess.run')
    @patch('b601_rs_teleop.Path.read_text', return_value='*header*\n0 mttcan can1')
    @patch('b601_rs_teleop.Path.exists', return_value=True)
    def test_platform_can_is_never_auto_configured(self, exists, read_text, run):
        with self.assertRaisesRegex(RuntimeError, '拒绝自动配置'):
            ensure_peak_can_ready('can1')
        run.assert_not_called()

    def test_only_read_commands(self):
        for code in (8, 13, 17, 23, 24):
            with self.assertRaises(ValueError):
                query_packet(code, 0)

    def test_split_packet_and_noise(self):
        packet = reply(16,b'\x02'+struct.pack('<ih',-123,0))
        buf = bytearray(b'noise'+packet[:3])
        self.assertIsNone(take_packet(buf))
        buf.extend(packet[3:])
        self.assertEqual(take_packet(buf), (16, b'\x02'+struct.pack('<ih',-123,0)))
        self.assertEqual(buf, b'')

    def test_bad_checksum_frame_does_not_hide_next_valid_frame(self):
        data = bytearray(reply(1,b'\x00'))
        data[-1] ^= 1
        data.extend(reply(16,b'\x02'+struct.pack('<ih',-123,0)))
        self.assertEqual(take_packet(data),(16,b'\x02'+struct.pack('<ih',-123,0)))
        self.assertEqual(data, bytearray())

    def test_zero_is_fresh_and_not_cached(self):
        reader = FreshStarReader.__new__(FreshStarReader)
        reader.query = Mock(side_effect=lambda code,i: bytes((i,))+struct.pack('<ih',0,0))
        self.assertEqual(reader.read_positions_sequential(), [0]*7)
        self.assertEqual(reader.query.call_count, 7)
        reader.query.side_effect = RuntimeError('timeout')
        with self.assertRaises(RuntimeError):
            reader.read_positions_sequential()

    def sync_reader(self, ids=range(7), angle=0):
        reader=FreshStarReader.__new__(FreshStarReader)
        reader.serial=Mock()
        reader.serial.in_waiting=200
        packets=b''.join(reply(22,bytes((i,))+bytes(9)+struct.pack('<ih',angle,0)) for i in ids)
        reader.serial.read.side_effect=lambda _: packets if reader.serial.read.call_count==1 else b''
        return reader

    def test_sync_zero_and_out_of_order_packets(self):
        reader=self.sync_reader(reversed(range(7)))
        self.assertEqual(reader.read_positions(),[0]*7)
        reader.serial.write.assert_called_once_with(sync_monitor_packet())
        self.assertEqual(sync_monitor_packet()[2],25)
        self.assertEqual(sync_monitor_packet()[4:7],bytes((22,1,7)))

    def test_sync_retries_full_read_after_corrupt_batch(self):
        reader=self.sync_reader()
        good=b''.join(reply(22,bytes((i,))+bytes(9)+struct.pack('<ih',i*10,0)) for i in range(7))
        corrupt=bytearray(good)
        corrupt[-1] ^= 1
        # Keep the first corrupt batch isolated until the reader retries; the
        # second write then releases a clean batch.
        batches=[bytes(corrupt), good]
        sent=set()
        def read_batch(_):
            attempt=reader.serial.write.call_count
            if attempt in sent:
                return b''
            sent.add(attempt)
            return batches[attempt-1]
        reader.serial.read.side_effect=read_batch
        writes=[]
        reader.serial.write.side_effect=lambda packet: writes.append(packet)
        self.assertEqual(reader.read_positions(),[0,1,2,3,4,5,6])
        self.assertEqual(writes,[sync_monitor_packet(),sync_monitor_packet()])

    def test_sync_duplicate_or_missing_id_not_replaced_with_cache(self):
        reader=self.sync_reader([0,1,2,3,4,5,5])
        with self.assertRaises(RuntimeError):reader.read_positions()
        reader=self.sync_reader([0,1,2,3,4,5])
        with self.assertRaises(RuntimeError):reader.read_positions()

    def test_sync_out_of_range_rejected(self):
        reader=self.sync_reader(angle=10000)
        with self.assertRaises(RuntimeError):reader.read_positions()

    def test_wrong_id_rejected(self):
        reader = FreshStarReader.__new__(FreshStarReader)
        reader.serial = Mock()
        reader.serial.in_waiting = 6
        reader.serial.read.return_value = reply(1, b'\x05')
        with self.assertRaises(RuntimeError):
            reader.query(1, 0)


class MappingTests(unittest.TestCase):
    def setUp(self):
        self.c = load_config(DEFAULT_CONFIG)
        # Keep coverage of legacy configurations using seven relative joints.
        self.c.pop('gripper_mapping', None)

    def test_relative_alignment_has_no_initial_jump(self):
        m = RelativeMapper(self.c, 10, set(range(7)))
        leader = [91,-2,0,5,-3,3,0]
        follower = [-3,-.2,-.03,0,-3,2,0]
        m.rebase(leader, follower)
        self.assertEqual(m.target(leader),follower)

    def test_signs_and_single_joint(self):
        m = RelativeMapper(self.c,10,{2})
        m.rebase([0]*7,[0,20,20,0,0,0,0])
        goal=m.target([5]*7)
        self.assertEqual(goal,[0,20,15,0,0,0,0])

    def test_soft_limits_and_local_span(self):
        m = RelativeMapper(self.c,10,set(range(7)))
        m.rebase([0]*7,[135,160,170,0,0,0,0])
        goal=m.target([100,100,-100,100,100,100,100])
        self.assertEqual(goal,[140,165,175,-10,-10,10,10])

    def test_reset_and_invalid_start_rejected(self):
        m = RelativeMapper(self.c,10,set(range(7)))
        with self.assertRaises(RuntimeError):m.rebase([0]*7,[180]*7)
        m.rebase([0]*7,[0]*7)
        with self.assertRaises(RuntimeError):m.target([360]*7)

    def test_rate_bounds_and_goal_stop(self):
        t=Trajectory([0]*7,10,20)
        old=[0]*7
        for _ in range(150):
            pos=t.step([5]*7,.1,[(-10,10)]*7)
            self.assertTrue(all(abs(a-b)<=1.00001 for a,b in zip(pos,old)))
            self.assertTrue(all(0<=x<=5 for x in pos))
            old=list(pos)
        for v in pos:
            self.assertAlmostEqual(v,5)

    def test_default_full_range_replays_user_base_motion(self):
        args=parse_args(['--live'])
        self.assertTrue(args.full_range)
        self.assertEqual(args.joint,'all')
        self.assertIsNone(args.span_deg)
        m=RelativeMapper(self.c,None,set(range(7)))
        m.rebase([-.8,-2.6,-.4,6.4,10.1,10.5,.4],[-2.568,-.056,-.027,.132,-3.087,1.69,.209])
        target=m.target([-61.1,-2.5,-.1,4.5,36.1,0,21])
        self.assertAlmostEqual(target[0],-62.868)
        self.assertAlmostEqual(target[5],-8.81)
        self.assertGreater(target[6],20)
        self.assertAlmostEqual(m.target([-200]*7)[0],-140)

    def test_explicit_local_debug_range_is_retained(self):
        args=parse_args(['--live','--joint','1','--span-deg','10'])
        self.assertFalse(args.full_range)
        self.assertEqual(args.span_deg,10)

    def test_faster_default_profile(self):
        a=parse_args([])
        self.assertEqual((a.speed,a.accel,a.hz),(120,240,30))
        self.assertEqual((a.j4_speed,a.j4_accel),(150,300))
        self.assertEqual((a.j7_speed,a.j7_accel),(180,450))
        self.assertEqual(a.channel,'auto')

    def test_j4_profile_can_be_adjusted_independently(self):
        args=parse_args(['--j4-speed','135','--j4-accel','275'])
        vmax=[args.speed]*6+[args.j7_speed]
        amax=[args.accel]*6+[args.j7_accel]
        vmax[3],amax[3]=args.j4_speed,args.j4_accel
        self.assertEqual((vmax[3],amax[3]),(135,275))
        self.assertEqual((vmax[2],amax[2]),(120,240))

    def test_per_joint_trajectory_profile_keeps_speed_and_acceleration_bounds(self):
        t=Trajectory([0]*7,[100]*6+[150],[180]*6+[360])
        bounds=[(-1000,1000)]*7
        accel_limits=[180]*6+[360]
        for _ in range(20):
            old=list(t.position)
            old_velocity=list(t.velocity)
            t.step([1000]*7,.05,bounds)
            for i,limit in enumerate([100]*6+[150]):
                self.assertLessEqual(abs(t.position[i]-old[i]),limit*.05+1e-8)
                self.assertLessEqual(abs(t.velocity[i]-old_velocity[i]),accel_limits[i]*.05+1e-8)
        self.assertGreater(t.position[6],t.position[0])

    def test_goal_arrival_and_reversal_have_no_velocity_snap(self):
        t=Trajectory([0]*7,100,180)
        bounds=[(-200,200)]*7
        for k in range(240):
            goal=[50 if k<8 else -20]*7
            previous=list(t.velocity)
            previous_position=list(t.position)
            t.step(goal,.066,bounds)
            for i in range(7):
                self.assertLessEqual(abs(t.velocity[i]-previous[i]),180*.066+1e-8)
                self.assertLessEqual(abs(t.position[i]-previous_position[i]),100*.066+1e-8)
        self.assertAlmostEqual(t.position[0],-20)
        self.assertAlmostEqual(t.velocity[0],0)

    def test_gripper_error_above_old_limit_does_not_abort(self):
        t=Trajectory([0]*6+[100],100,240)
        command,limited=t.limit_lead([0]*6+[35],self.c['max_command_lead_deg'],
                                     [(-140,140)]*6+[(0,265)])
        self.assertEqual(limited,[7])
        self.assertEqual(command[6],35+self.c['max_command_lead_deg'][6])
        self.assertEqual(t.velocity[6],0)

    def test_stalled_joint_does_not_accumulate_target(self):
        t=Trajectory([0]*7,60,120)
        for _ in range(100):
            t.step([90]*7,.05,[(-140,140)]*7)
            command,_=t.limit_lead([0]*7,self.c['max_command_lead_deg'],[(-140,140)]*7)
            self.assertTrue(all(abs(x)<=limit for x,limit in zip(command,self.c['max_command_lead_deg'])))
        # The same lead bound travels with measured feedback; it is not a
        # fixed ±10-degree movement range like the removed commissioning span.
        actual=[0]*7
        for _ in range(150):
            t.step([90]*7,.05,[(-140,140)]*7)
            command,_=t.limit_lead(actual,self.c['max_command_lead_deg'],[(-140,140)]*7)
            actual=list(command)
        self.assertTrue(all(x>80 for x in actual))

    def test_gripper_scale_maps_55_deg_to_follower_limit(self):
        m=RelativeMapper(self.c,None,set(range(7)))
        m.rebase([0]*7,[0]*7)
        target=m.target([0,0,0,0,0,0,55])
        self.assertAlmostEqual(target[6],min(55*self.c['scales'][6],self.c['limits_deg'][6][1]))
        self.assertEqual(target[6],340)


class GripperEndpointTests(unittest.TestCase):
    def setUp(self):
        self.c = load_config(DEFAULT_CONFIG)

    def mapper(self, follower_gripper=9.366946187191228, span=None, selected=None):
        m = RelativeMapper(self.c, span, set(range(7)) if selected is None else selected)
        m.rebase([5.4,-2.5,-.2,5,-5.3,-.2,-.3],
                 [1.777,-.056,.823,-1.07,5.267,-1.063,follower_gripper])
        return m

    def test_user_closed_samples_no_longer_preserve_9_degree_gap(self):
        m = self.mapper()
        for angle in (-.5,-.3,0,.2,.9,1,1.5):
            leader=list(m.leader0)
            leader[6]=angle
            self.assertEqual(m.target(leader)[6],-3)
            self.assertEqual(m.target(leader)[:6],m.follower0[:6])

    def test_resume_does_not_redefine_partly_open_as_closed(self):
        for follower in (0,9.37,19.62,100,300):
            m=self.mapper(follower)
            self.assertEqual(m.target(m.leader0)[6],-3)
            opened=list(m.leader0)
            opened[6]=55
            self.assertEqual(m.target(opened)[6],340)

    def test_opening_is_continuous_bounded_and_monotonic(self):
        m=self.mapper()
        goals=[]
        for angle in (1.5,1.6,10,25,55,60):
            leader=list(m.leader0)
            leader[6]=angle
            goals.append(m.target(leader)[6])
        self.assertEqual(goals,sorted(goals))
        self.assertTrue(all(-3<=v<=340 for v in goals))
        self.assertAlmostEqual(goals[1],-3+.1*343/53.5)

    def test_single_joint_and_local_span_remain_effective(self):
        m=self.mapper(selected={0})
        self.assertEqual(m.target(m.leader0)[6],m.follower0[6])
        m=self.mapper(follower_gripper=100,span=10)
        self.assertEqual(m.target(m.leader0)[6],90)

    def test_starting_closure_is_rate_limited_and_torque_bounded(self):
        m=self.mapper()
        trajectory=Trajectory(m.follower0,100,240)
        bounds=[m.envelope(i) for i in range(7)]
        first=trajectory.step(m.target(m.leader0),.1,bounds)
        self.assertGreater(first[6],-3)
        self.assertLessEqual(abs(first[6]-m.follower0[6]),2.4+1e-8)
        h=Hardware(SimpleNamespace(),self.c)
        h.motors={j:Mock() for j in range(1,8)}
        h.send([0]*6+[-3],[0]*6+[19.62])
        self.assertAlmostEqual(h.gripper_torque_command_nm,-1.8)

    def test_invalid_endpoints_are_rejected_before_hardware(self):
        raw=json.loads(DEFAULT_CONFIG.read_text())
        for changes in ({'leader_open_deg':1.5}, {'follower_closed_deg':-20},
                        {'leader_open_deg':10}, {'follower_open_deg':400}):
            c=copy.deepcopy(raw)
            c['gripper_mapping'].update(changes)
            with tempfile.TemporaryDirectory() as directory:
                p=Path(directory)/'config.json'
                p.write_text(json.dumps(c))
                with self.assertRaises(ValueError):
                    load_config(p)


class LifecycleTests(unittest.TestCase):
    def hardware(self):
        h=Hardware(SimpleNamespace(),load_config(DEFAULT_CONFIG))
        h.controller=Mock()
        h.leader=Mock()
        h.motors={j:Mock() for j in range(1,8)}
        return h

    def test_preview_close_never_disables_or_writes(self):
        h=self.hardware()
        h.close()
        for m in h.motors.values():
            m.disable.assert_not_called()
            m.enable.assert_not_called()
            m.send_mit.assert_not_called()
            m.robstride_write_param_u32.assert_not_called()

    def test_partial_enable_failure_disables_every_motor(self):
        h=self.hardware()
        h.touched=True
        h.motors[3].enable.side_effect=RuntimeError('failure')
        with self.assertRaises(RuntimeError):h.enable_at([0]*7)
        h.close()
        for m in h.motors.values():self.assertEqual(m.disable.call_count,3)

    def test_failed_disable_does_not_remove_watchdog(self):
        h=self.hardware()
        h.touched=True
        h.saved={1:{'mode':0,'timeout':0}}
        h.motors[1].disable.side_effect=RuntimeError('CAN broken')
        self.assertTrue(h.close())
        h.motors[1].robstride_write_param_u32.assert_not_called()

    def test_gripper_host_torque_command_is_clamped(self):
        h=self.hardware()
        h.send([90]*7,[0]*7)
        args=h.motors[7].send_mit.call_args.args
        self.assertEqual(args[:4],(0,0,0,1.5))
        self.assertLessEqual(abs(args[4]),1.8)

    def test_gripper_measured_velocity_applies_braking(self):
        h=self.hardware()
        h.gripper_velocity_rad_s=1
        h.send([0]*7,[0]*7)
        self.assertAlmostEqual(h.motors[7].send_mit.call_args.args[4],-.6)


class FreshParallelReadTests(unittest.TestCase):
    def test_batch_backend_is_authoritative_and_failure_never_falls_back(self):
        h=self.hardware()
        h.leader=None
        h.feedback_reader=Mock()
        windows=[[1,2]]*7
        h.feedback_reader.read.return_value=([float(i) for i in range(7)],.75,windows)
        try:
            _,actual=h.read()
            self.assertEqual(actual,list(range(7)))
            self.assertEqual(h.joint_read_windows_ns['follower'],windows)
            h.feedback_reader.read.side_effect=RuntimeError('CAN batch incomplete')
            with self.assertRaisesRegex(RuntimeError,'incomplete'): h.read()
            for motor in h.motors.values(): motor.robstride_get_param_f32.assert_not_called()
            reader=h.feedback_reader
        finally:
            h.close()
        reader.close.assert_called_once()

    def test_autonomous_read_uses_fresh_can_without_touching_uart(self):
        h=self.hardware()
        uart=h.leader
        h.leader=None
        try:
            _, actual=h.read()
            self.assertEqual(len(actual),7)
            self.assertIsNone(h.joint_read_windows_ns['leader'])
            self.assertIsNone(h.leader_reader_pool)
            uart.read_positions.assert_not_called()
            for i,x in enumerate(actual):
                self.assertAlmostEqual(x,i+1)
        finally:
            h.close()

    def hardware(self):
        h=Hardware(SimpleNamespace(),load_config(DEFAULT_CONFIG))
        h.leader=Mock()
        h.motors={j:Mock() for j in range(1,8)}
        for j,m in h.motors.items():
            m.robstride_get_param_f32.side_effect=lambda param,timeout,j=j: (
                math.radians(j) if param==0x7019 else .75)
        return h

    def test_uart_batch_overlaps_can_but_every_result_is_fresh(self):
        h=self.hardware()
        uart_started=threading.Event()
        can_started=threading.Event()
        def leader():
            uart_started.set()
            if not can_started.wait(1):
                raise RuntimeError('CAN did not overlap UART')
            return list(range(7))
        def first_can(param,timeout):
            if not uart_started.wait(1):
                raise RuntimeError('UART did not overlap CAN')
            can_started.set()
            return math.radians(1)
        h.leader.read_positions.side_effect=leader
        h.motors[1].robstride_get_param_f32.side_effect=first_can
        try:
            a,b=h.read()
            self.assertEqual(a,list(range(7)))
            for i,v in enumerate(b):
                self.assertAlmostEqual(v,i+1)
            self.assertEqual(h.gripper_velocity_rad_s,.75)
            windows=h.joint_read_windows_ns
            self.assertLessEqual(windows['leader'][0],windows['follower'][0][1])
            self.assertLessEqual(windows['follower'][0][0],windows['leader'][1])
            for m in h.motors.values():
                m.enable.assert_not_called()
                m.send_mit.assert_not_called()
        finally:
            h.close()
        self.assertIsNone(h.leader_reader_pool)

    def test_uart_failure_is_propagated_without_cached_angles(self):
        h=self.hardware()
        h.leader.read_positions.side_effect=RuntimeError('UART missing ID')
        try:
            with self.assertRaisesRegex(RuntimeError,'UART missing ID'):
                h.read()
        finally:
            h.close()

    def test_uart_io_failure_can_return_only_fresh_follower_for_fault_hold(self):
        h=self.hardware()
        h.leader.read_positions.side_effect=OSError(5,'Input/output error')
        try:
            leader,actual=h.read(allow_leader_fault=True)
            self.assertIsNone(leader)
            self.assertEqual(len(actual),7)
            self.assertIn('Input/output error',h.leader_error)
            self.assertIsNone(h.joint_read_windows_ns['leader'])
            self.assertIsNotNone(h.read_window_ns)
            reads=h.leader.read_positions.call_count
            _,fresh=h.read(allow_leader_fault=True,skip_leader=True)
            self.assertEqual(h.leader.read_positions.call_count,reads)
            self.assertEqual(fresh,actual)
            for m in h.motors.values():
                m.disable.assert_not_called()
        finally:
            h.close()

    def test_can_failure_cannot_be_masked_as_recoverable_leader_failure(self):
        h=self.hardware()
        h.motors[3].robstride_get_param_f32.side_effect=OSError('CAN timeout')
        try:
            with self.assertRaisesRegex(OSError,'CAN timeout'):
                h.read(allow_leader_fault=True)
        finally:
            h.close()

    def test_can_failure_does_not_wait_for_uart_and_disables_before_join(self):
        h=self.hardware()
        started=threading.Event()
        release=threading.Event()
        h.touched=True
        def leader():
            started.set()
            release.wait(2)
            return [0]*7
        h.leader.read_positions.side_effect=leader
        h.motors[3].robstride_get_param_f32.side_effect=RuntimeError('CAN timeout')
        try:
            with self.assertRaisesRegex(RuntimeError,'CAN timeout'):
                h.read()
            self.assertTrue(started.wait(1))
            self.assertFalse(h.pending_leader.done())
            # A fake worker pool asserts that motor shutdown precedes the join.
            pool=h.leader_reader_pool
            def join(**kwargs):
                self.assertEqual(h.motors[7].disable.call_count,3)
                release.set()
                pool.shutdown(**kwargs)
            h.leader_reader_pool=SimpleNamespace(shutdown=join)
        finally:
            h.close()
            release.set()

    def test_background_reconnect_cannot_block_fresh_can_hold(self):
        h=self.hardware()
        h.args.port='fake-UART'
        started=threading.Event(); release=threading.Event()
        reader=Mock()
        def positions():
            started.set()
            release.wait(2)
            return [12.]*7
        reader.read_positions.side_effect=positions
        try:
            with patch('star_arm_readonly.FreshStarReader',return_value=reader):
                h.request_leader_recovery()
                self.assertTrue(started.wait(1))
                for _ in range(3):
                    leader,actual=h.read(allow_leader_fault=True,skip_leader=True)
                    self.assertIsNone(leader)
                    self.assertEqual(len(actual),7)
                    self.assertIsNone(h.take_leader_recovery())
                release.set()
                h.leader_recovery.result(timeout=1)
                self.assertEqual(h.take_leader_recovery()[0],[12.]*7)
                for m in h.motors.values():m.disable.assert_not_called()
        finally:
            release.set()
            h.close()


class VelocityFeedforwardTests(unittest.TestCase):
    def setUp(self):
        self.c=load_config(DEFAULT_CONFIG)
        self.h=Hardware(SimpleNamespace(),self.c)
        self.h.motors={j:Mock() for j in range(1,8)}
        self.bounds=[(-140,140)]*7

    def test_motion_uses_partial_signed_velocity_but_no_gripper_velocity(self):
        velocity=[40,-40,20,-20,10,-10,100]
        self.h.send([0]*7,[0]*7,velocity)
        for i in range(6):
            args=self.h.motors[i+1].send_mit.call_args.args
            self.assertAlmostEqual(args[1],math.radians(velocity[i]*self.c['velocity_feedforward_ratio'][i]))
            self.assertEqual(args[2:4],(self.c['kp'][i],self.c['kd'][i]))
        self.assertEqual(self.h.motors[7].send_mit.call_args.args,(0,0,0,1.5,0))

    def test_hold_and_new_alignment_have_zero_velocity_target(self):
        trajectory=Trajectory([0]*7,100,240)
        self.assertEqual(trajectory.command_velocity,[0]*7)
        trajectory.step([30]*7,.1,self.bounds)
        hold=list(trajectory.position)
        trajectory.velocity=[0]*7  # P/space in the running program
        command=trajectory.step(hold,.1,self.bounds)
        self.h.send(command,command,trajectory.command_velocity)
        for i in range(6):
            self.assertEqual(self.h.motors[i+1].send_mit.call_args.args[1],0)

    def test_small_goal_approaches_smoothly_and_pause_has_zero_velocity(self):
        trajectory=Trajectory([0]*7,100,240)
        command=trajectory.step([.1]*7,.1,self.bounds)
        self.assertTrue(all(0<v<.1 for v in command))
        self.assertTrue(all(0<v<1 for v in trajectory.command_velocity))
        trajectory.velocity=[0]*7
        trajectory.step(command,.1,self.bounds)
        self.assertEqual(trajectory.command_velocity,[0]*7)

    def test_lead_guard_removes_assistance_and_bounds_stalled_commands(self):
        trajectory=Trajectory([0]*7,100,240)
        for _ in range(50):
            trajectory.step([100]*7,.1,self.bounds)
            command,limited=trajectory.limit_lead([0]*7,self.c['max_command_lead_deg'],self.bounds)
            for j in limited:
                self.assertEqual(trajectory.command_velocity[j-1],0)
            self.assertTrue(all(abs(p)<=v for p,v in zip(command,self.c['max_command_lead_deg'])))

    def test_soft_boundary_and_reverse_motion_have_bounded_velocity(self):
        trajectory=Trajectory([0]*7,100,240)
        previous=[0]*7
        for goal in ([200]*7,[-200]*7):
            for _ in range(90):
                command=trajectory.step(goal,.09,self.bounds)
                for i in range(7):
                    self.assertLessEqual(abs(command[i]-previous[i]),9+1e-8)
                    self.assertLessEqual(abs(trajectory.command_velocity[i]),100)
                    self.assertTrue(-140<=command[i]<=140)
                previous=command
        self.assertEqual(trajectory.command_velocity,[0]*7)

    def test_invalid_ratios_rejected_and_legacy_config_defaults_to_zero(self):
        raw=json.loads(DEFAULT_CONFIG.read_text())
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'config.json'
            for values in ([0]*6,[.6]*6+[0],[-.1]*6+[0],[0]*6+[.1],[float('nan')]*7):
                raw['velocity_feedforward_ratio']=values
                p.write_text(json.dumps(raw))
                with self.assertRaises((ValueError,RuntimeError)):load_config(p)
            raw.pop('velocity_feedforward_ratio')
            p.write_text(json.dumps(raw))
            self.assertEqual(load_config(p)['velocity_feedforward_ratio'],[0]*7)

if __name__=='__main__':
    unittest.main()
