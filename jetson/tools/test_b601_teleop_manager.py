from b601_control_io import runtime_path
import sys
import tempfile
import time
from pathlib import Path
import unittest
from unittest.mock import patch

from b601_teleop_manager import TeleopManager, READY_MESSAGE

FAKE = r'''
import os,sys,tty,time
assert sys.stdin.isatty()
time.sleep(.15)
tty.setcbreak(sys.stdin.fileno())
print('遥操作硬件就绪，等待开始。',flush=True)
while True:
 k=os.read(sys.stdin.fileno(),1).decode()
 print('received:'+k,flush=True)
 if k=='r': print('已开始；模拟跟随',flush=True)
 if k=='p': print('已暂停，模拟保持',flush=True)
 if k=='h': print('已暂停，模拟直接使能保持',flush=True)
 if k=='q': break
'''


def wait_for(fn, timeout=3):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        if fn(): return
        time.sleep(.015)
    raise AssertionError('状态未在期限内出现')


class ManagerTests(unittest.TestCase):
    def test_fault_hold_is_visible_and_only_explicit_resume_requests_recovery(self):
        self.manager.start(auto_follow=False)
        wait_for(lambda:self.manager.status()['ready'])
        pid=self.manager.proc.pid
        packet={'context':{'process_id':pid},'publish_mono_ns':time.monotonic_ns(),
                'tracking':True,'paused':True,'fault_hold':True,'fault_hold_reason':'UART disconnected'}
        self.manager.telemetry_getter=lambda:packet
        status=self.manager.status()
        self.assertEqual(status['state'],'fault_hold')
        self.assertTrue(status['paused'])
        with self.assertRaisesRegex(RuntimeError,'暂停保持'):
            self.manager.submit_vla_step([0.]*7,[0.]*7)
        self.assertNotIn('received:r',self.manager.status()['output_tail'])
        self.manager.resume()
        wait_for(lambda:'received:r' in self.manager.status()['output_tail'])

    def test_autonomous_connection_cannot_resume_leader_following(self):
        self.manager.start(auto_follow=False, autonomous=True)
        wait_for(lambda:self.manager.status()['ready'])
        self.assertTrue(self.manager.status()['autonomous'])
        self.manager.send_key('h')
        wait_for(lambda:self.manager.status()['state']=='paused')
        with self.assertRaisesRegex(RuntimeError,'自主模型控制'):
            self.manager.resume()
        self.assertNotIn('received:r',self.manager.status()['output_tail'])

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='b601-manager-test-')
        self.root=Path(self.temp.name)
        self.root_patch=patch('b601_teleop_manager.ROOT', self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.manager=TeleopManager([sys.executable,'-u','-c',FAKE],self.root,
                                  stop_file=self.root/'stop')

    def tearDown(self):
        self.manager.close()
        self.temp.cleanup()

    def test_start_waits_for_ready_and_pause_resume_stop(self):
        self.manager.start()
        self.assertNotIn('received:r',self.manager.status()['output_tail'])
        wait_for(lambda:self.manager.status()['state']=='following')
        with self.assertRaises(RuntimeError): self.manager.start()
        self.manager.toggle()
        wait_for(lambda:self.manager.status()['state']=='paused')
        self.manager.toggle()
        wait_for(lambda:self.manager.status()['state']=='following')
        self.manager.stop()
        wait_for(lambda:not self.manager.status()['running'])
        self.assertIn('received:q',self.manager.status()['output_tail'])
        self.assertTrue((self.root/'stop').exists())

    def test_explicit_pause_and_resume_are_not_toggles(self):
        self.manager.start()
        wait_for(lambda:self.manager.status()['state']=='following')
        self.manager.pause()
        wait_for(lambda:self.manager.status()['state']=='paused')
        self.manager.pause()
        self.assertEqual(self.manager.status()['state'],'paused')
        self.manager.resume()
        wait_for(lambda:self.manager.status()['state']=='following')
        self.manager.resume()
        self.assertEqual(self.manager.status()['state'],'following')

    def test_resume_is_blocked_while_vla_step_is_active(self):
        import json
        self.manager.start()
        wait_for(lambda:self.manager.status()['state']=='following')
        self.manager.pause()
        wait_for(lambda:self.manager.status()['state']=='paused')
        path=runtime_path(self.manager.proc.pid, 'status.json')
        path.write_text(json.dumps({'state':'executing'}))
        self.addCleanup(path.unlink, missing_ok=True)
        with self.assertRaisesRegex(RuntimeError,'VLA单步尚未结束'):
            self.manager.resume()
        path.unlink()
        self.manager.resume()
        wait_for(lambda:self.manager.status()['state']=='following')

    def test_never_autostarts_after_failed_initialization(self):
        self.manager.command=[sys.executable,'-u','-c',"print('错误：通信失败',flush=True);raise SystemExit(2)"]
        self.manager.start()
        wait_for(lambda:self.manager.status()['state']=='error')
        self.assertIn('通信失败',self.manager.status()['error'])
        self.assertNotIn('received:r',self.manager.status()['output_tail'])

    def test_preview_connection_without_enabling(self):
        self.manager.start(auto_follow=False)
        wait_for(lambda:self.manager.status()['ready'])
        self.assertEqual(self.manager.status()['state'],'waiting')
        self.assertNotIn('received:r',self.manager.status()['output_tail'])
        with self.assertRaises(ValueError): self.manager.send_key('arbitrary command')

    def test_vla_hold_key_enables_hold_without_any_follow_key(self):
        self.manager.start(auto_follow=False)
        wait_for(lambda:self.manager.status()['ready'])
        self.manager.send_key('h')
        wait_for(lambda:self.manager.status()['state']=='paused')
        self.assertNotIn('received:r',self.manager.status()['output_tail'])
        self.assertEqual(self.manager.status()['control_epoch'],1)

    def test_pause_while_vla_executing_sends_abort_even_when_already_paused(self):
        import json
        self.manager.start(auto_follow=False)
        wait_for(lambda:self.manager.status()['ready'])
        self.manager.send_key('h')
        wait_for(lambda:self.manager.status()['state']=='paused')
        path=runtime_path(self.manager.proc.pid, 'status.json')
        abort=runtime_path(self.manager.proc.pid, 'abort')
        path.write_text(json.dumps({'state':'executing','id':'test'}))
        self.addCleanup(path.unlink,missing_ok=True)
        self.addCleanup(abort.unlink,missing_ok=True)
        self.manager.pause()
        wait_for(lambda:'received:p' in self.manager.status()['output_tail'])
        self.assertTrue(abort.exists())

    def test_stop_before_ready_cancels_autofollow(self):
        self.manager.start()
        self.manager.stop()
        wait_for(lambda:self.manager.status()['ready'])
        time.sleep(.2)
        self.assertNotIn('received:r',self.manager.status()['output_tail'])

    def test_named_rolling_request_requires_fresh_state_and_exact_previous_id(self):
        import json
        import threading
        self.manager.start(auto_follow=False)
        wait_for(lambda:self.manager.status()['ready'])
        self.manager.send_key('h')
        wait_for(lambda:self.manager.status()['state']=='paused')
        pid=self.manager.proc.pid
        packet={'context':{'process_id':pid},'publish_mono_ns':time.monotonic_ns(),
                'live':True,'tracking':True,'paused':True}
        self.manager.telemetry_getter=lambda:packet
        status_path=runtime_path(pid, 'status.json')
        request_path=runtime_path(pid, 'json')
        self.addCleanup(status_path.unlink,missing_ok=True)
        self.addCleanup(request_path.unlink,missing_ok=True)
        status_path.write_text(json.dumps({'id':'old','state':'executing'}))
        args=([0.]*7,[0.]*7,[0.]*7)
        with self.assertRaisesRegex(RuntimeError,'旧轨迹'):
            self.manager.submit_vla_step(*args,trajectory_deg=[[0.]*7],replace_id='wrong',expected_mono_ns=time.monotonic_ns())
        with self.assertRaisesRegex(ValueError,'新鲜'):
            self.manager.submit_vla_step(*args,trajectory_deg=[[0.]*7],replace_id='old',expected_mono_ns=0)
        self.assertFalse(request_path.exists())
        self.manager.submit_vla_step(*args,trajectory_deg=[[0.]*7],replace_id='old',expected_mono_ns=time.monotonic_ns())
        request=json.loads(request_path.read_text())
        self.assertEqual(request['replace_id'],'old')
        self.assertGreater(request['expected_mono_ns'],0)


if __name__=='__main__': unittest.main()
