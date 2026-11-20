"""Hardware-free state machine tests; never opens a serial port or CAN bus."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from b601_vla_runner import VlaRunner, Cancelled, latest_worker_progress
from b601_vla_motion import DEFAULT_PROFILE


class Recorder:
    def __init__(self):
        self.lock = threading.RLock()
        self.active = self.finalizing = None
        self.pose = [10., 20., 30., 0., 5., -5., 12.]
        self.stale = False

    @property
    def latest_telemetry(self):
        return {'context': {'process_id': 123}, 'publish_mono_ns': 0 if self.stale else time.monotonic_ns(),
                'live': True, 'tracking': True, 'paused': True,
                'follower_deg': list(self.pose), 'command_deg': list(self.pose)}


class Frames:
    stale = False

    def snapshot(self, name, age):
        return None if self.stale else ('image', time.monotonic_ns())


class Controller:
    def __init__(self, recorder):
        self.recorder = recorder
        self.running, self.ready = False, False
        self.state, self.epoch = 'stopped', 0
        self.keys, self.requests = [], []
        self.aborts = self.stops = 0
        self.step = {'state': 'idle'}
        self.finish_step = True
        self.instant_ready = True

    def status(self):
        return {'running': self.running, 'state': self.state, 'ready': self.ready,
                'pid': 123 if self.running else None, 'control_epoch': self.epoch}

    def start(self, auto_follow=True, autonomous=False):
        assert not auto_follow
        self.autonomous = autonomous
        self.running, self.ready = True, self.instant_ready
        self.state = 'waiting' if self.ready else 'starting'

    def send_key(self, key):
        self.keys.append(key)
        self.epoch += 1
        self.state = 'paused'

    def submit_vla_step(self, action, expected, origin, trajectory_deg=None):
        self.last_trajectory=trajectory_deg
        self.requests.append((action, expected, origin))
        self.step = {'id': str(len(self.requests)), 'state': 'complete' if self.finish_step else 'executing',
                     'target_deg': list(self.recorder.pose)}
        return self.step

    def vla_step_status(self):
        return copy.deepcopy(self.step)

    def abort_vla_step(self):
        self.aborts += 1
        self.epoch += 1
        self.step['state'] = 'aborted'
        self.state = 'paused'

    def stop(self):
        self.stops += 1
        self.running = False
        self.epoch += 1


def wait_for(fn):
    deadline = time.monotonic()+2
    while time.monotonic() < deadline:
        if fn():
            return
        time.sleep(.005)
    raise AssertionError('未到预期状态')


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.recorder, self.frames = Recorder(), Frames()
        self.controller = Controller(self.recorder)
        self.entered = threading.Event()
        self.release = threading.Event()

    def predictor(self, packet, frames, task, cancel):
        self.entered.set()
        while not self.release.wait(.005):
            if cancel.is_set():
                raise Cancelled()
        return {'action_deg': packet['follower_deg'], 'device': 'fake', 'inference_seconds': .01}

    def runner(self, predictor=None):
        obj = VlaRunner(self.controller, self.recorder, self.frames, predictor or self.predictor, motion_profile=DEFAULT_PROFILE)
        self.addCleanup(obj.close)
        return obj

    def test_stop_during_sensor_validation_is_cancelled_and_preserves_diagnostic(self):
        obj = self.runner()
        self.controller.running = self.controller.ready = True
        self.controller.state = 'paused'
        recorded = []
        obj._record = lambda event, **data: recorded.append((event, data))
        def interrupted(*args, **kwargs):
            obj.stop()
            raise RuntimeError('从臂不再处于暂停保持；本次VLA测试已终止')
        obj._sensors = interrupted
        obj._run('test')
        self.assertEqual(obj.status()['state'], 'stopped')
        self.assertFalse(obj.status()['active'])
        self.assertEqual(self.controller.aborts, 1)
        self.assertTrue(any(event == 'cancelled_during_validation' for event, _ in recorded))
        self.assertEqual(self.controller.stops, 0)

    def test_sensor_failure_without_user_stop_remains_error_and_requests_hold(self):
        obj = self.runner()
        self.controller.running = self.controller.ready = True
        self.controller.state = 'paused'
        def failed(*args, **kwargs):
            raise RuntimeError('fresh CAN feedback lost')
        obj._sensors = failed
        obj._run('test')
        self.assertEqual(obj.status()['state'], 'error')
        self.assertIn('fresh CAN feedback lost', obj.status()['message'])
        self.assertEqual(self.controller.aborts, 1)
        self.assertEqual(self.controller.stops, 0)

    def test_one_click_connects_in_hold_and_completes_bounded_session(self):
        self.release.set()
        obj = self.runner()
        obj.start()
        wait_for(lambda: obj.status()['state'] == 'complete')
        self.assertEqual(self.controller.keys, ['h'])
        self.assertEqual(len(self.controller.requests), 10)
        self.assertEqual(obj.status()['steps'], 10)
        self.assertTrue(all(req[2] == self.recorder.pose for req in self.controller.requests))
        self.assertTrue(self.controller.autonomous)

    def test_executes_five_actions_then_reobserves_and_records_both_counts(self):
        count = []
        def predict(packet, frames, task, cancel):
            count.append(1)
            return {'action_chunk_deg': [list(packet['follower_deg']) for _ in range(5)]}
        obj = self.runner(predict)
        obj.configured_profile = {**DEFAULT_PROFILE, 'action_steps': 5, 'max_steps': 2,
                                  'task_bounds_deg': [[-100, 100]]*7}
        obj.start()
        wait_for(lambda: obj.status()['state'] == 'complete')
        self.assertEqual(len(count), 2)
        self.assertEqual(len(self.controller.requests), 10)
        self.assertEqual(obj.status()['steps'], 10)
        self.assertEqual(obj.status()['rounds'], 2)

    def test_continuous_plan_submits_all_fifty_in_one_control_request(self):
        def predict(packet,frames,task,cancel):
            return {'action_chunk_deg':[list(packet['follower_deg']) for _ in range(50)]}
        obj=self.runner(predict)
        obj.configured_profile={**DEFAULT_PROFILE,'continuous_plan':True,'action_steps':50,'max_steps':2}
        obj.start()
        wait_for(lambda:obj.status()['state']=='complete')
        self.assertEqual(len(self.controller.requests),2)
        self.assertEqual(len(self.controller.last_trajectory),50)
        self.assertEqual(obj.status()['steps'],100)

    def test_fifteen_step_selection_reobserves_without_consuming_old_suffix(self):
        calls=[]
        def predict(packet,frames,task,cancel):
            calls.append(1)
            return {'action_chunk_deg':[[float(len(calls))]*7 for _ in range(50)]}
        obj=self.runner(predict)
        obj.configured_profile={**DEFAULT_PROFILE,'continuous_plan':True,'action_steps':50,'max_steps':2}
        obj.start(action_steps=15)
        wait_for(lambda:obj.status()['state']=='complete')
        self.assertEqual(len(calls),2)
        self.assertEqual(len(self.controller.last_trajectory),15)
        self.assertEqual(self.controller.last_trajectory,[[2.]*7 for _ in range(15)])
        self.assertEqual(obj.status()['steps'],30)
        self.assertEqual(obj.motion_profile['speed_deg_s'],DEFAULT_PROFILE['speed_deg_s'])
        self.assertEqual(obj.configured_profile['action_steps'],50)

    def test_invalid_observation_selection_cannot_start_controller(self):
        obj=self.runner()
        obj.configured_profile={**DEFAULT_PROFILE,'continuous_plan':True}
        for v in (True,0,16,51,'15',15.0):
            with self.subTest(value=v),self.assertRaises(ValueError):obj.start(action_steps=v)
        self.assertFalse(self.controller.running)
        self.assertEqual(self.controller.requests,[])

    def test_recurring_pose_warning_never_stops_or_claims_grasp(self):
        obj=self.runner()
        samples=[[0.]*7,[20.,0.,0.,0.,0.,0.,40.],[1.]*7]
        for _ in range(4):obj._record_plan_motion(samples)
        self.assertIn('抓空',obj.status()['repetition_warning'])
        self.assertEqual(self.controller.aborts,0)
        for n in range(4):obj._record_plan_motion([[n*10.]*7,[(n+1)*10.]*7])
        self.assertEqual(obj.status()['repetition_warning'],'')

    def test_grasp_miss_stops_before_saving_and_drops_late_prediction(self):
        import numpy as np
        obj=self.runner()
        obj.start()
        self.assertTrue(self.entered.wait(1))
        def snapshot(name,age):
            self.assertGreater(self.controller.aborts,0)
            return (np.zeros((4,4,3),dtype=np.uint8),time.monotonic_ns()) if name=='hand' else None
        obj.frames.snapshot=snapshot
        with tempfile.TemporaryDirectory() as tmp:
            obj.run_dir=Path(tmp)
            result=obj.report_grasp_miss()
            obj.thread.join(1)
            folder=Path(result['grasp_miss_path'])
            evidence=json.loads((folder/'evidence.json').read_text())
            self.assertEqual(evidence['label'],'human_reported_grasp_miss')
            self.assertFalse(evidence['images']['top']['available'])
            self.assertTrue((folder/'hand.jpg').is_file())
            self.assertTrue(evidence['telemetry_matches_controller'])
            obj.run_dir=None
        self.assertEqual(self.controller.requests,[])
        self.assertFalse(obj.status()['active'])

    def test_stop_during_continuous_plan_aborts_without_another_plan(self):
        def predict(packet,frames,task,cancel):
            return {'action_chunk_deg':[list(packet['follower_deg']) for _ in range(50)]}
        obj=self.runner(predict)
        obj.configured_profile={**DEFAULT_PROFILE,'continuous_plan':True,'action_steps':50}
        self.controller.finish_step=False
        obj.start()
        wait_for(lambda:len(self.controller.requests)==1)
        obj.stop()
        obj.thread.join(1)
        self.assertEqual(len(self.controller.requests),1)
        self.assertEqual(self.controller.step['state'],'aborted')

    def test_stop_after_first_chunk_action_cannot_submit_the_rest(self):
        def predict(packet, frames, task, cancel):
            return {'action_chunk_deg': [list(packet['follower_deg']) for _ in range(5)]}
        obj = self.runner(predict)
        obj.configured_profile = {**DEFAULT_PROFILE, 'action_steps': 5}
        original = self.controller.submit_vla_step
        def submit(*args):
            result = original(*args)
            obj.stop()
            return result
        self.controller.submit_vla_step = submit
        obj.start()
        wait_for(lambda: obj.status()['state'] == 'stopped')
        self.assertEqual(len(self.controller.requests), 1)

    def test_command_completion_does_not_claim_measured_motion(self):
        obj = self.runner()
        self.controller.start(auto_follow=False)
        self.controller.send_key('h')
        actual = list(self.recorder.pose)
        target = [v+.45 for v in actual]
        obj._settle_step(target, actual, actual)
        self.assertEqual(obj.status()['net_motion_deg'], [0.]*6)
        self.assertEqual(obj.status()['actual_delta_deg'], [0.]*6)
        self.assertAlmostEqual(obj.status()['tracking_error_deg'][2], .45)

    def test_failed_settling_reports_joint_actual_and_target(self):
        obj = self.runner()
        self.controller.start(auto_follow=False)
        self.controller.send_key('h')
        actual = list(self.recorder.pose)
        target = list(actual)
        target[2] += 1.1
        with patch('b601_vla_runner.time.monotonic', side_effect=[0, 5]):
            with self.assertRaisesRegex(RuntimeError, 'J3指令已发完但实测未到位'):
                obj._settle_step(target, actual, actual)

    def test_stop_during_inference_drops_late_output(self):
        obj = self.runner()
        obj.start()
        self.assertTrue(self.entered.wait(1))
        obj.stop()
        self.release.set()
        obj.thread.join(1)
        self.assertEqual(self.controller.requests, [])
        self.assertGreaterEqual(self.controller.aborts, 1)
        self.assertEqual(obj.status()['state'], 'stopped')

    def test_model_preparation_precedes_capture_and_origin(self):
        captured=[]
        def predict(packet, frames, task, cancel):
            captured.append(list(packet['follower_deg']))
            return {'action_deg':list(packet['follower_deg']), 'device':'fake'}
        obj=self.runner(predict)
        obj.injected_predictor=False
        recorder=self.recorder
        class Client:
            def request(self, request, cancel, progress=None):
                self.request_value=request
                recorder.pose[2]+=2.
                return {'prepared':True,'device':'fake'}
            def close(self):pass
        obj.inference_client=Client()
        obj._wait_stationary=lambda: None
        with tempfile.TemporaryDirectory() as tmp:
            package=Path(tmp);(package/'best_model').mkdir();(package/'best_model/model.safetensors').touch()
            spec = {**obj.model_spec, 'policy_type': 'smolvla', 'motion_profile': '/home/jetson/Desktop/b601-rs/calibration/vla_motion_config.json', 'model_dir': str(package/'best_model')}
            self.recorder.collection_profile = {'id': spec['collection_profile']}
            with patch('b601_vla_runner.PYTHON',Path(sys.executable)),patch('b601_vla_runner.load_model_spec',return_value=spec):
                obj.start()
                wait_for(lambda:obj.status()['state']=='complete')
        self.assertEqual(obj.inference_client.request_value,{'warmup_only':True,'model_id':spec['id'],'fallback_cpu':False})
        self.assertEqual(captured[0][2],32.)
        self.assertEqual(self.controller.requests[0][2][2],32.)

    def test_stop_during_model_preparation_cannot_submit_action(self):
        obj=self.runner()
        obj.injected_predictor=False
        entered=threading.Event()
        class Client:
            def request(self, request, cancel, progress=None):
                entered.set()
                while not cancel.wait(.01):pass
                raise Cancelled()
            def close(self):pass
        obj.inference_client=Client()
        with tempfile.TemporaryDirectory() as tmp:
            package=Path(tmp);(package/'best_model').mkdir();(package/'best_model/model.safetensors').touch()
            spec = {**obj.model_spec, 'policy_type': 'smolvla', 'motion_profile': '/home/jetson/Desktop/b601-rs/calibration/vla_motion_config.json', 'model_dir': str(package/'best_model')}
            self.recorder.collection_profile = {'id': spec['collection_profile']}
            with patch('b601_vla_runner.PYTHON',Path(sys.executable)),patch('b601_vla_runner.load_model_spec',return_value=spec):
                obj.start()
                self.assertTrue(entered.wait(1))
                obj.stop()
                obj.thread.join(1)
        self.assertEqual(self.controller.requests,[])
        self.assertEqual(obj.status()['state'],'stopped')

    def test_stop_before_ready_never_sends_enable_key(self):
        self.controller.instant_ready = False
        obj = self.runner()
        obj.start()
        wait_for(lambda: self.controller.running)
        obj.stop()
        self.controller.ready = True
        obj.thread.join(1)
        self.assertEqual(self.controller.keys, [])
        self.assertEqual(self.controller.stops, 1)

    def test_abort_waits_for_paused_manager_ack(self):
        obj = self.runner()
        self.controller.state = 'pausing'
        self.controller.step = {'state': 'aborted'}
        finished = threading.Event()
        thread = threading.Thread(target=lambda: (obj._wait_abort_hold(), finished.set()))
        thread.start()
        self.assertFalse(finished.wait(.08))
        self.controller.state = 'paused'
        self.assertTrue(finished.wait(1))
        thread.join(1)

    def test_wrong_collection_cannot_connect_robot(self):
        obj = self.runner()
        obj.injected_predictor = False
        self.recorder.collection_profile = {'id': 'insertion'}
        spec = {**obj.model_spec, 'collection_profile': 'pick_place'}
        with patch('b601_vla_runner.load_model_spec', return_value=spec):
            with self.assertRaisesRegex(ValueError, '对应任务'):
                obj.start()
        self.assertFalse(self.controller.running)
        self.assertEqual(self.controller.requests, [])

    def test_stop_during_execution_aborts_no_second_action(self):
        self.controller.finish_step = False
        self.release.set()
        obj = self.runner()
        obj.start()
        wait_for(lambda: len(self.controller.requests) == 1)
        obj.stop()
        obj.thread.join(1)
        self.assertEqual(len(self.controller.requests), 1)
        self.assertEqual(self.controller.step['state'], 'aborted')

    def test_control_epoch_change_invalidates_prediction(self):
        obj = self.runner()
        obj.start()
        self.assertTrue(self.entered.wait(1))
        self.controller.epoch += 1
        self.release.set()
        wait_for(lambda: obj.status()['state'] == 'error')
        self.assertEqual(self.controller.requests, [])

    def test_recording_or_stale_cameras_block_start(self):
        obj = self.runner()
        self.recorder.active = object()
        with self.assertRaisesRegex(RuntimeError, '数据采集'):
            obj.start()
        self.recorder.active = None
        self.frames.stale = True
        with self.assertRaisesRegex(RuntimeError, '相机'):
            obj.start()
        self.assertFalse(self.controller.running)

    def test_telemetry_loss_during_inference_drops_result(self):
        obj = self.runner()
        obj.start()
        self.assertTrue(self.entered.wait(1))
        self.recorder.stale = True
        self.release.set()
        wait_for(lambda: obj.status()['state'] == 'error')
        self.assertEqual(self.controller.requests, [])

    def test_recording_started_during_inference_blocks_commit(self):
        obj = self.runner()
        obj.start()
        self.assertTrue(self.entered.wait(1))
        self.recorder.active = object()
        self.release.set()
        wait_for(lambda: obj.status()['state'] == 'error')
        self.assertEqual(self.controller.requests, [])

    def test_double_start_rejected_while_worker_active(self):
        obj = self.runner()
        obj.start()
        self.assertTrue(self.entered.wait(1))
        with self.assertRaisesRegex(RuntimeError, '尚未结束'):
            obj.start()

    def test_worker_progress_ignores_library_logs_and_invalid_json(self):
        log = 'library warning\nVLA_PROGRESS '+json.dumps({'message':'正在GPU推理', 'device':'cuda'})+'\nVLA_PROGRESS {bad\n'
        self.assertEqual(latest_worker_progress(log)['device'], 'cuda')
        self.assertIsNone(latest_worker_progress('some warnings'))

    def test_worker_subprocess_progress_result_and_log_are_preserved(self):
        import numpy as np
        obj = self.runner()
        real_popen = subprocess.Popen
        script = '''import json,sys,time
request=json.loads(sys.stdin.readline())
print('VLA_PROGRESS '+json.dumps({'request_id':request['request_id'],'message':'本轮GPU失败，正在CPU推理','device':'cpu','gpu_error':'allocation failed'}),flush=True)
time.sleep(.15)
print('VLA_RESULT '+json.dumps({'request_id':request['request_id'],'action_deg':[0.]*7,'device':'cpu','gpu_error':'allocation failed','inference_seconds':.01}),flush=True)
'''
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(obj.inference_client,'log_dir',Path(tmp)), \
             patch('b601_vla_inference_client.subprocess.Popen',side_effect=lambda cmd,**kw:real_popen([sys.executable,'-u','-c',script],**kw)):
            result = obj._predict(self.recorder.latest_telemetry,
                                  {k:(np.zeros((4,4,3),dtype=np.uint8),time.monotonic_ns()) for k in ('top','hand')},
                                  'test',obj.cancel)
            self.assertEqual(result['device'],'cpu')
            self.assertEqual(result['gpu_error'],'allocation failed')
            self.assertEqual(obj.status()['worker_progress']['device'],'cpu')
            obj.inference_client.close()
            logs=list(Path(tmp).glob('*.log'))
            self.assertEqual(len(logs),1)
            self.assertIn('VLA_RESULT',logs[0].read_text())


if __name__ == '__main__':
    unittest.main()
