import json
import queue
import tempfile
import time
from pathlib import Path
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from b601_dataset import DatasetRecorder
from b601_dataset_web import get_dataset_response
from b601_telemetry import TelemetryPublisher
from b601_collection_profiles import get_profile, load_profile_name, render_panel


def packet(seq, paused=False):
    ns = time.monotonic_ns()
    return {'schema':'b601-telemetry-v1','session':'test-only','seq':seq,
            'read_window_ns':[ns-1_000_000,ns],'send_window_ns':[ns,ns],
            'leader_deg':[float(seq)]*7,'follower_deg':[float(seq)]*7,
            'command_deg':[float(seq)+.5]*7,'live':True,'tracking':True,
            'paused':paused,'motor_commands_sent':True,
            'context':{'config':{},'arguments':{}},'mit_commands':[[0,0,0,0,0]]*7}


class DatasetTests(unittest.TestCase):
    def test_fault_hold_feedback_remains_visible_and_cannot_start_demonstration(self):
        self.prime()
        p=packet(2,paused=True)
        p.update(fault_hold=True,fault_hold_reason='UART disconnected',leader_read=False,leader_deg=None)
        self.rec.on_telemetry(p)
        self.assertTrue(self.rec.status()['teleop']['fault_hold'])
        self.assertEqual(self.rec.status()['teleop']['follower_deg'],[2.]*7)
        with self.assertRaisesRegex(ValueError,'故障保持'):
            self.rec.start_episode()
        p.update(seq=3,paused=False)
        with self.assertRaisesRegex(ValueError,'Invalid telemetry'):
            self.rec.on_telemetry(p)

    def test_autonomous_follower_feedback_is_valid_without_fabricating_leader(self):
        self.prime()
        p=packet(2,paused=True)
        p.update(leader_read=False,leader_deg=None)
        p['context']['arguments']['autonomous']=True
        self.rec.on_telemetry(p)
        self.assertEqual(self.rec.latest_telemetry['seq'],2)
        self.assertIsNone(self.rec.latest_telemetry['leader_deg'])
        self.assertEqual(self.rec.latest_telemetry['follower_deg'],[2.]*7)
        with self.assertRaisesRegex(ValueError,'自主模型执行不能记录'):
            self.rec.start_episode()
        p['paused']=False
        self.rec.on_telemetry({**p,'seq':3})
        with self.assertRaisesRegex(ValueError,'自主模型执行不能记录'):
            self.rec.start_episode()

    def test_missing_leader_is_rejected_without_explicit_autonomous_mode(self):
        p=packet(1);p['leader_deg']=None
        with self.assertRaisesRegex(ValueError,'Invalid telemetry'):
            self.rec.on_telemetry(p)
        p['leader_read']=False
        with self.assertRaisesRegex(ValueError,'Invalid telemetry'):
            self.rec.on_telemetry(p)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='b601-recorder-test-')
        self.root = Path(self.temp.name)
        self.rec = DatasetRecorder(self.root, enable_socket=False, min_free_bytes=0)
        self.depth = np.array([[0,100,1500,2300,65000]],dtype=np.uint16)
        self.rgb = np.zeros((12,16,3),dtype=np.uint8)

    def tearDown(self):
        self.rec.close()
        self.temp.cleanup()

    def prime(self, seq=1, paused=False):
        for name in ('hand','top'):
            self.rec.on_frame(name,self.rgb,encoding='bgr8')
        self.rec.on_frame('depth',self.depth,encoding='16UC1')
        self.rec.on_telemetry(packet(seq,paused))

    def read_rows(self, folder, name):
        return [json.loads(line) for line in (folder/(name+'.jsonl')).read_text().splitlines()]

    def test_refuses_missing_or_paused_inputs(self):
        with self.assertRaisesRegex(ValueError,'等待新数据'):
            self.rec.start_episode()
        self.prime(paused=True)
        with self.assertRaisesRegex(ValueError,'未处于跟随'):
            self.rec.start_episode()
        self.assertFalse(self.rec.status()['recording'])

    def test_actual_cycles_depth_integrity_and_replay(self):
        self.prime()
        folder=Path(self.rec.start_episode()['folder'])
        for name in ('hand','top'):
            self.rec.on_frame(name,self.rgb)
        self.rec.on_frame('depth',self.depth,encoding='16UC1')
        p=packet(2);self.rec.on_telemetry(p);self.rec.on_telemetry(p)
        self.rec.mark('退回纠错','插偏后退出')
        self.rec.on_telemetry(packet(4))
        result=self.rec.finish_episode('success','纠错后到位')
        self.assertEqual(result['counts']['teleop'],2)
        self.assertEqual(result['quality']['telemetry_sequence_gaps'],1)
        stored=self.read_rows(folder,'depth')[0]
        np.testing.assert_array_equal(cv2.imread(str(folder/stored['file']),cv2.IMREAD_UNCHANGED),self.depth)
        rows=self.read_rows(folder,'samples')
        self.assertEqual(len(rows),2)
        self.assertTrue(rows[1]['images']['depth']['reused'])
        self.assertTrue(all(r['image_timing_ok'] for r in rows))
        self.assertEqual(self.read_rows(folder,'events')[1]['phase'],'退回纠错')
        self.assertTrue((folder/'teleop-context-1.json').exists())
        code,mime,body=get_dataset_response('/api/episode?episode='+folder.name,self.root)
        self.assertEqual(code,200)
        self.assertEqual(len(json.loads(body)['samples']),2)
        bad=get_dataset_response('/recorded?episode='+folder.name+'&file=../../etc/passwd',self.root)
        self.assertEqual(bad[0],400)

    def test_stale_camera_is_reported_not_filled(self):
        self.prime()
        folder=Path(self.rec.start_episode()['folder'])
        self.rec.on_telemetry(packet(2))
        result=self.rec.finish_episode('failure','模拟缺图')
        self.assertEqual(result['quality']['missing_or_stale_image_matches']['hand'],1)
        self.assertIsNone(self.read_rows(folder,'samples')[0]['images']['hand']['file'])

    def test_write_error_is_not_reported_as_success(self):
        self.prime()
        folder=Path(self.rec.start_episode()['folder'])
        with patch('b601_dataset.cv2.imwrite',return_value=False):
            self.rec.on_frame('hand',self.rgb)
            self.rec.queue.join()
            with self.assertRaises(OSError):
                self.rec.finish_episode('success')
        meta=json.loads((folder/'metadata.json').read_text())
        self.assertEqual(meta['status'],'saved_with_errors')
        self.assertTrue(meta['write_errors'])

    def test_dropped_queue_is_visible(self):
        self.prime()
        self.rec.start_episode()
        with patch.object(self.rec.queue,'put_nowait',side_effect=queue.Full):
            self.rec.on_frame('hand',self.rgb)
        result=self.rec.finish_episode('unlabeled')
        self.assertEqual(result['quality']['queue_dropped']['hand'],1)

    def test_no_listener_does_not_block_control(self):
        pub=TelemetryPublisher('offline-test',{}, {})
        started=time.monotonic()
        with patch('b601_telemetry.SOCKET_PATH', str(self.root/'missing.sock')):
            for _ in range(20):
                pub.publish(follower_deg=[0]*7)
        self.assertLess(time.monotonic()-started,1)
        pub.close()

    def test_pick_place_destination_defaults_phases_and_metadata(self):
        profile = get_profile('pick_place')
        profile['root'] = str(self.root / 'pick-place')
        settings = self.root / 'settings.json'
        self.rec.set_collection_profile(profile, settings)
        self.assertEqual(load_profile_name(settings), 'pick_place')
        self.assertEqual(self.rec.status()['dataset_root'], profile['root'])
        self.prime()
        folder = Path(self.rec.start_episode()['folder'])
        for phase in profile['phases']:
            self.rec.mark(phase)
        self.rec.on_telemetry(packet(2))
        self.rec.finish_episode('success')
        meta = json.loads((folder / 'metadata.json').read_text())
        self.assertEqual(folder.parent, Path(profile['root']))
        self.assertEqual(meta['task'], profile['task'])
        self.assertEqual(meta['success_criterion'], profile['success_criterion'])
        self.assertEqual(meta['dataset']['id'], 'pick_place')
        self.assertEqual([r['phase'] for r in self.read_rows(folder, 'events') if r['type']=='phase'], profile['phases'])

    def test_switch_refused_while_recording_or_finalizing(self):
        self.prime()
        original = self.rec.root
        self.rec.start_episode()
        profile = {**get_profile('pick_place'), 'root': str(self.root / 'other')}
        with self.assertRaisesRegex(ValueError, '不能切换'):
            self.rec.set_collection_profile(profile)
        self.rec.finish_episode('unlabeled')
        with self.rec.lock:
            self.rec.finalizing = {'simulated': True}
        try:
            with self.assertRaisesRegex(ValueError, '不能切换'):
                self.rec.set_collection_profile(profile)
        finally:
            self.rec.finalizing = None
        self.assertEqual(self.rec.root, original)

    def test_switch_leaves_old_data_and_explicit_replay_uses_its_dataset(self):
        insertion = {**get_profile('insertion'), 'root': str(self.root / 'old')}
        pick = {**get_profile('pick_place'), 'root': str(self.root / 'new')}
        self.rec.set_collection_profile(insertion)
        self.prime()
        folder = Path(self.rec.start_episode()['folder'])
        self.rec.on_telemetry(packet(2))
        self.rec.finish_episode('success')
        before = (folder / 'metadata.json').read_bytes()
        self.rec.set_collection_profile(pick)
        self.assertEqual((folder / 'metadata.json').read_bytes(), before)
        self.assertIsNone(self.rec.status()['last_result'])
        with patch('b601_dataset_web.get_profile', return_value=insertion):
            response = get_dataset_response('/api/episode?episode='+folder.name+'&dataset=insertion', self.rec.root)
        self.assertEqual(response[0], 200)
        self.assertEqual(json.loads(response[2])['metadata']['task'], '网线插接')
        self.assertEqual(get_dataset_response('/review?episode='+folder.name+'&dataset=../../etc', self.rec.root)[0],400)

    def test_explicit_task_and_criterion_survive_profile_defaults(self):
        self.rec.set_collection_profile({**get_profile('pick_place'), 'root': str(self.root)})
        self.prime()
        folder = Path(self.rec.start_episode('将红色方块放入右侧托盘', '方块A', '方块在托盘内')['folder'])
        self.rec.finish_episode('unlabeled')
        meta = json.loads((folder / 'metadata.json').read_text())
        self.assertEqual(meta['task'], '将红色方块放入右侧托盘')
        self.assertEqual(meta['success_criterion'], '方块在托盘内')

    def test_panel_renders_profile_and_escapes_script_content(self):
        panel = render_panel('</script>', get_profile('pick_place'))
        self.assertNotIn('__COLLECTION_', panel)
        self.assertNotIn('__RECORD_TOKEN__', panel)
        self.assertIn('b601-pick-place', panel)
        self.assertIn('\\u003c/script>', panel)

    def test_jitter_does_not_halve_saved_fps_and_no_burst_after_gap(self):
        self.rec.close()
        self.rec = DatasetRecorder(self.root, enable_socket=False, min_free_bytes=0,
                                   camera_hz=15, hand_camera_hz=30)
        self.prime()
        self.rec.start_episode()
        start = time.monotonic_ns()
        offered = []
        with patch.object(self.rec, '_offer', side_effect=lambda kind,ep,data: offered.append((kind,data['received_mono_ns']))):
            for i in range(300):
                ns = start + round(i*1e9/30) + (-800_000 if i%2 else 800_000)
                self.rec.on_frame('hand', self.rgb, received_ns=ns)
                self.rec.on_frame('top', self.rgb, received_ns=ns)
            hand = [ns for kind,ns in offered if kind=='hand']
            top = [ns for kind,ns in offered if kind=='top']
            self.assertEqual(len(hand), 300)
            self.assertEqual(len(top), 150)
            future = start + 20_000_000_000
            self.rec.on_frame('hand', self.rgb, received_ns=future)
            self.rec.on_frame('hand', self.rgb, received_ns=future+1_000_000)
            self.assertEqual(sum(kind=='hand' for kind,_ in offered),301)
        result = self.rec.finish_episode('unlabeled')
        meta = json.loads((Path(result['folder'])/'metadata.json').read_text())
        self.assertAlmostEqual(meta['camera_hz_caps']['hand'],30,places=4)
        self.assertAlmostEqual(meta['camera_hz_caps']['top'],15,places=4)


if __name__=='__main__':
    unittest.main()
