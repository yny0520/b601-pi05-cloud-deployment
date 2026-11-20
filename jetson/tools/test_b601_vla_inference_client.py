import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest

from b601_vla_inference_client import InferenceClient, Cancelled


SCRIPT = '''import json,os,sys,time
for line in sys.stdin:
 r=json.loads(line)
 if r.get('boot_delay'):time.sleep(r['boot_delay'])
 print('VLA_PROGRESS '+json.dumps({'request_id':r['request_id'],'message':'testing','device':'cuda'}),flush=True)
 if r.get('wait'):time.sleep(30)
 if r.get('fail'):
  print('simulated allocation failure',file=sys.stderr,flush=True)
  raise SystemExit(2)
 print('VLA_RESULT '+json.dumps({'request_id':'stale','action_deg':[99]*7}),flush=True)
 print('VLA_RESULT '+json.dumps({'request_id':r['request_id'],'action_deg':[0]*7,'device':'cuda','pid':os.getpid()}),flush=True)
'''


class ClientTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.client=InferenceClient([sys.executable,'-u','-c',SCRIPT],self.temp.name)
        self.cancel=threading.Event()

    def tearDown(self):
        self.client.close()
        self.temp.cleanup()

    def test_reuses_same_process_and_ignores_stale_result(self):
        progress=[]
        a=self.client.request({},self.cancel,progress.append)
        b=self.client.request({},self.cancel,progress.append)
        self.assertEqual(a['pid'],b['pid'])
        self.assertNotEqual(a['request_id'],b['request_id'])
        self.assertEqual(b['action_deg'],[0]*7)
        self.assertEqual(len(progress),2)

    def test_cancel_during_request_then_close_releases_process(self):
        timer=threading.Timer(.1,self.cancel.set)
        timer.start()
        try:
            with self.assertRaises(Cancelled):self.client.request({'wait':True},self.cancel)
            proc=self.client.proc
            self.client.close()
            self.assertIsNotNone(proc.poll())
        finally:timer.cancel()

    def test_cancel_before_start_does_not_spawn_process(self):
        self.cancel.set()
        with self.assertRaises(Cancelled):self.client.request({},self.cancel)
        self.assertIsNone(self.client.proc)

    def test_silent_startup_reports_elapsed_time_without_claiming_cuda_ready(self):
        progress=[]
        self.client.request({'warmup_only':True,'boot_delay':1.15},self.cancel,progress.append)
        initial=[p for p in progress if p.get('stage')=='initializing']
        self.assertTrue(initial)
        self.assertGreaterEqual(initial[0]['request_elapsed_seconds'],1.)
        self.assertIsNone(initial[0]['device'])
        self.assertIn('尚未提交运动',initial[0]['message'])

    def test_stop_still_cancels_during_silent_dependency_import(self):
        timer=threading.Timer(.15,self.cancel.set)
        timer.start()
        try:
            with self.assertRaises(Cancelled):
                self.client.request({'warmup_only':True,'boot_delay':30},self.cancel)
        finally:timer.cancel()

    def test_worker_failure_is_reported_without_reusing_old_output(self):
        self.client.request({},self.cancel)
        with self.assertRaisesRegex(RuntimeError,'allocation failure'):
            self.client.request({'fail':True},self.cancel)

    def test_log_is_saved_on_close(self):
        self.client.request({},self.cancel)
        self.client.close()
        logs=list(Path(self.temp.name).glob('*.log'))
        self.assertEqual(len(logs),1)
        self.assertIn('VLA_RESULT',logs[0].read_text())


if __name__=='__main__':unittest.main()
