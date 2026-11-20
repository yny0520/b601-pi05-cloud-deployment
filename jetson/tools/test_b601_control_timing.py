import threading
import time
import unittest
from b601_control_timing import ControlTiming


class TimingTests(unittest.TestCase):
    def test_slow_native_like_phase_reports_owner_stack_once_and_idle_is_ignored(self):
        events=[]
        ready=threading.Event()
        def emit(event, **data):
            events.append((event,data));ready.set()
        t=ControlTiming(emit, threshold_s=.03)
        try:
            t.mark('sdk_send_mit_J2')
            self.assertTrue(ready.wait(.5))
            t.mark('idle');time.sleep(.08)
            self.assertEqual(len(events),1)
            self.assertEqual(events[0][1]['phase'],'sdk_send_mit_J2')
            self.assertTrue(any('test_slow_native_like_phase' in line for line in events[0][1]['stack']))
        finally:t.close()
        self.assertFalse(t.thread.is_alive())
