"""Storage stalls must not block the command loop; no robot devices used."""
import io
import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from b601_control_io import BufferedJournal, runtime_path, atomic_runtime_json


class ControlIOTests(unittest.TestCase):
    def test_volatile_command_and_ack_never_call_fsync(self):
        p=runtime_path(os.getpid(), 'status.json')
        try:
            with patch('os.fsync', side_effect=AssertionError('disk sync in control loop')):
                atomic_runtime_json(p, {'state':'complete','id':'test'})
            self.assertEqual(json.loads(p.read_text())['state'], 'complete')
            self.assertTrue(str(p).startswith('/dev/shm/'))
        finally:
            p.unlink(missing_ok=True)

    def test_blocked_diagnostic_disk_has_bounded_memory_and_nonblocking_enqueue(self):
        entered, release=threading.Event(),threading.Event()
        class SlowDisk(io.StringIO):
            def write(self, text):
                entered.set()
                release.wait(3)
                return super().write(text)
        stream=SlowDisk()
        journal=BufferedJournal(None,capacity=2,stream=stream)
        try:
            journal.write('first\n')
            self.assertTrue(entered.wait(1))
            start=time.monotonic()
            for i in range(1000):journal.write('sample\n')
            self.assertLess(time.monotonic()-start, .3)
            self.assertEqual(journal.pending.qsize(), 2)
            self.assertGreater(journal.dropped, 990)
        finally:
            release.set()
            journal.flush()
            journal.close()

    def test_invalid_json_cannot_replace_last_good_ack(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'ack.json'
            atomic_runtime_json(p, {'state':'paused'})
            with self.assertRaises(ValueError):atomic_runtime_json(p, {'bad':float('nan')})
            self.assertEqual(json.loads(p.read_text()), {'state':'paused'})
            self.assertEqual([x.name for x in Path(d).iterdir()], ['ack.json'])


if __name__ == '__main__':unittest.main()
