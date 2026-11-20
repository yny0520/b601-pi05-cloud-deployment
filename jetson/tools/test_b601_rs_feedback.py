"""Protocol safety tests, without motors or CAN devices."""
import math
import struct
import unittest
from unittest.mock import patch

from b601_rs_feedback import BatchFeedback, FRAME, EFF, HOST, QUERIES, TIMESTAMPNS


class FakeSocket:
    def __init__(self):
        self.sent = []
        self.responses = []
        self.transform = lambda rows: rows

    def send(self, raw):
        self.sent.append(raw)
        if len(self.sent) == len(QUERIES):
            import time
            for motor, param in reversed(QUERIES):
                value = math.radians(motor) if param == 0x7019 else .75
                data = struct.pack('<H2xf', param, value)
                self.responses.append(((17 << 24) | (motor << 8) | HOST,
                                       data, time.time_ns(), time.monotonic_ns()))
            self.responses = self.transform(self.responses)
        return len(raw)


class BatchTests(unittest.TestCase):
    def reader(self, transform=None):
        r = BatchFeedback.__new__(BatchFeedback)
        r.failed = False
        r.timeout_s = .002
        r.sock = FakeSocket()
        if transform:
            r.sock.transform = transform
        def receive():
            if not r.sock.responses:
                raise BlockingIOError()
            return r.sock.responses.pop(0)
        r.receive = receive
        return r

    def run_read(self, r):
        with patch('b601_rs_feedback.select.select', side_effect=lambda *a: ([r.sock], [], []) if r.sock.responses else ([], [], [])):
            return r.read()

    def test_out_of_order_fresh_multi_turn_positions_and_only_read_requests(self):
        r = self.reader()
        angles, velocity, windows = self.run_read(r)
        for j, v in enumerate(angles, 1):
            self.assertAlmostEqual(v, j, places=5)
        self.assertEqual(velocity, .75)
        self.assertEqual(len(windows), 7)
        for raw, (motor, param) in zip(r.sock.sent, QUERIES):
            can_id, dlc, data = FRAME.unpack(raw)
            self.assertEqual(can_id, EFF | (17 << 24) | (HOST << 8) | motor)
            self.assertEqual((dlc, data), (8, struct.pack('<H6x', param)))

    def test_missing_wrong_motor_wrong_param_wrong_host_and_stale_never_substitute(self):
        for change in ('missing', 'motor', 'param', 'host', 'stale'):
            def transform(rows):
                cid, data, wall, mono = rows[0]
                if change == 'missing': return rows[1:]
                if change == 'motor': cid = (cid & ~0xFF00) | (8 << 8)
                if change == 'param': data = struct.pack('<H', 0x701A) + data[2:]
                if change == 'host': cid = cid & ~255
                if change == 'stale': wall = 0
                return [(cid, data, wall, mono)] + rows[1:]
            with self.subTest(change=change):
                r = self.reader(transform)
                with self.assertRaisesRegex(RuntimeError, '超时'): self.run_read(r)
                self.assertTrue(r.failed)
                with self.assertRaisesRegex(RuntimeError, '已失效'): r.read()

    def test_status_nonfinite_duplicate_and_fault_are_fatal(self):
        for change in ('status', 'nan', 'duplicate', 'fault', 'status_fault'):
            def transform(rows):
                cid, data, wall, mono = rows[0]
                if change == 'duplicate': return [rows[0]] + rows
                if change == 'status': cid |= 1 << 16
                if change == 'nan': data = data[:4] + struct.pack('<f', float('nan'))
                if change == 'fault':
                    cid = (21 << 24) | (2 << 8) | HOST
                    data = struct.pack('<II', 16416, 0)
                if change == 'status_fault': cid = (24 << 24) | (1 << 20) | (2 << 8) | HOST
                return [(cid, data, wall, mono)] + rows[1:]
            with self.subTest(change=change):
                r = self.reader(transform)
                with self.assertRaises(RuntimeError): self.run_read(r)
                self.assertTrue(r.failed)

    def test_queued_fault_is_not_discarded_before_requests(self):
        r = self.reader()
        r.sock.responses = [((21 << 24) | (2 << 8) | HOST, struct.pack('<II', 16416, 0), 0, 0)]
        with self.assertRaisesRegex(RuntimeError, '故障'): self.run_read(r)
        self.assertFalse(r.sock.sent)

    def test_kernel_timestamp_required_and_truncation_rejected(self):
        r = BatchFeedback.__new__(BatchFeedback)
        from unittest.mock import Mock
        r.sock = Mock()
        raw = FRAME.pack(EFF | (17 << 24) | (1 << 8) | HOST, 8, bytes(8))
        r.sock.recvmsg.return_value = (raw, [], 0, None)
        with self.assertRaisesRegex(RuntimeError, '时间戳'): r.receive()
        r.sock.recvmsg.return_value = (raw, [(1, TIMESTAMPNS, struct.pack('@ll', 123, 456))], 0, None)
        self.assertEqual(r.receive()[2], 123000000456)


if __name__ == '__main__': unittest.main()
