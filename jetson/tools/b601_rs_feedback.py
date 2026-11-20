"""Fresh RobStride parameter batches over a dedicated, read-only SocketCAN path.

MotorBridge remains the command/configuration owner. Never fall back to its
cached MIT position: that position may wrap and is not multi-turn mechPos.
"""
from __future__ import annotations

import math
import select
import socket
import struct
import time

FRAME = struct.Struct('=IB3x8s')
EFF = 0x80000000
HOST = 0xFD
TIMESTAMPNS = getattr(socket, 'SO_TIMESTAMPNS', 35)  # Linux LP64 SO_TIMESTAMPNS_OLD
QUERIES = tuple((j, 0x7019) for j in range(1, 8)) + ((7, 0x701B),)


class BatchFeedback:
    def __init__(self, channel, timeout_s=.08):
        self.timeout_s = timeout_s
        self.failed = False
        self.sock = socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        try:
            self.sock.setsockopt(socket.SOL_SOCKET, TIMESTAMPNS, 1)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 262144)
            # Parameter responses, MIT/status feedback and fault reports only.
            filters = b''.join(struct.pack('=II', EFF | (kind << 24), EFF | (31 << 24))
                               for kind in (17, 2, 21, 24))
            self.sock.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FILTER, filters)
            self.sock.bind((channel,))
            self.sock.setblocking(False)
        except BaseException:
            self.sock.close()
            raise

    def close(self):
        self.sock.close()

    def receive(self):
        raw, ancillary, flags, _ = self.sock.recvmsg(FRAME.size, 128)
        if flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC) or len(raw) != FRAME.size:
            raise RuntimeError('CAN反馈帧/时间戳截断')
        can_id, dlc, data = FRAME.unpack(raw)
        if not can_id & EFF or can_id & 0x60000000 or dlc != 8:
            raise RuntimeError('CAN反馈格式无效')
        stamp = None
        for level, kind, value in ancillary:
            if level == socket.SOL_SOCKET and kind == TIMESTAMPNS and len(value) >= 16:
                sec, ns = struct.unpack('@ll', value[:16])
                stamp = sec * 1_000_000_000 + ns
        if stamp is None:
            raise RuntimeError('CAN反馈缺少内核接收时间戳')
        return can_id & 0x1FFFFFFF, data, stamp, time.monotonic_ns()

    @staticmethod
    def check_report(can_id, data):
        kind = (can_id >> 24) & 31
        motor = (can_id >> 8) & 255
        host = can_id & 255
        if not 1 <= motor <= 7 or host not in (0xFD, 0xFE, 0xFF):
            return
        if kind == 21:
            fault, warning = struct.unpack('<II', data)
            if fault or warning:
                raise RuntimeError(f'J{motor}新鲜故障/警告：{fault}/{warning}；不自动清故障')
        if kind in (2, 24):
            fault_bits = (can_id >> 16) & 0x3F
            if fault_bits:
                raise RuntimeError(f'J{motor}新鲜状态故障位：0x{fault_bits:02x}；不自动清故障')

    def read(self):
        if self.failed:
            raise RuntimeError('CAN批量读取已失效，禁止接受超时后的迟到回复')
        try:
            # Discard previous parameter replies; never discard fault reports.
            drained = 0
            while True:
                try:
                    can_id, data, _, _ = self.receive()
                except BlockingIOError:
                    break
                self.check_report(can_id, data)
                drained += 1
                if drained > 4096:
                    raise RuntimeError('CAN反馈队列持续拥塞')
            deadline = time.monotonic() + self.timeout_s
            requests = {}
            for motor, param in QUERIES:
                wall_ns, mono_ns = time.time_ns(), time.monotonic_ns()
                payload = struct.pack('<H', param) + bytes(6)
                self.sock.send(FRAME.pack(EFF | (17 << 24) | (HOST << 8) | motor, 8, payload))
                requests[motor, param] = (wall_ns, mono_ns)
            values, windows = {}, {}
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(f'CAN新鲜反馈超时，缺少{sorted(set(QUERIES)-set(values))}')
                if not select.select([self.sock], [], [], remaining)[0]:
                    continue
                can_id, data, wall_ns, mono_ns = self.receive()
                self.check_report(can_id, data)
                if (can_id >> 24) & 31 != 17:
                    continue
                key = ((can_id >> 8) & 255, struct.unpack_from('<H', data)[0])
                if can_id & 255 not in (0xFD, 0xFE, 0xFF) or key not in requests:
                    continue
                if wall_ns < requests[key][0]:
                    continue
                if (can_id >> 16) & 255:
                    raise RuntimeError(f'CAN参数读取状态错误：{key}')
                if key in values:
                    raise RuntimeError(f'CAN参数出现重复回复：{key}')
                value = struct.unpack_from('<f', data, 4)[0]
                if not math.isfinite(value):
                    raise RuntimeError(f'CAN参数非有限值：{key}')
                values[key] = value
                windows[key] = [requests[key][1], mono_ns]
                if len(values) == len(QUERIES):
                    return ([math.degrees(values[j, 0x7019]) for j in range(1, 8)],
                            values[7, 0x701B], [windows[j, 0x7019] for j in range(1, 8)])
        except BaseException:
            # The protocol has no transaction ID. Reusing a timed-out reader
            # could mistake a late response for the next cycle's response.
            self.failed = True
            raise
