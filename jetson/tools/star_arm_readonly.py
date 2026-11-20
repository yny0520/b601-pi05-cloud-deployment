"""Fresh FashionStar angle queries with no servo-control commands.

Framing/angle decoding follows motorbridge/motorbridge-smart-servo:
smart_servo_vendors/fashionstar/src/protocol.rs (1/ping, 16/angle and
25/subcommand22 sync monitor). The latter queries all seven in one request.
Unlike the SDK's filtered monitor, a valid zero is a fresh measurement. There
is no cache: timeout, mismatched ID or malformed reply raises. A bad checksum
is discarded so a single noisy UART frame does not stop teleoperation.
"""
import struct
import time

REQUEST = b'\x12\x4c'
RESPONSE = b'\x05\x1c'


def query_packet(code, servo_id):
    if code not in (1, 16) or servo_id not in range(7):
        raise ValueError('只允许ID0..6的ping或多圈角度查询')
    body = REQUEST + bytes((code, 1, servo_id))
    return body + bytes((sum(body) & 255,))


def sync_monitor_packet():
    params = bytes((22, 1, 7, *range(7)))
    body = REQUEST + bytes((25, len(params))) + params
    return body + bytes((sum(body) & 255,))


def take_packet(buffer):
    while True:
        start = buffer.find(RESPONSE)
        if start < 0:
            if len(buffer)>1:
                del buffer[:-1]
            return None
        if start:
            del buffer[:start]
        if len(buffer)<4:
            return None
        size = buffer[3]
        if size == 0 or size > 32:
            del buffer[0]
            continue
        if len(buffer)<size+5:
            return None
        packet = bytes(buffer[:size+5])
        if sum(packet[:-1]) & 255 != packet[-1]:
            # A UART frame can be corrupted by line noise while the next
            # frame is already in the receive buffer. Drop only this frame
            # and keep scanning; the caller still fails on a real timeout or
            # missing/duplicate servo response.
            del buffer[0]
            continue
        del buffer[:size+5]
        return packet[2], packet[4:-1]


class FreshStarReader:
    def __init__(self, port):
        import serial
        self.serial = serial.Serial(port, 1_000_000, timeout=.002,
                                    write_timeout=.03, exclusive=True)

    def query(self, code, servo_id):
        self.serial.reset_input_buffer()
        self.serial.write(query_packet(code, servo_id))
        deadline = time.monotonic()+.04
        buffer = bytearray()
        while time.monotonic()<deadline:
            buffer.extend(self.serial.read(max(1,self.serial.in_waiting)))
            packet = take_packet(buffer)
            if packet is None:
                continue
            reply_code, params = packet
            if reply_code != code or not params or params[0] != servo_id:
                raise RuntimeError(f'主臂ID{servo_id}回复指令/ID不符')
            return params
        raise RuntimeError(f'主臂ID{servo_id}查询超时（无新鲜回复）')

    def ping(self, servo_id):
        return self.query(1, servo_id) == bytes((servo_id,))

    def read_positions_sequential(self):
        positions = []
        for i in range(7):
            params = self.query(16, i)
            if len(params)<7:
                raise RuntimeError(f'主臂ID{i}角度回复长度不足')
            angle = struct.unpack('<i',params[1:5])[0]/10.0
            if abs(angle)>720:
                raise RuntimeError(f'主臂ID{i}角度{angle}°超过两圈，先核对计圈/姿态')
            positions.append(angle)
        return positions

    def read_positions(self):
        # A corrupt frame is discarded by take_packet(). Retry one complete
        # batch when noise causes the first batch to miss an ID.
        for attempt in range(2):
            self.serial.reset_input_buffer()
            self.serial.write(sync_monitor_packet())
            deadline = time.monotonic()+.08
            buffer = bytearray()
            angles = {}
            while time.monotonic()<deadline:
                buffer.extend(self.serial.read(max(1, self.serial.in_waiting)))
                while True:
                    packet = take_packet(buffer)
                    if packet is None:
                        break
                    code, params = packet
                    if code != 22 or len(params)<14 or params[0] not in range(7):
                        raise RuntimeError('主臂批量监测回复指令/ID/长度不符')
                    servo_id = params[0]
                    if servo_id in angles:
                        raise RuntimeError(f'主臂ID{servo_id}在同一轮重复回复')
                    angle = struct.unpack('<i',params[10:14])[0]/10.0
                    if abs(angle)>720:
                        raise RuntimeError(f'主臂ID{servo_id}角度超出两圈，先核对计圈/姿态')
                    angles[servo_id] = angle
                if len(angles)==7:
                    return [angles[i] for i in range(7)]
        missing = sorted(set(range(7))-set(angles))
        raise RuntimeError(f'主臂批量查询超时，缺少新鲜回复ID{missing}')

    def close(self):
        self.serial.close()
