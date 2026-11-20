"""Best-effort local telemetry. No robot or recording control in this module."""
import json
import os
import socket
import time

SOCKET_PATH = '/tmp/b601-rs-telemetry.sock'


class TelemetryPublisher:
    def __init__(self, session, config, arguments):
        self.session = session
        self.context = {'process_id': os.getpid(), 'config': config, 'arguments': {k: str(v) if hasattr(v, '__fspath__') else v
                                                       for k, v in arguments.items()}}
        self.sequence = 0
        self.dropped = 0
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.sock.setblocking(False)

    def publish(self, **sample):
        # Nonblocking socket: disk/UI/collector failure cannot stall teleoperation.
        try:
            self.sequence += 1
            packet = {'schema': 'b601-telemetry-v1', 'session': self.session,
                      'seq': self.sequence, 'publish_mono_ns': time.monotonic_ns(),
                      'wall_ns': time.time_ns(), 'publisher_dropped': self.dropped,
                      'context': self.context, **sample}
            self.sock.sendto(json.dumps(packet, ensure_ascii=False, allow_nan=False,
                                       default=str).encode(), SOCKET_PATH)
        except (OSError, ValueError, TypeError):
            self.dropped += 1

    def close(self):
        self.sock.close()
