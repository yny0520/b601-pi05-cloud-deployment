"""Bounded phase diagnostics; never sends commands or refreshes a watchdog."""
import sys
import threading
import time
import traceback


class ControlTiming:
    def __init__(self, emit, threshold_s=.15):
        self.emit = emit
        self.threshold_s = threshold_s
        self.owner = threading.get_ident()
        self.phase = ('idle', time.monotonic())
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._watch, name='b601-phase-watch', daemon=True)
        self.thread.start()

    def mark(self, name):
        self.phase = (name, time.monotonic())

    def _watch(self):
        reported = None
        while not self.stop_event.wait(.025):
            phase = self.phase
            if phase[0] == 'idle' or phase == reported or time.monotonic()-phase[1] < self.threshold_s:
                continue
            frame = sys._current_frames().get(self.owner)
            self.emit('control_phase_stall', phase=phase[0], elapsed_s=time.monotonic()-phase[1],
                      stack=traceback.format_stack(frame, limit=10) if frame else [])
            reported = phase

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=.2)
