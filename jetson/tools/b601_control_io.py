"""Volatile command transport and bounded diagnostic writes for the CAN loop."""
from functools import lru_cache
import json
import os
from pathlib import Path
import queue
import stat
import threading
import uuid


@lru_cache(maxsize=1)
def runtime_root():
    root = Path('/dev/shm') / f'b601-rs-{os.getuid()}'
    root.mkdir(mode=0o700, exist_ok=True)
    info = root.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise RuntimeError('控制临时目录权限无效')
    return root


def runtime_path(pid, suffix):
    if type(pid) is not int or pid <= 0 or suffix not in ('json', 'status.json', 'abort'):
        raise ValueError('控制临时文件标识无效')
    return runtime_root() / f'b601-vla-step-{pid}.{suffix}'


def atomic_runtime_json(path, value):
    """Atomic visibility only: ephemeral control messages need no disk fsync."""
    path = Path(path)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('x') as f:
            json.dump(value, f, ensure_ascii=False, allow_nan=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class BufferedJournal:
    """Slow/full diagnostic storage must never wait on the motor send thread.

    flush/fileno are for the pre-enable settings backup; close is post-disable.
    During control, write only enqueues bounded text and records dropped lines.
    """
    def __init__(self, path, capacity=256, stream=None):
        self.stream = stream if stream is not None else Path(path).open('a', buffering=1)
        self.pending = queue.Queue(maxsize=capacity)
        self.dropped = 0
        self.error = None
        self.thread = threading.Thread(target=self._drain, name='b601-journal', daemon=True)
        self.thread.start()

    def _drain(self):
        while True:
            line = self.pending.get()
            try:
                if line is None:
                    return
                try:
                    self.stream.write(line)
                except Exception as exc:
                    self.error = str(exc)
            finally:
                self.pending.task_done()

    def write(self, line):
        try:
            self.pending.put_nowait(line)
        except queue.Full:
            self.dropped += 1

    def flush(self):
        self.pending.join()
        if self.error:
            raise RuntimeError('诊断日志写入失败：'+self.error)
        self.stream.flush()

    def fileno(self):
        return self.stream.fileno()

    def close(self):
        # Called only after hardware.close; an unavailable disk must not retain
        # a live controller. Give the daemon a bounded opportunity to finish.
        try:
            self.pending.put(None, timeout=.5)
        except queue.Full:
            return
        self.thread.join(timeout=1)
        if not self.thread.is_alive():
            self.stream.close()
