"""One persistent inference subprocess per live test; independent of motor I/O."""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import uuid

PYTHON = Path('/home/jetson/miniforge3/envs/lerobot/bin/python')
DIAGNOSTICS = Path('/home/jetson/Desktop/b601-rs/diagnostics/vla-inference')


class Cancelled(Exception):
    pass


def latest_worker_progress(log):
    for line in reversed(log.splitlines()):
        if line.startswith('VLA_PROGRESS '):
            try:
                result = json.loads(line[len('VLA_PROGRESS '):])
            except ValueError:
                continue
            if isinstance(result, dict) and isinstance(result.get('message'), str):
                return result
    return None


class InferenceClient:
    def __init__(self, command=None, log_dir=None):
        self.command = command or [str(PYTHON), str(Path(__file__).with_name('b601_smolvla_live_predict.py')), '--serve']
        self.log_dir = Path(log_dir or DIAGNOSTICS)
        self.proc = self.output = None
        self.offset = 0
        self.pending = ''

    def request(self, request, cancel, progress=None, timeout=240):
        if cancel.is_set():
            raise Cancelled()
        if self.proc is None:
            self.output = tempfile.TemporaryFile(mode='w+')
            try:
                self.proc = subprocess.Popen(self.command, stdin=subprocess.PIPE,
                                             stdout=self.output, stderr=self.output, text=True)
            except BaseException:
                self.output.close()
                self.output = None
                raise
        if self.proc.poll() is not None:
            raise RuntimeError('模型进程已退出：'+self._tail())
        request_id = uuid.uuid4().hex
        payload = {**request, 'request_id': request_id}
        try:
            self.proc.stdin.write(json.dumps(payload, ensure_ascii=False)+'\n')
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError('无法提交模型请求：'+self._tail()) from exc
        started = last_report = time.monotonic()
        deadline = started+timeout
        last_progress = {'request_id': request_id, 'stage': 'initializing',
                         'device': None, 'message':
                         ('正在启动模型依赖；首次约1–2分钟，当前尚未提交运动'
                          if request.get('warmup_only') else '正在准备本轮模型推理')}
        while True:
            if cancel.wait(.03):
                raise Cancelled()
            size = os.fstat(self.output.fileno()).st_size
            chunk = os.pread(self.output.fileno(), size-self.offset, self.offset)
            self.offset += len(chunk)
            self.pending += chunk.decode(errors='replace')
            while '\n' in self.pending:
                line, self.pending = self.pending.split('\n', 1)
                if not line.startswith(('VLA_RESULT ', 'VLA_PROGRESS ')):
                    continue
                try:
                    value = json.loads(line.split(' ', 1)[1])
                except ValueError:
                    continue
                # Every reply must belong to this request, never a stale round.
                if not isinstance(value, dict) or value.get('request_id') != request_id:
                    continue
                if line.startswith('VLA_RESULT '):
                    return value
                last_progress = value
                if progress:
                    progress({**value, 'request_elapsed_seconds': time.monotonic()-started})
                    last_report = time.monotonic()
            if self.proc.poll() is not None:
                raise RuntimeError('模型进程失败：'+self._tail())
            if time.monotonic() > deadline:
                raise RuntimeError('模型推理超过240秒；本轮结果作废')
            if progress and time.monotonic()-last_report >= 1.:
                # Python/Torch imports can be silent for tens of seconds on NX.
                # A heartbeat means the process is alive, not that CUDA is ready.
                progress({**last_progress, 'request_elapsed_seconds': time.monotonic()-started})
                last_report = time.monotonic()

    def _tail(self):
        if self.output is None:
            return ''
        size = os.fstat(self.output.fileno()).st_size
        return os.pread(self.output.fileno(), min(size, 4000), max(0, size-4000)).decode(errors='replace')

    def close(self):
        proc, output = self.proc, self.output
        if proc is None:
            return
        try:
            if proc.poll() is None:
                # This is exclusively the model process, never a motor controller.
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2)
            try:
                self.log_dir.mkdir(parents=True, exist_ok=True)
                size = os.fstat(output.fileno()).st_size
                data = os.pread(output.fileno(), min(size, 2_000_000), max(0, size-2_000_000))
                (self.log_dir/(time.strftime('%Y%m%d-%H%M%S')+f'-{proc.pid}.log')).write_bytes(data)
            except OSError:
                pass
        finally:
            if proc.stdin:
                proc.stdin.close()
            output.close()
            self.proc = self.output = None
            self.offset, self.pending = 0, ''
