#!/usr/bin/env python3
"""Run hardware-free regression tests only while no controller owns the arm."""
import fcntl
import os
from pathlib import Path
import subprocess
import sys


def main():
    workspace = Path(__file__).resolve().parents[1]
    lockpath = Path('/home/jetson/Desktop/b601-rs/teleop.lock')
    with lockpath.open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('控制器正在运行，拒绝执行回归测试；先停止并确认机械臂已可靠支撑。')
        env = {**os.environ, 'OPENBLAS_NUM_THREADS':'1', 'OMP_NUM_THREADS':'1',
               'MKL_NUM_THREADS':'1', 'NUMEXPR_NUM_THREADS':'1'}
        patterns = sys.argv[1:] or ['test_b601_control_io.py', 'test_b601_rs*.py',
                    'test_b601_teleop_manager.py', 'test_b601_vla*.py',
                    'test_b601_fault_hold.py', 'test_b601_dataset.py']
        for pattern in patterns:
            subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
                            'tools', '-p', pattern], cwd=workspace, env=env, check=True)


if __name__ == '__main__':
    main()
