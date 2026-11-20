#!/usr/bin/env python3
"""Supervise one authorized Pi run; return promptly after its own termination.

This does not replace the local controller or alter its motion/fault limits.
User stop, controller/PID/epoch changes and faults revoke automatic recovery.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import time
import urllib.request
import uuid

from b601_control_io import atomic_runtime_json, runtime_path


def owned_hold(v, pid, epoch):
    t = v['teleop']
    return (t.get('running') and t.get('pid') == pid
            and t.get('control_epoch') == epoch and t.get('state') == 'paused'
            and not t.get('fault_hold') and v.get('feedback_fresh'))


def can_return(v, pid, epoch, own_stop=False):
    return (owned_hold(v, pid, epoch) and not v['run'].get('active')
            and (own_stop or v['run'].get('state') == 'complete'))


def return_trajectory(current_command, home):
    goal = list(home)
    goal[6] = current_command[6]
    return goal, [[a + (b-a)*(i+1)/16 for a, b in zip(current_command, goal)]
                  for i in range(16)]


class Demo:
    def __init__(self, base, out):
        self.base, self.out = base, out
        html = urllib.request.urlopen(base+'/', timeout=3).read().decode()
        self.token = json.loads(re.search(r'const recordToken=(.*?);', html).group(1))
        self.samples = []
        self.images = ThreadPoolExecutor(max_workers=2)
        self.image_jobs = []

    def get(self, path):
        return json.load(urllib.request.urlopen(self.base+path, timeout=2))

    def post(self, path, data):
        req = urllib.request.Request(self.base+path, data=json.dumps(data).encode(),
                                     headers={'Content-Type': 'application/json',
                                              'X-Record-Token': self.token})
        return json.load(urllib.request.urlopen(req, timeout=5))

    def sample(self, stage):
        v, p = self.get('/api/vla/status'), self.get('/api/status')
        v['teleop'].pop('output_tail', None)
        self.samples.append({'mono': time.monotonic(), 'stage': stage,
                             'vla': v, 'packet': p['teleop']})
        return v, p['teleop']

    def snapshot(self, camera, label):
        with urllib.request.urlopen(self.base+'/stream/'+camera, timeout=3) as r:
            data = b''
            while len(data) < 4_000_000:
                data += r.read(4096)
                a = data.find(b'\xff\xd8')
                z = data.find(b'\xff\xd9', max(0, a))
                if a >= 0 and z >= 0:
                    payload = data[a:z+2]
                    (self.out/(camera+'-'+label+'.jpg')).write_bytes(payload)
                    (self.out/(camera+'-latest.jpg')).write_bytes(payload)
                    return
            raise RuntimeError('invalid camera stream')

    def photos(self, label):
        for job in self.image_jobs:
            if job.done():
                try:
                    job.result()
                except Exception as exc:
                    print('IMAGE_ERROR', str(exc), flush=True)
        self.image_jobs = [j for j in self.image_jobs if not j.done()]
        if not self.image_jobs:
            self.image_jobs = [self.images.submit(self.snapshot, n, label)
                               for n in ['top', 'hand']]

    def recover(self, pid, epoch, home, own_stop):
        v, p = self.sample('before_return')
        if not can_return(v, pid, epoch, own_stop):
            return {'completed': False, 'reason': 'recovery authorization/state revoked'}
        goal, actions = return_trajectory(p['command_deg'], home)
        ident = uuid.uuid4().hex
        path = runtime_path(pid, 'json')
        if path.exists() or v['step'].get('state') in ('pending', 'executing'):
            return {'completed': False, 'reason': 'pending motion; no replacement'}
        atomic_runtime_json(path, {'id': ident, 'created_mono_ns': time.monotonic_ns(),
                                  'action_deg': actions[0], 'expected_state_deg': p['follower_deg'],
                                  'session_origin_deg': p['follower_deg'], 'trajectory_deg': actions})
        started, last = time.monotonic(), 0
        while time.monotonic()-started < 35:
            v, p = self.sample('return')
            if not owned_hold(v, pid, epoch) or v['run'].get('active'):
                raise RuntimeError('control changed during return; no further commands')
            step = v['step']
            if step.get('id') == ident:
                if step['state'] in ('aborted', 'rejected'):
                    return {'completed': False, 'reason': step}
                if step['state'] == 'complete':
                    return {'completed': True, 'goal': goal, 'actual': p['follower_deg'],
                            'elapsed_s': time.monotonic()-started}
            if time.monotonic()-last > 2:
                print('RETURN', [round(x, 2) for x in p['follower_deg']], flush=True)
                last = time.monotonic()
            time.sleep(.15)
        self.post('/api/teleop/vla-abort', {})
        return {'completed': False, 'reason': 'existing 35-second return timeout'}

    def run(self):
        v, p = self.sample('before')
        pid, epoch = v['teleop'].get('pid'), v['teleop'].get('control_epoch')
        if not owned_hold(v, pid, epoch) or v['run'].get('active'):
            raise RuntimeError('requires current autonomous healthy hold, no model run')
        if not v['teleop'].get('autonomous') or v['model']['id'] != 'pi05_b601_lora_6000':
            raise RuntimeError('unexpected controller or model')
        home = list(p['command_deg'])
        self.snapshot('top', 'before')
        self.snapshot('hand', 'before')
        result = {'pid': pid, 'home': home, 'thermal_warning_c': 50,
                  'thermal_return_c': 80, 'own_stop': False}
        started = time.monotonic()
        try:
            r = self.post('/api/vla/start', {'action_steps': 16})
            result['run_id'] = r['id']
            print('START', r['id'], flush=True)
            lastprint = lastphoto = 0
            while True:
                v, p = self.sample('model')
                now = time.monotonic()
                if v['run'].get('id') != r['id']:
                    raise RuntimeError('model session changed')
                if not owned_hold(v, pid, epoch):
                    raise RuntimeError('controller, feedback or epoch changed')
                temps = [(m or {}).get('temperature_mos_c') for m in p.get('motor_feedback') or []]
                if now-lastprint > 2:
                    print(json.dumps({'elapsed': round(now-started, 1), 'state': v['run']['state'],
                                      'rounds': v['run'].get('rounds'), 'points': v['run']['steps'],
                                      'pose': p['follower_deg'], 'temps': temps,
                                      'warning': any(t is not None and t >= 50 for t in temps)},
                                     ensure_ascii=False), flush=True)
                    lastprint = now
                if not v['run']['active']:
                    result['reason'] = v['run']['state']+': '+v['run']['message']
                    break
                if any(t is not None and t >= 80 for t in temps) or now-started > 190:
                    result['reason'] = ('temperature reached 80C; controlled return'
                                        if any(t is not None and t >= 80 for t in temps)
                                        else 'outer monitor time budget')
                    self.post('/api/vla/stop', {})
                    result['own_stop'] = True
                    epoch += 1
                    deadline = time.monotonic()+3
                    while True:
                        v, p = self.sample('cancel_wait')
                        if v['run'].get('id') != r['id']:
                            raise RuntimeError('model changed during cancellation')
                        # Our own stop sends P and advances the control epoch.
                        if (v['teleop'].get('pid') != pid or not v['teleop'].get('running')
                                or v['teleop'].get('control_epoch') != epoch
                                or v['teleop'].get('fault_hold')):
                            raise RuntimeError('controller lost during cancellation')
                        if (v['run']['state'] == 'stopped' and not v['run']['active']
                                and v['teleop']['state'] == 'paused'
                                and v['step'].get('state') not in ('pending', 'executing')
                                and not runtime_path(pid, 'abort').exists()):
                            break
                        if time.monotonic() > deadline:
                            raise RuntimeError('cancel not acknowledged')
                        time.sleep(.1)
                    break
                if now-lastphoto > 4:
                    self.photos('t%03d' % (now-started))
                    lastphoto = now
                time.sleep(.25)
            result['model_seconds'] = time.monotonic()-started
            result['rounds'], result['points'] = v['run'].get('rounds'), v['run']['steps']
            # Perform return before slow diagnostic writes or camera waits.
            result['return'] = self.recover(pid, epoch, home, result['own_stop'])
        except BaseException as exc:
            result['error'] = str(exc)
            # A user/control change must never launch a new motion or re-enable.
            try:
                v = self.get('/api/vla/status')
                if v['run'].get('id') == result.get('run_id') and v['run'].get('active'):
                    self.post('/api/vla/stop', {})
            except Exception:
                pass
        finally:
            v, p = self.sample('final')
            result['final_controller'] = v['teleop']
            result['final_temperatures'] = [(m or {}).get('temperature_mos_c')
                                            for m in p.get('motor_feedback') or []]
            result['final_pose'] = p.get('follower_deg')
            self.images.shutdown(wait=True)
            for job in self.image_jobs:
                try:
                    job.result()
                except Exception:
                    pass
            for name in ['top', 'hand']:
                try:
                    self.snapshot(name, 'after')
                except Exception:
                    pass
            (self.out/'samples.json').write_text(json.dumps(self.samples, ensure_ascii=False))
            (self.out/'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
            print('END', json.dumps(result, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='start the authorized real robot run')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--base', default='http://127.0.0.1:8766')
    args = parser.parse_args()
    if not args.execute:
        parser.error('real motion requires --execute')
    args.out.mkdir(parents=True, exist_ok=False)
    Demo(args.base, args.out).run()


if __name__ == '__main__':
    main()
