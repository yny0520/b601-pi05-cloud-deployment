#!/usr/bin/env python3
"""Portable, static-only verification. Never imports robot/model runtime code."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT/'FILE_MANIFEST.json').read_text(encoding='utf-8'))
for entry in manifest['files']:
    path = ROOT/entry['path']
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
        raise SystemExit('SHA256 mismatch: '+entry['path'])
count = 0
for path in ROOT.rglob('*.py'):
    ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    count += 1
for path in ROOT.rglob('*.json'):
    json.loads(path.read_text(encoding='utf-8'))
modules = {p.stem for p in (ROOT/'jetson/tools').glob('*.py')}
for path in (ROOT/'jetson/tools').glob('*.py'):
    for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
        names = [node.module] if isinstance(node, ast.ImportFrom) and node.module else [n.name for n in node.names] if isinstance(node, ast.Import) else []
        for name in names:
            if name.startswith(('b601_', 'star_arm_', 'camera_preview_')) and name not in modules:
                raise SystemExit('Missing local dependency '+name+' in '+path.name)
print(f'PASS: {len(manifest["files"])} hashes; {count} Python sources; JSON and local imports checked. No runtime imports, network or motor access.')
