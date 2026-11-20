"""Read-only replay routes, restricted to individual dataset episodes."""
import json
from pathlib import Path
import re
from urllib.parse import urlsplit, parse_qs
import cv2
import numpy as np
from b601_collection_profiles import get_profile


def json_reply(code, payload):
    return code, 'application/json; charset=utf-8', json.dumps(payload, ensure_ascii=False).encode()


def get_dataset_response(url, root):
    parts = urlsplit(url)
    if parts.path not in ('/review', '/api/episode', '/recorded'):
        return None
    try:
        params = parse_qs(parts.query)
        if 'dataset' in params:
            root = get_profile(params['dataset'][0])['root']
        name = params.get('episode', [''])[0]
        if not re.fullmatch(r'\d{8}-\d{6}-\d{6}', name):
            raise ValueError('无效的采集集编号')
        folder = Path(root)/name
        if not folder.is_dir():
            raise ValueError('该集不存在')
        if parts.path == '/review':
            return 200, 'text/html; charset=utf-8', Path(__file__).with_name('b601_review.html').read_bytes()
        if parts.path == '/api/episode':
            metadata = json.loads((folder/'metadata.json').read_text())
            if metadata.get('status') == 'recording':
                raise ValueError('请先结束本集再回放')
            rows = [json.loads(line) for line in (folder/'samples.jsonl').read_text().splitlines() if line.strip()]
            events = [json.loads(line) for line in (folder/'events.jsonl').read_text().splitlines() if line.strip()]
            return json_reply(200, {'metadata': metadata, 'samples': rows, 'events': events})
        filename = params.get('file', [''])[0]
        path = (folder/filename).resolve()
        if not path.is_relative_to(folder.resolve()) or len(Path(filename).parts) != 2:
            raise ValueError('不允许访问该路径')
        if Path(filename).parts[0] not in ('hand','top','depth') or path.suffix not in ('.jpg','.png','.npy'):
            raise ValueError('不支持的图像')
        if not path.is_file():
            raise ValueError('图像不存在')
        if Path(filename).parts[0] == 'depth':
            raw = np.load(path, allow_pickle=False) if path.suffix == '.npy' else cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if raw is None:
                raise ValueError('无法读取深度图')
            meters = raw.astype(np.float32)*.001 if raw.dtype == np.uint16 else raw.astype(np.float32)
            valid = np.isfinite(meters) & (meters>0)
            scaled = np.rint(np.clip(np.where(valid,meters,0),0,2)*127.5).astype(np.uint8)
            heat = cv2.applyColorMap(scaled,cv2.COLORMAP_JET)
            heat[~valid] = 0
            heat[valid & (meters>2)] = (70,70,70)
            ok,encoded = cv2.imencode('.jpg',heat)
            if not ok:
                raise ValueError('无法生成深度预览')
            return 200, 'image/jpeg', encoded.tobytes()
        return 200, 'image/jpeg', path.read_bytes()
    except (OSError, ValueError, KeyError) as exc:
        return json_reply(400, {'error': str(exc)})
