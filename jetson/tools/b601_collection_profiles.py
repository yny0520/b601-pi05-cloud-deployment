"""Known collection tasks and destinations; no hardware or arbitrary web paths."""
import copy
import json
from pathlib import Path

DATASETS = Path('/home/jetson/Desktop/b601-rs/datasets')
SETTINGS = DATASETS.parent / 'calibration/collection_settings.json'
PROFILES = {
    'pick_place': {
        'id': 'pick_place', 'label': 'Pick and Place · 抓取放置',
        'root': str(DATASETS / 'b601-pick-place'),
        'task': '抓起桌面上的物体，搬运到目标区域并放下',
        'success_criterion': '物体放到目标区域内，松爪后稳定放置且夹爪移开',
        'setup_placeholder': '例如：物体=方块；起点=左侧标记；目标=右侧托盘；相机位置v1',
        'note_placeholder': '例如：抓空、掉落、未放入目标区、重新抓取',
        'phases': ['接近', '抓取', '抬起', '搬运', '放置', '松爪', '退回纠错', '验证'],
    },
    'insertion': {
        'id': 'insertion', 'label': '网线插接',
        'root': str(DATASETS / 'b601-insertion'),
        'task': '网线插接', 'success_criterion': '人工确认插接到位',
        'setup_placeholder': '例如：插座A、正向、线缆位置1',
        'note_placeholder': '例如：插偏、线缆拖拽、夹持打滑、接触后退回',
        'phases': ['接近', '对孔', '插接', '退回纠错', '布线', '验证'],
    },
}


def get_profile(name):
    if name not in PROFILES:
        raise ValueError('未知采集任务；请选择抓取放置或网线插接')
    return copy.deepcopy(PROFILES[name])


def load_profile_name(path=SETTINGS):
    try:
        name = json.loads(Path(path).read_text())['profile']
    except FileNotFoundError:
        return 'insertion'
    get_profile(name)
    return name


def render_panel(token, profile):
    def js(value):
        return json.dumps(value, ensure_ascii=False).replace('<', '\\u003c')
    return (Path(__file__).with_name('b601_collection_panel.html').read_text()
            .replace('__RECORD_TOKEN__', js(token))
            .replace('__COLLECTION_PROFILES__', js(PROFILES))
            .replace('__COLLECTION_PROFILE__', js(profile)))
