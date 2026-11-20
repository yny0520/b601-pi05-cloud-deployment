"""Stateless smoothing within one already-predicted joint-position path.

No sensor filtering or delayed commands: this inspects future points from the
same model chunk, never future camera observations. Each submitted prefix is
filtered independently; its endpoints and every gripper reference stay exact.
"""
import math
from statistics import median


def filter_action_path(actions, max_correction_deg=2.):
    if (not isinstance(actions, list) or not 1 <= len(actions) <= 50 or
            any(not isinstance(a, list) or len(a) != 7 or
                any(type(v) not in (int, float) or not math.isfinite(v) for v in a)
                for a in actions)):
        raise ValueError('滤波输入必须为1至50个有限的七维动作')
    if (type(max_correction_deg) not in (int, float) or
            not math.isfinite(max_correction_deg) or not 0 < max_correction_deg <= 3):
        raise ValueError('动作滤波单点修正量须在0至3°内')
    result = [list(a) for a in actions]
    if len(actions) < 3:
        return result
    despiked = [list(a) for a in actions]
    for t in range(1, len(actions)-1):
        for j in range(6):
            despiked[t][j] = median([actions[t-1][j], actions[t][j], actions[t+1][j]])
    for t in range(1, len(actions)-1):
        for j in range(6):
            smooth = (despiked[t-1][j]+2*despiked[t][j]+despiked[t+1][j])/4
            raw = actions[t][j]
            result[t][j] = max(raw-max_correction_deg, min(raw+max_correction_deg, smooth))
    return result
