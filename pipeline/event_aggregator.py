"""事件聚合器：把逐帧跌倒分数聚合成片段级跌倒事件。

流水线角色：检测/pose/时序/规则各通道融合后，每帧产出一个 fall_score∈[0,1]；
本模块负责：
  1. 滑动平均平滑（抑制单帧抖动）
  2. 滞回阈值（score≥hi 开启候选事件，score<lo 结束）防止边界震荡切碎事件
  3. 间隙合并（相邻事件间隔 < merge_gap 视为同一次跌倒）
  4. 最短持续过滤（短于 min_dur 的候选视为误报丢弃）
  5. 事件置信度 = 窗口内平滑分数的最大值
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FrameScore:
    t: float        # 帧时间戳（秒）
    score: float    # 融合后的跌倒分数 [0,1]


@dataclass(frozen=True)
class AggEvent:
    t_start: float
    t_end: float
    score: float


def smooth(scores: list[float], win: int = 5) -> list[float]:
    """居中滑动平均。win<=1 时原样返回。"""
    if win <= 1 or len(scores) < win:
        return list(scores)
    half = win // 2
    out = []
    for i in range(len(scores)):
        lo, hi = max(0, i - half), min(len(scores), i + half + 1)
        out.append(sum(scores[lo:hi]) / (hi - lo))
    return out


def aggregate(
    frames: list[FrameScore],
    smooth_win: int = 5,
    th_hi: float = 0.6,
    th_lo: float = 0.4,
    merge_gap: float = 1.0,
    min_dur: float = 0.3,
) -> list[AggEvent]:
    """逐帧分数 → 跌倒事件列表。"""
    if not frames:
        return []
    sm = smooth([f.score for f in frames], smooth_win)

    # 滞回阈值切候选段
    segs: list[list[int]] = []
    cur: list[int] | None = None
    for i, s in enumerate(sm):
        if cur is None:
            if s >= th_hi:
                cur = [i]
        else:
            cur.append(i)
            if s < th_lo:
                segs.append(cur)
                cur = None
    if cur:
        segs.append(cur)

    events = [
        AggEvent(frames[seg[0]].t, frames[seg[-1]].t, max(sm[i] for i in seg))
        for seg in segs
    ]

    # 间隙合并
    merged: list[AggEvent] = []
    for ev in events:
        if merged and ev.t_start - merged[-1].t_end < merge_gap:
            prev = merged[-1]
            merged[-1] = AggEvent(prev.t_start, ev.t_end, max(prev.score, ev.score))
        else:
            merged.append(ev)

    # 最短持续过滤
    return [ev for ev in merged if ev.t_end - ev.t_start >= min_dur]
