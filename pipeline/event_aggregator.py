"""事件聚合器：把逐帧跌倒分数聚合成片段级跌倒事件。

流水线角色：检测/pose/时序/规则各通道融合后，每帧产出一个 fall_score∈[0,1]；
本模块负责：
  1. 因果滑动平均平滑（抑制单帧抖动，不读取未来帧）
  2. 滞回阈值（score≥hi 开启候选事件，score<lo 结束）防止边界震荡切碎事件
  3. 间隙合并（相邻事件间隔 < merge_gap 视为同一次跌倒）
  4. 最短持续过滤（短于 min_dur 的候选视为误报丢弃）
  5. 事件置信度 = 窗口内平滑分数的最大值
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise


@dataclass(frozen=True)
class FrameScore:
    t: float        # 帧时间戳（秒）
    score: float | None  # 融合分数 [0,1]；None 表示该帧不可观测

    def __post_init__(self) -> None:
        if not math.isfinite(self.t):
            raise ValueError("帧时间戳必须是有限数值")
        if self.score is not None and (
            not math.isfinite(self.score) or not 0.0 <= self.score <= 1.0
        ):
            raise ValueError("帧分数必须是 [0,1] 内的有限数值或 None")


@dataclass(frozen=True)
class AggEvent:
    t_start: float
    t_end: float
    score: float


@dataclass(frozen=True)
class TrackedAggEvent:
    track_id: int
    t_start: float
    t_end: float
    score: float


def smooth(scores: list[float], win: int = 5) -> list[float]:
    """因果滑动平均；时刻 i 只使用当前及之前的分数。"""
    if win < 1 or win % 2 == 0:
        raise ValueError("win 必须是正奇数")
    if win == 1:
        return list(scores)
    out = []
    for i in range(len(scores)):
        lo = max(0, i - win + 1)
        out.append(sum(scores[lo : i + 1]) / (i - lo + 1))
    return out


def aggregate(
    frames: list[FrameScore],
    smooth_win: int = 5,
    th_hi: float = 0.6,
    th_lo: float = 0.4,
    merge_gap: float = 1.0,
    min_dur: float = 0.3,
    max_unobserved_gap: float = 0.5,
) -> list[AggEvent]:
    """逐帧分数 → 跌倒事件列表。"""
    if smooth_win < 1 or smooth_win % 2 == 0:
        raise ValueError("smooth_win 必须是正奇数")
    if not 0.0 <= th_lo < th_hi <= 1.0:
        raise ValueError("阈值必须满足 0 <= th_lo < th_hi <= 1")
    if merge_gap < 0 or min_dur < 0 or max_unobserved_gap < 0:
        raise ValueError("时间间隔参数不能为负数")
    if any(b.t <= a.t for a, b in pairwise(frames)):
        raise ValueError("帧时间戳必须严格递增")
    if not frames:
        return []
    observed_history: list[float] = []
    sm: list[float | None] = []
    last_observed_t: float | None = None
    for frame in frames:
        if frame.score is None:
            sm.append(None)
            continue
        if last_observed_t is not None and frame.t - last_observed_t > max_unobserved_gap:
            observed_history.clear()
        observed_history.append(frame.score)
        observed_history = observed_history[-smooth_win:]
        sm.append(sum(observed_history) / len(observed_history))
        last_observed_t = frame.t

    # 滞回阈值切候选段
    segs: list[list[int]] = []
    cur: list[int] | None = None
    for i, s in enumerate(sm):
        if s is None:
            if cur is not None and frames[i].t - frames[cur[-1]].t > max_unobserved_gap:
                segs.append(cur)
                cur = None
            continue
        if cur is None:
            if s >= th_hi:
                cur = [i]
        else:
            if s < th_lo:
                segs.append(cur)
                cur = None
            else:
                cur.append(i)
    if cur:
        segs.append(cur)

    events = [
        AggEvent(
            frames[seg[0]].t,
            frames[seg[-1]].t,
            max(float(sm[i]) for i in seg if sm[i] is not None),
        )
        for seg in segs
    ]
    events = [
        event
        for event in events
        if event.t_end - event.t_start + 1e-9 >= min_dur
    ]

    # 间隙合并
    merged: list[AggEvent] = []
    for ev in events:
        if merged and ev.t_start - merged[-1].t_end < merge_gap:
            prev = merged[-1]
            merged[-1] = AggEvent(prev.t_start, ev.t_end, max(prev.score, ev.score))
        else:
            merged.append(ev)

    return merged


def aggregate_tracks(
    scores_by_track: dict[int, list[FrameScore]],
    **aggregate_kwargs,
) -> list[TrackedAggEvent]:
    """分别聚合每条人物轨迹，事件保留 track_id。"""
    events = [
        TrackedAggEvent(track_id, event.t_start, event.t_end, event.score)
        for track_id, frames in scores_by_track.items()
        for event in aggregate(frames, **aggregate_kwargs)
    ]
    return sorted(events, key=lambda event: (event.track_id, event.t_start, event.t_end))
