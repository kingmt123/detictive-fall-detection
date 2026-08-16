"""pipeline/event_aggregator.py 的 toy 测试。"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.event_aggregator import FrameScore, aggregate, aggregate_tracks, smooth


def seq(pairs):
    return [FrameScore(t, s) for t, s in pairs]


def test_empty():
    assert aggregate([]) == []


def test_frame_score_and_aggregate_parameters_are_validated():
    with pytest.raises(ValueError):
        FrameScore(0.0, 1.1)
    with pytest.raises(ValueError):
        FrameScore(float("nan"), 0.5)
    with pytest.raises(ValueError):
        aggregate(seq([(0.1, 0.5), (0.0, 0.6)]))
    with pytest.raises(ValueError):
        aggregate(seq([(0.0, 0.5)]), smooth_win=2)
    with pytest.raises(ValueError):
        aggregate(seq([(0.0, 0.5)]), th_hi=0.3, th_lo=0.4)


def test_clean_fall_produces_single_event():
    # 2s 安静(0.1) → 1s 跌倒(0.9) → 2s 安静
    frames = seq([(t * 0.1, 0.9 if 2.0 <= t * 0.1 < 3.0 else 0.1) for t in range(50)])
    evs = aggregate(frames)
    assert len(evs) == 1
    # 5 帧因果均值允许最多约 0.4s 的在线确认延迟，但绝不提前读取未来帧。
    assert 2.0 <= evs[0].t_start <= 2.4
    assert 2.9 <= evs[0].t_end <= 3.3
    assert evs[0].score > 0.8


def test_hysteresis_prevents_fragmentation():
    # 分数在高阈值附近震荡（0.9, 0.5, 0.9...）：滞回应保持为一个事件
    frames = seq([(t * 0.1, 0.9 if t % 2 == 0 else 0.5) for t in range(20)])
    evs = aggregate(frames, smooth_win=1)
    assert len(evs) == 1


def test_close_events_merged():
    # 两段高分间隔 0.5s < merge_gap=1.0 → 合并为一次跌倒
    pairs = [(0.0, 0.9), (0.1, 0.9), (0.2, 0.1), (0.3, 0.1), (0.4, 0.1),
             (0.5, 0.9), (0.6, 0.9), (0.7, 0.1), (0.8, 0.1)]
    evs = aggregate(seq(pairs), smooth_win=1, th_hi=0.6, th_lo=0.4,
                    merge_gap=1.0, min_dur=0.0)
    assert len(evs) == 1
    assert evs[0].t_start == 0.0 and evs[0].t_end == 0.6


def test_distant_events_not_merged():
    pairs = [(0.0, 0.9), (0.1, 0.9), (0.2, 0.1),
             (3.0, 0.9), (3.1, 0.9), (3.2, 0.1)]
    evs = aggregate(seq(pairs), smooth_win=1, th_hi=0.6, th_lo=0.4,
                    merge_gap=1.0, min_dur=0.0)
    assert len(evs) == 2


def test_short_blip_filtered():
    # 单帧尖峰（0.1s < min_dur=0.3）视为误报
    pairs = [(0.0, 0.1), (0.1, 0.95), (0.2, 0.1)]
    assert aggregate(seq(pairs), smooth_win=1) == []


def test_two_short_blips_do_not_become_valid_only_by_merging_gap():
    pairs = [
        (0.0, 0.9), (0.1, 0.1),
        (0.5, 0.9), (0.6, 0.1),
    ]
    assert aggregate(
        seq(pairs),
        smooth_win=1,
        th_hi=0.6,
        th_lo=0.4,
        merge_gap=1.0,
        min_dur=0.2,
    ) == []


def test_min_duration_tolerates_decimal_timestamp_roundoff():
    frames = seq([(3.7, 0.9), (3.8, 0.9), (3.9, 0.1)])
    events = aggregate(
        frames, smooth_win=1, th_hi=0.6, th_lo=0.4, merge_gap=0.0, min_dur=0.1
    )
    assert len(events) == 1


def test_event_ends_at_last_frame_above_low_threshold():
    frames = seq([(0.0, 0.9), (0.1, 0.7), (0.2, 0.1)])
    event = aggregate(frames, smooth_win=1, min_dur=0.0)[0]
    assert event.t_end == 0.1


def test_short_unobserved_gap_is_unknown_not_a_zero_score():
    frames = [
        FrameScore(0.0, 0.9),
        FrameScore(0.1, None),
        FrameScore(0.2, 0.9),
        FrameScore(0.3, 0.1),
    ]
    event = aggregate(
        frames,
        smooth_win=1,
        min_dur=0.0,
        max_unobserved_gap=0.25,
    )[0]
    assert event.t_start == 0.0
    assert event.t_end == 0.2


def test_long_unobserved_gap_closes_active_event():
    frames = [
        FrameScore(0.0, 0.9),
        FrameScore(0.1, None),
        FrameScore(0.6, None),
        FrameScore(1.0, 0.9),
        FrameScore(1.1, 0.1),
    ]
    events = aggregate(
        frames,
        smooth_win=1,
        min_dur=0.0,
        merge_gap=0.2,
        max_unobserved_gap=0.25,
    )
    assert [(event.t_start, event.t_end) for event in events] == [(0.0, 0.0), (1.0, 1.0)]


def test_aggregate_tracks_preserves_person_identity():
    scores = {
        7: seq([(0.0, 0.9), (0.1, 0.9), (0.2, 0.1)]),
        3: seq([(0.0, 0.1), (0.1, 0.9), (0.2, 0.9), (0.3, 0.1)]),
    }
    events = aggregate_tracks(scores, smooth_win=1, min_dur=0.0)
    assert [(event.track_id, event.t_start, event.t_end) for event in events] == [
        (3, 0.1, 0.2),
        (7, 0.0, 0.1),
    ]


def test_score_is_window_max():
    pairs = [(0.0, 0.7), (0.1, 0.95), (0.2, 0.7), (0.3, 0.1)]
    evs = aggregate(seq(pairs), smooth_win=1, th_hi=0.6, th_lo=0.4, min_dur=0.0)
    assert len(evs) == 1 and abs(evs[0].score - 0.95) < 1e-9


def test_smooth_window():
    assert smooth([1.0, 0.0, 0.0, 0.0, 1.0], win=3)[2] < 0.4
    assert smooth([0.5, 0.5], win=1) == [0.5, 0.5]


def test_smooth_is_causal_and_rejects_even_windows():
    assert smooth([0.0, 0.0, 1.0], win=3)[:2] == [0.0, 0.0]
    with pytest.raises(ValueError):
        smooth([0.0, 1.0], win=2)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
