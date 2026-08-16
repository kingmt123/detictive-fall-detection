"""pipeline/event_aggregator.py 的 toy 测试。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.event_aggregator import FrameScore, aggregate, smooth


def seq(pairs):
    return [FrameScore(t, s) for t, s in pairs]


def test_empty():
    assert aggregate([]) == []


def test_clean_fall_produces_single_event():
    # 2s 安静(0.1) → 1s 跌倒(0.9) → 2s 安静
    frames = seq([(t * 0.1, 0.9 if 2.0 <= t * 0.1 < 3.0 else 0.1) for t in range(50)])
    evs = aggregate(frames)
    assert len(evs) == 1
    assert abs(evs[0].t_start - 2.0) < 0.3 and abs(evs[0].t_end - 3.0) < 0.3
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
    assert evs[0].t_start == 0.0 and evs[0].t_end == 0.7  # 滞回延伸至跌破 th_lo 的帧


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


def test_score_is_window_max():
    pairs = [(0.0, 0.7), (0.1, 0.95), (0.2, 0.7), (0.3, 0.1)]
    evs = aggregate(seq(pairs), smooth_win=1, th_hi=0.6, th_lo=0.4, min_dur=0.0)
    assert len(evs) == 1 and abs(evs[0].score - 0.95) < 1e-9


def test_smooth_window():
    assert smooth([1.0, 0.0, 0.0, 0.0, 1.0], win=3)[2] < 0.4
    assert smooth([0.5, 0.5], win=1) == [0.5, 0.5]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
