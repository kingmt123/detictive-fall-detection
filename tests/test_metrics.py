"""eval/metrics.py 的 toy 测试：用手工构造的预测验证指标计算正确性。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.metrics import Event, competition_map, match_events, temporal_iou


def test_temporal_iou():
    a = Event("c1", 0.0, 2.0)
    b = Event("c1", 1.0, 3.0)
    assert abs(temporal_iou(a, b) - 1.0 / 3.0) < 1e-9
    assert temporal_iou(a, Event("c1", 5.0, 6.0)) == 0.0
    assert temporal_iou(a, a) == 1.0


def test_perfect_prediction_map_is_one():
    gts = [Event("c1", 2.0, 4.0), Event("c2", 1.0, 3.0)]
    preds = [Event("c1", 2.1, 4.1, 0.99), Event("c2", 0.9, 3.2, 0.98)]
    r = competition_map(gts, preds)
    assert r["p_at_r90"] == 1.0 and r["p_at_r95"] == 1.0 and r["map"] == 1.0


def test_all_missed_map_is_zero():
    gts = [Event("c1", 2.0, 4.0)]
    r = competition_map(gts, [])
    assert r["map"] == 0.0


def test_low_score_fp_never_hurts():
    # 10 个 GT 全部被高分命中：低分误报不影响（max-precision 约定下被过滤）
    gts = [Event(f"c{i}", 0.0, 1.0) for i in range(10)]
    preds = [Event(f"c{i}", 0.0, 1.0, 0.9) for i in range(10)]
    preds.append(Event("cX", 5.0, 6.0, 0.1))  # 低分误报
    r = competition_map(gts, preds)
    assert r["p_at_r90"] == 1.0 and r["p_at_r95"] == 1.0


def test_high_score_fp_hurts():
    # 高分误报分数高于全部正确预测：达到满召回必须纳入它 → precision=10/11
    gts = [Event(f"c{i}", 0.0, 1.0) for i in range(10)]
    preds = [Event(f"c{i}", 0.0, 1.0, 0.9) for i in range(10)]
    preds.insert(0, Event("cX", 5.0, 6.0, 0.95))  # 高分误报
    r = competition_map(gts, preds)
    assert abs(r["p_at_r90"] - 10 / 11) < 1e-9
    assert abs(r["p_at_r95"] - 10 / 11) < 1e-9


def test_recall_ceiling_limits_p_at_r95():
    # 10 个 GT，最高分只能召回 9 个（1 个预测分数为 0 以下不可能——
    # 用只有 9 个预测模拟漏检）：recall 上限 0.9 → P@R95 = 0
    gts = [Event(f"c{i}", 0.0, 1.0) for i in range(10)]
    preds = [Event(f"c{i}", 0.0, 1.0, 0.9) for i in range(9)]
    r = competition_map(gts, preds)
    assert r["p_at_r90"] == 1.0
    assert r["p_at_r95"] == 0.0
    assert r["map"] == 0.5


def test_match_is_one_to_one_and_highest_score_wins():
    g = Event("c1", 0.0, 2.0)
    # 两个预测都覆盖同一 GT：高分者匹配，低分者计 FP
    p_hi = Event("c1", 0.0, 2.0, 0.9)
    p_lo = Event("c1", 0.5, 2.5, 0.5)
    matches, fps, fns = match_events([g], [p_lo, p_hi])
    assert len(matches) == 1 and matches[0][1] is p_hi
    assert len(fps) == 1 and fps[0] is p_lo
    assert len(fns) == 0


def test_cross_clip_never_matches():
    g = Event("c1", 0.0, 2.0)
    p = Event("c2", 0.0, 2.0, 0.9)  # 区间完全相同但 clip 不同
    matches, fps, fns = match_events([g], [p])
    assert not matches and len(fps) == 1 and len(fns) == 1


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
