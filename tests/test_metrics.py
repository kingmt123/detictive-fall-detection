"""eval/metrics.py 的 toy 测试：用手工构造的预测验证指标计算正确性。"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.metrics import (
    Event,
    clip_pr_curve,
    competition_map,
    match_events,
    temporal_iou,
)


def test_competition_map_requires_explicit_protocol():
    with pytest.raises(TypeError):
        competition_map([], [])


def test_temporal_iou():
    a = Event("c1", 0.0, 2.0)
    b = Event("c1", 1.0, 3.0)
    assert abs(temporal_iou(a, b) - 1.0 / 3.0) < 1e-9
    assert temporal_iou(a, Event("c1", 5.0, 6.0)) == 0.0
    assert temporal_iou(a, a) == 1.0


def test_event_rejects_zero_duration_and_non_finite_values():
    with pytest.raises(ValueError):
        Event("c1", 1.0, 1.0)
    with pytest.raises(ValueError):
        Event("c1", 0.0, float("nan"))
    with pytest.raises(ValueError):
        Event("c1", 0.0, 1.0, float("nan"))
    with pytest.raises(ValueError):
        Event("", 0.0, 1.0)


def test_match_events_rejects_invalid_iou_threshold():
    with pytest.raises(ValueError):
        match_events([], [], match_iou=-0.1)
    with pytest.raises(ValueError):
        match_events([], [], match_iou=1.1)


def test_clip_curve_rejects_non_boolean_labels():
    with pytest.raises(TypeError):
        clip_pr_curve({"fall-1": 1}, {"fall-1": 0.8})


def test_clip_curve_rejects_non_finite_scores():
    with pytest.raises(ValueError):
        clip_pr_curve({"fall-1": True}, {"fall-1": float("nan")})


def test_perfect_prediction_map_is_one():
    gts = [Event("c1", 2.0, 4.0), Event("c2", 1.0, 3.0)]
    preds = [Event("c1", 2.1, 4.1, 0.99), Event("c2", 0.9, 3.2, 0.98)]
    r = competition_map(gts, preds, mode="event")
    assert r["p_at_r90"] == 1.0 and r["p_at_r95"] == 1.0 and r["map"] == 1.0
    assert r["map_percent"] == 100.0


def test_clip_level_curve_counts_negative_clips_as_false_positives():
    curve = clip_pr_curve(
        {"fall-1": True, "fall-2": True, "adl-1": False},
        {"fall-1": 0.9, "fall-2": 0.8, "adl-1": 0.95},
    )
    full_recall = [p for p in curve if p.recall == 1.0]
    assert len(full_recall) == 1
    assert full_recall[0].tp == 2
    assert full_recall[0].fp == 1
    assert full_recall[0].fn == 0
    assert abs(full_recall[0].precision - 2 / 3) < 1e-9


def test_clip_level_curve_rejects_missing_predictions():
    with pytest.raises(ValueError, match="缺少 clip"):
        clip_pr_curve(
            {"fall-1": True, "adl-1": False},
            {"fall-1": 0.9},
        )


def test_clip_level_competition_map_supports_official_percentage_scale():
    result = competition_map(
        {"fall-1": True, "fall-2": True, "adl-1": False},
        {"fall-1": 0.9, "fall-2": 0.8, "adl-1": 0.1},
        mode="clip",
    )
    assert result["map"] == 1.0
    assert result["map_percent"] == 100.0


def test_all_missed_map_is_zero():
    gts = [Event("c1", 2.0, 4.0)]
    r = competition_map(gts, [], mode="event")
    assert r["map"] == 0.0


def test_low_score_fp_never_hurts():
    # 10 个 GT 全部被高分命中：低分误报不影响（max-precision 约定下被过滤）
    gts = [Event(f"c{i}", 0.0, 1.0) for i in range(10)]
    preds = [Event(f"c{i}", 0.0, 1.0, 0.9) for i in range(10)]
    preds.append(Event("cX", 5.0, 6.0, 0.1))  # 低分误报
    r = competition_map(gts, preds, mode="event")
    assert r["p_at_r90"] == 1.0 and r["p_at_r95"] == 1.0


def test_high_score_fp_hurts():
    # 高分误报分数高于全部正确预测：达到满召回必须纳入它 → precision=10/11
    gts = [Event(f"c{i}", 0.0, 1.0) for i in range(10)]
    preds = [Event(f"c{i}", 0.0, 1.0, 0.9) for i in range(10)]
    preds.insert(0, Event("cX", 5.0, 6.0, 0.95))  # 高分误报
    r = competition_map(gts, preds, mode="event")
    assert abs(r["p_at_r90"] - 10 / 11) < 1e-9
    assert abs(r["p_at_r95"] - 10 / 11) < 1e-9


def test_recall_ceiling_limits_p_at_r95():
    # 10 个 GT，最高分只能召回 9 个（1 个预测分数为 0 以下不可能——
    # 用只有 9 个预测模拟漏检）：recall 上限 0.9 → P@R95 = 0
    gts = [Event(f"c{i}", 0.0, 1.0) for i in range(10)]
    preds = [Event(f"c{i}", 0.0, 1.0, 0.9) for i in range(9)]
    r = competition_map(gts, preds, mode="event")
    assert r["p_at_r90"] == 1.0
    assert r["p_at_r95"] == 0.0
    assert r["map"] == 0.5
    assert r["map_percent"] == 50.0


def test_match_is_one_to_one_and_highest_score_wins():
    g = Event("c1", 0.0, 2.0)
    # 两个预测都覆盖同一 GT：高分者匹配，低分者计 FP
    p_hi = Event("c1", 0.0, 2.0, 0.9)
    p_lo = Event("c1", 0.5, 2.5, 0.5)
    matches, fps, fns = match_events([g], [p_lo, p_hi])
    assert len(matches) == 1 and matches[0][1] is p_hi
    assert len(fps) == 1 and fps[0] is p_lo
    assert len(fns) == 0


def test_equal_score_matching_is_order_invariant_and_maximal():
    g1 = Event("c1", 0.0, 2.0)
    g2 = Event("c1", 1.0, 3.0)
    shared = Event("c1", 0.5, 2.5, 0.9)
    only_g1 = Event("c1", 0.0, 1.0, 0.9)

    forward = match_events([g1, g2], [shared, only_g1], match_iou=0.3)
    reversed_gt = match_events([g2, g1], [shared, only_g1], match_iou=0.3)
    reversed_pred = match_events([g1, g2], [only_g1, shared], match_iou=0.3)

    assert len(forward[0]) == 2
    assert len(reversed_gt[0]) == 2
    assert len(reversed_pred[0]) == 2
    expected = {(gt.t_start, pred.t_start) for gt, pred in forward[0]}
    assert expected == {
        (gt.t_start, pred.t_start) for gt, pred in reversed_gt[0]
    }
    assert expected == {
        (gt.t_start, pred.t_start) for gt, pred in reversed_pred[0]
    }
    assert not forward[1] and not forward[2]
    assert not reversed_gt[1] and not reversed_gt[2]


def test_event_mode_rejects_non_event_elements():
    with pytest.raises(TypeError, match="Event"):
        competition_map(["not-an-event"], [], mode="event")


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
