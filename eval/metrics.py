"""赛题评测指标：视频片段级跌倒事件检测。

赛题定义（见赛题 docx 第 24~34 行）:
- TP: 真实跌倒并且检测为跌倒
- FP: 日常行为误检测为跌倒
- FN: 真实跌倒但是没检测到（漏检）
- MAP = (P@R90 + P@R95) / 2，即召回率 90% 与 95% 时精准率的均值

事件匹配协议（本地评测集自建，尽量贴近赛题"片段级"语义）:
- GT 事件与预测事件属于同一 clip，且时间区间 IoU >= match_iou（默认 0.3）则匹配；
- 一个 GT 最多匹配一个预测（取分数最高者），一个预测最多匹配一个 GT；
- 未匹配的预测计 FP，未匹配的 GT 计 FN。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Event:
    """一个跌倒事件（GT 或预测）。预测需带 score，GT 的 score 置 1.0。"""
    clip_id: str
    t_start: float
    t_end: float
    score: float = 1.0

    def __post_init__(self):
        if not isinstance(self.clip_id, str) or not self.clip_id:
            raise ValueError("clip_id 必须是非空字符串")
        if not all(math.isfinite(value) for value in (self.t_start, self.t_end, self.score)):
            raise ValueError("事件时间与分数必须是有限数值")
        if self.t_end <= self.t_start:
            raise ValueError(f"t_end({self.t_end}) 必须大于 t_start({self.t_start})")


def temporal_iou(a: Event, b: Event) -> float:
    inter = max(0.0, min(a.t_end, b.t_end) - max(a.t_start, b.t_start))
    union = max(a.t_end, b.t_end) - min(a.t_start, b.t_start)
    return inter / union if union > 0 else 0.0


def match_events(
    gts: list[Event], preds: list[Event], match_iou: float = 0.3
) -> tuple[list[tuple[Event, Event]], list[Event], list[Event]]:
    """最大基数一对一匹配；高分预测优先，同分时不依赖输入排列。

    Returns: (matches, unmatched_preds[FP], unmatched_gts[FN])
    """
    if not 0.0 <= match_iou <= 1.0:
        raise ValueError("match_iou 必须在 [0, 1] 内")
    pred_order = sorted(
        range(len(preds)),
        key=lambda i: (
            -preds[i].score,
            preds[i].clip_id,
            preds[i].t_start,
            preds[i].t_end,
        ),
    )
    adjacency: dict[int, list[int]] = {}
    for pred_i in pred_order:
        pred = preds[pred_i]
        candidates = [
            (gt_i, temporal_iou(gt, pred))
            for gt_i, gt in enumerate(gts)
            if gt.clip_id == pred.clip_id and temporal_iou(gt, pred) >= match_iou
        ]
        adjacency[pred_i] = [
            gt_i
            for gt_i, _ in sorted(
                candidates,
                key=lambda item: (
                    -item[1],
                    gts[item[0]].clip_id,
                    gts[item[0]].t_start,
                    gts[item[0]].t_end,
                ),
            )
        ]

    gt_to_pred: dict[int, int] = {}

    def assign(pred_i: int, seen_gts: set[int]) -> bool:
        for gt_i in adjacency[pred_i]:
            if gt_i in seen_gts:
                continue
            seen_gts.add(gt_i)
            owner = gt_to_pred.get(gt_i)
            if owner is None or assign(owner, seen_gts):
                gt_to_pred[gt_i] = pred_i
                return True
        return False

    for pred_i in pred_order:
        assign(pred_i, set())

    matched_pred = set(gt_to_pred.values())
    matches = [(gts[gt_i], preds[pred_i]) for gt_i, pred_i in gt_to_pred.items()]
    unmatched_preds = [pred for i, pred in enumerate(preds) if i not in matched_pred]
    unmatched_gts = [gt for i, gt in enumerate(gts) if i not in gt_to_pred]
    return matches, unmatched_preds, unmatched_gts


@dataclass
class PRPoint:
    threshold: float
    precision: float
    recall: float
    tp: int
    fp: int
    fn: int


def pr_curve(
    gts: list[Event], preds: list[Event], match_iou: float = 0.3
) -> list[PRPoint]:
    """对预测分数扫阈值，产出 P-R 曲线上的点（按阈值降序）。"""
    if not 0.0 <= match_iou <= 1.0:
        raise ValueError("match_iou 必须在 [0, 1] 内")
    n_gt = len(gts)
    points: list[PRPoint] = []
    scores = sorted({p.score for p in preds}, reverse=True)
    for th in scores:
        kept = [p for p in preds if p.score >= th]
        matches, fps, fns = match_events(gts, kept, match_iou)
        tp = len(matches)
        fp = len(fps)
        fn = len(fns)
        recall = tp / n_gt if n_gt else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        points.append(PRPoint(th, precision, recall, tp, fp, fn))
    return points


def clip_pr_curve(
    gt_by_clip: dict[str, bool], pred_scores: dict[str, float]
) -> list[PRPoint]:
    """整段视频二分类的 P-R 曲线。

    所有 GT clip 必须显式提供预测分数。缺失预测和不在 GT 的额外预测都会被拒绝，
    防止批处理失败或拼写错误被静默计入指标。
    """
    unknown = set(pred_scores) - set(gt_by_clip)
    if unknown:
        raise ValueError(f"预测包含未知 clip: {sorted(unknown)[:5]}")
    missing = set(gt_by_clip) - set(pred_scores)
    if missing:
        raise ValueError(f"预测缺少 clip: {sorted(missing)[:5]}")
    if any(type(label) is not bool for label in gt_by_clip.values()):
        raise TypeError("clip 标签必须是 bool")
    scores = {clip: float(pred_scores[clip]) for clip in gt_by_clip}
    if any(not math.isfinite(score) for score in scores.values()):
        raise ValueError("clip 预测分数必须是有限数值")
    thresholds = sorted(set(scores.values()), reverse=True)
    n_positive = sum(gt_by_clip.values())
    points: list[PRPoint] = []
    for threshold in thresholds:
        predicted = {clip for clip, score in scores.items() if score >= threshold}
        tp = sum(gt_by_clip[clip] for clip in predicted)
        fp = len(predicted) - tp
        fn = n_positive - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / n_positive if n_positive else 0.0
        points.append(PRPoint(threshold, precision, recall, tp, fp, fn))
    return points


def precision_at_recall(curve: list[PRPoint], target_recall: float) -> float:
    """召回率 >= target 的所有工作点中的最大精准率；达不到该召回则返回 0。"""
    feasible = [pt.precision for pt in curve if pt.recall >= target_recall]
    return max(feasible) if feasible else 0.0


def competition_map(
    gts: list[Event] | dict[str, bool],
    preds: list[Event] | dict[str, float],
    *,
    mode: Literal["event", "clip"],
    match_iou: float = 0.3,
) -> dict:
    """赛题 MAP；同时返回内部比例值和官方公式使用的百分数值。

    ``event`` 是本项目的代理事件协议；``clip`` 用于官方若采用整段二分类时。
    """
    if mode == "event":
        if not isinstance(gts, list) or not isinstance(preds, list):
            raise TypeError("event 模式要求 Event 列表")
        if any(not isinstance(event, Event) for event in [*gts, *preds]):
            raise TypeError("event 模式要求 Event 列表")
        curve = pr_curve(gts, preds, match_iou)
    elif mode == "clip":
        if not isinstance(gts, dict) or not isinstance(preds, dict):
            raise TypeError("clip 模式要求 clip->label/score 字典")
        curve = clip_pr_curve(gts, preds)
    else:
        raise ValueError(f"未知评测模式: {mode}")
    p90 = precision_at_recall(curve, 0.90)
    p95 = precision_at_recall(curve, 0.95)
    map_ratio = (p90 + p95) / 2
    return {
        "p_at_r90": p90,
        "p_at_r95": p95,
        "map": map_ratio,
        "map_percent": map_ratio * 100.0,
        "curve": curve,
        "mode": mode,
    }
