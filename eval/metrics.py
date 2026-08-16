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

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Event:
    """一个跌倒事件（GT 或预测）。预测需带 score，GT 的 score 置 1.0。"""
    clip_id: str
    t_start: float
    t_end: float
    score: float = 1.0

    def __post_init__(self):
        if self.t_end < self.t_start:
            raise ValueError(f"t_end({self.t_end}) < t_start({self.t_start})")


def temporal_iou(a: Event, b: Event) -> float:
    inter = max(0.0, min(a.t_end, b.t_end) - max(a.t_start, b.t_start))
    union = max(a.t_end, b.t_end) - min(a.t_start, b.t_start)
    return inter / union if union > 0 else 0.0


def match_events(
    gts: list[Event], preds: list[Event], match_iou: float = 0.3
) -> tuple[list[tuple[Event, Event]], list[Event], list[Event]]:
    """贪心匹配：按预测分数从高到低，匹配 IoU 最大且达标的未匹配 GT。

    Returns: (matches, unmatched_preds[FP], unmatched_gts[FN])
    """
    matched_gt: set[int] = set()
    matches: list[tuple[Event, Event]] = []
    unmatched_preds: list[Event] = []
    for p in sorted(preds, key=lambda e: -e.score):
        best_i, best_iou = -1, match_iou
        for i, g in enumerate(gts):
            if i in matched_gt or g.clip_id != p.clip_id:
                continue
            iou = temporal_iou(g, p)
            if iou >= best_iou:
                best_i, best_iou = i, iou
        if best_i >= 0:
            matched_gt.add(best_i)
            matches.append((gts[best_i], p))
        else:
            unmatched_preds.append(p)
    unmatched_gts = [g for i, g in enumerate(gts) if i not in matched_gt]
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


def precision_at_recall(curve: list[PRPoint], target_recall: float) -> float:
    """召回率 >= target 的所有工作点中的最大精准率；达不到该召回则返回 0。"""
    feasible = [pt.precision for pt in curve if pt.recall >= target_recall]
    return max(feasible) if feasible else 0.0


def competition_map(
    gts: list[Event], preds: list[Event], match_iou: float = 0.3
) -> dict:
    """赛题 MAP = (P@R90 + P@R95) / 2。"""
    curve = pr_curve(gts, preds, match_iou)
    p90 = precision_at_recall(curve, 0.90)
    p95 = precision_at_recall(curve, 0.95)
    return {"p_at_r90": p90, "p_at_r95": p95, "map": (p90 + p95) / 2, "curve": curve}
