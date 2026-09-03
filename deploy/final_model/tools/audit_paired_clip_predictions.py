"""Audit two cache-ordered clip prediction files with paired group bootstrap."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from tools.train_long_context_event_oracle import template_group
from tools.train_tcn import _atomic_json, _sha256_file


def metrics_from_arrays(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.bool_)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or scores.shape != labels.shape or not labels.any():
        raise ValueError("labels/scores 必须为同形一维且包含正例")
    order = np.argsort(-scores, kind="stable")
    sorted_labels = labels[order]
    sorted_scores = scores[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores) != 0), labels.size - 1]
    tp = np.cumsum(sorted_labels)[ends]
    predicted = ends + 1
    recall = tp / labels.sum()
    precision = tp / predicted

    def at_recall(target: float) -> float:
        feasible = precision[recall >= target]
        return float(feasible.max()) if feasible.size else 0.0

    p90 = at_recall(0.90)
    p95 = at_recall(0.95)
    return {
        "clip_map_percent": 50.0 * (p90 + p95),
        "clip_p_at_r90": p90,
        "clip_p_at_r95": p95,
    }


def _operating_point(
    clip_ids: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    target: float,
) -> dict[str, Any]:
    order = np.argsort(-scores, kind="stable")
    sorted_labels = labels[order].astype(bool)
    sorted_scores = scores[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores) != 0), labels.size - 1]
    tp = np.cumsum(sorted_labels)[ends]
    predicted = ends + 1
    recall = tp / labels.sum()
    precision = tp / predicted
    feasible = np.flatnonzero(recall >= target)
    if feasible.size == 0:
        raise ValueError("预测无法达到目标 recall")
    best = feasible[int(np.argmax(precision[feasible]))]
    threshold = float(sorted_scores[ends[best]])
    positive = scores >= threshold
    false_positive_ids = clip_ids[positive & ~labels.astype(bool)]
    activities = Counter(str(value).split("/", 1)[0] for value in false_positive_ids)
    return {
        "target_recall": target,
        "threshold": threshold,
        "precision": float(precision[best]),
        "tp": int(tp[best]),
        "fp": int(predicted[best] - tp[best]),
        "fn": int(labels.sum() - tp[best]),
        "false_positive_by_activity": dict(sorted(activities.items())),
        "false_positive_clip_ids": false_positive_ids.tolist(),
    }


def paired_group_bootstrap(
    clip_ids: np.ndarray,
    labels: np.ndarray,
    baseline: np.ndarray,
    challenger: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, list[float]]:
    groups = np.asarray([template_group(str(clip_id)) for clip_id in clip_ids])
    rows = [np.flatnonzero(groups == group) for group in sorted(set(groups.tolist()))]
    rng = np.random.default_rng(seed)
    names = ("clip_map_percent", "clip_p_at_r90", "clip_p_at_r95")
    deltas = np.empty((replicates, len(names)), dtype=np.float64)
    for replicate in range(replicates):
        indices = np.concatenate(
            [rows[index] for index in rng.integers(0, len(rows), len(rows))]
        )
        base_metrics = metrics_from_arrays(labels[indices], baseline[indices])
        challenge_metrics = metrics_from_arrays(labels[indices], challenger[indices])
        deltas[replicate] = [
            challenge_metrics[name] - base_metrics[name] for name in names
        ]
    return {
        name: [
            float(np.quantile(deltas[:, index], 0.025)),
            float(np.quantile(deltas[:, index], 0.975)),
        ]
        for index, name in enumerate(names)
    }


def _load(
    path: Path, *, score_key: str = "score"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        clip_key = "clip_id" if "clip_id" in payload.files else "clip_ids"
        label_key = "label" if "label" in payload.files else "labels"
        if score_key not in payload.files:
            raise ValueError(f"预测文件缺少 score key: {score_key}")
        return (
            np.asarray(payload[clip_key]).astype(str),
            np.asarray(payload[label_key], dtype=np.float64),
            np.asarray(payload[score_key], dtype=np.float64),
        )


def _alignment_order(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Return candidate row indices aligned to unique reference clip IDs."""
    reference = np.asarray(reference).astype(str)
    candidate = np.asarray(candidate).astype(str)
    if reference.ndim != 1 or candidate.ndim != 1:
        raise ValueError("clip ID 必须为一维")
    if len(set(reference.tolist())) != reference.size:
        raise ValueError("baseline clip ID 不唯一")
    if len(set(candidate.tolist())) != candidate.size:
        raise ValueError("challenger clip ID 不唯一")
    lookup = {clip_id: index for index, clip_id in enumerate(candidate.tolist())}
    if reference.size != candidate.size or any(
        clip_id not in lookup for clip_id in reference.tolist()
    ):
        raise ValueError("baseline/challenger clip ID 集合不一致")
    return np.asarray([lookup[clip_id] for clip_id in reference.tolist()], dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--challenger", type=Path, required=True)
    parser.add_argument("--baseline-score-key", default="score")
    parser.add_argument("--challenger-score-key", default="score")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260828)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("审计输出已存在，拒绝覆盖")
    baseline_ids, baseline_labels, baseline_scores = _load(
        args.baseline, score_key=args.baseline_score_key
    )
    challenger_ids, challenger_labels, challenger_scores = _load(
        args.challenger, score_key=args.challenger_score_key
    )
    alignment = _alignment_order(baseline_ids, challenger_ids)
    challenger_ids = challenger_ids[alignment]
    challenger_labels = challenger_labels[alignment]
    challenger_scores = challenger_scores[alignment]
    if not np.array_equal(baseline_ids, challenger_ids):
        raise AssertionError("clip ID 对齐失败")
    if not np.array_equal(baseline_labels, challenger_labels):
        raise ValueError("baseline/challenger 标签不一致")
    baseline_metrics = metrics_from_arrays(baseline_labels, baseline_scores)
    challenger_metrics = metrics_from_arrays(challenger_labels, challenger_scores)
    metric_names = ("clip_map_percent", "clip_p_at_r90", "clip_p_at_r95")
    baseline_points = {
        name: _operating_point(
            baseline_ids, baseline_labels, baseline_scores, target
        )
        for name, target in (("r90", 0.90), ("r95", 0.95))
    }
    challenger_points = {
        name: _operating_point(
            challenger_ids, challenger_labels, challenger_scores, target
        )
        for name, target in (("r90", 0.90), ("r95", 0.95))
    }
    baseline_r95_fp = set(baseline_points["r95"]["false_positive_clip_ids"])
    challenger_r95_fp = set(challenger_points["r95"]["false_positive_clip_ids"])
    report = {
        "protocol": "paired_random_validation_prediction_audit_v1",
        "clips": int(baseline_ids.size),
        "baseline": baseline_metrics,
        "challenger": challenger_metrics,
        "delta": {
            name: challenger_metrics[name] - baseline_metrics[name]
            for name in metric_names
        },
        "score_pearson": float(np.corrcoef(baseline_scores, challenger_scores)[0, 1]),
        "error_pearson": float(
            np.corrcoef(
                baseline_scores - baseline_labels,
                challenger_scores - challenger_labels,
            )[0, 1]
        ),
        "operating_points": {
            "baseline": baseline_points,
            "challenger": challenger_points,
        },
        "r95_false_positive_set": {
            "rescued_by_challenger": sorted(baseline_r95_fp - challenger_r95_fp),
            "added_by_challenger": sorted(challenger_r95_fp - baseline_r95_fp),
            "intersection": len(baseline_r95_fp & challenger_r95_fp),
            "jaccard": len(baseline_r95_fp & challenger_r95_fp)
            / len(baseline_r95_fp | challenger_r95_fp),
        },
        "template_group_paired_bootstrap_delta_95ci": paired_group_bootstrap(
            baseline_ids,
            baseline_labels,
            baseline_scores,
            challenger_scores,
            replicates=args.bootstrap_replicates,
            seed=args.seed,
        ),
        "baseline_sha256": _sha256_file(args.baseline),
        "challenger_sha256": _sha256_file(args.challenger),
        "baseline_score_key": args.baseline_score_key,
        "challenger_score_key": args.challenger_score_key,
        "bootstrap_replicates": args.bootstrap_replicates,
        "validation_used_for_selection": False,
        "test_accessed": False,
    }
    _atomic_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
