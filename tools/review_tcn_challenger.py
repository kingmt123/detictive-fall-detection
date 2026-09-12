"""对冻结 validation 预测执行 TCN challenger 五折与 bootstrap review。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from eval.tcn_ablation import atomic_json, metric_summary, sha256_file

PROTOCOL = "fall_tcn_challenger_review_v1"


def _load_predictions(path: Path) -> tuple[dict[str, bool], dict[str, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("split") != "val":
        raise ValueError(f"review 只接受 validation 报告: {path}")
    rows = payload.get("predictions")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"报告缺少 predictions: {path}")
    labels: dict[str, bool] = {}
    scores: dict[str, float] = {}
    for row in rows:
        clip_id = row["clip_id"]
        if clip_id in labels:
            raise ValueError(f"重复 clip_id: {clip_id}")
        labels[clip_id] = bool(row["has_fall"])
        scores[clip_id] = float(row["tcn_score"])
    return labels, scores


def _delta(
    labels: dict[str, bool],
    baseline_scores: dict[str, float],
    challenger_scores: dict[str, float],
) -> dict[str, float]:
    baseline = metric_summary(labels, baseline_scores)
    challenger = metric_summary(labels, challenger_scores)
    return {
        "map": challenger["map"] - baseline["map"],
        "p_at_r95": challenger["p_at_r95"] - baseline["p_at_r95"],
    }


def review_predictions(
    labels: dict[str, bool],
    baseline_scores: dict[str, float],
    challenger_scores: dict[str, float],
    *,
    folds: int,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    if folds < 2 or bootstrap_samples < 1:
        raise ValueError("folds 必须至少为 2，bootstrap_samples 必须为正")
    if labels.keys() != baseline_scores.keys() or labels.keys() != challenger_scores.keys():
        raise ValueError("baseline/challenger clip 集合必须完全一致")
    if not any(labels.values()) or all(labels.values()):
        raise ValueError("review 必须同时包含正负 clip")

    fold_rows = []
    for fold in range(folds):
        clip_ids = [
            clip_id
            for clip_id in labels
            if int(hashlib.sha256(clip_id.encode()).hexdigest(), 16) % folds == fold
        ]
        subset_labels = {clip_id: labels[clip_id] for clip_id in clip_ids}
        if not subset_labels or not any(subset_labels.values()) or all(subset_labels.values()):
            raise ValueError(f"fold {fold} 缺少正类或负类")
        delta = _delta(
            subset_labels,
            {clip_id: baseline_scores[clip_id] for clip_id in clip_ids},
            {clip_id: challenger_scores[clip_id] for clip_id in clip_ids},
        )
        fold_rows.append({"fold": fold, "clip_count": len(clip_ids), "delta": delta})

    positive_ids = np.asarray([clip_id for clip_id, value in labels.items() if value])
    negative_ids = np.asarray([clip_id for clip_id, value in labels.items() if not value])
    rng = np.random.default_rng(seed)
    bootstrap_delta = np.empty((bootstrap_samples, 2), dtype=np.float64)
    for index in range(bootstrap_samples):
        sampled = np.concatenate(
            (
                rng.choice(positive_ids, len(positive_ids), replace=True),
                rng.choice(negative_ids, len(negative_ids), replace=True),
            )
        )
        sampled_labels = {f"sample-{i}": labels[clip_id] for i, clip_id in enumerate(sampled)}
        sampled_baseline = {
            f"sample-{i}": baseline_scores[clip_id]
            for i, clip_id in enumerate(sampled)
        }
        sampled_challenger = {
            f"sample-{i}": challenger_scores[clip_id]
            for i, clip_id in enumerate(sampled)
        }
        delta = _delta(sampled_labels, sampled_baseline, sampled_challenger)
        bootstrap_delta[index] = delta["map"], delta["p_at_r95"]

    names = ("map", "p_at_r95")
    bootstrap = {
        name: {
            "mean": float(np.mean(bootstrap_delta[:, position])),
            "median": float(np.median(bootstrap_delta[:, position])),
            "ci95": np.percentile(
                bootstrap_delta[:, position], [2.5, 97.5]
            ).tolist(),
            "positive_fraction": float(
                np.mean(bootstrap_delta[:, position] > 0.0)
            ),
        }
        for position, name in enumerate(names)
    }
    full_delta = _delta(labels, baseline_scores, challenger_scores)
    fold_joint_wins = sum(
        row["delta"]["map"] > 0.0 and row["delta"]["p_at_r95"] > 0.0
        for row in fold_rows
    )
    return {
        "full_delta": full_delta,
        "folds": fold_rows,
        "fold_joint_wins": fold_joint_wins,
        "bootstrap": bootstrap,
        "review_pass": bool(
            full_delta["map"] >= 0.03
            and full_delta["p_at_r95"] >= 0.03
            and fold_joint_wins >= folds - 1
            and bootstrap["map"]["ci95"][0] >= 0.0
            and bootstrap["p_at_r95"]["ci95"][0] >= 0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--challenger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260818)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("输出已存在，拒绝覆盖")
    labels, baseline_scores = _load_predictions(args.baseline)
    challenger_labels, challenger_scores = _load_predictions(args.challenger)
    if challenger_labels != labels:
        raise ValueError("baseline/challenger 标签不一致")
    result = review_predictions(
        labels,
        baseline_scores,
        challenger_scores,
        folds=args.folds,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    payload = {
        "protocol": PROTOCOL,
        "split": "val",
        "clip_count": len(labels),
        "positive_clips": sum(labels.values()),
        "fold_count": args.folds,
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        **result,
        "artifact_sha256": {
            "baseline": sha256_file(args.baseline),
            "challenger": sha256_file(args.challenger),
            "review_code": sha256_file(Path(__file__)),
        },
        "test_accessed": False,
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
