"""Audit a fixed PCA residual adapter with crossed unseen-subject/action holdouts."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from tools.audit_paired_clip_predictions import metrics_from_arrays
from tools.evaluate_caucafall_pca_residual_adapter import _adapter_scores, _sigmoid
from tools.train_tcn import _atomic_json, _sha256_file

METRICS = ("clip_map_percent", "clip_p_at_r90", "clip_p_at_r95")


def activity_identity(clip_id: str) -> str:
    """Return the class-qualified activity while removing subject suffixes."""
    class_name, name = str(clip_id).split("/", 1)
    activity = re.sub(r"S\d+_subject_\d+$", "", name)
    if not activity:
        raise ValueError(f"CAUCAFall activity identity 无效: {clip_id}")
    return f"{class_name}/{activity}"


def _metric_delta(
    baseline: dict[str, float], candidate: dict[str, float]
) -> dict[str, float]:
    return {name: candidate[name] - baseline[name] for name in METRICS}


def _multiway_bootstrap(
    labels: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    subject_index: np.ndarray,
    positive_activity_index: np.ndarray,
    negative_activity_index: np.ndarray,
    *,
    subject_count: int,
    positive_activity_count: int,
    negative_activity_count: int,
    replicates: int,
    seed: int,
) -> dict[str, list[float]]:
    cell_rows = defaultdict(list)
    for row, cell in enumerate(
        zip(
            subject_index.tolist(),
            positive_activity_index.tolist(),
            negative_activity_index.tolist(),
            strict=True,
        )
    ):
        cell_rows[cell].append(row)
    rng = np.random.default_rng(seed)
    deltas = np.empty((replicates, len(METRICS)), dtype=np.float64)
    for replicate in range(replicates):
        subjects = rng.integers(0, subject_count, subject_count)
        positives = rng.integers(
            0, positive_activity_count, positive_activity_count
        )
        negatives = rng.integers(
            0, negative_activity_count, negative_activity_count
        )
        rows = np.concatenate(
            [
                np.asarray(cell_rows[(subject, positive, negative)], dtype=np.int64)
                for subject in subjects
                for positive in positives
                for negative in negatives
            ]
        )
        base_metrics = metrics_from_arrays(labels[rows], baseline[rows])
        candidate_metrics = metrics_from_arrays(labels[rows], candidate[rows])
        deltas[replicate] = [
            candidate_metrics[name] - base_metrics[name] for name in METRICS
        ]
    return {
        name: [
            float(np.quantile(deltas[:, index], 0.025)),
            float(np.quantile(deltas[:, index], 0.975)),
        ]
        for index, name in enumerate(METRICS)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--nested-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260831)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("PCA residual crossed holdout 输出已存在，拒绝覆盖")
    if args.bootstrap_replicates < 100:
        raise ValueError("PCA residual crossed holdout bootstrap 次数过少")
    lock = json.loads(args.protocol_lock.read_text(encoding="utf-8"))
    if (
        lock.get("protocol") != "caucafall_pca_residual_crossed_holdout_v1"
        or lock.get("status") != "locked_before_shortcut_audit"
        or lock.get("fixed_candidate") != "pca:8:ridge:0.3"
    ):
        raise ValueError("PCA residual crossed holdout protocol lock 无效")
    for path_text, expected in lock.get("artifact_sha256", {}).items():
        path = Path(path_text)
        if not path.is_file() or _sha256_file(path) != expected:
            raise ValueError(f"PCA residual crossed holdout artifact hash 不匹配: {path}")
    nested = json.loads(args.nested_result.read_text(encoding="utf-8"))
    if (
        nested.get("decision") != "future_external_candidate"
        or nested.get("selection_counts") != {"pca:8:ridge:0.3": 10}
    ):
        raise ValueError("PCA residual crossed holdout fixed candidate 来源无效")
    with np.load(args.embeddings, allow_pickle=False) as payload:
        clip_ids = np.asarray(payload["clip_id"]).astype(str)
        labels = np.asarray(payload["label"], dtype=np.uint8)
        groups = np.asarray(payload["group_id"]).astype(str)
        logits = np.asarray(payload["member_logits"], dtype=np.float64)
        features = np.concatenate(
            [payload["short_embedding"], payload["dense_embedding"]], axis=1
        ).astype(np.float64)
    activities = np.asarray([activity_identity(clip_id) for clip_id in clip_ids])
    subjects = sorted(set(groups.tolist()))
    positive_activities = sorted(set(activities[labels == 1].tolist()))
    negative_activities = sorted(set(activities[labels == 0].tolist()))
    if (
        features.shape != (100, 768)
        or len(subjects) != 10
        or len(positive_activities) != 5
        or len(negative_activities) != 5
    ):
        raise ValueError("PCA residual crossed holdout identity 无效")
    anchor_logits = logits.mean(axis=1)
    rows = []
    for subject_index, subject in enumerate(subjects):
        for positive_index, positive_activity in enumerate(positive_activities):
            for negative_index, negative_activity in enumerate(negative_activities):
                held = (groups == subject) & np.isin(
                    activities, (positive_activity, negative_activity)
                )
                train = (groups != subject) & ~np.isin(
                    activities, (positive_activity, negative_activity)
                )
                if held.sum() != 2 or train.sum() != 72 or labels[held].sum() != 1:
                    raise ValueError("PCA residual crossed holdout split 无效")
                candidate_scores, _ = _adapter_scores(
                    features[train],
                    features[held],
                    anchor_logits[train],
                    anchor_logits[held],
                    labels[train],
                    "pca:8:ridge:0.3",
                )
                held_indices = np.flatnonzero(held)
                for local_index, clip_index in enumerate(held_indices):
                    rows.append(
                        {
                            "clip_index": int(clip_index),
                            "subject_index": subject_index,
                            "positive_activity_index": positive_index,
                            "negative_activity_index": negative_index,
                            "candidate_score": float(candidate_scores[local_index]),
                        }
                    )
    clip_index = np.asarray([row["clip_index"] for row in rows], dtype=np.int64)
    repeated_labels = labels[clip_index]
    baseline = _sigmoid(anchor_logits[clip_index])
    candidate = np.asarray([row["candidate_score"] for row in rows])
    subject_index = np.asarray([row["subject_index"] for row in rows])
    positive_index = np.asarray([row["positive_activity_index"] for row in rows])
    negative_index = np.asarray([row["negative_activity_index"] for row in rows])
    if len(rows) != 500 or not np.isfinite(candidate).all():
        raise ValueError("PCA residual crossed holdout coverage 无效")
    baseline_metrics = metrics_from_arrays(repeated_labels, baseline)
    candidate_metrics = metrics_from_arrays(repeated_labels, candidate)
    delta = _metric_delta(baseline_metrics, candidate_metrics)
    bootstrap = _multiway_bootstrap(
        repeated_labels,
        baseline,
        candidate,
        subject_index,
        positive_index,
        negative_index,
        subject_count=10,
        positive_activity_count=5,
        negative_activity_count=5,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    subject_deltas = []
    for index, subject in enumerate(subjects):
        mask = subject_index == index
        subject_deltas.append(
            {
                "subject": subject,
                "delta": _metric_delta(
                    metrics_from_arrays(repeated_labels[mask], baseline[mask]),
                    metrics_from_arrays(repeated_labels[mask], candidate[mask]),
                ),
            }
        )
    pair_deltas = []
    for positive_index_value, positive_activity in enumerate(positive_activities):
        for negative_index_value, negative_activity in enumerate(negative_activities):
            mask = (positive_index == positive_index_value) & (
                negative_index == negative_index_value
            )
            pair_deltas.append(
                {
                    "positive_activity": positive_activity,
                    "negative_activity": negative_activity,
                    "delta": _metric_delta(
                        metrics_from_arrays(repeated_labels[mask], baseline[mask]),
                        metrics_from_arrays(repeated_labels[mask], candidate[mask]),
                    ),
                }
            )
    subject_map = [row["delta"]["clip_map_percent"] for row in subject_deltas]
    pair_map = [row["delta"]["clip_map_percent"] for row in pair_deltas]
    gates = {
        "global_map_improves": delta["clip_map_percent"] > 0.0,
        "global_p90_nonnegative": delta["clip_p_at_r90"] >= 0.0,
        "global_p95_nonnegative": delta["clip_p_at_r95"] >= 0.0,
        "multiway_bootstrap_map_lower_positive": bootstrap["clip_map_percent"][0]
        > 0.0,
        "at_least_eight_subject_map_nonnegative": sum(value >= 0 for value in subject_map)
        >= 8,
        "at_least_twenty_activity_pair_map_nonnegative": sum(
            value >= 0 for value in pair_map
        )
        >= 20,
        "no_subject_map_regression_over_two_points": min(subject_map) >= -2.0,
    }
    report = {
        "protocol": "caucafall_pca_residual_crossed_holdout_v1",
        "qualification": "development_shortcut_audit_only",
        "fixed_candidate": "pca:8:ridge:0.3",
        "split": "exclude_held_subject_and_one_positive_plus_one_negative_activity",
        "training_clips_per_fit": 72,
        "fits": 250,
        "scored_rows": 500,
        "incumbent_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "delta": delta,
        "paired_subject_positive_activity_negative_activity_bootstrap_delta_95ci": bootstrap,
        "subject_deltas": subject_deltas,
        "activity_pair_deltas": pair_deltas,
        "shortcut_audit_gates": gates,
        "passes_all_shortcut_audit_gates": all(gates.values()),
        "decision": (
            "retain_future_external_candidate"
            if all(gates.values())
            else "reject_as_subject_or_activity_shortcut"
        ),
        "inputs": {
            "protocol_lock_sha256": _sha256_file(args.protocol_lock),
            "embeddings_sha256": _sha256_file(args.embeddings),
            "nested_result_sha256": _sha256_file(args.nested_result),
        },
        "test_accessed": False,
        "limitations": [
            "crossed cells reuse clips and are handled with a three-axis cluster bootstrap",
            "CAUCAFall remains development-only",
            "passing still requires a new untouched external dataset",
        ],
    }
    _atomic_json(args.output, report)
    print(json.dumps({"output": str(args.output), "decision": report["decision"]}))


if __name__ == "__main__":
    main()
