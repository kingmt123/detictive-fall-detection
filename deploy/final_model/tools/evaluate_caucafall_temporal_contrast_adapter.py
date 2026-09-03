"""Nested LOSO audit of a dense-minus-short temporal-contrast residual adapter."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from tools.audit_paired_clip_predictions import metrics_from_arrays
from tools.evaluate_caucafall_pca_residual_adapter import (
    CANDIDATES,
    METRICS,
    _adapter_scores,
    _group_bootstrap_deltas,
    _metric_delta,
    _sigmoid,
    inner_loso_scores,
    select_candidate,
)
from tools.train_tcn import _atomic_json, _sha256_file


def temporal_contrast_features(short: np.ndarray, dense: np.ndarray) -> np.ndarray:
    """Create a scale-contrast representation while rejecting incompatible spaces."""
    short = np.asarray(short, dtype=np.float64)
    dense = np.asarray(dense, dtype=np.float64)
    if short.ndim != 2 or short.shape != dense.shape or not np.isfinite(short).all():
        raise ValueError("temporal contrast embedding shape 无效")
    if not np.isfinite(dense).all():
        raise ValueError("temporal contrast embedding value 无效")
    return dense - short


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260831)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("temporal contrast adapter 输出已存在，拒绝覆盖")
    lock = json.loads(args.protocol_lock.read_text(encoding="utf-8"))
    if (
        lock.get("protocol") != "caucafall_temporal_contrast_adapter_nested_v1"
        or lock.get("status") != "locked_before_development_evaluation"
        or lock.get("candidates") != list(CANDIDATES)
    ):
        raise ValueError("temporal contrast adapter protocol lock 无效")
    for path_text, expected in lock.get("artifact_sha256", {}).items():
        path = Path(path_text)
        if not path.is_file() or _sha256_file(path) != expected:
            raise ValueError(f"temporal contrast adapter artifact hash 不匹配: {path}")
    with np.load(args.embeddings, allow_pickle=False) as payload:
        clip_ids = np.asarray(payload["clip_id"]).astype(str)
        labels = np.asarray(payload["label"], dtype=np.uint8)
        groups = np.asarray(payload["group_id"]).astype(str)
        logits = np.asarray(payload["member_logits"], dtype=np.float64)
        features = temporal_contrast_features(
            payload["short_embedding"], payload["dense_embedding"]
        )
    if (
        clip_ids.shape != (100,)
        or logits.shape != (100, 7)
        or features.shape != (100, 384)
        or labels.sum() != 50
        or len(set(groups.tolist())) != 10
    ):
        raise ValueError("temporal contrast adapter CAUCAFall identity 无效")
    anchor_logits = logits.mean(axis=1)
    incumbent = _sigmoid(anchor_logits)
    crossfit = np.full(labels.size, np.nan)
    selections = []
    outer_deltas = []
    for outer_index, held_group in enumerate(sorted(set(groups.tolist()))):
        held = groups == held_group
        train = ~held
        inner_scores = inner_loso_scores(
            features[train], anchor_logits[train], labels[train], groups[train]
        )
        selected, candidates = select_candidate(
            labels[train],
            groups[train],
            inner_scores,
            bootstrap_replicates=args.bootstrap_replicates,
            seed=args.bootstrap_seed + outer_index * 100,
        )
        crossfit[held], _ = _adapter_scores(
            features[train],
            features[held],
            anchor_logits[train],
            anchor_logits[held],
            labels[train],
            selected,
        )
        outer_delta = _metric_delta(
            metrics_from_arrays(labels[held], incumbent[held]),
            metrics_from_arrays(labels[held], crossfit[held]),
        )
        outer_deltas.append(outer_delta)
        selections.append(
            {
                "held_subject": held_group,
                "selected_candidate": selected,
                "inner_candidates": candidates,
                "held_delta": outer_delta,
            }
        )
    if not np.isfinite(crossfit).all():
        raise ValueError("temporal contrast adapter outer LOSO 覆盖不完整")
    counts = Counter(row["selected_candidate"] for row in selections)
    consensus_name, consensus_count = counts.most_common(1)[0]
    consensus = consensus_name != "anchor" and consensus_count >= 8
    baseline_metrics = metrics_from_arrays(labels, incumbent)
    candidate_metrics = metrics_from_arrays(labels, crossfit)
    delta = _metric_delta(baseline_metrics, candidate_metrics)
    bootstrap_deltas = _group_bootstrap_deltas(
        labels,
        incumbent,
        crossfit,
        groups,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    bootstrap = {
        name: [
            float(np.quantile(bootstrap_deltas[:, index], 0.025)),
            float(np.quantile(bootstrap_deltas[:, index], 0.975)),
        ]
        for index, name in enumerate(METRICS)
    }
    subject_map = [row["clip_map_percent"] for row in outer_deltas]
    gates = {
        "exact_nonanchor_eight_of_ten_consensus": consensus,
        "global_map_improves": delta["clip_map_percent"] > 0.0,
        "global_p90_nonnegative": delta["clip_p_at_r90"] >= 0.0,
        "global_p95_nonnegative": delta["clip_p_at_r95"] >= 0.0,
        "map_bootstrap_lower_positive": bootstrap["clip_map_percent"][0] > 0.0,
        "at_least_eight_subject_map_nonnegative": sum(value >= 0 for value in subject_map)
        >= 8,
        "no_subject_map_regression_over_two_points": min(subject_map) >= -2.0,
    }
    report = {
        "protocol": "caucafall_temporal_contrast_adapter_nested_v1",
        "qualification": "posthoc_development_nested_loso_only",
        "architecture": "equal_logit_anchor_plus_PCA_of_dense_minus_short_embedding",
        "candidates": list(CANDIDATES),
        "selection_counts": dict(sorted(counts.items())),
        "selections": selections,
        "incumbent_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "delta": delta,
        "paired_subject_bootstrap_delta_95ci": bootstrap,
        "development_gates": gates,
        "passes_all_development_gates": all(gates.values()),
        "decision": (
            "requires_crossed_shortcut_audit"
            if all(gates.values())
            else "stop_temporal_contrast_adapter"
        ),
        "inputs": {
            "protocol_lock_sha256": _sha256_file(args.protocol_lock),
            "embeddings_sha256": _sha256_file(args.embeddings),
        },
        "test_accessed": False,
        "limitations": [
            "route was proposed after inspecting the joint PCA shortcut audit",
            "CAUCAFall is development-only",
            "GMDCSA24 and all reserved tests are excluded",
        ],
    }
    _atomic_json(args.output, report)
    print(json.dumps({"output": str(args.output), "decision": report["decision"]}))


if __name__ == "__main__":
    main()
