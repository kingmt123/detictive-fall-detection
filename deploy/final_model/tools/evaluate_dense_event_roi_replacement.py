"""Evaluate fivefold event-triplet ROI heads as a fixed Dense-head replacement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools.audit_paired_clip_predictions import (
    _operating_point,
    metrics_from_arrays,
    paired_group_bootstrap,
)
from tools.audit_probability_rmpts_errors import error_transitions
from tools.evaluate_phase_motion_replacement import equal_logit_fusion
from tools.evaluate_rmpts_diagnostic import load_member_matrix
from tools.train_tcn import _atomic_json, _sha256_file


def _load_challenger(paths: list[Path]) -> tuple[dict[str, float], dict[str, int]]:
    scores: dict[str, float] = {}
    folds: dict[str, int] = {}
    for path in paths:
        with np.load(path, allow_pickle=False) as payload:
            ids = np.asarray(payload["clip_id"]).astype(str)
            values = np.asarray(payload["score"], dtype=np.float64)
            fold_values = np.asarray(payload["fold"], dtype=np.int64)
        for clip_id, score, fold in zip(ids, values, fold_values, strict=True):
            if clip_id in scores:
                raise ValueError(f"challenger OOF clip 重复: {clip_id}")
            scores[clip_id] = float(score)
            folds[clip_id] = int(fold)
    return scores, folds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--member-report", type=Path, required=True)
    parser.add_argument("--dense-oof", type=Path, required=True)
    parser.add_argument("--challenger-oof", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Dense event ROI replacement 输出已存在，拒绝覆盖")
    if len(args.challenger_oof) != 5:
        raise ValueError("Dense event ROI replacement 必须提供五折 challenger")
    with np.load(args.dense_oof, allow_pickle=False) as payload:
        clip_ids = np.asarray(payload["clip_id"]).astype(str)
        labels = np.asarray(payload["label"], dtype=np.uint8)
        folds = np.asarray(payload["fold"], dtype=np.int64)
        dense_scores = np.asarray(payload["dense48_score"], dtype=np.float64)
        expected_baseline = np.asarray(payload["score"], dtype=np.float64)
    short_logits, _member_names = load_member_matrix(args.member_report, clip_ids, labels)
    short_scores = 1.0 / (1.0 + np.exp(-np.clip(short_logits, -30.0, 30.0)))
    challenger_lookup, challenger_folds = _load_challenger(args.challenger_oof)
    if set(challenger_lookup) != set(clip_ids.tolist()):
        raise ValueError("challenger 五折 OOF clip 集合不完整")
    if any(challenger_folds[clip_id] != int(fold) for clip_id, fold in zip(clip_ids, folds, strict=True)):
        raise ValueError("challenger fold 身份不一致")
    event_scores = np.asarray([challenger_lookup[clip_id] for clip_id in clip_ids])
    baseline = equal_logit_fusion(np.column_stack((short_scores, dense_scores)))
    reconstruction_error = float(np.max(np.abs(baseline - expected_baseline)))
    if reconstruction_error > 3e-6:
        raise ValueError(f"七头 baseline 重建不一致: {reconstruction_error}")
    challenger = equal_logit_fusion(np.column_stack((short_scores, event_scores)))
    names = ("clip_map_percent", "clip_p_at_r90", "clip_p_at_r95")
    baseline_metrics = metrics_from_arrays(labels, baseline)
    challenger_metrics = metrics_from_arrays(labels, challenger)
    fold_results = {}
    for fold in range(5):
        selected = folds == fold
        base = metrics_from_arrays(labels[selected], baseline[selected])
        challenge = metrics_from_arrays(labels[selected], challenger[selected])
        fold_results[str(fold)] = {
            "baseline": base,
            "challenger": challenge,
            "delta": {name: challenge[name] - base[name] for name in names},
        }
    baseline_point = _operating_point(clip_ids, labels, baseline, 0.95)
    challenger_point = _operating_point(clip_ids, labels, challenger, 0.95)
    transitions = error_transitions(
        clip_ids,
        labels,
        folds,
        baseline >= float(baseline_point["threshold"]),
        challenger >= float(challenger_point["threshold"]),
    )
    report = {
        "protocol": "edgefall_dense_event_triplet_fixed_replacement_fivefold_v1",
        "replacement": "original_dense48_head_to_event_triplet_dense48_head",
        "clips": len(clip_ids),
        "baseline": baseline_metrics,
        "challenger": challenger_metrics,
        "delta": {
            name: challenger_metrics[name] - baseline_metrics[name] for name in names
        },
        "folds": fold_results,
        "fold_nonnegative_counts": {
            name: sum(fold_results[str(fold)]["delta"][name] >= 0 for fold in range(5))
            for name in names
        },
        "dense_member_comparison": {
            "original_metrics": metrics_from_arrays(labels, dense_scores),
            "event_triplet_metrics": metrics_from_arrays(labels, event_scores),
            "score_pearson": float(np.corrcoef(dense_scores, event_scores)[0, 1]),
            "error_pearson": float(
                np.corrcoef(dense_scores - labels, event_scores - labels)[0, 1]
            ),
        },
        "paired_template_group_bootstrap_delta_95ci": paired_group_bootstrap(
            clip_ids,
            labels,
            baseline,
            challenger,
            replicates=args.bootstrap_replicates,
            seed=20260829,
        ),
        "r95_operating_points": {
            "baseline": baseline_point,
            "challenger": challenger_point,
        },
        "r95_transitions": transitions,
        "baseline_reconstruction_max_abs_error": reconstruction_error,
        "input_sha256": {
            "member_report": _sha256_file(args.member_report),
            "dense_oof": _sha256_file(args.dense_oof),
            "challenger_oof": [_sha256_file(path) for path in args.challenger_oof],
        },
        "test_accessed": False,
    }
    _atomic_json(args.output, report)
    print(
        json.dumps(
            {
                "delta": report["delta"],
                "fold_nonnegative_counts": report["fold_nonnegative_counts"],
                "dense_member_comparison": report["dense_member_comparison"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
