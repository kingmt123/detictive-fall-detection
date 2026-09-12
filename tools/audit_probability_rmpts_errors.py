"""Audit R95 rescues and introduced errors for fixed probability-domain RMPTS."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from models.edgefall_ensemble import (
    rmpts_fuse_numpy_logits,
    rmpts_fuse_numpy_probabilities,
)
from tools.audit_paired_clip_predictions import _operating_point, metrics_from_arrays
from tools.evaluate_rmpts_diagnostic import _logit, load_member_matrix
from tools.train_tcn import _atomic_json, _sha256_file


def error_transitions(
    clip_ids: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    baseline_positive: np.ndarray,
    challenger_positive: np.ndarray,
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=bool)
    masks = {
        "positive_rescued": labels & ~baseline_positive & challenger_positive,
        "positive_introduced": labels & baseline_positive & ~challenger_positive,
        "negative_fp_removed": ~labels & baseline_positive & ~challenger_positive,
        "negative_fp_introduced": ~labels & ~baseline_positive & challenger_positive,
    }
    result = {}
    for name, mask in masks.items():
        selected_ids = clip_ids[mask]
        result[name] = {
            "count": int(mask.sum()),
            "clip_ids": selected_ids.tolist(),
            "by_activity": dict(
                sorted(Counter(item.split("/", 1)[0] for item in selected_ids).items())
            ),
            "by_fold": {
                str(fold): int(np.count_nonzero(mask & (folds == fold)))
                for fold in sorted(set(folds.tolist()))
            },
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--member-report", type=Path, required=True)
    parser.add_argument("--dense-oof", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Probability-RMPTS error audit 输出已存在")
    with np.load(args.dense_oof, allow_pickle=False) as payload:
        required = {"clip_id", "label", "fold", "dense48_score"}
        if not required.issubset(payload.files):
            raise ValueError("dense OOF 缺少 Probability-RMPTS audit 字段")
        clip_ids = np.asarray(payload["clip_id"]).astype(str)
        labels = np.asarray(payload["label"], dtype=np.uint8)
        folds = np.asarray(payload["fold"], dtype=np.int64)
        dense_probability = np.asarray(payload["dense48_score"], dtype=np.float64)
    short_logits, _ = load_member_matrix(args.member_report, clip_ids, labels)
    short_probability = 1.0 / (
        1.0 + np.exp(-np.clip(short_logits, -30.0, 30.0))
    )
    dense_logits = _logit(dense_probability)
    incumbent_logit = rmpts_fuse_numpy_logits(
        short_logits[:, (0, 2, 4)],
        short_logits[:, (1, 3, 5)],
        dense_logits,
        delta=Fraction(0, 1),
    )
    incumbent = 1.0 / (1.0 + np.exp(-np.clip(incumbent_logit, -30.0, 30.0)))
    challenger = rmpts_fuse_numpy_probabilities(
        short_probability[:, (0, 2, 4)],
        short_probability[:, (1, 3, 5)],
        dense_probability,
        delta=Fraction(1, 7),
    )
    baseline_point = _operating_point(clip_ids, labels, incumbent, 0.95)
    challenger_point = _operating_point(clip_ids, labels, challenger, 0.95)
    transitions = error_transitions(
        clip_ids,
        labels,
        folds,
        incumbent >= float(baseline_point["threshold"]),
        challenger >= float(challenger_point["threshold"]),
    )
    report = {
        "protocol": "edgefall_probability_rmpts_r95_error_audit_v1",
        "qualification": "same_oof_diagnostic_only",
        "incumbent": {
            "fusion": "logit:0/1",
            "metrics": metrics_from_arrays(labels, incumbent),
            "r95_operating_point": baseline_point,
        },
        "challenger": {
            "fusion": "probability:1/7",
            "metrics": metrics_from_arrays(labels, challenger),
            "r95_operating_point": challenger_point,
        },
        "transitions": transitions,
        "net_r95_error_changes": {
            "positive": (
                transitions["positive_rescued"]["count"]
                - transitions["positive_introduced"]["count"]
            ),
            "negative": (
                transitions["negative_fp_removed"]["count"]
                - transitions["negative_fp_introduced"]["count"]
            ),
        },
        "member_report_sha256": _sha256_file(args.member_report),
        "dense_oof_sha256": _sha256_file(args.dense_oof),
        "test_accessed": False,
        "limitations": [
            "both fusion domain and delta were proposed after inspecting this historical OOF",
            "each method uses its own deterministic R95 operating point",
            "formal replacement still requires the locked retrospective nested protocol",
        ],
    }
    _atomic_json(args.output, report)
    print(json.dumps({"output": str(args.output.resolve()), "net": report["net_r95_error_changes"]}))


if __name__ == "__main__":
    main()
