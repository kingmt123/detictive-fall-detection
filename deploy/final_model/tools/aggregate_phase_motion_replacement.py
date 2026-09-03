"""Aggregate five fixed-member Phase-Motion replacement folds."""

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
from tools.train_tcn import _atomic_json, _sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", type=Path, action="append", required=True)
    parser.add_argument("--fold-report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Phase-Motion aggregate 输出已存在，拒绝覆盖")
    if len(args.prediction) != 5 or len(args.fold_report) != 5:
        raise ValueError("Phase-Motion aggregate 必须恰好提供五折")
    arrays = []
    reports = []
    for prediction, report_path in zip(args.prediction, args.fold_report, strict=True):
        with np.load(prediction, allow_pickle=False) as payload:
            arrays.append(
                tuple(
                    np.asarray(payload[name]).copy()
                    for name in ("clip_id", "label", "baseline", "challenger", "fold")
                )
            )
        reports.append(json.loads(report_path.read_text(encoding="utf-8")))
    clip_ids = np.concatenate([item[0] for item in arrays]).astype(str)
    labels = np.concatenate([item[1] for item in arrays]).astype(np.uint8)
    baseline = np.concatenate([item[2] for item in arrays]).astype(np.float64)
    challenger = np.concatenate([item[3] for item in arrays]).astype(np.float64)
    folds = np.concatenate([item[4] for item in arrays]).astype(np.int64)
    if len(set(clip_ids.tolist())) != len(clip_ids) or len(clip_ids) != 9600:
        raise ValueError("Phase-Motion aggregate clip 身份不完整或重复")
    if set(folds.tolist()) != set(range(5)):
        raise ValueError("Phase-Motion aggregate fold 不完整")
    removed = {report["replacement"]["removed_member"] for report in reports}
    if len(removed) != 1:
        raise ValueError("五折没有固定替换同一 member")
    baseline_metrics = metrics_from_arrays(labels, baseline)
    challenger_metrics = metrics_from_arrays(labels, challenger)
    names = ("clip_map_percent", "clip_p_at_r90", "clip_p_at_r95")
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
    base_point = _operating_point(clip_ids, labels, baseline, 0.95)
    challenge_point = _operating_point(clip_ids, labels, challenger, 0.95)
    transitions = error_transitions(
        clip_ids,
        labels,
        folds,
        baseline >= float(base_point["threshold"]),
        challenger >= float(challenge_point["threshold"]),
    )
    report = {
        "protocol": "phase_motion_native48_fixed_member_fivefold_replacement_v1",
        "clips": len(clip_ids),
        "removed_member": next(iter(removed)),
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
        "paired_template_group_bootstrap_delta_95ci": paired_group_bootstrap(
            clip_ids,
            labels,
            baseline,
            challenger,
            replicates=args.bootstrap_replicates,
            seed=20260829,
        ),
        "r95_operating_points": {"baseline": base_point, "challenger": challenge_point},
        "r95_transitions": transitions,
        "input_sha256": {
            "predictions": [_sha256_file(path) for path in args.prediction],
            "reports": [_sha256_file(path) for path in args.fold_report],
        },
        "test_accessed": False,
    }
    _atomic_json(args.output, report)
    print(
        json.dumps(
            {
                "delta": report["delta"],
                "fold_nonnegative_counts": report["fold_nonnegative_counts"],
                "removed_member": report["removed_member"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
