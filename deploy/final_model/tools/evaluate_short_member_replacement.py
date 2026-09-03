"""Evaluate a short-window member replacement under the frozen final fusion."""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path

import numpy as np

from models.edgefall_ensemble import rmpts_fuse_numpy_probabilities
from tools.audit_paired_clip_predictions import (
    metrics_from_arrays,
    paired_group_bootstrap,
)
from tools.evaluate_dense_resolution_replacement import load_challenger
from tools.evaluate_phase_motion_replacement import equal_logit_fusion
from tools.evaluate_probability_rmpts_calibration_crossfit import calibration_crossfit
from tools.evaluate_rmpts_diagnostic import _logit, load_member_matrix
from tools.train_tcn import _atomic_json, _sha256_file

MEMBER_NAMES = (
    "F1-320-seed1",
    "F1-640-seed1",
    "F1-320-seed2",
    "F1-640-seed2",
    "F1-320-seed3",
    "F1-640-seed3",
)


def probability_rmpts(short_probability: np.ndarray, dense: np.ndarray) -> np.ndarray:
    return rmpts_fuse_numpy_probabilities(
        short_probability[:, (0, 2, 4)],
        short_probability[:, (1, 3, 5)],
        dense,
        delta=Fraction(1, 7),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--member-report", type=Path, required=True)
    parser.add_argument("--dense-oof", type=Path, required=True)
    parser.add_argument("--challenger-oof", type=Path, action="append", required=True)
    parser.add_argument("--replace-member", choices=MEMBER_NAMES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("short member replacement 输出已存在，拒绝覆盖")
    with np.load(args.dense_oof, allow_pickle=False) as payload:
        required = {"clip_id", "label", "fold", "dense48_score"}
        if not required.issubset(payload.files):
            raise ValueError("dense OOF 缺少 short replacement 字段")
        all_ids = np.asarray(payload["clip_id"]).astype(str)
        all_labels = np.asarray(payload["label"], dtype=np.uint8)
        all_folds = np.asarray(payload["fold"], dtype=np.int64)
        all_dense = np.asarray(payload["dense48_score"], dtype=np.float64)
    reference_folds = dict(zip(all_ids.tolist(), all_folds.tolist(), strict=True))
    clip_ids, labels, challenger_scores, folds = load_challenger(
        args.challenger_oof, reference_folds
    )
    index = {clip_id: offset for offset, clip_id in enumerate(all_ids)}
    order = np.asarray([index[clip_id] for clip_id in clip_ids], dtype=np.int64)
    if not np.array_equal(labels, all_labels[order]):
        raise ValueError("short challenger label 身份不一致")
    short_logits, _ = load_member_matrix(args.member_report, all_ids, all_labels)
    selected_logits = short_logits[order]
    short_probability = 1.0 / (1.0 + np.exp(-np.clip(selected_logits, -30.0, 30.0)))
    replacement_index = MEMBER_NAMES.index(args.replace_member)
    challenger_probability = short_probability.copy()
    challenger_probability[:, replacement_index] = challenger_scores
    dense_probability = all_dense[order]
    baseline_equal_logit = equal_logit_fusion(
        np.column_stack((short_probability, dense_probability))
    )
    challenger_equal_logit = equal_logit_fusion(
        np.column_stack((challenger_probability, dense_probability))
    )
    baseline_rmpts = probability_rmpts(short_probability, dense_probability)
    challenger_rmpts = probability_rmpts(challenger_probability, dense_probability)
    names = ("clip_map_percent", "clip_p_at_r90", "clip_p_at_r95")

    def comparison(baseline: np.ndarray, challenger: np.ndarray, seed: int) -> dict:
        baseline_metrics = metrics_from_arrays(labels, baseline)
        challenger_metrics = metrics_from_arrays(labels, challenger)
        return {
            "baseline": baseline_metrics,
            "challenger": challenger_metrics,
            "delta": {
                name: challenger_metrics[name] - baseline_metrics[name] for name in names
            },
            "paired_template_group_bootstrap_delta_95ci": paired_group_bootstrap(
                clip_ids,
                labels,
                baseline,
                challenger,
                replicates=args.bootstrap_replicates,
                seed=seed,
            ),
        }

    report = {
        "protocol": "edgefall_short_member_fixed_replacement_v1",
        "replace_member": args.replace_member,
        "folds_present": sorted(set(folds.tolist())),
        "clips": int(labels.size),
        "equal_logit": comparison(baseline_equal_logit, challenger_equal_logit, 20260829),
        "probability_rmpts_1_7": comparison(baseline_rmpts, challenger_rmpts, 20260830),
        "member": {
            "baseline": metrics_from_arrays(
                labels, short_probability[:, replacement_index]
            ),
            "challenger": metrics_from_arrays(labels, challenger_scores),
            "score_pearson": float(
                np.corrcoef(short_probability[:, replacement_index], challenger_scores)[0, 1]
            ),
        },
        "input_sha256": {
            "member_report": _sha256_file(args.member_report),
            "dense_oof": _sha256_file(args.dense_oof),
            "challenger_oof": [_sha256_file(path) for path in args.challenger_oof],
        },
        "test_accessed": False,
    }
    if sorted(set(folds.tolist())) == list(range(5)):
        baseline_calibrated, baseline_details = calibration_crossfit(
            labels,
            folds,
            selected_logits,
            _logit(dense_probability),
            "global",
        )
        challenger_logits = selected_logits.copy()
        challenger_logits[:, replacement_index] = _logit(challenger_scores)
        challenger_calibrated, challenger_details = calibration_crossfit(
            labels,
            folds,
            challenger_logits,
            _logit(dense_probability),
            "global",
        )
        report["frozen_probability_rmpts_1_7_shared_temperature"] = {
            **comparison(baseline_calibrated, challenger_calibrated, 20260831),
            "baseline_folds": baseline_details,
            "challenger_folds": challenger_details,
        }
    _atomic_json(args.output, report)
    print(json.dumps(report["probability_rmpts_1_7"]["delta"], sort_keys=True))


if __name__ == "__main__":
    main()
