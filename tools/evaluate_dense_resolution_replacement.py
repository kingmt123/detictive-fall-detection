"""Evaluate one dense temporal/ROI head as a fixed Dense48-320 replacement."""

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
from tools.evaluate_phase_motion_replacement import equal_logit_fusion
from tools.evaluate_probability_rmpts_calibration_crossfit import calibration_crossfit
from tools.evaluate_rmpts_diagnostic import _logit, load_member_matrix
from tools.train_tcn import _atomic_json, _sha256_file


def load_challenger(
    paths: list[Path], reference_folds: dict[str, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows: dict[str, tuple[int, float, int]] = {}
    for path in paths:
        with np.load(path, allow_pickle=False) as payload:
            ids = np.asarray(payload["clip_id"]).astype(str)
            labels = np.asarray(payload["label"], dtype=np.uint8)
            scores = np.asarray(payload["score"], dtype=np.float64)
            folds = np.asarray(payload["fold"], dtype=np.int64)
        for clip_id, label, score, fold in zip(ids, labels, scores, folds, strict=True):
            if clip_id in rows:
                raise ValueError(f"Dense48-640 clip 重复: {clip_id}")
            if clip_id not in reference_folds or reference_folds[clip_id] != int(fold):
                raise ValueError(f"Dense48-640 fold 身份不一致: {clip_id}")
            rows[clip_id] = (int(label), float(score), int(fold))
    ordered_ids = np.asarray(sorted(rows))
    return (
        ordered_ids,
        np.asarray([rows[item][0] for item in ordered_ids], dtype=np.uint8),
        np.asarray([rows[item][1] for item in ordered_ids], dtype=np.float64),
        np.asarray([rows[item][2] for item in ordered_ids], dtype=np.int64),
    )


def shared_temperature_crossfit_comparison(
    labels: np.ndarray,
    folds: np.ndarray,
    short_logits: np.ndarray,
    baseline_dense_probability: np.ndarray,
    challenger_dense_probability: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict], list[dict]]:
    """Apply the frozen global-temperature protocol to two complete OOF systems."""
    if sorted(set(np.asarray(folds, dtype=np.int64).tolist())) != list(range(5)):
        raise ValueError("共享温度替换比较需要完整五折 challenger")
    baseline_scores, baseline_details = calibration_crossfit(
        labels,
        folds,
        short_logits,
        _logit(baseline_dense_probability),
        "global",
    )
    challenger_scores, challenger_details = calibration_crossfit(
        labels,
        folds,
        short_logits,
        _logit(challenger_dense_probability),
        "global",
    )
    return baseline_scores, challenger_scores, baseline_details, challenger_details


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--member-report", type=Path, required=True)
    parser.add_argument("--dense-oof", type=Path, required=True)
    parser.add_argument("--challenger-oof", type=Path, action="append", required=True)
    parser.add_argument(
        "--replacement",
        choices=("dense48_640", "dense64_320"),
        default="dense48_640",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Dense48 resolution replacement 输出已存在，拒绝覆盖")
    if not 1 <= len(args.challenger_oof) <= 5:
        raise ValueError("Dense48 resolution replacement 需要一至五折 challenger")
    with np.load(args.dense_oof, allow_pickle=False) as payload:
        required = {"clip_id", "label", "fold", "score", "dense48_score"}
        if not required.issubset(payload.files):
            raise ValueError("dense OOF 缺少 resolution replacement 字段")
        all_ids = np.asarray(payload["clip_id"]).astype(str)
        all_labels = np.asarray(payload["label"], dtype=np.uint8)
        all_folds = np.asarray(payload["fold"], dtype=np.int64)
        all_dense = np.asarray(payload["dense48_score"], dtype=np.float64)
        expected_baseline = np.asarray(payload["score"], dtype=np.float64)
    reference_folds = {
        clip_id: int(fold) for clip_id, fold in zip(all_ids, all_folds, strict=True)
    }
    clip_ids, labels, challenger_dense, folds = load_challenger(
        args.challenger_oof, reference_folds
    )
    index = {clip_id: offset for offset, clip_id in enumerate(all_ids)}
    order = np.asarray([index[item] for item in clip_ids], dtype=np.int64)
    if not np.array_equal(labels, all_labels[order]):
        raise ValueError("Dense48-640 label 身份不一致")
    short_logits, _ = load_member_matrix(args.member_report, all_ids, all_labels)
    short_scores = 1.0 / (1.0 + np.exp(-np.clip(short_logits[order], -30.0, 30.0)))
    baseline = equal_logit_fusion(np.column_stack((short_scores, all_dense[order])))
    reconstruction_error = float(np.max(np.abs(baseline - expected_baseline[order])))
    if reconstruction_error > 3e-6:
        raise ValueError(f"七头 baseline 重建不一致: {reconstruction_error}")
    challenger = equal_logit_fusion(np.column_stack((short_scores, challenger_dense)))
    baseline_probability_rmpts = rmpts_fuse_numpy_probabilities(
        short_scores[:, (0, 2, 4)],
        short_scores[:, (1, 3, 5)],
        all_dense[order],
        delta=Fraction(1, 7),
    )
    challenger_probability_rmpts = rmpts_fuse_numpy_probabilities(
        short_scores[:, (0, 2, 4)],
        short_scores[:, (1, 3, 5)],
        challenger_dense,
        delta=Fraction(1, 7),
    )
    names = ("clip_map_percent", "clip_p_at_r90", "clip_p_at_r95")
    baseline_metrics = metrics_from_arrays(labels, baseline)
    challenger_metrics = metrics_from_arrays(labels, challenger)
    dense_320_metrics = metrics_from_arrays(labels, all_dense[order])
    dense_640_metrics = metrics_from_arrays(labels, challenger_dense)
    report = {
        "protocol": "edgefall_dense_fixed_replacement_canary_v2",
        "replacement": f"dense48_320_to_{args.replacement}",
        "folds_present": sorted(set(folds.tolist())),
        "clips": int(labels.size),
        "baseline": baseline_metrics,
        "challenger": challenger_metrics,
        "delta": {
            name: challenger_metrics[name] - baseline_metrics[name] for name in names
        },
        "probability_rmpts_1_7": {
            "baseline": metrics_from_arrays(labels, baseline_probability_rmpts),
            "challenger": metrics_from_arrays(labels, challenger_probability_rmpts),
            "delta": {
                name: (
                    metrics_from_arrays(labels, challenger_probability_rmpts)[name]
                    - metrics_from_arrays(labels, baseline_probability_rmpts)[name]
                )
                for name in names
            },
            "paired_template_group_bootstrap_delta_95ci": paired_group_bootstrap(
                clip_ids,
                labels,
                baseline_probability_rmpts,
                challenger_probability_rmpts,
                replicates=args.bootstrap_replicates,
                seed=20260830,
            ),
        },
        "dense_member": {
            "dense48_320": dense_320_metrics,
            "dense48_640": dense_640_metrics,
            "delta": {
                name: dense_640_metrics[name] - dense_320_metrics[name] for name in names
            },
            "score_pearson": float(np.corrcoef(all_dense[order], challenger_dense)[0, 1]),
            "error_pearson": float(
                np.corrcoef(all_dense[order] - labels, challenger_dense - labels)[0, 1]
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
        "baseline_reconstruction_max_abs_error": reconstruction_error,
        "input_sha256": {
            "member_report": _sha256_file(args.member_report),
            "dense_oof": _sha256_file(args.dense_oof),
            "challenger_oof": [_sha256_file(path) for path in args.challenger_oof],
        },
        "test_accessed": False,
    }
    if sorted(set(folds.tolist())) == list(range(5)):
        (
            baseline_shared_temperature,
            challenger_shared_temperature,
            baseline_temperature_folds,
            challenger_temperature_folds,
        ) = shared_temperature_crossfit_comparison(
            labels,
            folds,
            short_logits[order],
            all_dense[order],
            challenger_dense,
        )
        baseline_temperature_metrics = metrics_from_arrays(
            labels, baseline_shared_temperature
        )
        challenger_temperature_metrics = metrics_from_arrays(
            labels, challenger_shared_temperature
        )
        report["frozen_probability_rmpts_1_7_shared_temperature"] = {
            "baseline": baseline_temperature_metrics,
            "challenger": challenger_temperature_metrics,
            "delta": {
                name: (
                    challenger_temperature_metrics[name]
                    - baseline_temperature_metrics[name]
                )
                for name in names
            },
            "baseline_folds": baseline_temperature_folds,
            "challenger_folds": challenger_temperature_folds,
            "paired_template_group_bootstrap_delta_95ci": paired_group_bootstrap(
                clip_ids,
                labels,
                baseline_shared_temperature,
                challenger_shared_temperature,
                replicates=args.bootstrap_replicates,
                seed=20260831,
            ),
        }
    _atomic_json(args.output, report)
    print(json.dumps({"delta": report["delta"], "dense_member": report["dense_member"]}))


if __name__ == "__main__":
    main()
