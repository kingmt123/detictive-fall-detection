"""Cross-fit one shared scale and bias before Probability-RMPTS fusion."""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from models.edgefall_ensemble import rmpts_fuse_numpy_probabilities
from tools.audit_paired_clip_predictions import (
    metrics_from_arrays,
    paired_group_bootstrap,
)
from tools.evaluate_probability_rmpts_calibration_crossfit import (
    calibrated_probability_rmpts,
    calibration_crossfit,
)
from tools.evaluate_rmpts_diagnostic import _logit, load_member_matrix
from tools.train_tcn import _atomic_json, _sha256_file

SCALE_BOUNDS = (0.25, 4.0)
BIAS_BOUNDS = (-4.0, 4.0)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))


def fit_shared_platt(logits: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    values = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != targets.size:
        raise ValueError("shared Platt logits/labels shape 不匹配")
    if not np.isfinite(values).all() or np.unique(targets).tolist() != [0.0, 1.0]:
        raise ValueError("shared Platt 输入无效")
    targets = np.repeat(targets, values.shape[1])
    values = values.reshape(-1)
    scale, bias = 1.0, 0.0

    def loss(candidate_scale: float, candidate_bias: float) -> float:
        linear = np.clip(candidate_scale * values + candidate_bias, -60.0, 60.0)
        return float(np.mean(np.logaddexp(0.0, linear) - targets * linear))

    for _ in range(80):
        probability = _sigmoid(scale * values + bias)
        residual = probability - targets
        weight = probability * (1.0 - probability)
        gradient = np.asarray(
            [np.mean(values * residual), np.mean(residual)], dtype=np.float64
        )
        hessian = np.asarray(
            [
                [np.mean(values * values * weight), np.mean(values * weight)],
                [np.mean(values * weight), np.mean(weight)],
            ],
            dtype=np.float64,
        )
        hessian += np.eye(2) * 1e-10
        step = np.linalg.solve(hessian, gradient)
        old_loss = loss(scale, bias)
        accepted = False
        multiplier = 1.0
        for _ in range(20):
            candidate_scale = float(
                np.clip(scale - multiplier * step[0], *SCALE_BOUNDS)
            )
            candidate_bias = float(np.clip(bias - multiplier * step[1], *BIAS_BOUNDS))
            if loss(candidate_scale, candidate_bias) <= old_loss + 1e-12:
                accepted = True
                break
            multiplier *= 0.5
        if not accepted or max(abs(candidate_scale - scale), abs(candidate_bias - bias)) < 1e-10:
            break
        scale, bias = candidate_scale, candidate_bias
    return scale, bias


def apply_shared_platt(
    short_logits: np.ndarray,
    dense_logits: np.ndarray,
    *,
    scale: float,
    bias: float,
) -> np.ndarray:
    short_probability = _sigmoid(scale * short_logits + bias)
    dense_probability = _sigmoid(scale * dense_logits + bias)
    return rmpts_fuse_numpy_probabilities(
        short_probability[:, (0, 2, 4)],
        short_probability[:, (1, 3, 5)],
        dense_probability,
        delta=Fraction(1, 7),
    )


def shared_platt_crossfit(
    labels: np.ndarray,
    folds: np.ndarray,
    short_logits: np.ndarray,
    dense_logits: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    all_logits = np.column_stack((short_logits, dense_logits))
    scores = np.full(labels.size, np.nan, dtype=np.float64)
    details = []
    for fold in sorted(set(folds.tolist())):
        inner = folds != fold
        held = folds == fold
        scale, bias = fit_shared_platt(all_logits[inner], labels[inner])
        held_scores = apply_shared_platt(
            short_logits[held], dense_logits[held], scale=scale, bias=bias
        )
        scores[held] = held_scores
        details.append(
            {
                "outer_fold": int(fold),
                "scale": scale,
                "bias": bias,
                "held_metrics": metrics_from_arrays(labels[held], held_scores),
            }
        )
    if not np.isfinite(scores).all():
        raise AssertionError("shared Platt crossfit 未覆盖全部 rows")
    return scores, details


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--member-report", type=Path, required=True)
    parser.add_argument("--dense-oof", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260829)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("shared Platt 输出已存在，拒绝覆盖")
    with np.load(args.dense_oof, allow_pickle=False) as payload:
        clip_ids = np.asarray(payload["clip_id"]).astype(str)
        labels = np.asarray(payload["label"], dtype=np.uint8)
        folds = np.asarray(payload["fold"], dtype=np.int64)
        dense_logits = _logit(np.asarray(payload["dense48_score"]))
    short_logits, _ = load_member_matrix(args.member_report, clip_ids, labels)
    baseline = calibrated_probability_rmpts(
        short_logits, dense_logits, np.ones(7, dtype=np.float64)
    )
    temperature_scores, temperature_folds = calibration_crossfit(
        labels, folds, short_logits, dense_logits, "global"
    )
    platt_scores, platt_folds = shared_platt_crossfit(
        labels, folds, short_logits, dense_logits
    )
    report = {
        "protocol": "edgefall_probability_rmpts_shared_platt_crossfit_surrogate_v1",
        "qualification": "development_surrogate_only_base_members_not_outer_isolated",
        "fixed_delta": "1/7",
        "baseline_metrics": metrics_from_arrays(labels, baseline),
        "shared_temperature": {
            "metrics": metrics_from_arrays(labels, temperature_scores),
            "folds": temperature_folds,
        },
        "shared_platt": {
            "metrics": metrics_from_arrays(labels, platt_scores),
            "folds": platt_folds,
            "paired_group_bootstrap_delta_vs_temperature_95ci": paired_group_bootstrap(
                clip_ids,
                labels,
                temperature_scores,
                platt_scores,
                replicates=args.bootstrap_replicates,
                seed=args.bootstrap_seed,
            ),
        },
        "member_report_sha256": _sha256_file(args.member_report),
        "dense_oof_sha256": _sha256_file(args.dense_oof),
        "test_accessed": False,
        "limitations": [
            "base predictions were not trained with the held calibration fold excluded",
            "formal promotion requires outer-isolated nested evaluation",
        ],
    }
    _atomic_json(args.output, report)
    print(json.dumps({"output": str(args.output.resolve()), "metrics": report["shared_platt"]["metrics"]}))


if __name__ == "__main__":
    main()
