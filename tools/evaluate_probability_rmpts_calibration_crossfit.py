"""Evaluate temperature-calibrated Probability-RMPTS with fold isolation.

This is a development surrogate over the existing first-layer OOF predictions.
For every held fold, calibration parameters are fitted on the other four folds
with binary cross-entropy and then applied to the held fold.  The base members
are not outer-isolated, so this report cannot qualify a deployment model.
"""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from models.edgefall_ensemble import RMPTS_GRID, rmpts_fuse_numpy_probabilities
from tools.audit_paired_clip_predictions import (
    metrics_from_arrays,
    paired_group_bootstrap,
)
from tools.evaluate_rmpts_crossfit import select_rmpts_delta
from tools.evaluate_rmpts_diagnostic import _logit, load_member_matrix
from tools.train_tcn import _atomic_json, _sha256_file

CALIBRATION_VARIANTS = ("global", "family", "member")
INVERSE_TEMPERATURE_BOUNDS = (0.25, 4.0)
FIXED_DELTA = Fraction(1, 7)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    inverse_temperature_bounds: tuple[float, float] = INVERSE_TEMPERATURE_BOUNDS,
) -> float:
    """Fit one positive temperature by convex BCE minimization."""
    values = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.float64)
    if values.ndim not in (1, 2) or values.shape[0] != targets.size:
        raise ValueError("temperature logits/labels shape 不匹配")
    if not np.isfinite(values).all() or not np.isin(targets, (0.0, 1.0)).all():
        raise ValueError("temperature 输入必须有限且标签为二元")
    if np.unique(targets).size != 2:
        raise ValueError("temperature 拟合需要正负样本")
    if values.ndim == 2:
        targets = np.repeat(targets, values.shape[1])
        values = values.reshape(-1)
    low, high = map(float, inverse_temperature_bounds)
    if not 0.0 < low < high:
        raise ValueError("inverse temperature bounds 无效")

    def gradient(scale: float) -> float:
        return float(np.mean(values * (_sigmoid(scale * values) - targets)))

    if gradient(low) >= 0.0:
        inverse_temperature = low
    elif gradient(high) <= 0.0:
        inverse_temperature = high
    else:
        for _ in range(80):
            middle = 0.5 * (low + high)
            if gradient(middle) < 0.0:
                low = middle
            else:
                high = middle
        inverse_temperature = 0.5 * (low + high)
    return 1.0 / inverse_temperature


def fit_variant_temperatures(
    short_logits: np.ndarray,
    dense_logits: np.ndarray,
    labels: np.ndarray,
    variant: str,
) -> np.ndarray:
    """Return temperatures in frozen member order: six short then dense."""
    if short_logits.shape != (labels.size, 6) or dense_logits.shape != (labels.size,):
        raise ValueError("Probability-RMPTS calibration shape 不匹配")
    if variant not in CALIBRATION_VARIANTS:
        raise ValueError(f"未知 calibration variant: {variant}")
    all_logits = np.column_stack((short_logits, dense_logits))
    if variant == "global":
        temperature = fit_temperature(all_logits, labels)
        return np.full(7, temperature, dtype=np.float64)
    if variant == "family":
        short_320 = fit_temperature(short_logits[:, (0, 2, 4)], labels)
        short_640 = fit_temperature(short_logits[:, (1, 3, 5)], labels)
        dense = fit_temperature(dense_logits, labels)
        return np.asarray(
            [short_320, short_640, short_320, short_640, short_320, short_640, dense],
            dtype=np.float64,
        )
    return np.asarray(
        [fit_temperature(all_logits[:, index], labels) for index in range(7)],
        dtype=np.float64,
    )


def calibrated_probability_rmpts(
    short_logits: np.ndarray,
    dense_logits: np.ndarray,
    temperatures: np.ndarray,
) -> np.ndarray:
    temperatures = np.asarray(temperatures, dtype=np.float64)
    if temperatures.shape != (7,) or not np.isfinite(temperatures).all():
        raise ValueError("Probability-RMPTS temperatures 必须为七个有限值")
    if np.any(temperatures <= 0.0):
        raise ValueError("Probability-RMPTS temperatures 必须为正")
    short_probability = _sigmoid(short_logits / temperatures[:6])
    dense_probability = _sigmoid(dense_logits / temperatures[6])
    return rmpts_fuse_numpy_probabilities(
        short_probability[:, (0, 2, 4)],
        short_probability[:, (1, 3, 5)],
        dense_probability,
        delta=FIXED_DELTA,
    )


def calibrated_probability_rmpts_grid(
    short_logits: np.ndarray,
    dense_logits: np.ndarray,
    temperatures: np.ndarray,
) -> dict[Fraction, np.ndarray]:
    temperatures = np.asarray(temperatures, dtype=np.float64)
    if temperatures.shape != (7,) or np.any(temperatures <= 0.0):
        raise ValueError("Probability-RMPTS grid temperatures 无效")
    short_probability = _sigmoid(short_logits / temperatures[:6])
    dense_probability = _sigmoid(dense_logits / temperatures[6])
    return {
        delta: rmpts_fuse_numpy_probabilities(
            short_probability[:, (0, 2, 4)],
            short_probability[:, (1, 3, 5)],
            dense_probability,
            delta=delta,
        )
        for delta in RMPTS_GRID
    }


def calibration_crossfit(
    labels: np.ndarray,
    folds: np.ndarray,
    short_logits: np.ndarray,
    dense_logits: np.ndarray,
    variant: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    scores = np.full(labels.size, np.nan, dtype=np.float64)
    fold_details = []
    unique_folds = sorted(set(np.asarray(folds, dtype=np.int64).tolist()))
    if len(unique_folds) != 5:
        raise ValueError("Probability-RMPTS calibration 需要完整五折")
    for outer_fold in unique_folds:
        inner = folds != outer_fold
        held = folds == outer_fold
        temperatures = fit_variant_temperatures(
            short_logits[inner], dense_logits[inner], labels[inner], variant
        )
        held_scores = calibrated_probability_rmpts(
            short_logits[held], dense_logits[held], temperatures
        )
        fixed_scores = calibrated_probability_rmpts(
            short_logits[held], dense_logits[held], np.ones(7, dtype=np.float64)
        )
        scores[held] = held_scores
        held_metrics = metrics_from_arrays(labels[held], held_scores)
        fixed_metrics = metrics_from_arrays(labels[held], fixed_scores)
        fold_details.append(
            {
                "outer_fold": int(outer_fold),
                "temperatures": temperatures.tolist(),
                "held_metrics": held_metrics,
                "held_delta_vs_fixed": {
                    key: held_metrics[key] - fixed_metrics[key] for key in held_metrics
                },
            }
        )
    if not np.isfinite(scores).all():
        raise AssertionError("Probability-RMPTS calibration 未覆盖全部 rows")
    return scores, fold_details


def global_temperature_delta_crossfit(
    clip_ids: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    short_logits: np.ndarray,
    dense_logits: np.ndarray,
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Fit one temperature and select the locked delta grid inside four folds."""
    scores = np.full(labels.size, np.nan, dtype=np.float64)
    fold_details = []
    for outer_fold in sorted(set(folds.tolist())):
        inner = folds != outer_fold
        held = folds == outer_fold
        temperatures = fit_variant_temperatures(
            short_logits[inner], dense_logits[inner], labels[inner], "global"
        )
        inner_grid = calibrated_probability_rmpts_grid(
            short_logits[inner], dense_logits[inner], temperatures
        )
        selected, candidates = select_rmpts_delta(
            clip_ids[inner],
            labels[inner],
            inner_grid,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=bootstrap_seed + int(outer_fold) * 100,
        )
        held_grid = calibrated_probability_rmpts_grid(
            short_logits[held], dense_logits[held], temperatures
        )
        held_scores = held_grid[selected]
        scores[held] = held_scores
        fold_details.append(
            {
                "outer_fold": int(outer_fold),
                "temperature": float(temperatures[0]),
                "selected_delta": f"{selected.numerator}/{selected.denominator}",
                "inner_candidates": candidates,
                "held_metrics": metrics_from_arrays(labels[held], held_scores),
            }
        )
    if not np.isfinite(scores).all():
        raise AssertionError("global temperature delta crossfit 未覆盖全部 rows")
    return scores, fold_details


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--member-report", type=Path, required=True)
    parser.add_argument("--dense-oof", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260829)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Probability-RMPTS calibration 输出已存在，拒绝覆盖")
    if args.bootstrap_replicates < 100:
        raise ValueError("Probability-RMPTS calibration bootstrap 至少需要 100 次")
    with np.load(args.dense_oof, allow_pickle=False) as payload:
        required = {"clip_id", "label", "fold", "score", "dense48_score"}
        if not required.issubset(payload.files):
            raise ValueError("dense OOF 缺少 Probability-RMPTS calibration 字段")
        clip_ids = np.asarray(payload["clip_id"]).astype(str)
        labels = np.asarray(payload["label"], dtype=np.uint8)
        folds = np.asarray(payload["fold"], dtype=np.int64)
        incumbent_scores = np.asarray(payload["score"], dtype=np.float64)
        dense_logits = _logit(np.asarray(payload["dense48_score"]))
    if len(set(clip_ids.tolist())) != clip_ids.size:
        raise ValueError("Probability-RMPTS calibration clip_id 必须唯一")
    short_logits, _ = load_member_matrix(args.member_report, clip_ids, labels)
    identity_temperatures = np.ones(7, dtype=np.float64)
    fixed_scores = calibrated_probability_rmpts(
        short_logits, dense_logits, identity_temperatures
    )
    variants = []
    for offset, variant in enumerate(CALIBRATION_VARIANTS):
        scores, fold_details = calibration_crossfit(
            labels, folds, short_logits, dense_logits, variant
        )
        metrics = metrics_from_arrays(labels, scores)
        variants.append(
            {
                "variant": variant,
                "metrics": metrics,
                "delta_vs_fixed_probability_rmpts": {
                    key: metrics[key] - metrics_from_arrays(labels, fixed_scores)[key]
                    for key in metrics
                },
                "paired_group_bootstrap_delta_vs_fixed_95ci": paired_group_bootstrap(
                    clip_ids,
                    labels,
                    fixed_scores,
                    scores,
                    replicates=args.bootstrap_replicates,
                    seed=args.bootstrap_seed + offset,
                ),
                "folds": fold_details,
            }
        )
    selected_scores, selected_folds = global_temperature_delta_crossfit(
        clip_ids,
        labels,
        folds,
        short_logits,
        dense_logits,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed + 1000,
    )
    selected_metrics = metrics_from_arrays(labels, selected_scores)
    fixed_metrics = metrics_from_arrays(labels, fixed_scores)
    report = {
        "protocol": "edgefall_probability_rmpts_temperature_crossfit_surrogate_v1",
        "qualification": "development_surrogate_only_base_members_not_outer_isolated",
        "fixed_delta": "1/7",
        "selection": None,
        "incumbent_equal_logit_metrics": metrics_from_arrays(labels, incumbent_scores),
        "fixed_uncalibrated_probability_rmpts_metrics": metrics_from_arrays(
            labels, fixed_scores
        ),
        "variants": variants,
        "global_temperature_locked_delta_crossfit": {
            "metrics": selected_metrics,
            "delta_vs_fixed_probability_rmpts": {
                key: selected_metrics[key] - fixed_metrics[key] for key in selected_metrics
            },
            "paired_group_bootstrap_delta_vs_fixed_95ci": paired_group_bootstrap(
                clip_ids,
                labels,
                fixed_scores,
                selected_scores,
                replicates=args.bootstrap_replicates,
                seed=args.bootstrap_seed + 2000,
            ),
            "folds": selected_folds,
        },
        "inverse_temperature_bounds": list(INVERSE_TEMPERATURE_BOUNDS),
        "member_report_sha256": _sha256_file(args.member_report),
        "dense_oof_sha256": _sha256_file(args.dense_oof),
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.bootstrap_seed,
        "test_accessed": False,
        "limitations": [
            "base predictions were not trained with the held calibration fold excluded",
            "variants are reported without selecting a deployment model",
            "formal promotion requires outer-isolated nested evaluation",
        ],
    }
    _atomic_json(args.output, report)
    print(json.dumps({"output": str(args.output.resolve()), "variants": len(variants)}))


if __name__ == "__main__":
    main()
