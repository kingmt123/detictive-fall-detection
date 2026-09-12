"""Run fast, diagnostic-only robust fusion experiments on existing OOF logits."""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from models.edgefall_ensemble import (
    RMPTS_GRID,
    rmpts_fuse_numpy_logits,
    rmpts_fuse_numpy_probabilities,
)
from tools.audit_paired_clip_predictions import (
    metrics_from_arrays,
    paired_group_bootstrap,
)
from tools.evaluate_rmpts_crossfit import select_rmpts_delta
from tools.evaluate_rmpts_diagnostic import _logit, load_member_matrix
from tools.train_tcn import _atomic_json, _sha256_file


def fit_logit_scale(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    regularization: float = 0.01,
    iterations: int = 50,
) -> float:
    """Fit one positive confidence scale with bounded Newton updates."""
    values = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.float64)
    if values.ndim != 1 or targets.shape != values.shape:
        raise ValueError("logit scale 输入 shape 不匹配")
    if not np.isfinite(values).all() or not set(np.unique(targets)).issubset({0.0, 1.0}):
        raise ValueError("logit scale 输入无效")
    scale = 1.0
    for _ in range(iterations):
        bounded = np.clip(scale * values, -30.0, 30.0)
        probability = 1.0 / (1.0 + np.exp(-bounded))
        gradient = np.mean(values * (probability - targets)) + regularization * (
            scale - 1.0
        )
        hessian = np.mean(values * values * probability * (1.0 - probability)) + regularization
        updated = float(np.clip(scale - gradient / hessian, 0.25, 4.0))
        if abs(updated - scale) < 1e-10:
            break
        scale = updated
    return scale


def fast_fusion_scores(
    short_logits: np.ndarray,
    dense_logits: np.ndarray,
) -> dict[str, np.ndarray]:
    if short_logits.ndim != 2 or short_logits.shape[1] != 6:
        raise ValueError("快速融合要求六个短窗 logit")
    if dense_logits.shape != (short_logits.shape[0],):
        raise ValueError("dense logit shape 不匹配")
    seven = np.column_stack((short_logits, dense_logits))
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(seven, -30.0, 30.0)))
    rmpts_probability_weights = np.asarray(
        [2 / 21, 1 / 7, 2 / 21, 1 / 7, 2 / 21, 1 / 7, 2 / 7],
        dtype=np.float64,
    )
    if not np.isclose(rmpts_probability_weights.sum(), 1.0):
        raise AssertionError("概率域 RMPTS 权重和不为 1")

    def probability_logit(values: np.ndarray) -> np.ndarray:
        clipped = np.clip(values, 1e-7, 1.0 - 1e-7)
        return np.log(clipped / (1.0 - clipped))

    return {
        "incumbent_equal_logit": seven.mean(axis=1),
        "equal_probability": probability_logit(probabilities.mean(axis=1)),
        "rmpts_delta_1_7": rmpts_fuse_numpy_logits(
            short_logits[:, (0, 2, 4)],
            short_logits[:, (1, 3, 5)],
            dense_logits,
            delta=Fraction(1, 7),
        ),
        "rmpts_probability_1_7": probability_logit(
            probabilities @ rmpts_probability_weights
        ),
        "family_median_logit": (
            (3 / 7) * np.median(short_logits[:, (0, 2, 4)], axis=1)
            + (3 / 7) * np.median(short_logits[:, (1, 3, 5)], axis=1)
            + (1 / 7) * dense_logits
        ),
        "median_logit": np.median(seven, axis=1),
        "trimmed_mean_logit": (
            seven.sum(axis=1) - seven.min(axis=1) - seven.max(axis=1)
        )
        / 5.0,
    }


def crossfit_scaled_scores(
    labels: np.ndarray,
    folds: np.ndarray,
    short_logits: np.ndarray,
    dense_logits: np.ndarray,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    seven = np.column_stack((short_logits, dense_logits))
    equal = np.full(labels.size, np.nan, dtype=np.float64)
    rmpts = np.full(labels.size, np.nan, dtype=np.float64)
    evidence = []
    for fold in sorted(set(folds.tolist())):
        inner = folds != fold
        heldout = folds == fold
        scales = np.asarray(
            [fit_logit_scale(seven[inner, index], labels[inner]) for index in range(7)]
        )
        calibrated = seven[heldout] * scales
        equal[heldout] = calibrated.mean(axis=1)
        rmpts[heldout] = rmpts_fuse_numpy_logits(
            calibrated[:, (0, 2, 4)],
            calibrated[:, (1, 3, 5)],
            calibrated[:, 6],
            delta=Fraction(1, 7),
        )
        evidence.append({"outer_fold": int(fold), "member_scales": scales.tolist()})
    if not np.isfinite(equal).all() or not np.isfinite(rmpts).all():
        raise AssertionError("cross-fit calibration 未覆盖全部 OOF")
    return {
        "crossfit_scaled_equal_logit": equal,
        "crossfit_scaled_rmpts_1_7": rmpts,
    }, evidence


def probability_rmpts_grid(
    short_logits: np.ndarray, dense_logits: np.ndarray
) -> dict[Fraction, np.ndarray]:
    seven = np.column_stack((short_logits, dense_logits))
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(seven, -30.0, 30.0)))
    result = {}
    for delta in RMPTS_GRID:
        result[delta] = rmpts_fuse_numpy_probabilities(
            probabilities[:, (0, 2, 4)],
            probabilities[:, (1, 3, 5)],
            probabilities[:, 6],
            delta=delta,
        )
    return result


def crossfit_probability_rmpts(
    clip_ids: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    short_logits: np.ndarray,
    dense_logits: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    grid = probability_rmpts_grid(short_logits, dense_logits)
    crossfit = np.full(labels.size, np.nan, dtype=np.float64)
    evidence = []
    for fold in sorted(set(folds.tolist())):
        inner = folds != fold
        heldout = folds == fold
        selected, candidates = select_rmpts_delta(
            clip_ids[inner],
            labels[inner],
            {delta: scores[inner] for delta, scores in grid.items()},
            bootstrap_replicates=100,
            bootstrap_seed=20260829 + int(fold) * 100,
        )
        crossfit[heldout] = grid[selected][heldout]
        evidence.append(
            {
                "outer_fold": int(fold),
                "selected_delta": f"{selected.numerator}/{selected.denominator}",
                "inner_candidates": candidates,
            }
        )
    if not np.isfinite(crossfit).all():
        raise AssertionError("probability RMPTS cross-fit 未覆盖全部 OOF")
    return crossfit, evidence


def _score_report(labels: np.ndarray, folds: np.ndarray, logits: np.ndarray) -> dict[str, Any]:
    scores = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
    return {
        "metrics": metrics_from_arrays(labels, scores),
        "per_fold": [
            {
                "fold": int(fold),
                **metrics_from_arrays(labels[folds == fold], scores[folds == fold]),
            }
            for fold in sorted(set(folds.tolist()))
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--member-report", type=Path, required=True)
    parser.add_argument("--dense-oof", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("快速融合报告已存在，拒绝覆盖")
    with np.load(args.dense_oof, allow_pickle=False) as payload:
        required = {"clip_id", "label", "fold", "dense48_score"}
        if not required.issubset(payload.files):
            raise ValueError("dense OOF 缺少快速融合字段")
        clip_ids = np.asarray(payload["clip_id"]).astype(str)
        labels = np.asarray(payload["label"], dtype=np.uint8)
        folds = np.asarray(payload["fold"], dtype=np.int64)
        dense_logits = _logit(np.asarray(payload["dense48_score"]))
    short_logits, _ = load_member_matrix(args.member_report, clip_ids, labels)
    candidates = fast_fusion_scores(short_logits, dense_logits)
    scaled, calibration = crossfit_scaled_scores(
        labels, folds, short_logits, dense_logits
    )
    candidates.update(scaled)
    probability_crossfit, probability_selection = crossfit_probability_rmpts(
        clip_ids, labels, folds, short_logits, dense_logits
    )
    candidates["crossfit_probability_rmpts"] = _logit(probability_crossfit)
    baseline = _score_report(labels, folds, candidates["incumbent_equal_logit"])
    rows = {}
    for name, logits in candidates.items():
        row = _score_report(labels, folds, logits)
        row["delta_vs_incumbent"] = {
            key: row["metrics"][key] - baseline["metrics"][key]
            for key in baseline["metrics"]
        }
        rows[name] = row
    baseline_scores = 1.0 / (
        1.0 + np.exp(-np.clip(candidates["incumbent_equal_logit"], -30.0, 30.0))
    )
    promising_bootstrap = {}
    for offset, name in enumerate(
        ("rmpts_probability_1_7", "crossfit_probability_rmpts")
    ):
        candidate_scores = 1.0 / (
            1.0 + np.exp(-np.clip(candidates[name], -30.0, 30.0))
        )
        promising_bootstrap[name] = paired_group_bootstrap(
            clip_ids,
            labels,
            baseline_scores,
            candidate_scores,
            replicates=2000,
            seed=20260829 + offset,
        )
    report = {
        "protocol": "edgefall_rmpts_fast_fusion_diagnostic_v1",
        "qualification": "diagnostic_only_outer_isolated_base_members_required",
        "selected_candidate": None,
        "clips": int(labels.size),
        "member_report_sha256": _sha256_file(args.member_report),
        "dense_oof_sha256": _sha256_file(args.dense_oof),
        "candidates": rows,
        "crossfit_calibration": calibration,
        "crossfit_probability_selection": probability_selection,
        "promising_candidate_paired_bootstrap_delta_95ci": promising_bootstrap,
        "test_accessed": False,
        "limitations": [
            "existing base OOF members are not outer-isolated for calibration selection",
            "no candidate may replace the incumbent from this diagnostic report",
        ],
    }
    _atomic_json(args.output, report)
    print(json.dumps({"output": str(args.output.resolve()), "candidates": len(rows)}))


if __name__ == "__main__":
    main()
