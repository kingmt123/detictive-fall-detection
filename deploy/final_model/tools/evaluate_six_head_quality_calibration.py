"""Nested train-OOF quality calibration for the fixed six-head EdgeFall score."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from tools.evaluate_nested_oof_stacking import _metrics_from_arrays
from tools.train_long_context_event_oracle import template_group
from tools.train_tcn import _atomic_json, _sha256_file


def quality_features(scores: np.ndarray, quality: np.ndarray) -> np.ndarray:
    """Return base logit, five quality values, and their score interactions."""
    scores = np.asarray(scores, dtype=np.float64)
    quality = np.asarray(quality, dtype=np.float64)
    if scores.ndim != 1 or quality.shape != (scores.size, 5):
        raise ValueError("six-head score/quality shape 不匹配")
    if not np.isfinite(scores).all() or not np.isfinite(quality).all():
        raise ValueError("six-head score/quality 必须为有限数值")
    clipped = np.clip(scores, 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped))[:, None]
    return np.concatenate((logits, quality, logits * quality), axis=1)


def fit_weighted_ridge(
    features: np.ndarray, labels: np.ndarray, *, l2: float
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Fit deterministic class-balanced ridge with an unpenalized intercept."""
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if features.ndim != 2 or labels.shape != (features.shape[0],):
        raise ValueError("ridge features/labels shape 不匹配")
    if l2 < 0.0 or not np.isfinite(l2):
        raise ValueError("ridge l2 无效")
    positives = float(labels.sum())
    if positives <= 0.0 or positives >= labels.size:
        raise ValueError("ridge 训练必须同时包含正负样本")
    mean = features.mean(axis=0)
    scale = features.std(axis=0).clip(min=1e-6)
    standardized = (features - mean) / scale
    design = np.column_stack((standardized, np.ones(labels.size)))
    sample_weights = np.where(labels > 0.5, (labels.size - positives) / positives, 1.0)
    root = np.sqrt(sample_weights)
    weighted_design = design * root[:, None]
    penalty = np.diag([l2] * features.shape[1] + [0.0])
    coefficients = np.linalg.solve(
        weighted_design.T @ weighted_design + penalty,
        weighted_design.T @ (labels * root),
    )
    return coefficients[:-1], float(coefficients[-1]), mean, scale


def apply_weighted_ridge(
    features: np.ndarray,
    coefficients: np.ndarray,
    intercept: float,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    linear = ((np.asarray(features) - mean) / scale) @ coefficients + intercept
    return (1.0 / (1.0 + np.exp(-np.clip(linear, -40.0, 40.0)))).astype(
        np.float32
    )


def nested_quality_calibration(
    features: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    *,
    l2_grid: tuple[float, ...],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Cross-fit every outer fold and select L2 only on its inner folds."""
    labels = np.asarray(labels, dtype=np.uint8)
    folds = np.asarray(folds, dtype=np.int64)
    outer_folds = sorted(set(folds.tolist()))
    if len(outer_folds) < 3:
        raise ValueError("quality calibration 至少需要三个 folds")
    predictions = np.full(labels.size, np.nan, dtype=np.float32)
    details: list[dict[str, Any]] = []
    for outer_fold in outer_folds:
        candidates = []
        for l2 in l2_grid:
            inner_predictions = np.full(labels.size, np.nan, dtype=np.float32)
            for inner_fold in outer_folds:
                if inner_fold == outer_fold:
                    continue
                fit = (folds != outer_fold) & (folds != inner_fold)
                held = (folds != outer_fold) & (folds == inner_fold)
                parameters = fit_weighted_ridge(features[fit], labels[fit], l2=l2)
                inner_predictions[held] = apply_weighted_ridge(
                    features[held], *parameters
                )
            inner = folds != outer_fold
            candidates.append(
                {"l2": l2, "metrics": _metrics_from_arrays(labels[inner], inner_predictions[inner])}
            )
        selected = max(
            candidates,
            key=lambda item: (
                item["metrics"]["clip_map_percent"],
                item["metrics"]["clip_p_at_r95"],
            ),
        )
        fit = folds != outer_fold
        held = folds == outer_fold
        parameters = fit_weighted_ridge(
            features[fit], labels[fit], l2=float(selected["l2"])
        )
        predictions[held] = apply_weighted_ridge(features[held], *parameters)
        details.append(
            {
                "outer_fold": outer_fold,
                "selected_l2": selected["l2"],
                "inner_candidates": candidates,
                "coefficients": parameters[0].tolist(),
            }
        )
    if not np.isfinite(predictions).all():
        raise RuntimeError("quality calibration meta-OOF 不完整")
    return predictions, details


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument("--roi-cache", type=Path, required=True)
    parser.add_argument("--score-key", default="fixed_equal_logit_average")
    parser.add_argument("--l2-grid", type=float, nargs="+", default=[0.1, 1.0, 10.0, 100.0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prediction_path = args.output.with_name(f"{args.output.stem}_meta_oof_predictions.npz")
    if args.output.exists() or prediction_path.exists():
        raise FileExistsError("quality calibration 输出已存在，拒绝覆盖")
    l2_grid = tuple(float(value) for value in args.l2_grid)
    if not 1 <= len(l2_grid) <= 5 or any(value < 0.0 for value in l2_grid):
        raise ValueError("l2 grid 无效")
    with np.load(args.oof, allow_pickle=False) as payload:
        required = {"clip_id", "label", "fold", args.score_key}
        if not required.issubset(payload.files):
            raise ValueError("six-head OOF 缺少所需字段")
        clip_ids = np.asarray(payload["clip_id"])
        labels = np.asarray(payload["label"], dtype=np.uint8)
        folds = np.asarray(payload["fold"], dtype=np.int64)
        base_scores = np.asarray(payload[args.score_key], dtype=np.float32)
    with np.load(args.roi_cache, allow_pickle=False) as payload:
        if not {"clip_ids", "quality"}.issubset(payload.files):
            raise ValueError("ROI cache 缺少 clip_ids/quality")
        roi_ids = [str(value) for value in payload["clip_ids"]]
        roi_quality = np.asarray(payload["quality"], dtype=np.float32)
    roi_index = {clip_id: index for index, clip_id in enumerate(roi_ids)}
    if len(roi_index) != len(roi_ids) or any(str(item) not in roi_index for item in clip_ids):
        raise ValueError("ROI quality 无法与 six-head OOF 对齐")
    quality = np.stack([roi_quality[roi_index[str(item)]] for item in clip_ids])
    features = quality_features(base_scores, quality)
    calibrated, details = nested_quality_calibration(
        features, labels, folds, l2_grid=l2_grid
    )
    base_metrics = _metrics_from_arrays(labels, base_scores)
    calibrated_metrics = _metrics_from_arrays(labels, calibrated)
    per_fold = []
    for fold in sorted(set(folds.tolist())):
        mask = folds == fold
        baseline = _metrics_from_arrays(labels[mask], base_scores[mask])
        candidate = _metrics_from_arrays(labels[mask], calibrated[mask])
        per_fold.append(
            {
                "fold": fold,
                "baseline": baseline,
                "candidate": candidate,
                "delta": {key: candidate[key] - baseline[key] for key in baseline},
            }
        )
    np.savez_compressed(
        prediction_path,
        clip_id=clip_ids,
        label=labels,
        fold=folds,
        fixed_equal_logit_average=base_scores,
        quality_calibrated=calibrated,
    )
    report = {
        "protocol": "edgefall_six_head_nested_quality_calibration_v1",
        "selection_split": "train template-grouped OOF only",
        "validation_uses": 0,
        "test_accessed": False,
        "features": [
            "six_head_logit",
            "track_exact_fraction",
            "keypoint_quality_mean",
            "keypoint_quality_min",
            "bbox_area_fraction",
            "bbox_step",
            "six_head_logit_x_each_quality",
        ],
        "l2_grid": list(l2_grid),
        "baseline": base_metrics,
        "candidate": calibrated_metrics,
        "delta": {key: calibrated_metrics[key] - base_metrics[key] for key in base_metrics},
        "per_fold": per_fold,
        "fit_details": details,
        "oof_sha256": _sha256_file(args.oof),
        "roi_cache_sha256": _sha256_file(args.roi_cache),
        "template_groups": len({template_group(str(item)) for item in clip_ids}),
        "meta_oof_predictions": str(prediction_path.resolve()),
    }
    _atomic_json(args.output, report)
    print(json.dumps(report["delta"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
