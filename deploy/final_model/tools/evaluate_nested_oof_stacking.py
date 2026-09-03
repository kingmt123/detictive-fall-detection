"""Nested template-group OOF audit for pre-registered C5 static fusions.

This script never loads random validation or a test cache.  It consumes only
complete first-layer OOF predictions, then produces second-layer (meta) OOF
scores.  Consequently its reported metrics are suitable for selecting whether
C5 itself is credible, rather than merely fitting C5 on all labels and
re-reporting the same examples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from models.oof_stacking import apply_nonnegative_l2, fit_nonnegative_l2_logistic
from models.tcn_dataset import load_window_cache
from tools.evaluate_oof_multistream_fusion import _logit
from tools.fit_nonnegative_oof_stacking import _load_member_oof
from tools.train_long_context_event_oracle import template_group
from tools.train_tcn import _atomic_json


def _metrics_from_arrays(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Compute the competition clip metric in O(n log n), including tied scores."""
    labels = np.asarray(labels, dtype=np.uint8)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or scores.shape != labels.shape:
        raise ValueError("指标 labels/scores shape 不一致")
    if not labels.size or not labels.any() or labels.all() or not np.isfinite(scores).all():
        raise ValueError("指标必须有有限 score 且同时包含正负样本")
    order = np.argsort(-scores, kind="mergesort")
    sorted_score = scores[order]
    sorted_label = labels[order]
    group_end = np.r_[np.flatnonzero(sorted_score[:-1] != sorted_score[1:]), labels.size - 1]
    cumulative_tp = np.cumsum(sorted_label)[group_end].astype(np.float64)
    predicted = group_end.astype(np.float64) + 1.0
    precision = cumulative_tp / predicted
    recall = cumulative_tp / float(labels.sum())

    def precision_at(target: float) -> float:
        feasible = precision[recall >= target]
        return float(feasible.max()) if feasible.size else 0.0

    p90 = precision_at(0.90)
    p95 = precision_at(0.95)
    return {
        "clip_map": (p90 + p95) / 2.0,
        "clip_map_percent": (p90 + p95) * 50.0,
        "clip_p_at_r90": p90,
        "clip_p_at_r95": p95,
    }


def _cdf_transform(train: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Map values to a frozen empirical OOF CDF learned from ``train`` only."""
    train = np.asarray(train, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if train.ndim != 2 or values.ndim != 2 or train.shape[1] != values.shape[1]:
        raise ValueError("CDF 特征 shape 不一致")
    if not train.size or not np.isfinite(train).all() or not np.isfinite(values).all():
        raise ValueError("CDF 特征必须为非空有限数值")
    ordered = np.sort(train, axis=0)
    ranks = np.empty_like(values)
    for member in range(train.shape[1]):
        ranks[:, member] = np.searchsorted(
            ordered[:, member], values[:, member], side="right"
        ) / float(train.shape[0])
    return ranks


def _select_l2(
    features: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    *,
    outer_fold: int,
    l2_grid: tuple[float, ...],
    transform: str,
) -> tuple[float, list[dict[str, Any]]]:
    """Choose L2 only through inner grouped cross-fitting of meta training rows."""
    candidates: list[dict[str, Any]] = []
    inner_folds = sorted(set(folds[folds != outer_fold].tolist()))
    for l2 in l2_grid:
        predicted = np.full(labels.size, np.nan, dtype=np.float32)
        for inner_fold in inner_folds:
            fit_mask = (folds != outer_fold) & (folds != inner_fold)
            held_mask = (folds != outer_fold) & (folds == inner_fold)
            fit_features = features[fit_mask]
            held_features = features[held_mask]
            if transform == "cdf":
                held_features = _cdf_transform(fit_features, held_features)
                fit_features = _cdf_transform(fit_features, fit_features)
            weights, intercept, mean, scale = fit_nonnegative_l2_logistic(
                fit_features, labels[fit_mask], l2=l2
            )
            predicted[held_mask] = apply_nonnegative_l2(
                held_features, weights, intercept, mean, scale
            )
        inner_mask = folds != outer_fold
        metrics = _metrics_from_arrays(labels[inner_mask], predicted[inner_mask])
        candidates.append({"l2": l2, "inner_meta_oof": metrics})
    best = max(
        candidates,
        key=lambda item: (
            item["inner_meta_oof"]["clip_map_percent"],
            item["inner_meta_oof"]["clip_p_at_r95"],
            item["inner_meta_oof"]["clip_p_at_r90"],
        ),
    )
    return float(best["l2"]), candidates


def _fit_outer_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    *,
    outer_fold: int,
    l2_grid: tuple[float, ...],
    transform: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    fit_mask = folds != outer_fold
    held_mask = folds == outer_fold
    selected_l2, candidates = _select_l2(
        features, labels, folds, outer_fold=outer_fold, l2_grid=l2_grid, transform=transform
    )
    fit_features = features[fit_mask]
    held_features = features[held_mask]
    if transform == "cdf":
        held_features = _cdf_transform(fit_features, held_features)
        fit_features = _cdf_transform(fit_features, fit_features)
    weights, intercept, mean, scale = fit_nonnegative_l2_logistic(
        fit_features, labels[fit_mask], l2=selected_l2
    )
    scores = apply_nonnegative_l2(held_features, weights, intercept, mean, scale)
    return scores, {
        "outer_fold": outer_fold,
        "selected_l2": selected_l2,
        "inner_candidates": candidates,
        "weights": weights.tolist(),
        "intercept": intercept,
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
    }


def _bootstrap_deltas(
    labels: np.ndarray,
    folds: np.ndarray,
    clips: list[dict[str, Any]],
    scores_by_method: dict[str, np.ndarray],
    *,
    replicates: int,
    seed: int,
) -> dict[str, dict[str, list[float]]]:
    """Template-group paired bootstrap CIs against Primary's raw OOF scores."""
    groups = np.asarray([template_group(str(clip["clip_id"])) for clip in clips])
    unique_groups = np.asarray(sorted(set(groups.tolist())))
    group_rows = [np.flatnonzero(groups == group) for group in unique_groups]
    rng = np.random.default_rng(seed)
    metric_names = ("clip_map_percent", "clip_p_at_r90", "clip_p_at_r95")
    deltas = {
        method: np.empty((replicates, len(metric_names)), dtype=np.float64)
        for method in scores_by_method
        if method != "Primary"
    }
    for replicate in range(replicates):
        indices = np.concatenate(
            [group_rows[index] for index in rng.integers(0, len(group_rows), len(group_rows))]
        )
        primary = _metrics_from_arrays(labels[indices], scores_by_method["Primary"][indices])
        for method, result in deltas.items():
            metrics = _metrics_from_arrays(labels[indices], scores_by_method[method][indices])
            result[replicate] = [metrics[name] - primary[name] for name in metric_names]
    return {
        method: {
            name: [float(np.quantile(values[:, index], 0.025)), float(np.quantile(values[:, index], 0.975))]
            for index, name in enumerate(metric_names)
        }
        for method, values in deltas.items()
    }


def _summary(
    labels: np.ndarray, folds: np.ndarray, scores: np.ndarray
) -> dict[str, Any]:
    overall = _metrics_from_arrays(labels, scores)
    per_fold = []
    for fold in sorted(set(folds.tolist())):
        mask = folds == fold
        per_fold.append({"fold": int(fold), **_metrics_from_arrays(labels[mask], scores[mask])})
    return {
        "overall": overall,
        "per_outer_fold": per_fold,
        "fold_mean": {
            name: float(np.mean([row[name] for row in per_fold])) for name in overall
        },
        "fold_std": {
            name: float(np.std([row[name] for row in per_fold], ddof=1)) for name in overall
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument(
        "--member", nargs=2, action="append", metavar=("NAME", "OOF_GLOB"), required=True
    )
    parser.add_argument("--l2-grid", type=float, nargs="+", default=[1e-4, 1e-3, 1e-2, 1e-1, 1.0])
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260827)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("nested stacking 输出已存在，拒绝覆盖")
    prediction_path = args.output.with_name(f"{args.output.stem}_meta_oof_predictions.npz")
    if prediction_path.exists():
        raise FileExistsError("meta OOF prediction 输出已存在，拒绝覆盖")
    l2_grid = tuple(float(value) for value in args.l2_grid)
    if not 1 <= len(l2_grid) <= 5 or any(value < 0.0 for value in l2_grid):
        raise ValueError("L2 grid 必须含 1 至 5 个非负值")
    if args.bootstrap_replicates < 100:
        raise ValueError("bootstrap 至少需要 100 次重复")
    train = load_window_cache(args.train_cache, verify_hashes=True)
    if train.metadata.get("split") != "train":
        raise ValueError("nested stacking 只允许 train template OOF")
    clips = train.metadata["clips"]
    names: list[str] = []
    raw_scores: list[np.ndarray] = []
    member_sources = []
    folds: np.ndarray | None = None
    for name, pattern in args.member:
        if name in names:
            raise ValueError(f"stacking member 名称重复: {name}")
        scores, member_folds, sources = _load_member_oof(pattern, clips)
        if folds is None:
            folds = member_folds
        elif not np.array_equal(folds, member_folds):
            raise ValueError("stacking 成员的 OOF fold assignment 不一致")
        names.append(name)
        raw_scores.append(scores)
        member_sources.append({"name": name, "oof": sources})
    if not 2 <= len(names) <= 12:
        raise ValueError("nested stacking 需要 2 到 12 个成员")
    assert folds is not None
    outer_folds = sorted(set(folds.tolist()))
    if len(outer_folds) < 3:
        raise ValueError("双层交叉拟合至少需要 3 个 template folds")
    labels = np.asarray([bool(clip["has_fall"]) for clip in clips], dtype=np.uint8)
    features = np.column_stack([_logit(scores) for scores in raw_scores])
    scores_by_method: dict[str, np.ndarray] = {"Primary": raw_scores[0]}
    scores_by_method["fixed_probability_average"] = np.mean(raw_scores, axis=0).astype(np.float32)
    mean_logit = np.mean(features, axis=1)
    scores_by_method["fixed_equal_logit_average"] = (
        1.0 / (1.0 + np.exp(-mean_logit))
    ).astype(np.float32)
    scores_by_method["standardized_equal_average"] = np.full(labels.size, np.nan, dtype=np.float32)
    scores_by_method["nonnegative_l2_logistic"] = np.full(labels.size, np.nan, dtype=np.float32)
    scores_by_method["oof_cdf_rank_average"] = np.full(labels.size, np.nan, dtype=np.float32)
    scores_by_method["oof_cdf_nonnegative_l2_logistic"] = np.full(labels.size, np.nan, dtype=np.float32)
    fit_details: dict[str, list[dict[str, Any]]] = {
        "nonnegative_l2_logistic": [],
        "oof_cdf_nonnegative_l2_logistic": [],
    }

    for outer_fold in outer_folds:
        fit_mask = folds != outer_fold
        held_mask = folds == outer_fold
        fit_features = features[fit_mask]
        held_features = features[held_mask]
        mean = fit_features.mean(axis=0)
        scale = fit_features.std(axis=0).clip(min=1e-6)
        scores_by_method["standardized_equal_average"][held_mask] = np.mean(
            (held_features - mean) / scale, axis=1
        )
        cdf_held = _cdf_transform(fit_features, held_features)
        scores_by_method["oof_cdf_rank_average"][held_mask] = np.mean(cdf_held, axis=1)
        logistic_scores, logistic_detail = _fit_outer_logistic(
            features, labels, folds, outer_fold=outer_fold, l2_grid=l2_grid, transform="identity"
        )
        cdf_scores, cdf_detail = _fit_outer_logistic(
            features, labels, folds, outer_fold=outer_fold, l2_grid=l2_grid, transform="cdf"
        )
        scores_by_method["nonnegative_l2_logistic"][held_mask] = logistic_scores
        scores_by_method["oof_cdf_nonnegative_l2_logistic"][held_mask] = cdf_scores
        fit_details["nonnegative_l2_logistic"].append(logistic_detail)
        fit_details["oof_cdf_nonnegative_l2_logistic"].append(cdf_detail)

    if any(np.any(~np.isfinite(score)) for score in scores_by_method.values()):
        raise RuntimeError("meta OOF 预测不完整")
    summaries = {name: _summary(labels, folds, score) for name, score in scores_by_method.items()}
    primary_metrics = summaries["Primary"]["overall"]
    deltas = {
        name: {metric: value - primary_metrics[metric] for metric, value in summary["overall"].items()}
        for name, summary in summaries.items()
        if name != "Primary"
    }
    correlations = np.corrcoef(np.column_stack(raw_scores), rowvar=False)
    bootstrap_ci = _bootstrap_deltas(
        labels, folds, clips, scores_by_method,
        replicates=args.bootstrap_replicates, seed=args.bootstrap_seed,
    )
    np.savez_compressed(
        prediction_path,
        clip_id=np.asarray([str(clip["clip_id"]) for clip in clips]),
        label=labels,
        fold=folds,
        **{name: score.astype(np.float32) for name, score in scores_by_method.items()},
    )
    report = {
        "protocol": "nested_template_group_oof_static_fusion_v1",
        "selection_split": "train template-grouped OOF only",
        "validation_uses": 0,
        "test_accessed": False,
        "pre_registered_methods": [
            "fixed_probability_average",
            "fixed_equal_logit_average",
            "standardized_equal_average",
            "nonnegative_l2_logistic",
            "oof_cdf_rank_average",
            "oof_cdf_nonnegative_l2_logistic",
        ],
        "members": member_sources,
        "member_order": names,
        "member_score_pearson": correlations.tolist(),
        "l2_grid": list(l2_grid),
        "fit_details": fit_details,
        "metrics": summaries,
        "delta_vs_primary": deltas,
        "template_group_paired_bootstrap_delta_95ci": bootstrap_ci,
        "fold_assignment_sha256": hashlib.sha256(folds.tobytes()).hexdigest(),
        "cache_signature": train.metadata.get("signature_sha256"),
        "meta_oof_predictions": {
            "path": str(prediction_path.resolve()),
            "sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
        },
    }
    _atomic_json(args.output, report)
    print(json.dumps({"metrics": summaries, "delta_vs_primary": deltas}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
