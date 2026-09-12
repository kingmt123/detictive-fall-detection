"""Leak-free nonnegative L2 stacking utilities."""

from __future__ import annotations

from itertools import combinations

import numpy as np


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def fit_nonnegative_l2_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    l2: float = 1e-3,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Fit deterministic class-balanced nonnegative L2 logistic stacking.

    ``fit_nonnegative_l2`` is retained for backward compatibility with the
    already materialized ridge-stacking experiment.  This function is the
    logistic objective described by the C5 protocol.  With at most 12 members,
    exact active-set enumeration is small and avoids adding a solver dependency.
    """
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if features.ndim != 2 or not 1 <= features.shape[1] <= 12:
        raise ValueError("OOF stacking 需要 1 到 12 个二维特征")
    if labels.shape != (features.shape[0],) or not np.all(
        (labels == 0.0) | (labels == 1.0)
    ):
        raise ValueError("OOF stacking 标签必须为同长度二分类数组")
    if l2 < 0.0:
        raise ValueError("l2 不能为负")
    positives = float(labels.sum())
    if positives == 0.0 or positives == labels.size:
        raise ValueError("OOF stacking 必须同时包含正负样本")

    mean = features.mean(axis=0)
    scale = features.std(axis=0).clip(min=1e-6)
    standardized = (features - mean) / scale
    sample_weight = np.where(labels == 1.0, (labels.size - positives) / positives, 1.0)
    member_count = features.shape[1]
    best: tuple[float, np.ndarray, float] | None = None

    for active_count in range(member_count + 1):
        for active in combinations(range(member_count), active_count):
            active_array = np.asarray(active, dtype=np.int64)
            active_features = standardized[:, active_array]
            design = np.column_stack((active_features, np.ones(labels.size)))
            coefficients = np.zeros(active_count + 1, dtype=np.float64)

            # Damped Newton optimisation of a strictly convex objective.  The
            # intercept is deliberately neither penalised nor sign constrained.
            for _ in range(100):
                logits = design @ coefficients
                probability = _sigmoid(logits)
                gradient = design.T @ (sample_weight * (probability - labels))
                gradient[:-1] += l2 * coefficients[:-1]
                curvature = sample_weight * probability * (1.0 - probability)
                hessian = design.T @ (design * curvature[:, None])
                if active_count:
                    hessian[:-1, :-1] += l2 * np.eye(active_count)
                try:
                    step = np.linalg.solve(hessian, gradient)
                except np.linalg.LinAlgError:
                    break
                objective = float(
                    np.sum(sample_weight * np.logaddexp(0.0, logits)
                    - sample_weight * labels * logits)
                    + 0.5 * l2 * np.square(coefficients[:-1]).sum()
                )
                step_scale = 1.0
                while step_scale >= 1e-8:
                    candidate = coefficients - step_scale * step
                    candidate_logits = design @ candidate
                    candidate_objective = float(
                        np.sum(
                            sample_weight * np.logaddexp(0.0, candidate_logits)
                            - sample_weight * labels * candidate_logits
                        )
                        + 0.5 * l2 * np.square(candidate[:-1]).sum()
                    )
                    if candidate_objective <= objective + 1e-12:
                        coefficients = candidate
                        break
                    step_scale *= 0.5
                else:
                    break
                if np.linalg.vector_norm(step_scale * step) < 1e-8:
                    break

            active_weights = coefficients[:-1]
            if np.any(active_weights < -1e-8):
                continue
            active_weights = active_weights.clip(min=0.0)
            logits = active_features @ active_weights + coefficients[-1]
            objective = float(
                np.sum(sample_weight * np.logaddexp(0.0, logits)
                - sample_weight * labels * logits)
                + 0.5 * l2 * np.square(active_weights).sum()
            )
            weights = np.zeros(member_count, dtype=np.float64)
            weights[active_array] = active_weights
            if best is None or objective < best[0]:
                best = objective, weights, float(coefficients[-1])
    if best is None:
        raise RuntimeError("nonnegative L2 logistic active-set 求解失败")
    return (
        best[1].astype(np.float32),
        best[2],
        mean.astype(np.float32),
        scale.astype(np.float32),
    )


def fit_nonnegative_l2(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    l2: float = 1e-3,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Fit class-balanced ridge stacking with nonnegative member weights.

    The small ensemble size makes exhaustive active-set enumeration exact and
    deterministic. The intercept is not penalized or sign constrained.
    """
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if features.ndim != 2 or not 1 <= features.shape[1] <= 12:
        raise ValueError("OOF stacking 需要 1 到 12 个二维特征")
    if labels.shape != (features.shape[0],) or not np.all(
        (labels == 0.0) | (labels == 1.0)
    ):
        raise ValueError("OOF stacking 标签必须为同长度二分类数组")
    if l2 < 0.0:
        raise ValueError("l2 不能为负")
    positives = float(labels.sum())
    if positives == 0.0 or positives == labels.size:
        raise ValueError("OOF stacking 必须同时包含正负样本")
    mean = features.mean(axis=0)
    scale = features.std(axis=0).clip(min=1e-6)
    standardized = (features - mean) / scale
    sample_weight = np.where(labels == 1.0, (labels.size - positives) / positives, 1.0)
    root_weight = np.sqrt(sample_weight)
    member_count = features.shape[1]
    best: tuple[float, np.ndarray, float] | None = None
    for active_count in range(member_count + 1):
        for active in combinations(range(member_count), active_count):
            active_array = np.asarray(active, dtype=np.int64)
            active_features = standardized[:, active_array]
            design = np.column_stack((active_features, np.ones(labels.size)))
            weighted_design = design * root_weight[:, None]
            weighted_labels = labels * root_weight
            penalty = np.diag([l2] * active_count + [0.0])
            coefficients = np.linalg.solve(
                weighted_design.T @ weighted_design + penalty,
                weighted_design.T @ weighted_labels,
            )
            active_weights = coefficients[:-1]
            if np.any(active_weights < -1e-10):
                continue
            active_weights = active_weights.clip(min=0.0)
            intercept = float(coefficients[-1])
            residual = active_features @ active_weights + intercept - labels
            objective = float(
                np.sum(sample_weight * np.square(residual))
                + l2 * np.square(active_weights).sum()
            )
            weights = np.zeros(member_count, dtype=np.float64)
            weights[active_array] = active_weights
            if best is None or objective < best[0]:
                best = objective, weights, intercept
    if best is None:
        raise RuntimeError("nonnegative L2 active-set 求解失败")
    return (
        best[1].astype(np.float32),
        best[2],
        mean.astype(np.float32),
        scale.astype(np.float32),
    )


def apply_nonnegative_l2(
    features: np.ndarray,
    weights: np.ndarray,
    intercept: float,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    """Apply fitted stacking and return a monotonic sigmoid score."""
    features = np.asarray(features, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    if features.ndim != 2 or features.shape[1:] != weights.shape:
        raise ValueError("stacking 特征与权重 shape 不匹配")
    if mean.shape != weights.shape or scale.shape != weights.shape or np.any(scale <= 0):
        raise ValueError("stacking 标准化参数无效")
    linear = ((features - mean) / scale) @ weights + float(intercept)
    return _sigmoid(linear).astype(np.float32)
