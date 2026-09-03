"""Evaluate a non-neural two-feature torso-tilt/bbox-aspect fall baseline.

This is intentionally narrower than PIFR: it tests only the user's two-feature
hypothesis with a regularized linear logistic classifier.  Thresholds and three
linear coefficients are fit on the train split only; val and the fixed train
hard-negative cohort are reporting-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from eval.metrics import competition_map
from models.tcn_dataset import WindowMemmapCache, load_window_cache
from tools.build_tcn_multistream_sidecar import load_sidecar
from tools.train_tcn import _atomic_json, select_training_indices


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"不是合法 JSON: {path}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"JSON 对象预期: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pifr_two_features(pose: np.ndarray, geometry: np.ndarray) -> np.ndarray:
    """Return final-frame normalized torso tilt and log bbox width/height.

    ``tilt`` is zero for an upright nose-to-mid-hip axis and one for a horizontal
    axis.  Invalid nose/hip/bbox observations map to zero rather than silently
    producing a high fall score.
    """
    if pose.ndim != 4 or pose.shape[-2:] != (17, 3):
        raise ValueError("pose 必须为 (B,T,17,3)")
    if geometry.shape != (*pose.shape[:2], 6):
        raise ValueError("geometry 必须为 (B,T,6)")
    endpoint = pose[:, -1]
    nose = endpoint[:, 0, :2]
    nose_visible = endpoint[:, 0, 2] > 0
    hip_confidence = endpoint[:, (11, 12), 2] > 0
    hip_weight = hip_confidence.astype(np.float32)
    hip_count = hip_weight.sum(axis=1, keepdims=True)
    mid_hip = (endpoint[:, (11, 12), :2] * hip_weight[..., None]).sum(axis=1)
    mid_hip /= np.maximum(hip_count, 1.0)
    vector = mid_hip - nose
    norm = np.linalg.vector_norm(vector, axis=1)
    valid_torso = nose_visible & (hip_count[:, 0] > 0) & (norm > 1e-6)
    upright_cosine = np.clip(np.abs(vector[:, 1]) / np.maximum(norm, 1e-6), 0.0, 1.0)
    tilt = np.arccos(upright_cosine) / (np.pi / 2.0)
    endpoint_geometry = geometry[:, -1]
    bbox_visible = endpoint_geometry[:, 5] > 0
    aspect = np.log(
        np.maximum(endpoint_geometry[:, 2], 1e-6)
        / np.maximum(endpoint_geometry[:, 3], 1e-6)
    )
    tilt = np.where(valid_torso, tilt, 0.0)
    aspect = np.where(bbox_visible, aspect, 0.0)
    return np.column_stack((tilt, aspect)).astype(np.float32, copy=False)


def _fit_linear_logistic(
    features: np.ndarray, labels: np.ndarray, *, l2: float = 1e-3
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit weighted 2D logistic regression with deterministic Newton updates."""
    if features.ndim != 2 or features.shape[1] != 2:
        raise ValueError("线性基线需要 (N,2) 特征")
    labels = np.asarray(labels, dtype=np.float64)
    if labels.shape != (features.shape[0],) or not np.all((labels == 0) | (labels == 1)):
        raise ValueError("labels 必须是与特征对齐的二分类向量")
    positive_count = int(np.count_nonzero(labels))
    negative_count = int(labels.size - positive_count)
    if not positive_count or not negative_count:
        raise ValueError("线性基线训练需要正负样本")
    mean = features.mean(axis=0, dtype=np.float64)
    scale = features.std(axis=0, dtype=np.float64).clip(min=1e-6)
    x = (features.astype(np.float64) - mean) / scale
    design = np.column_stack((x, np.ones(x.shape[0], dtype=np.float64)))
    sample_weight = np.where(labels == 1, negative_count / positive_count, 1.0)
    coefficients = np.zeros(3, dtype=np.float64)
    penalty = np.diag((l2, l2, 0.0))
    for _ in range(30):
        logits = np.clip(design @ coefficients, -40.0, 40.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        gradient = design.T @ (sample_weight * (probabilities - labels)) + penalty @ coefficients
        curvature = sample_weight * probabilities * (1.0 - probabilities)
        hessian = design.T @ (design * curvature[:, None]) + penalty
        step = np.linalg.solve(hessian, gradient)
        coefficients -= step
        if np.linalg.vector_norm(step) < 1e-8:
            break
    return coefficients.astype(np.float32), mean.astype(np.float32), scale.astype(np.float32)


def _probabilities(features: np.ndarray, coefficients: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    standardized = (features - mean) / scale
    logits = np.clip(standardized @ coefficients[:2] + coefficients[2], -40.0, 40.0)
    return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)


def _cache_probabilities(
    cache: WindowMemmapCache,
    geometry: np.ndarray,
    coefficients: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    if batch_size < 1:
        raise ValueError("batch_size 必须为正")
    pose = cache.array("features")
    scores = np.empty(cache.sample_count, dtype=np.float32)
    for start in range(0, cache.sample_count, batch_size):
        end = min(start + batch_size, cache.sample_count)
        features = pifr_two_features(
            np.asarray(pose[start:end], dtype=np.float32),
            np.asarray(geometry[start:end], dtype=np.float32),
        )
        scores[start:end] = _probabilities(features, coefficients, mean, scale)
    return scores


def _clip_scores(cache: WindowMemmapCache, scores: np.ndarray) -> dict[str, float]:
    if scores.shape != (cache.sample_count,):
        raise ValueError("窗口分数与 cache 样本数不一致")
    maxima = np.full(len(cache.metadata["clips"]), -np.inf, dtype=np.float32)
    np.maximum.at(maxima, cache.array("clip_indices"), scores)
    if not np.all(np.isfinite(maxima)):
        raise ValueError("存在没有窗口分数的 clip")
    return {
        clip["clip_id"]: float(maxima[index])
        for index, clip in enumerate(cache.metadata["clips"])
    }


def _metrics(cache: WindowMemmapCache, clip_scores: dict[str, float]) -> tuple[dict[str, Any], float]:
    ground_truth = {clip["clip_id"]: bool(clip["has_fall"]) for clip in cache.metadata["clips"]}
    result = competition_map(ground_truth, clip_scores, mode="clip")
    curve = result["curve"]

    def best_point(recall: float) -> dict[str, Any]:
        candidates = [point for point in curve if point.recall >= recall]
        if not candidates:
            raise ValueError(f"无法达到目标召回率 {recall}")
        point = max(candidates, key=lambda value: (value.precision, value.threshold))
        return {
            "threshold": float(point.threshold),
            "precision": float(point.precision),
            "recall": float(point.recall),
            "tp": point.tp,
            "fp": point.fp,
            "fn": point.fn,
        }

    metrics = {
        "clip_map": float(result["map"]),
        "clip_map_percent": float(result["map_percent"]),
        "clip_p_at_r90": float(result["p_at_r90"]),
        "clip_p_at_r95": float(result["p_at_r95"]),
        "clip_r90_point": best_point(0.90),
        "clip_r95_point": best_point(0.95),
    }
    return metrics, metrics["clip_r90_point"]["threshold"]


def _hard_negative_report(
    cohort: dict[str, Any], clip_scores: dict[str, float], threshold: float
) -> dict[str, Any]:
    groups = cohort.get("clips_by_activity")
    if not isinstance(groups, dict) or not groups:
        raise ValueError("cohort 缺少 clips_by_activity")
    all_ids = {clip_id for values in groups.values() for clip_id in values}
    if len(all_ids) != int(cohort.get("clip_count", -1)):
        raise ValueError("cohort clip_count 与 clips_by_activity 不一致")
    if missing := all_ids - set(clip_scores):
        raise ValueError(f"cohort clip 不在训练 cache: {sorted(missing)[:3]}")
    report_groups: dict[str, Any] = {}
    for activity, clip_ids in groups.items():
        values = np.asarray([clip_scores[clip_id] for clip_id in clip_ids], dtype=np.float32)
        count = int(np.count_nonzero(values >= threshold))
        report_groups[activity] = {
            "clip_count": int(values.size),
            "false_positive_count": count,
            "false_positive_rate": count / values.size,
            "score_mean": float(values.mean()),
            "score_p95": float(np.quantile(values, 0.95)),
        }
    values = np.asarray([clip_scores[clip_id] for clip_id in all_ids], dtype=np.float32)
    count = int(np.count_nonzero(values >= threshold))
    return {
        "threshold": threshold,
        "clip_count": int(values.size),
        "false_positive_count": count,
        "false_positive_rate": count / values.size,
        "groups": report_groups,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--train-sidecar", type=Path, required=True)
    parser.add_argument("--val-sidecar", type=Path, required=True)
    parser.add_argument("--hard-cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8192)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"输出已存在，拒绝覆盖: {args.output}")
    train = load_window_cache(args.train_cache)
    val = load_window_cache(args.val_cache)
    if train.metadata.get("split") != "train" or val.metadata.get("split") != "val":
        raise ValueError("PIFR 基线只允许 train 拟合与 val 评估")
    cohort = _load_json(args.hard_cohort)
    if cohort.get("split") != "train":
        raise ValueError("PIFR 基线的困难负例 cohort 必须来自 train")
    train_geometry = load_sidecar(args.train_sidecar, train)
    val_geometry = load_sidecar(args.val_sidecar, val)
    selected = select_training_indices(
        np.asarray(train.array("labels")), negative_ratio=2.0, seed=20260817
    )
    fit_features = pifr_two_features(
        np.asarray(train.array("features")[selected], dtype=np.float32),
        np.asarray(train_geometry[selected], dtype=np.float32),
    )
    coefficients, mean, scale = _fit_linear_logistic(
        fit_features, np.asarray(train.array("labels")[selected], dtype=np.uint8)
    )
    val_scores = _cache_probabilities(
        val, val_geometry, coefficients, mean, scale, batch_size=args.batch_size
    )
    val_metrics, threshold = _metrics(val, _clip_scores(val, val_scores))
    train_scores = _cache_probabilities(
        train, train_geometry, coefficients, mean, scale, batch_size=args.batch_size
    )
    train_clip_scores = _clip_scores(train, train_scores)
    report = {
        "protocol": "pifr_two_feature_linear_baseline_v1",
        "features": ["nose_to_mid_hip_tilt", "log_bbox_width_height_ratio"],
        "classifier": "L2-regularized weighted linear logistic regression (no neural network)",
        "coefficients": coefficients.tolist(),
        "standardization": {"mean": mean.tolist(), "scale": scale.tolist()},
        "train_fit_samples": int(selected.size),
        "val": val_metrics,
        "hard_negative": _hard_negative_report(cohort, train_clip_scores, threshold),
        "train_cache_signature_sha256": train.metadata.get("signature_sha256"),
        "val_cache_signature_sha256": val.metadata.get("signature_sha256"),
        "hard_cohort_sha256": _sha256_file(args.hard_cohort),
    }
    _atomic_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
