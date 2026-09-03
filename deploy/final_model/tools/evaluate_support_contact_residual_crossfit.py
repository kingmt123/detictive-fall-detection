"""Cross-fit a low-dimensional support/contact residual over seven-head logits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from models.tcn_dataset import load_window_cache
from tools.audit_paired_clip_predictions import (
    metrics_from_arrays,
    paired_group_bootstrap,
)
from tools.build_tcn_multistream_sidecar import load_sidecar
from tools.evaluate_gmdcsa24_seven_head import _atomic_npz
from tools.evaluate_ofsyn_frozen_temporal_contrast_adapter import gate_decision
from tools.evaluate_rmpts_diagnostic import _logit, load_member_matrix
from tools.train_tcn import _atomic_json, _sha256_file

PROTOCOL = "edgefall_support_contact_residual_crossfit_v1"
RESIDUAL_L2 = 10.0
FEATURE_NAMES = (
    "log_sequence_length",
    "observed_rate",
    "joint_visibility_mean",
    "joint_visibility_final",
    "bbox_cy_end_minus_start",
    "bbox_cy_range",
    "bbox_cy_max_down_step",
    "bbox_cy_max_up_step",
    "bbox_cy_final_stillness",
    "bbox_bottom_end_minus_start",
    "bbox_bottom_floor_gap",
    "bbox_bottom_contact_fraction",
    "bbox_height_end_minus_start",
    "bbox_height_range",
    "aspect_end_minus_start",
    "aspect_range",
    "hip_y_end_minus_start",
    "hip_y_range",
    "hip_y_max_down_step",
    "hip_y_final_stillness",
    "ankle_to_bottom_median",
    "ankle_to_bottom_final",
    "wrist_to_bottom_q10",
    "wrist_to_bottom_final",
    "low_wrist_support_fraction",
    "hand_foot_distance_q10",
    "hand_foot_contact_fraction",
    "torso_vertical_span_median",
    "torso_vertical_span_final",
    "max_descent_time_fraction",
    "post_descent_stillness",
    "descent_to_stillness_ratio",
)


def _finite(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    return result[np.isfinite(result)]


def _stat(values: np.ndarray, mode: str, default: float = 0.0) -> float:
    finite = _finite(values)
    if finite.size == 0:
        return default
    if mode == "mean":
        return float(finite.mean())
    if mode == "median":
        return float(np.median(finite))
    if mode == "q10":
        return float(np.quantile(finite, 0.1))
    if mode == "range":
        return float(finite.max() - finite.min())
    raise ValueError(f"unknown statistic: {mode}")


def _endpoint_change(values: np.ndarray) -> float:
    finite = _finite(values)
    return float(finite[-1] - finite[0]) if finite.size >= 2 else 0.0


def _final_mean(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    tail = array[max(0, array.size * 3 // 4) :]
    return _stat(tail, "mean")


def _stillness(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    tail = array[max(0, array.size * 3 // 4) :]
    valid = np.isfinite(tail[1:]) & np.isfinite(tail[:-1])
    return float(np.abs(np.diff(tail)[valid]).mean()) if valid.any() else 0.0


def _joint_mean(sequence: np.ndarray, indices: tuple[int, ...]) -> np.ndarray:
    selected = sequence[:, indices]
    valid = selected[..., 2] >= 0.1
    values = np.where(valid, selected[..., 1], np.nan)
    counts = np.sum(np.isfinite(values), axis=1)
    totals = np.nansum(values, axis=1)
    return np.divide(
        totals,
        counts,
        out=np.full(sequence.shape[0], np.nan, dtype=np.float64),
        where=counts > 0,
    )


def support_contact_features(pose: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """Build fixed causal-sequence contact summaries from endpoint frames."""
    joints = np.asarray(pose, dtype=np.float64)
    boxes = np.asarray(bbox, dtype=np.float64)
    if joints.ndim != 3 or joints.shape[1:] != (17, 3):
        raise ValueError("support/contact pose shape invalid")
    if boxes.shape != (joints.shape[0], 6) or joints.shape[0] < 1:
        raise ValueError("support/contact bbox shape invalid")
    observed = boxes[:, 5] > 0.5
    if not observed.any():
        return np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    joints = joints[observed]
    boxes = boxes[observed]
    cy = boxes[:, 1]
    height = np.maximum(boxes[:, 3], 1e-6)
    bottom = cy + 0.5 * height
    aspect = boxes[:, 4]
    visibility = (joints[..., 2] >= 0.1).mean(axis=1)
    wrists = _joint_mean(joints, (9, 10))
    hips = _joint_mean(joints, (11, 12))
    ankles = _joint_mean(joints, (15, 16))
    shoulders = _joint_mean(joints, (5, 6))
    hip_absolute = cy + hips * height
    wrist_bottom = 0.5 - wrists
    ankle_bottom = 0.5 - ankles
    hand_foot = np.abs(wrists - ankles)
    torso_span = hips - shoulders
    low_wrist = np.isfinite(wrists) & (wrists >= 0.25)
    hand_contact = np.isfinite(hand_foot) & (hand_foot <= 0.18)
    floor_level = float(np.quantile(bottom, 0.95))
    floor_contact = bottom >= floor_level - 0.02
    cy_step = np.diff(cy)
    hip_step = np.diff(hip_absolute)
    descent_series = np.where(np.isfinite(hip_step), hip_step, -np.inf)
    if descent_series.size and np.isfinite(descent_series).any():
        peak = int(np.argmax(descent_series)) + 1
        max_descent = float(np.max(descent_series))
    else:
        peak = 0
        max_descent = 0.0
    post = hip_absolute[peak:]
    post_valid = np.isfinite(post[1:]) & np.isfinite(post[:-1])
    post_stillness = (
        float(np.abs(np.diff(post)[post_valid]).mean()) if post_valid.any() else 0.0
    )
    features = np.asarray(
        [
            np.log1p(joints.shape[0]),
            float(observed.mean()),
            float(visibility.mean()),
            _final_mean(visibility),
            _endpoint_change(cy),
            _stat(cy, "range"),
            float(max(cy_step.max(initial=0.0), 0.0)),
            float(max((-cy_step).max(initial=0.0), 0.0)),
            _stillness(cy),
            _endpoint_change(bottom),
            float(floor_level - _final_mean(bottom)),
            float(floor_contact.mean()),
            _endpoint_change(height),
            _stat(height, "range"),
            _endpoint_change(aspect),
            _stat(aspect, "range"),
            _endpoint_change(hip_absolute),
            _stat(hip_absolute, "range"),
            max_descent,
            _stillness(hip_absolute),
            _stat(ankle_bottom, "median", 0.5),
            _final_mean(ankle_bottom),
            _stat(wrist_bottom, "q10", 0.5),
            _final_mean(wrist_bottom),
            float(low_wrist.mean()),
            _stat(hand_foot, "q10", 1.0),
            float(hand_contact.mean()),
            _stat(torso_span, "median"),
            _final_mean(torso_span),
            float(peak / max(joints.shape[0] - 1, 1)),
            post_stillness,
            float(max_descent / max(post_stillness, 1e-4)),
        ],
        dtype=np.float64,
    )
    if features.shape != (len(FEATURE_NAMES),) or not np.isfinite(features).all():
        raise ValueError("support/contact feature values invalid")
    return features


def fit_residual(
    features: np.ndarray,
    labels: np.ndarray,
    baseline_logits: np.ndarray,
    *,
    l2: float = RESIDUAL_L2,
) -> dict[str, np.ndarray]:
    """Fit a deterministic zero-initialized signed-quadratic logit residual."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    anchor = np.asarray(baseline_logits, dtype=np.float64)
    if x.ndim != 2 or y.shape != (x.shape[0],) or anchor.shape != y.shape:
        raise ValueError("support/contact residual training shape invalid")
    positives = float(y.sum())
    if positives <= 0.0 or positives >= y.size:
        raise ValueError("support/contact residual needs both classes")
    mean = x.mean(axis=0)
    scale = x.std(axis=0).clip(min=1e-6)
    z = (x - mean) / scale
    design = np.column_stack((z, z * np.abs(z), np.ones(x.shape[0])))
    design_t = torch.from_numpy(design)
    labels_t = torch.from_numpy(y)
    anchor_t = torch.from_numpy(anchor)
    sample_weight = np.where(y > 0.5, (y.size - positives) / positives, 1.0)
    weight_t = torch.from_numpy(sample_weight)
    coefficients = torch.zeros(design.shape[1], dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [coefficients], max_iter=100, tolerance_grad=1e-10, line_search_fn="strong_wolfe"
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        logits = anchor_t + design_t @ coefficients
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, labels_t, weight=weight_t, reduction="mean"
        )
        loss = loss + l2 * coefficients[:-1].square().sum() / x.shape[0]
        loss.backward()
        return loss

    optimizer.step(closure)
    return {
        "mean": mean,
        "scale": scale,
        "coefficients": coefficients.detach().numpy(),
    }


def apply_residual(
    features: np.ndarray,
    baseline_logits: np.ndarray,
    fitted: dict[str, np.ndarray],
) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    anchor = np.asarray(baseline_logits, dtype=np.float64)
    z = (x - fitted["mean"]) / fitted["scale"]
    design = np.column_stack((z, z * np.abs(z), np.ones(x.shape[0])))
    logits = anchor + design @ fitted["coefficients"]
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))


def extract_all_features(cache: Any, sidecar: np.ndarray) -> np.ndarray:
    pose_memmap = cache.array("features")
    rows = []
    for clip in cache.metadata["clips"]:
        start = int(clip["window_start"])
        count = int(clip["window_count"])
        if count <= 0:
            raise ValueError("support/contact clip has no windows")
        rows.append(
            support_contact_features(
                np.asarray(pose_memmap[start : start + count, -1]),
                np.asarray(sidecar[start : start + count, -1]),
            )
        )
    result = np.stack(rows)
    if result.shape != (len(rows), len(FEATURE_NAMES)):
        raise ValueError("support/contact full feature shape invalid")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--member-report", type=Path, required=True)
    parser.add_argument("--dense-oof", type=Path, required=True)
    parser.add_argument("--dense-cache", type=Path, required=True)
    parser.add_argument("--dense-sidecar", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260901)
    args = parser.parse_args()
    if args.output.exists() or args.predictions.exists():
        raise FileExistsError("support/contact residual output already exists")
    if args.bootstrap_replicates < 100:
        raise ValueError("support/contact bootstrap count invalid")
    lock = json.loads(args.protocol_lock.read_text(encoding="utf-8"))
    if (
        lock.get("protocol") != PROTOCOL
        or lock.get("status") != "locked_before_feature_extraction"
        or lock.get("residual_l2") != RESIDUAL_L2
        or lock.get("feature_names") != list(FEATURE_NAMES)
    ):
        raise ValueError("support/contact protocol lock invalid")
    for path_text, expected in lock.get("artifact_sha256", {}).items():
        path = Path(path_text)
        if not path.is_file() or _sha256_file(path) != expected:
            raise ValueError(f"support/contact artifact hash mismatch: {path}")
    with np.load(args.dense_oof, allow_pickle=False) as payload:
        clip_ids = np.asarray(payload["clip_id"]).astype(str)
        labels = np.asarray(payload["label"], dtype=np.uint8)
        folds = np.asarray(payload["fold"], dtype=np.int64)
        dense_score = np.asarray(payload["dense48_score"], dtype=np.float64)
    if clip_ids.shape != (9600,) or sorted(set(folds.tolist())) != list(range(5)):
        raise ValueError("support/contact OOF identity invalid")
    short_logits, _ = load_member_matrix(args.member_report, clip_ids, labels)
    member_logits = np.column_stack((short_logits, _logit(dense_score)))
    baseline_logits = member_logits.mean(axis=1)
    baseline = 1.0 / (1.0 + np.exp(-np.clip(baseline_logits, -30.0, 30.0)))
    cache = load_window_cache(args.dense_cache, verify_hashes=True)
    sidecar = load_sidecar(args.dense_sidecar, cache)
    cache_ids = [str(row["clip_id"]) for row in cache.metadata["clips"]]
    if set(cache_ids) != set(clip_ids.tolist()):
        raise ValueError("support/contact cache universe mismatch")
    cache_features = extract_all_features(cache, sidecar)
    cache_index = {clip_id: index for index, clip_id in enumerate(cache_ids)}
    features = cache_features[[cache_index[clip_id] for clip_id in clip_ids]]
    challenger = np.empty(labels.shape, dtype=np.float64)
    fold_models = []
    for fold in range(5):
        train = folds != fold
        held = ~train
        fitted = fit_residual(features[train], labels[train], baseline_logits[train])
        challenger[held] = apply_residual(features[held], baseline_logits[held], fitted)
        fold_models.append(
            {
                "fold": fold,
                "train_clips": int(train.sum()),
                "heldout_clips": int(held.sum()),
                "mean": fitted["mean"].tolist(),
                "scale": fitted["scale"].tolist(),
                "coefficients": fitted["coefficients"].tolist(),
            }
        )
        print(json.dumps({"stage": "crossfit", "fold": fold, "heldout": int(held.sum())}), flush=True)
    names = ("clip_map_percent", "clip_p_at_r90", "clip_p_at_r95")
    baseline_metrics = metrics_from_arrays(labels, baseline)
    challenger_metrics = metrics_from_arrays(labels, challenger)
    delta = {name: challenger_metrics[name] - baseline_metrics[name] for name in names}
    bootstrap = paired_group_bootstrap(
        clip_ids,
        labels,
        baseline,
        challenger,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    fold_rows = []
    for fold in range(5):
        held = folds == fold
        first = metrics_from_arrays(labels[held], baseline[held])
        second = metrics_from_arrays(labels[held], challenger[held])
        fold_rows.append(
            {
                "fold": fold,
                "delta": {name: second[name] - first[name] for name in names},
            }
        )
    gates = gate_decision(
        delta,
        bootstrap,
        [row["delta"]["clip_map_percent"] for row in fold_rows],
    )
    _atomic_npz(
        args.predictions,
        clip_id=clip_ids,
        label=labels,
        fold=folds,
        feature_names=np.asarray(FEATURE_NAMES),
        features=features.astype(np.float32),
        incumbent_score=baseline.astype(np.float32),
        challenger_score=challenger.astype(np.float32),
    )
    report = {
        "protocol": PROTOCOL,
        "qualification": "strict_template_grouped_crossfit_low_dimensional_residual",
        "feature_names": list(FEATURE_NAMES),
        "residual_l2": RESIDUAL_L2,
        "comparison": {
            "baseline": baseline_metrics,
            "challenger": challenger_metrics,
            "delta": delta,
            "paired_template_group_bootstrap_delta_95ci": bootstrap,
        },
        "folds": fold_rows,
        "fold_models": fold_models,
        "gates": gates,
        "decision": "promote" if gates["passes_all"] else "stop_and_switch_direction",
        "input_sha256": {
            "protocol_lock": _sha256_file(args.protocol_lock),
            "member_report": _sha256_file(args.member_report),
            "dense_oof": _sha256_file(args.dense_oof),
        },
        "predictions_sha256": _sha256_file(args.predictions),
        "test_accessed": False,
        "upfall_accessed": False,
        "limitations": [
            "support/contact variables are deterministic pose heuristics, not human labels",
            "the residual is cross-fit on OF-Syn development OOF",
            "a promoted route would still require final full-data residual fitting",
        ],
    }
    _atomic_json(args.output, report)
    print(json.dumps({"delta": delta, "gates": gates}, sort_keys=True))


if __name__ == "__main__":
    main()
