"""Cross-fit a tiny track-quality gate between primary and secondary actors."""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import torch

from models.edgefall_ensemble import rmpts_fuse_numpy_probabilities
from tools.audit_paired_clip_predictions import (
    metrics_from_arrays,
    paired_group_bootstrap,
)
from tools.evaluate_probability_rmpts_calibration_crossfit import calibration_crossfit
from tools.evaluate_rmpts_diagnostic import _logit, load_member_matrix
from tools.train_tcn import _atomic_json, _sha256_file

TRACK_FEATURE_NAMES = (
    "primary_mean_joint_confidence",
    "second_mean_joint_confidence",
    "second_to_primary_window_ratio",
    "second_observed_fraction",
)
SCORE_FEATURE_NAMES = (
    "alternate_minus_primary_mean_logit",
    "alternate_minus_primary_logit_std",
)
FIXED_DELTA = Fraction(1, 7)
GATE_L2 = 1.0


def _rmpts(short: np.ndarray, dense: np.ndarray) -> np.ndarray:
    return rmpts_fuse_numpy_probabilities(
        short[:, (0, 2, 4)],
        short[:, (1, 3, 5)],
        dense,
        delta=FIXED_DELTA,
    )


def _logit_array(probabilities: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    return np.log(values / (1.0 - values))


def _load_replay(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol") != "edgefall_top2_actor_confidence_trigger_error_canary_v1":
        raise ValueError("top-2 actor replay 协议不匹配")
    rows = {str(row["clip_id"]): row for row in payload.get("rows", [])}
    if len(rows) != len(payload.get("rows", [])):
        raise ValueError("top-2 actor replay clip 重复")
    return rows


def _load_track_features(path: Path) -> dict[str, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol") != "edgefall_top2_actor_confidence_trigger_full_train_audit_v1":
        raise ValueError("top-2 track audit 协议不匹配")
    result = {}
    for row in payload.get("rows", []):
        audit = row["track_audit"]
        primary = audit["primary"]
        second = audit["second"]
        result[str(row["clip_id"])] = np.asarray(
            [
                primary["mean_joint_confidence"],
                second["mean_joint_confidence"],
                audit["second_to_primary_window_ratio"],
                min(float(second["observed_frames"]) / 81.0, 1.0),
            ],
            dtype=np.float64,
        )
    return result


def fit_gate(
    features: np.ndarray,
    labels: np.ndarray,
    baseline_scores: np.ndarray,
    alternate_deltas: np.ndarray,
    *,
    l2: float = GATE_L2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a class-balanced sigmoid actor selector with fixed L2."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    baseline = np.asarray(baseline_scores, dtype=np.float64)
    delta = np.asarray(alternate_deltas, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] < 1:
        raise ValueError("actor gate feature shape 不匹配")
    if y.shape != (x.shape[0],) or baseline.shape != y.shape or delta.shape != y.shape:
        raise ValueError("actor gate arrays shape 不匹配")
    positives = float(y.sum())
    if positives <= 0.0 or positives >= y.size:
        raise ValueError("actor gate 训练需要正负样本")
    mean = x.mean(axis=0)
    scale = x.std(axis=0).clip(min=1e-6)
    design = np.column_stack(((x - mean) / scale, np.ones(x.shape[0])))
    design_t = torch.from_numpy(design)
    labels_t = torch.from_numpy(y)
    baseline_t = torch.from_numpy(baseline)
    delta_t = torch.from_numpy(delta)
    weights_t = torch.from_numpy(np.where(y > 0.5, (y.size - positives) / positives, 1.0))
    coefficients = torch.zeros(design.shape[1], dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [coefficients], max_iter=100, tolerance_grad=1e-10, line_search_fn="strong_wolfe"
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        alpha = torch.sigmoid(design_t @ coefficients)
        probability = (baseline_t + alpha * delta_t).clamp(1e-7, 1.0 - 1e-7)
        bce = -weights_t * (
            labels_t * torch.log(probability)
            + (1.0 - labels_t) * torch.log1p(-probability)
        )
        loss = bce.mean() + l2 * coefficients[:-1].square().sum() / x.shape[0]
        loss.backward()
        return loss

    optimizer.step(closure)
    return coefficients.detach().numpy(), mean, scale


def apply_gate(
    features: np.ndarray,
    coefficients: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    design = np.column_stack(((features - mean) / scale, np.ones(features.shape[0])))
    logits = np.clip(design @ coefficients, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-logits))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--member-report", type=Path, required=True)
    parser.add_argument("--dense-oof", type=Path, required=True)
    parser.add_argument("--replay-report", type=Path, required=True)
    parser.add_argument("--track-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--feature-set",
        choices=("track_only", "track_score_consensus"),
        default="track_only",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("actor set gate 输出已存在，拒绝覆盖")
    with np.load(args.dense_oof, allow_pickle=False) as payload:
        clip_ids = np.asarray(payload["clip_id"]).astype(str)
        labels = np.asarray(payload["label"], dtype=np.uint8)
        folds = np.asarray(payload["fold"], dtype=np.int64)
        dense = np.asarray(payload["dense48_score"], dtype=np.float64)
    if sorted(set(folds.tolist())) != list(range(5)):
        raise ValueError("actor set gate 需要完整五折 OOF")
    short_logits, _ = load_member_matrix(args.member_report, clip_ids, labels)
    short = 1.0 / (1.0 + np.exp(-np.clip(short_logits, -30.0, 30.0)))
    replay = _load_replay(args.replay_report)
    track_features = _load_track_features(args.track_audit)
    if set(replay) != set(track_features):
        raise ValueError("actor replay 与 track audit clip 集合不一致")
    index = {clip_id: offset for offset, clip_id in enumerate(clip_ids)}
    if not set(replay).issubset(index):
        raise ValueError("actor replay clip 不在 dense OOF universe")
    triggered_ids = np.asarray(sorted(replay))
    triggered_indices = np.asarray([index[item] for item in triggered_ids], dtype=np.int64)
    features = np.stack([track_features[item] for item in triggered_ids])
    alternate = np.asarray(
        [
            [member["alternate_actor_score"] for member in replay[item]["members"]]
            for item in triggered_ids
        ],
        dtype=np.float64,
    )
    feature_names = list(TRACK_FEATURE_NAMES)
    if args.feature_set == "track_score_consensus":
        primary_triggered = short[triggered_indices]
        primary_logits = _logit_array(primary_triggered)
        alternate_logits = _logit_array(alternate)
        score_features = np.column_stack(
            (
                alternate_logits.mean(axis=1) - primary_logits.mean(axis=1),
                alternate_logits.std(axis=1) - primary_logits.std(axis=1),
            )
        )
        features = np.column_stack((features, score_features))
        feature_names.extend(SCORE_FEATURE_NAMES)
    baseline = _rmpts(short, dense)
    all_alternate_short = short.copy()
    all_alternate_short[triggered_indices] = alternate
    all_alternate = _rmpts(all_alternate_short, dense)
    triggered_baseline = baseline[triggered_indices]
    triggered_delta = all_alternate[triggered_indices] - triggered_baseline
    candidate_short = short.copy()
    fold_details = []
    for outer_fold in range(5):
        inner = folds[triggered_indices] != outer_fold
        held = ~inner
        coefficients, mean, scale = fit_gate(
            features[inner],
            labels[triggered_indices][inner],
            triggered_baseline[inner],
            triggered_delta[inner],
        )
        alpha = apply_gate(features[held], coefficients, mean, scale)
        held_indices = triggered_indices[held]
        candidate_short[held_indices] = (
            short[held_indices]
            + alpha[:, None] * (alternate[held] - short[held_indices])
        )
        fold_details.append(
            {
                "outer_fold": outer_fold,
                "train_triggered": int(inner.sum()),
                "held_triggered": int(held.sum()),
                "coefficients": coefficients.tolist(),
                "feature_mean": mean.tolist(),
                "feature_scale": scale.tolist(),
                "held_alpha_mean": float(alpha.mean()),
                "held_alpha_min": float(alpha.min()),
                "held_alpha_max": float(alpha.max()),
            }
        )
    candidate = _rmpts(candidate_short, dense)
    baseline_calibrated, baseline_temperature_folds = calibration_crossfit(
        labels, folds, short_logits, _logit(dense), "global"
    )
    candidate_calibrated, candidate_temperature_folds = calibration_crossfit(
        labels, folds, _logit_array(candidate_short), _logit(dense), "global"
    )
    metric_names = ("clip_map_percent", "clip_p_at_r90", "clip_p_at_r95")

    def comparison(first: np.ndarray, second: np.ndarray, seed: int) -> dict[str, Any]:
        first_metrics = metrics_from_arrays(labels, first)
        second_metrics = metrics_from_arrays(labels, second)
        return {
            "baseline": first_metrics,
            "challenger": second_metrics,
            "delta": {
                name: second_metrics[name] - first_metrics[name] for name in metric_names
            },
            "paired_template_group_bootstrap_delta_95ci": paired_group_bootstrap(
                clip_ids,
                labels,
                first,
                second,
                replicates=args.bootstrap_replicates,
                seed=seed,
            ),
        }

    report = {
        "protocol": "edgefall_top2_actor_track_quality_set_gate_crossfit_v1",
        "qualification": "development_meta_oof_not_nested_base_training",
        "triggered_clips": int(triggered_ids.size),
        "feature_set": args.feature_set,
        "features": feature_names,
        "gate_l2": GATE_L2,
        "fixed_delta": "1/7",
        "uncalibrated_probability_rmpts": comparison(baseline, candidate, 20260829),
        "frozen_probability_rmpts_1_7_shared_temperature": {
            **comparison(baseline_calibrated, candidate_calibrated, 20260830),
            "baseline_temperature_folds": baseline_temperature_folds,
            "challenger_temperature_folds": candidate_temperature_folds,
        },
        "all_alternate_diagnostic": comparison(baseline, all_alternate, 20260831),
        "gate_folds": fold_details,
        "input_sha256": {
            "member_report": _sha256_file(args.member_report),
            "dense_oof": _sha256_file(args.dense_oof),
            "replay_report": _sha256_file(args.replay_report),
            "track_audit": _sha256_file(args.track_audit),
        },
        "test_accessed": False,
    }
    _atomic_json(args.output, report)
    print(
        json.dumps(
            report["frozen_probability_rmpts_1_7_shared_temperature"]["delta"],
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
