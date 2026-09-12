"""Audit whether deployable pose/track quality predicts Primary OOF errors.

The audit is deliberately read-only: it accepts only a train cache, its
sidecar, and complete template-grouped OOF predictions.  It never opens a
validation/test cache or invokes pose extraction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from models.tcn_dataset import WindowMemmapCache, load_window_cache
from tools.build_tcn_multistream_sidecar import load_sidecar
from tools.fit_nonnegative_oof_stacking import _load_member_oof
from tools.train_tcn import _atomic_json

QUALITY_DIRECTIONS = {
    "endpoint_joint_visibility": "low",
    "endpoint_mean_confidence": "low",
    "endpoint_track_observed": "low",
    "trailing_track_gap_frames": "high",
    "endpoint_bbox_step": "high",
}


def _trailing_gap(observed: np.ndarray) -> np.ndarray:
    """Number of consecutive unobserved frames ending each causal window."""
    observed = np.asarray(observed, dtype=bool)
    if observed.ndim != 2:
        raise ValueError("observed 必须是 (N,T)")
    return np.cumprod(~observed[:, ::-1], axis=1, dtype=np.int16).sum(axis=1)


def _clip_quality(cache: WindowMemmapCache, sidecar: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Summarise end-of-window input reliability for every clip without labels."""
    clips = cache.metadata["clips"]
    features = cache.array("features")
    values = {name: np.empty(len(clips), dtype=np.float32) for name in QUALITY_DIRECTIONS}
    for index, clip in enumerate(clips):
        start = int(clip["window_start"])
        end = start + int(clip["window_count"])
        pose = np.asarray(features[start:end], dtype=np.float32)
        geometry = np.asarray(sidecar[start:end], dtype=np.float32)
        confidence = pose[:, -1, :, 2]
        observed = geometry[..., 5] > 0.5
        values["endpoint_joint_visibility"][index] = float((confidence > 0).mean())
        values["endpoint_mean_confidence"][index] = float(confidence.mean())
        values["endpoint_track_observed"][index] = float(observed[:, -1].mean())
        values["trailing_track_gap_frames"][index] = float(_trailing_gap(observed).mean())
        center_step = np.linalg.norm(np.diff(geometry[..., :2], axis=1), axis=-1)
        valid_step = observed[:, 1:] & observed[:, :-1]
        values["endpoint_bbox_step"][index] = float(
            center_step[valid_step].mean() if valid_step.any() else 0.0
        )

    missing_count = 0
    missing_coordinate_nonzero = 0
    for start in range(0, cache.sample_count, 4096):
        pose = np.asarray(features[start : start + 4096], dtype=np.float32)
        missing = pose[..., 2] <= 0
        missing_count += int(missing.sum())
        missing_coordinate_nonzero += int(np.any(np.abs(pose[..., :2]) > 1e-7, axis=-1)[missing].sum())
    contract = {
        "missing_joint_count": missing_count,
        "missing_coordinate_nonzero_count": missing_coordinate_nonzero,
        "missing_coordinate_nonzero_rate": (
            missing_coordinate_nonzero / missing_count if missing_count else 0.0
        ),
    }
    return values, contract


def _r95_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    """Threshold with maximum precision among all points whose recall is >= .95."""
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    if not labels.any() or labels.all() or labels.shape != scores.shape:
        raise ValueError("R95 threshold 需要同长度正负 labels 和 scores")
    order = np.argsort(-scores, kind="mergesort")
    ordered_scores = scores[order]
    ordered_labels = labels[order]
    ends = np.r_[np.flatnonzero(ordered_scores[:-1] != ordered_scores[1:]), labels.size - 1]
    tp = np.cumsum(ordered_labels)[ends].astype(np.float64)
    precision = tp / (ends + 1.0)
    feasible = np.flatnonzero(tp / labels.sum() >= 0.95)
    if not feasible.size:
        return float(ordered_scores[-1])
    best = feasible[np.argmax(precision[feasible])]
    return float(ordered_scores[ends[best]])


def _enrichment(
    labels: np.ndarray,
    scores: np.ndarray,
    quality: np.ndarray,
    folds: np.ndarray,
    *,
    direction: str,
) -> dict[str, Any]:
    if direction not in {"low", "high"}:
        raise ValueError("quality direction 必须为 low/high")
    low, high = np.quantile(quality, (0.25, 0.75))
    bad = quality <= low if direction == "low" else quality >= high
    good = quality >= high if direction == "low" else quality <= low
    fold_rows = []
    for fold in sorted(set(folds.tolist())):
        held = folds == fold
        threshold = _r95_threshold(labels[held], scores[held])
        errors = (scores[held] >= threshold) != labels[held]
        held_bad, held_good = bad[held], good[held]
        bad_count, good_count = int(held_bad.sum()), int(held_good.sum())
        # Jeffreys smoothing keeps sparse R95 FN/FP strata finite.
        bad_rate = (float(errors[held_bad].sum()) + 0.5) / (bad_count + 1.0)
        good_rate = (float(errors[held_good].sum()) + 0.5) / (good_count + 1.0)
        fp_bad = held_bad & ~labels[held]
        fp_good = held_good & ~labels[held]
        fp_rate_bad = (float(errors[fp_bad].sum()) + 0.5) / (int(fp_bad.sum()) + 1.0)
        fp_rate_good = (float(errors[fp_good].sum()) + 0.5) / (int(fp_good.sum()) + 1.0)
        fold_rows.append(
            {
                "fold": int(fold),
                "r95_threshold": threshold,
                "bad_count": bad_count,
                "good_count": good_count,
                "bad_error_rate": bad_rate,
                "good_error_rate": good_rate,
                "error_rate_ratio": bad_rate / good_rate,
                "bad_fp_rate": fp_rate_bad,
                "good_fp_rate": fp_rate_good,
                "fp_rate_ratio": fp_rate_bad / fp_rate_good,
            }
        )
    stable = [
        row["error_rate_ratio"] >= 1.25 and row["bad_error_rate"] >= row["good_error_rate"] + 0.02
        for row in fold_rows
    ]
    return {
        "direction": direction,
        "global_bad_cutoff": float(low if direction == "low" else high),
        "global_good_cutoff": float(high if direction == "low" else low),
        "per_fold": fold_rows,
        "stable_enriched_fold_count": int(sum(stable)),
        "passes_p0_stability_gate": int(sum(stable)) >= 4,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--train-sidecar", type=Path, required=True)
    parser.add_argument("--primary-oof", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("quality audit 输出已存在，拒绝覆盖")
    cache = load_window_cache(args.train_cache, verify_hashes=True)
    if cache.metadata.get("split") != "train":
        raise ValueError("quality audit 只允许 train template OOF")
    clips = cache.metadata["clips"]
    scores, folds, sources = _load_member_oof(args.primary_oof, clips)
    labels = np.asarray([bool(clip["has_fall"]) for clip in clips])
    quality, contract = _clip_quality(cache, load_sidecar(args.train_sidecar, cache))
    audit = {
        name: _enrichment(labels, scores, value, folds, direction=QUALITY_DIRECTIONS[name])
        for name, value in quality.items()
    }
    report = {
        "protocol": "stage_s1_primary_template_oof_quality_audit_v1",
        "selection_split": "train template-grouped OOF only",
        "validation_uses": 0,
        "test_accessed": False,
        "primary_oof_sources": sources,
        "fold_assignment_sha256": hashlib.sha256(folds.tobytes()).hexdigest(),
        "cache_signature": cache.metadata.get("signature_sha256"),
        "input_contract": contract,
        "quality_summary": {
            name: {"mean": float(value.mean()), "p25": float(np.quantile(value, 0.25)), "p75": float(np.quantile(value, 0.75))}
            for name, value in quality.items()
        },
        "error_enrichment": audit,
        "p0_decision": {
            "passes": any(row["passes_p0_stability_gate"] for row in audit.values()),
            "passing_variables": [name for name, row in audit.items() if row["passes_p0_stability_gate"]],
            "gate": "at least one quality variable: bad-vs-good error ratio >=1.25 and +0.02 error-rate in >=4/5 OOF folds",
        },
    }
    _atomic_json(args.output, report)
    print(json.dumps(report["p0_decision"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
