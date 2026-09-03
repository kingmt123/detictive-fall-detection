"""Freeze one CAUCA-trained temporal-contrast adapter before UP-Fall inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools.evaluate_caucafall_pca_residual_adapter import _adapter_scores, _sigmoid
from tools.evaluate_caucafall_temporal_contrast_adapter import (
    temporal_contrast_features,
)
from tools.evaluate_gmdcsa24_seven_head import _atomic_npz
from tools.train_tcn import _atomic_json, _sha256_file

FIXED_CANDIDATE = "pca:8:ridge:1"


def score_frozen_adapter(
    short_embedding: np.ndarray,
    dense_embedding: np.ndarray,
    member_logits: np.ndarray,
    *,
    mean: np.ndarray,
    scale: np.ndarray,
    basis: np.ndarray,
    projected_scale: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:
    """Apply the exact frozen no-bias adapter to a new domain."""
    features = temporal_contrast_features(short_embedding, dense_embedding)
    logits = np.asarray(member_logits, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    basis = np.asarray(basis, dtype=np.float64)
    projected_scale = np.asarray(projected_scale, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    if (
        logits.shape != (features.shape[0], 7)
        or mean.shape != (features.shape[1],)
        or scale.shape != mean.shape
        or basis.shape != (8, features.shape[1])
        or projected_scale.shape != (8,)
        or beta.shape != (8,)
        or np.any(scale <= 0.0)
        or np.any(projected_scale <= 0.0)
    ):
        raise ValueError("frozen temporal-contrast adapter shape 无效")
    projected = ((features - mean) / scale) @ basis.T / projected_scale
    return _sigmoid(logits.mean(axis=1) + projected @ beta)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise FileExistsError("frozen temporal-contrast adapter 输出已存在，拒绝覆盖")
    lock = json.loads(args.protocol_lock.read_text(encoding="utf-8"))
    if (
        lock.get("protocol") != "caucafall_temporal_contrast_adapter_freeze_v1"
        or lock.get("status") != "locked_before_full_development_fit"
        or lock.get("fixed_candidate") != FIXED_CANDIDATE
    ):
        raise ValueError("frozen temporal-contrast adapter protocol lock 无效")
    for path_text, expected in lock.get("artifact_sha256", {}).items():
        path = Path(path_text)
        if not path.is_file() or _sha256_file(path) != expected:
            raise ValueError(f"frozen temporal-contrast artifact hash 不匹配: {path}")
    with np.load(args.embeddings, allow_pickle=False) as payload:
        labels = np.asarray(payload["label"], dtype=np.uint8)
        logits = np.asarray(payload["member_logits"], dtype=np.float64)
        short = np.asarray(payload["short_embedding"], dtype=np.float64)
        dense = np.asarray(payload["dense_embedding"], dtype=np.float64)
    features = temporal_contrast_features(short, dense)
    if (
        labels.shape != (100,)
        or labels.sum() != 50
        or logits.shape != (100, 7)
        or features.shape != (100, 384)
    ):
        raise ValueError("frozen temporal-contrast CAUCA identity 无效")
    anchor_logits = logits.mean(axis=1)
    _, fitted = _adapter_scores(
        features,
        features,
        anchor_logits,
        anchor_logits,
        labels,
        FIXED_CANDIDATE,
    )
    assert fitted is not None
    scores = score_frozen_adapter(
        short,
        dense,
        logits,
        mean=fitted["mean"],
        scale=fitted["scale"],
        basis=fitted["basis"],
        projected_scale=fitted["projected_scale"],
        beta=fitted["beta"],
    )
    _atomic_npz(
        args.output,
        candidate=np.asarray(FIXED_CANDIDATE),
        mean=fitted["mean"].astype(np.float64),
        scale=fitted["scale"].astype(np.float64),
        basis=fitted["basis"].astype(np.float64),
        projected_scale=fitted["projected_scale"].astype(np.float64),
        beta=fitted["beta"].astype(np.float64),
    )
    report = {
        "protocol": "caucafall_temporal_contrast_adapter_freeze_v1",
        "candidate": FIXED_CANDIDATE,
        "training_data": "CAUCAFall development-only 100 clips",
        "architecture": "equal-logit anchor plus PCA(dense-short) no-bias ridge residual",
        "parameter_shapes": {
            name: list(np.asarray(fitted[name]).shape)
            for name in ("mean", "scale", "basis", "projected_scale", "beta")
        },
        "development_fit_score_range": [float(scores.min()), float(scores.max())],
        "output_sha256": _sha256_file(args.output),
        "inputs": {
            "protocol_lock_sha256": _sha256_file(args.protocol_lock),
            "embeddings_sha256": _sha256_file(args.embeddings),
        },
        "upfall_inference_run": False,
        "upfall_metrics_inspected": False,
        "test_accessed": False,
        "limitations": [
            "the exact ridge was chosen after CAUCA development results were available",
            "CAUCAFall cannot count as external confirmation",
            "the frozen adapter requires untouched-domain confirmation",
        ],
    }
    _atomic_json(args.report, report)
    print(json.dumps({"output": str(args.output), "sha256": report["output_sha256"]}))


if __name__ == "__main__":
    main()
