"""Freeze the promoted CLIP event-semantic residual for final deployment fitting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools.evaluate_clip_event_semantic_residual import (
    FRAME_NAMES,
    PROMPTS,
    PROTOCOL,
)
from tools.evaluate_gmdcsa24_seven_head import _atomic_npz
from tools.evaluate_rmpts_diagnostic import _logit
from tools.evaluate_support_contact_residual_crossfit import RESIDUAL_L2, fit_residual
from tools.train_tcn import _atomic_json, _sha256_file

FREEZE_PROTOCOL = "edgefall_clip_event_semantic_residual_freeze_v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--crossfit-result", type=Path, required=True)
    parser.add_argument("--crossfit-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise FileExistsError("CLIP semantic residual freeze output already exists")
    lock = json.loads(args.protocol_lock.read_text(encoding="utf-8"))
    result = json.loads(args.crossfit_result.read_text(encoding="utf-8"))
    if (
        lock.get("protocol") != PROTOCOL
        or result.get("protocol") != PROTOCOL
        or result.get("decision") != "promote"
        or not result.get("gates", {}).get("passes_all")
        or result.get("input_sha256", {}).get("protocol_lock")
        != _sha256_file(args.protocol_lock)
        or result.get("predictions_sha256") != _sha256_file(args.crossfit_predictions)
    ):
        raise ValueError("CLIP semantic residual promotion evidence invalid")
    with np.load(args.crossfit_predictions, allow_pickle=False) as payload:
        clip_ids = np.asarray(payload["clip_id"]).astype(str)
        labels = np.asarray(payload["label"], dtype=np.uint8)
        folds = np.asarray(payload["fold"], dtype=np.int64)
        feature_names = np.asarray(payload["feature_names"]).astype(str)
        features = np.asarray(payload["features"], dtype=np.float64)
        incumbent = np.asarray(payload["incumbent_score"], dtype=np.float64)
    expected_names = np.asarray(
        [f"{frame}:{prompt}" for frame in FRAME_NAMES for prompt in PROMPTS]
    )
    if (
        clip_ids.shape != (9600,)
        or labels.shape != (9600,)
        or features.shape != (9600, expected_names.size)
        or not np.array_equal(feature_names, expected_names)
        or sorted(set(folds.tolist())) != list(range(5))
    ):
        raise ValueError("CLIP semantic residual freeze identity invalid")
    fitted = fit_residual(features, labels, _logit(incumbent), l2=RESIDUAL_L2)
    _atomic_npz(
        args.output,
        protocol=np.asarray(FREEZE_PROTOCOL),
        model_revision=np.asarray(lock["model_revision"]),
        frame_names=np.asarray(FRAME_NAMES),
        prompts=np.asarray(PROMPTS),
        feature_names=feature_names,
        mean=fitted["mean"],
        scale=fitted["scale"],
        coefficients=fitted["coefficients"],
        residual_l2=np.asarray(RESIDUAL_L2),
    )
    report = {
        "protocol": FREEZE_PROTOCOL,
        "source_protocol": PROTOCOL,
        "qualification": "full_oof_meta_fit_after_strict_crossfit_promotion",
        "clips": int(clip_ids.size),
        "folds_present": sorted(set(folds.tolist())),
        "model": lock["model"],
        "model_revision": lock["model_revision"],
        "frames": list(FRAME_NAMES),
        "prompts": list(PROMPTS),
        "feature_count": int(feature_names.size),
        "residual_l2": RESIDUAL_L2,
        "parameter_shapes": {
            name: list(np.asarray(fitted[name]).shape)
            for name in ("mean", "scale", "coefficients")
        },
        "input_sha256": {
            "protocol_lock": _sha256_file(args.protocol_lock),
            "crossfit_result": _sha256_file(args.crossfit_result),
            "crossfit_predictions": _sha256_file(args.crossfit_predictions),
        },
        "output_sha256": _sha256_file(args.output),
        "test_accessed": False,
        "upfall_accessed": False,
        "limitations": [
            "the frozen residual requires the frozen CLIP image encoder at deployment",
            "the residual parameters are fit on all development OOF rows after crossfit qualification",
            "external untouched confirmation remains separate from this deployment fit",
        ],
    }
    _atomic_json(args.report, report)
    print(json.dumps({"output": str(args.output), "sha256": report["output_sha256"]}))


if __name__ == "__main__":
    main()
