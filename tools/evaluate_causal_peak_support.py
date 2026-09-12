"""Evaluate max versus causal 2-of-3 peak support for fixed model routes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from models.clip_aggregator import (
    aggregate_window_logits_peak_support,
    aggregate_window_scores_max,
)
from models.tcn_dataset import load_window_cache
from tools.build_tcn_multistream_sidecar import load_sidecar
from tools.evaluate_clip_fusion_order import _metrics
from tools.evaluate_hard_negative_cohort import _model_from_run
from tools.train_tcn import _append_sidecar, _atomic_json, _materialize, predict_logits


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(values, -40, 40)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--weight", type=float, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("输出已存在，拒绝覆盖")
    if not (len(args.run) == len(args.checkpoint) == len(args.weight)):
        raise ValueError("route参数数量不匹配")
    weights = np.asarray(args.weight, dtype=np.float64)
    if np.any(weights < 0) or not np.isclose(weights.sum(), 1.0, atol=1e-8):
        raise ValueError("weights必须非负且和为1")
    cache = load_window_cache(args.cache, verify_hashes=True)
    if cache.metadata.get("split") == "test":
        raise ValueError("禁止使用test")
    features, _, clip_indices = _materialize(cache, None)
    features = _append_sidecar(features, load_sidecar(args.sidecar, cache), None)
    route_logits = []
    routes = []
    device = torch.device(args.device)
    for run_path, checkpoint in zip(args.run, args.checkpoint, strict=True):
        run = json.loads(run_path.read_text(encoding="utf-8"))
        model, _, _ = _model_from_run(
            model_kind="multiscale_multistream_tcn",
            run=run,
            checkpoint=checkpoint,
            device=device,
        )
        route_logits.append(
            predict_logits(model, features, device=device, batch_size=args.batch_size)
        )
        routes.append(
            {
                "run_sha256": hashlib.sha256(run_path.read_bytes()).hexdigest(),
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            }
        )
        del model
    route_logits_array = np.stack(route_logits)
    max_clip = np.stack(
        [
            aggregate_window_scores_max(route, clip_indices, len(cache.metadata["clips"]))
            for route in route_logits_array
        ]
    )
    support_clip = np.stack(
        [
            aggregate_window_logits_peak_support(
                route, clip_indices, len(cache.metadata["clips"]), support=2, horizon=3
            )
            for route in route_logits_array
        ]
    )
    max_scores = _sigmoid(np.sum(max_clip * weights[:, None], axis=0))
    support_scores = _sigmoid(np.sum(support_clip * weights[:, None], axis=0))
    report = {
        "protocol": "causal_2_of_3_peak_support_v1",
        "split": cache.metadata["split"],
        "routes": routes,
        "weights": args.weight,
        "max": _metrics(cache.metadata["clips"], max_scores),
        "causal_2_of_3": _metrics(cache.metadata["clips"], support_scores),
        "score_difference": {
            "changed_clips": int(np.count_nonzero(max_scores != support_scores)),
            "mean": float(np.mean(support_scores - max_scores)),
            "max_abs": float(np.max(np.abs(support_scores - max_scores))),
        },
        "test_accessed": False,
    }
    _atomic_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
