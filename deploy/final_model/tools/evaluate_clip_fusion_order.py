"""Compare fixed window-first and clip-first fusion on a non-test split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from eval.metrics import competition_map
from models.clip_aggregator import aggregate_window_scores_max
from models.tcn_dataset import load_window_cache
from tools.build_tcn_multistream_sidecar import load_sidecar
from tools.evaluate_hard_negative_cohort import _model_from_run
from tools.train_tcn import _append_sidecar, _atomic_json, _materialize, predict_logits


def compare_fusion_orders(
    route_window_logits: np.ndarray,
    clip_indices: np.ndarray,
    clip_count: int,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return window-first and clip-first probabilities for aligned routes."""
    logits = np.asarray(route_window_logits, dtype=np.float32)
    route_weights = np.asarray(weights, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[1] != np.asarray(clip_indices).size:
        raise ValueError("route_window_logits 必须为 (routes, windows)")
    if route_weights.shape != (logits.shape[0],):
        raise ValueError("weights 与 route 数量不匹配")
    if (
        np.any(route_weights < 0.0)
        or not np.isfinite(logits).all()
        or not np.isclose(route_weights.sum(), 1.0, rtol=0.0, atol=1e-8)
    ):
        raise ValueError("融合 logits/weights 无效")

    window_fused = np.sum(logits.astype(np.float64) * route_weights[:, None], axis=0)
    window_first_logits = aggregate_window_scores_max(
        window_fused, clip_indices, clip_count, require_all=True
    )
    route_clip_logits = np.stack(
        [
            aggregate_window_scores_max(
                route, clip_indices, clip_count, require_all=True
            )
            for route in logits
        ]
    )
    clip_first_logits = np.sum(
        route_clip_logits.astype(np.float64) * route_weights[:, None], axis=0
    )
    sigmoid = lambda values: 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))
    return (
        sigmoid(window_first_logits).astype(np.float32),
        sigmoid(clip_first_logits).astype(np.float32),
    )


def _metrics(clips: list[dict[str, Any]], scores: np.ndarray) -> dict[str, float]:
    labels = {str(clip["clip_id"]): bool(clip["has_fall"]) for clip in clips}
    predictions = {
        str(clip["clip_id"]): float(scores[index])
        for index, clip in enumerate(clips)
    }
    result = competition_map(labels, predictions, mode="clip")
    return {
        "clip_map": float(result["map"]),
        "clip_map_percent": float(result["map_percent"]),
        "clip_p_at_r90": float(result["p_at_r90"]),
        "clip_p_at_r95": float(result["p_at_r95"]),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--weight", type=float, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("输出已存在，拒绝覆盖")
    if not (len(args.run) == len(args.checkpoint) == len(args.weight)):
        raise ValueError("run/checkpoint/weight 数量必须一致")
    cache = load_window_cache(args.cache)
    if cache.metadata.get("split") == "test":
        raise ValueError("融合顺序比较禁止使用 test")
    features, _, clip_indices = _materialize(cache, None)
    features = _append_sidecar(features, load_sidecar(args.sidecar, cache), None)
    device = torch.device(args.device)
    route_logits: list[np.ndarray] = []
    routes = []
    for run_path, checkpoint in zip(args.run, args.checkpoint, strict=True):
        run = json.loads(run_path.read_text(encoding="utf-8"))
        model, _, _ = _model_from_run(
            model_kind="multiscale_multistream_tcn",
            run=run,
            checkpoint=checkpoint,
            device=device,
        )
        route_logits.append(
            predict_logits(
                model,
                features,
                device=device,
                batch_size=args.batch_size,
            )
        )
        routes.append(
            {
                "run_sha256": _sha256(run_path),
                "checkpoint_sha256": _sha256(checkpoint),
            }
        )
        del model
    window_first, clip_first = compare_fusion_orders(
        np.stack(route_logits),
        clip_indices,
        len(cache.metadata["clips"]),
        np.asarray(args.weight),
    )
    report = {
        "protocol": "fixed_logit_fusion_order_comparison_v1",
        "split": cache.metadata["split"],
        "cache_signature_sha256": cache.metadata["signature_sha256"],
        "weights": args.weight,
        "routes": routes,
        "window_first": _metrics(cache.metadata["clips"], window_first),
        "clip_first": _metrics(cache.metadata["clips"], clip_first),
        "score_difference": {
            "mean": float(np.mean(window_first - clip_first)),
            "max_abs": float(np.max(np.abs(window_first - clip_first))),
            "changed_clips": int(np.count_nonzero(window_first != clip_first)),
        },
        "test_accessed": False,
    }
    _atomic_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
