"""Evaluate one pre-registered two-model logit blend on a non-test split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from eval.metrics import competition_map
from models.tcn_dataset import load_window_cache
from tools.build_tcn_multistream_sidecar import load_sidecar
from tools.evaluate_hard_negative_cohort import _model_from_run
from tools.evaluate_oof_multistream_fusion import _clip_score_array, _logit
from tools.train_tcn import (
    _append_sidecar,
    _atomic_json,
    _materialize,
    predict_probabilities,
)


def fixed_logit_blend(
    primary: np.ndarray, expert: np.ndarray, *, primary_weight: float
) -> np.ndarray:
    """Blend calibrated clip scores without fitting any validation parameter."""
    return fixed_equal_expert_logit_blend(
        primary, [expert], primary_weight=primary_weight
    )


def fixed_equal_expert_logit_blend(
    primary: np.ndarray,
    experts: list[np.ndarray],
    *,
    primary_weight: float,
) -> np.ndarray:
    """Give the primary a fixed weight and divide the remainder across experts."""
    primary = np.asarray(primary, dtype=np.float64)
    expert_arrays = [np.asarray(expert, dtype=np.float64) for expert in experts]
    if not expert_arrays:
        raise ValueError("至少需要一个 expert")
    if primary.ndim != 1 or any(
        expert.shape != primary.shape for expert in expert_arrays
    ):
        raise ValueError("所有模型的 clip 分数必须为同形一维数组")
    if not 0.0 < primary_weight < 1.0:
        raise ValueError("primary_weight 必须位于 (0,1)")
    expert_weight = (1.0 - primary_weight) / len(expert_arrays)
    logits = primary_weight * _logit(primary)
    logits += expert_weight * sum(_logit(expert) for expert in expert_arrays)
    return (1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))).astype(np.float32)


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _metrics(clips: list[dict[str, Any]], scores: np.ndarray) -> dict[str, float]:
    labels = {str(clip["clip_id"]): bool(clip["has_fall"]) for clip in clips}
    predictions = {str(clip["clip_id"]): float(scores[index]) for index, clip in enumerate(clips)}
    result = competition_map(labels, predictions, mode="clip")
    return {
        "clip_map": float(result["map"]),
        "clip_map_percent": float(result["map_percent"]),
        "clip_p_at_r90": float(result["p_at_r90"]),
        "clip_p_at_r95": float(result["p_at_r95"]),
    }


def _pair_diagnostics(
    clips: list[dict[str, Any]], primary: np.ndarray, expert: np.ndarray
) -> dict[str, float]:
    labels = np.asarray([float(bool(clip["has_fall"])) for clip in clips])
    primary = np.asarray(primary, dtype=np.float64)
    expert = np.asarray(expert, dtype=np.float64)
    return {
        "score_pearson": float(np.corrcoef(primary, expert)[0, 1]),
        "error_pearson": float(
            np.corrcoef(primary - labels, expert - labels)[0, 1]
        ),
        "score_rmse": float(np.sqrt(np.mean(np.square(primary - expert)))),
    }


def _scores(
    *,
    model_kind: str,
    run_path: Path,
    checkpoint: Path,
    features: torch.Tensor,
    clip_indices: np.ndarray,
    clip_count: int,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    run = json.loads(Path(run_path).read_text(encoding="utf-8"))
    model, _, _ = _model_from_run(
        model_kind=model_kind, run=run, checkpoint=checkpoint, device=device
    )
    probabilities = predict_probabilities(model, features, device=device, batch_size=batch_size)
    return _clip_score_array(probabilities, clip_indices, clip_count)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--primary-run", type=Path, required=True)
    parser.add_argument("--primary-checkpoint", type=Path, required=True)
    parser.add_argument("--primary-kind", required=True)
    parser.add_argument("--expert-run", type=Path, required=True)
    parser.add_argument("--expert-checkpoint", type=Path, required=True)
    parser.add_argument("--expert-kind", required=True)
    parser.add_argument("--additional-expert-run", type=Path, action="append", default=[])
    parser.add_argument(
        "--additional-expert-checkpoint", type=Path, action="append", default=[]
    )
    parser.add_argument("--additional-expert-kind", action="append", default=[])
    parser.add_argument("--primary-weight", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--predictions-output",
        type=Path,
        help="可选：保存 cache-ordered 成员与融合 clip 分数，供配对审计",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("输出已存在，拒绝覆盖")
    if args.predictions_output is not None and args.predictions_output.exists():
        raise FileExistsError("预测输出已存在，拒绝覆盖")
    additional_count = len(args.additional_expert_run)
    if not (
        additional_count
        == len(args.additional_expert_checkpoint)
        == len(args.additional_expert_kind)
    ):
        raise ValueError("additional expert 的 run/checkpoint/kind 数量必须一致")
    cache = load_window_cache(args.cache)
    if cache.metadata.get("split") == "test":
        raise ValueError("固定融合禁止使用 test")
    features, _, clip_indices = _materialize(cache, None)
    features = _append_sidecar(features, load_sidecar(args.sidecar, cache), None)
    device = torch.device(args.device)
    primary = _scores(
        model_kind=args.primary_kind,
        run_path=args.primary_run,
        checkpoint=args.primary_checkpoint,
        features=features,
        clip_indices=clip_indices,
        clip_count=len(cache.metadata["clips"]),
        device=device,
        batch_size=args.batch_size,
    )
    expert = _scores(
        model_kind=args.expert_kind,
        run_path=args.expert_run,
        checkpoint=args.expert_checkpoint,
        features=features,
        clip_indices=clip_indices,
        clip_count=len(cache.metadata["clips"]),
        device=device,
        batch_size=args.batch_size,
    )
    additional_experts = [
        _scores(
            model_kind=kind,
            run_path=run,
            checkpoint=checkpoint,
            features=features,
            clip_indices=clip_indices,
            clip_count=len(cache.metadata["clips"]),
            device=device,
            batch_size=args.batch_size,
        )
        for run, checkpoint, kind in zip(
            args.additional_expert_run,
            args.additional_expert_checkpoint,
            args.additional_expert_kind,
            strict=True,
        )
    ]
    fused = fixed_equal_expert_logit_blend(
        primary, [expert, *additional_experts], primary_weight=args.primary_weight
    )
    expert_weight = (1.0 - args.primary_weight) / (1 + additional_count)
    report = {
        "protocol": "fixed_pre_registered_logit_blend_v1",
        "split": cache.metadata["split"],
        "cache_signature_sha256": cache.metadata["signature_sha256"],
        "primary": {
            "kind": args.primary_kind,
            "run_sha256": _sha256(args.primary_run),
            "checkpoint_sha256": _sha256(args.primary_checkpoint),
            "weight": args.primary_weight,
        },
        "expert": {
            "kind": args.expert_kind,
            "run_sha256": _sha256(args.expert_run),
            "checkpoint_sha256": _sha256(args.expert_checkpoint),
            "weight": expert_weight,
        },
        "additional_experts": [
            {
                "kind": kind,
                "run_sha256": _sha256(run),
                "checkpoint_sha256": _sha256(checkpoint),
                "weight": expert_weight,
            }
            for run, checkpoint, kind in zip(
                args.additional_expert_run,
                args.additional_expert_checkpoint,
                args.additional_expert_kind,
                strict=True,
            )
        ],
        "primary_metrics": _metrics(cache.metadata["clips"], primary),
        "expert_metrics": _metrics(cache.metadata["clips"], expert),
        "pair_diagnostics": _pair_diagnostics(
            cache.metadata["clips"], primary, expert
        ),
        "metrics": _metrics(cache.metadata["clips"], fused),
    }
    _atomic_json(args.output, report)
    if args.predictions_output is not None:
        args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.predictions_output,
            clip_id=np.asarray(
                [str(clip["clip_id"]) for clip in cache.metadata["clips"]]
            ),
            label=np.asarray(
                [float(bool(clip["has_fall"])) for clip in cache.metadata["clips"]]
            ),
            primary_score=primary,
            expert_score=expert,
            additional_expert_scores=np.stack(additional_experts),
            score=fused,
        )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
