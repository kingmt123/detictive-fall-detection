"""Evaluate the frozen label-blind seven-head deployment on OF-Syn val."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ultralytics import YOLO

from models.tcn_dataset import load_window_cache
from tools.benchmark_edgefall_seven_head_tail import _head, _load_checkpoint
from tools.build_tcn_multistream_sidecar import load_sidecar
from tools.evaluate_gmdcsa24_seven_head import _atomic_npz
from tools.train_edgefall_f1 import (
    F1Config,
    _head_scores,
    _state_dict_sha256,
    build_skeleton_model,
    clip_embeddings,
    metric_result,
)
from tools.train_tcn import _append_sidecar, _atomic_json, _materialize, _sha256_file

PROTOCOL = "edgefall_seven_head_label_blind_deployment_v1"


def equal_logit_probability(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 7 or not np.isfinite(values).all():
        raise ValueError("七头 logits 必须为有限的 [N,7]")
    mean_logit = values.mean(axis=1)
    return 1.0 / (1.0 + np.exp(-np.clip(mean_logit, -30.0, 30.0)))


def _validate_lock(path: Path) -> dict[str, Any]:
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock.get("protocol") != PROTOCOL or lock.get("status") != (
        "locked_before_label_blind_roi_extraction"
    ):
        raise ValueError("label-blind deployment protocol lock 无效")
    artifacts = lock.get("artifact_sha256")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("protocol lock 缺少 artifact hashes")
    for name, expected in artifacts.items():
        artifact = Path(name)
        if not artifact.is_file() or _sha256_file(artifact) != expected:
            raise ValueError(f"protocol artifact hash 不匹配: {name}")
    return lock


def _features(cache_path: Path, sidecar_path: Path) -> tuple[Any, torch.Tensor, np.ndarray]:
    cache = load_window_cache(cache_path, verify_hashes=True)
    if cache.metadata.get("dataset") != "of-syn" or cache.metadata.get("split") != "val":
        raise ValueError("label-blind deployment 只接受 OF-Syn val cache")
    sidecar = load_sidecar(sidecar_path, cache)
    indices = np.arange(int(cache.metadata["sample_count"]), dtype=np.int64)
    features, _, clip_indices = _materialize(cache, indices)
    return cache, _append_sidecar(features, sidecar, indices), clip_indices


def _roi(path: Path, clip_ids: list[str]) -> torch.Tensor:
    with np.load(path, allow_pickle=False) as payload:
        ids = [str(value) for value in payload["clip_ids"]]
        tokens = np.asarray(payload["tokens"], dtype=np.float32)
    if ids != clip_ids or tokens.shape != (len(clip_ids), 16, 2, 192):
        raise ValueError(f"label-blind ROI identity/shape 不匹配: {path}")
    return torch.from_numpy(tokens[:, :, 0].copy())


def _parameter_report(
    yolo_checkpoint: Path,
    skeleton: torch.nn.Module,
    heads: list[torch.nn.Module],
) -> dict[str, float | int | bool]:
    detector = YOLO(str(yolo_checkpoint)).model
    modules = [detector, skeleton, *heads]
    count = sum(parameter.numel() for module in modules for parameter in module.parameters())
    fp32_bytes = count * 4
    return {
        "parameter_count": count,
        "fp32_parameter_bytes": fp32_bytes,
        "fp32_parameter_mib": fp32_bytes / (1024**2),
        "parameters_le_20m": count <= 20_000_000,
        "fp32_parameters_le_80mb": fp32_bytes <= 80_000_000,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--short-cache", type=Path, required=True)
    parser.add_argument("--short-sidecar", type=Path, required=True)
    parser.add_argument("--dense-cache", type=Path, required=True)
    parser.add_argument("--dense-sidecar", type=Path, required=True)
    parser.add_argument("--roi-320", type=Path, required=True)
    parser.add_argument("--roi-640", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--dense-checkpoint", type=Path, required=True)
    parser.add_argument("--yolo-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--embedding-batch-size", type=int, default=1024)
    parser.add_argument("--head-batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.output.exists() or args.predictions.exists():
        raise FileExistsError("label-blind deployment 输出已存在，拒绝覆盖")
    if len(args.checkpoint) != 6:
        raise ValueError("必须按冻结顺序提供六个 short checkpoints")
    lock = _validate_lock(args.protocol_lock)
    amendment = json.loads(args.amendment.read_text(encoding="utf-8"))
    if amendment.get("parent") != str(args.protocol_lock).replace("\\", "/"):
        raise ValueError("deployment amendment parent 不匹配")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("请求 CUDA，但当前不可用")

    short_cache, short_features, short_window_clips = _features(
        args.short_cache, args.short_sidecar
    )
    dense_cache, dense_features, dense_window_clips = _features(
        args.dense_cache, args.dense_sidecar
    )
    clip_ids = [str(row["clip_id"]) for row in short_cache.metadata["clips"]]
    dense_ids = [str(row["clip_id"]) for row in dense_cache.metadata["clips"]]
    if clip_ids != dense_ids or len(clip_ids) != 1200:
        raise ValueError("short/dense validation clip identity 不匹配")
    roi_320 = _roi(args.roi_320, clip_ids)
    roi_640 = _roi(args.roi_640, clip_ids)

    checkpoints = [_load_checkpoint(path) for path in args.checkpoint]
    checkpoints.append(_load_checkpoint(args.dense_checkpoint))
    backbone_hashes = {
        _state_dict_sha256(checkpoint["skeleton_model_state"])
        for checkpoint in checkpoints
    }
    expected_backbone = lock["classifier"]["shared_skeleton_backbone_sha256"]
    if backbone_hashes != {expected_backbone}:
        raise ValueError("七头未共享冻结的单一 skeleton backbone")
    configs = [F1Config(**checkpoint["config"]) for checkpoint in checkpoints]
    if any(config != configs[0] for config in configs[1:]):
        raise ValueError("七头 config 不一致")
    skeleton = build_skeleton_model(configs[0]).to(device)
    skeleton.load_state_dict(checkpoints[0]["skeleton_model_state"], strict=True)
    heads = [_head(checkpoint).to(device).eval() for checkpoint in checkpoints]
    skeleton.eval()

    short_indices, short_embeddings_np = clip_embeddings(
        skeleton,
        short_features,
        short_window_clips,
        device=device,
        batch_size=args.embedding_batch_size,
    )
    dense_indices, dense_embeddings_np = clip_embeddings(
        skeleton,
        dense_features,
        dense_window_clips,
        device=device,
        batch_size=args.embedding_batch_size,
    )
    expected_indices = np.arange(len(clip_ids), dtype=np.int64)
    if not np.array_equal(short_indices, expected_indices) or not np.array_equal(
        dense_indices, expected_indices
    ):
        raise ValueError("七头 embedding 未覆盖全部 validation clips")
    short_embeddings = torch.from_numpy(short_embeddings_np)
    dense_embeddings = torch.from_numpy(dense_embeddings_np)
    logits = np.empty((len(clip_ids), 7), dtype=np.float64)
    all_indices = np.arange(len(clip_ids), dtype=np.int64)
    for index in range(6):
        roi = roi_320 if index % 2 == 0 else roi_640
        scores = _head_scores(
            heads[index],
            short_embeddings,
            roi,
            all_indices,
            device=device,
            batch_size=args.head_batch_size,
        )
        epsilon = np.finfo(np.float32).eps
        clipped = np.clip(scores, epsilon, 1.0 - epsilon)
        logits[:, index] = np.log(clipped / (1.0 - clipped))
    dense_scores = _head_scores(
        heads[6],
        dense_embeddings,
        roi_320,
        all_indices,
        device=device,
        batch_size=args.head_batch_size,
    )
    epsilon = np.finfo(np.float32).eps
    dense_clipped = np.clip(dense_scores, epsilon, 1.0 - epsilon)
    logits[:, 6] = np.log(dense_clipped / (1.0 - dense_clipped))
    scores = equal_logit_probability(logits)
    labels = np.asarray(
        [float(row["has_fall"]) for row in short_cache.metadata["clips"]],
        dtype=np.float64,
    )
    metrics = metric_result(clip_ids, labels, scores)
    parameters = _parameter_report(args.yolo_checkpoint, skeleton, heads)
    gates = {
        "complete_1200": len(scores) == 1200 and bool(np.isfinite(scores).all()),
        "map_above_fallback_development": metrics["clip_map_percent"] > 49.98701398357859,
        "parameters_le_20m": parameters["parameters_le_20m"],
        "fp32_parameters_le_80mb": parameters["fp32_parameters_le_80mb"],
        "v100_end_to_end_p95_le_100ms": None,
    }
    _atomic_npz(
        args.predictions,
        clip_id=np.asarray(clip_ids),
        label=labels.astype(np.uint8),
        member_logits=logits.astype(np.float32),
        score=scores.astype(np.float32),
    )
    result = {
        "protocol": PROTOCOL,
        "selection_is_label_blind": True,
        "event_proposal_protocol": lock["event_proposal"]["protocol"],
        "clips": len(clip_ids),
        "metrics": metrics,
        "parameters": parameters,
        "gates": gates,
        "qualification": "pending_v100_end_to_end_benchmark",
        "inputs": {
            "protocol_lock_sha256": _sha256_file(args.protocol_lock),
            "amendment_sha256": _sha256_file(args.amendment),
            "short_checkpoints": [
                {"path": str(path), "sha256": _sha256_file(path)}
                for path in args.checkpoint
            ],
            "dense_checkpoint": {
                "path": str(args.dense_checkpoint),
                "sha256": _sha256_file(args.dense_checkpoint),
            },
            "roi_320_sha256": _sha256_file(args.roi_320),
            "roi_640_sha256": _sha256_file(args.roi_640),
        },
        "predictions_sha256": _sha256_file(args.predictions),
        "test_accessed": False,
    }
    _atomic_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
