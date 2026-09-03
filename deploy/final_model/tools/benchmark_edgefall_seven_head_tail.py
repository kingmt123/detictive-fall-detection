"""Benchmark the frozen seven-head classifier tail on real cached inputs.

This intentionally excludes video decode, pose, tracking, and YOLO extraction.
It measures the two temporal encoders, seven ROI heads, and fixed fusion only.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from collections.abc import Callable
from dataclasses import fields
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from models.edgefall_ensemble import SevenHeadProbabilityRMPTSFusion
from models.edgefall_f1 import EdgeFallF1Head
from models.tcn_dataset import load_window_cache
from tools.build_tcn_multistream_sidecar import load_sidecar
from tools.train_edgefall_f1 import F1Config, build_skeleton_model
from tools.train_tcn import (
    _append_sidecar,
    _atomic_json,
    _materialize,
    _sha256_file,
)


def _config(payload: dict[str, Any]) -> F1Config:
    raw = payload.get("config")
    if not isinstance(raw, dict):
        raise TypeError("benchmark checkpoint 缺少 config")
    names = {field.name for field in fields(F1Config)}
    config = F1Config(**{name: raw[name] for name in names})
    config.validate()
    return config


def _load_checkpoint(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError("benchmark checkpoint 必须为 dict")
    required = {"skeleton_model_state", "fusion_state"}
    if not required.issubset(payload):
        raise ValueError(f"benchmark checkpoint 缺少字段: {sorted(required - set(payload))}")
    return payload


def _head(checkpoint: dict[str, Any]) -> EdgeFallF1Head:
    fusion_state = checkpoint["fusion_state"]
    projection = fusion_state.get("roi_encoder.input_projection.0.weight")
    if not isinstance(projection, torch.Tensor) or projection.ndim != 2:
        raise ValueError("benchmark checkpoint 无法推导 ROI token dim")
    inferred_roi_dim = int(projection.shape[1])
    persisted_roi_dim = int(checkpoint.get("roi_token_dim", inferred_roi_dim))
    if persisted_roi_dim != inferred_roi_dim:
        raise ValueError("benchmark checkpoint ROI token dim 签名冲突")
    model = EdgeFallF1Head(
        384,
        roi_dim=inferred_roi_dim,
        roi_token_layout=str(checkpoint.get("roi_token_layout", "direct")),
        roi_temporal_pooling=str(checkpoint.get("roi_temporal_pooling", "mean_max")),
    )
    model.load_state_dict(fusion_state, strict=True)
    return model


def _latency_summary(milliseconds: list[float]) -> dict[str, float]:
    values = np.asarray(milliseconds, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError("benchmark latency samples 无效")
    return {
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "mean_ms": float(values.mean()),
    }


def _measure(
    function: Callable[[], object],
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> dict[str, float]:
    for _ in range(warmup):
        function()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        samples.append((time.perf_counter() - started) * 1000.0)
    return _latency_summary(samples)


def _member_checkpoints(member_report: Path) -> list[Path]:
    report = json.loads(member_report.read_text(encoding="utf-8"))
    members = report.get("members")
    if not isinstance(members, list) or len(members) != 6:
        raise ValueError("benchmark member report 必须包含六个成员")
    paths = []
    for member in members:
        rows = member.get("oof") if isinstance(member, dict) else None
        if not isinstance(rows, list) or len(rows) != 5:
            raise ValueError("benchmark member OOF 必须完整五折")
        prediction = Path(rows[0]["path"])
        checkpoint = prediction.parent / "last.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"benchmark checkpoint 缺失: {checkpoint}")
        paths.append(checkpoint)
    return paths


def _real_window(cache_path: Path, sidecar_path: Path) -> torch.Tensor:
    cache = load_window_cache(cache_path, verify_hashes=True)
    sidecar = load_sidecar(sidecar_path, cache)
    features, _, _ = _materialize(cache, np.asarray([0], dtype=np.int64))
    return _append_sidecar(features, sidecar, np.asarray([0], dtype=np.int64))


def _real_roi(path: Path) -> torch.Tensor:
    with np.load(path, allow_pickle=False) as payload:
        tokens = np.asarray(payload["tokens"][0, :, 0], dtype=np.float32)
    if tokens.ndim != 2 or tokens.shape[1] != 192:
        raise ValueError("benchmark ROI token shape 必须为 (T,192)")
    return torch.from_numpy(tokens[None])


def _parameter_evidence(modules: list[nn.Module]) -> dict[str, float | int]:
    parameters = [parameter for module in modules for parameter in module.parameters()]
    count = sum(parameter.numel() for parameter in parameters)
    bytes_fp32 = count * 4
    return {
        "parameter_count": count,
        "fp32_parameter_bytes": bytes_fp32,
        "fp32_parameter_mib": bytes_fp32 / (1024**2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--member-report", type=Path, required=True)
    parser.add_argument("--dense-checkpoint", type=Path, required=True)
    parser.add_argument("--short-cache", type=Path, required=True)
    parser.add_argument("--short-sidecar", type=Path, required=True)
    parser.add_argument("--dense-cache", type=Path, required=True)
    parser.add_argument("--dense-sidecar", type=Path, required=True)
    parser.add_argument("--roi-320", type=Path, required=True)
    parser.add_argument("--roi-640", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=500)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("七头 tail benchmark 输出已存在，拒绝覆盖")
    if args.warmup < 1 or args.repeats < 10:
        raise ValueError("七头 tail benchmark warmup/repeats 过小")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("七头 tail benchmark 请求 CUDA，但当前不可用")

    short_paths = _member_checkpoints(args.member_report)
    short_checkpoints = [_load_checkpoint(path) for path in short_paths]
    dense_checkpoint = _load_checkpoint(args.dense_checkpoint)
    reference_config = _config(short_checkpoints[0])
    if any(_config(checkpoint) != reference_config for checkpoint in short_checkpoints):
        raise ValueError("六个短窗 checkpoint config 不一致")
    dense_config = _config(dense_checkpoint)
    if dense_config != reference_config:
        raise ValueError("dense/short checkpoint config 不一致")

    short_backbone = build_skeleton_model(reference_config)
    short_backbone.load_state_dict(short_checkpoints[0]["skeleton_model_state"], strict=True)
    dense_backbone = build_skeleton_model(dense_config)
    dense_backbone.load_state_dict(dense_checkpoint["skeleton_model_state"], strict=True)
    heads = [_head(checkpoint) for checkpoint in short_checkpoints]
    heads.append(_head(dense_checkpoint))
    probability_fusion = SevenHeadProbabilityRMPTSFusion(delta=Fraction(1, 7))
    modules: list[nn.Module] = [short_backbone, dense_backbone, *heads, probability_fusion]
    for module in modules:
        module.eval().to(device)

    short_x = _real_window(args.short_cache, args.short_sidecar).to(device)
    dense_x = _real_window(args.dense_cache, args.dense_sidecar).to(device)
    roi_320 = _real_roi(args.roi_320).to(device)
    roi_640 = _real_roi(args.roi_640).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        short_embedding = short_backbone.encode(short_x)
        dense_embedding = dense_backbone.encode(dense_x)

        def encode_short() -> torch.Tensor:
            return short_backbone.encode(short_x)

        def encode_dense() -> torch.Tensor:
            return dense_backbone.encode(dense_x)

        def short_320_heads() -> tuple[torch.Tensor, ...]:
            return tuple(heads[index](short_embedding, roi_320)[0] for index in (0, 2, 4))

        def short_640_heads() -> tuple[torch.Tensor, ...]:
            return tuple(heads[index](short_embedding, roi_640)[0] for index in (1, 3, 5))

        def dense_head() -> torch.Tensor:
            return heads[6](dense_embedding, roi_320)[0]

        def full_tail() -> torch.Tensor:
            e16 = short_backbone.encode(short_x)
            e48 = dense_backbone.encode(dense_x)
            logits_320 = torch.stack(
                [heads[index](e16, roi_320)[0] for index in (0, 2, 4)], dim=-1
            )
            logits_640 = torch.stack(
                [heads[index](e16, roi_640)[0] for index in (1, 3, 5)], dim=-1
            )
            dense_logit = heads[6](e48, roi_320)[0]
            return probability_fusion(
                torch.sigmoid(logits_320),
                torch.sigmoid(logits_640),
                torch.sigmoid(dense_logit),
            )

        stream_16 = torch.cuda.Stream(device=device) if device.type == "cuda" else None
        stream_48 = torch.cuda.Stream(device=device) if device.type == "cuda" else None

        def parallel_tail() -> torch.Tensor:
            if stream_16 is None or stream_48 is None:
                return full_tail()
            current = torch.cuda.current_stream(device)
            stream_16.wait_stream(current)
            stream_48.wait_stream(current)
            with torch.cuda.stream(stream_16):
                e16 = short_backbone.encode(short_x)
                logits_320 = torch.stack(
                    [heads[index](e16, roi_320)[0] for index in (0, 2, 4)], dim=-1
                )
                logits_640 = torch.stack(
                    [heads[index](e16, roi_640)[0] for index in (1, 3, 5)], dim=-1
                )
            with torch.cuda.stream(stream_48):
                e48 = dense_backbone.encode(dense_x)
                dense_logit = heads[6](e48, roi_320)[0]
            current.wait_stream(stream_16)
            current.wait_stream(stream_48)
            return probability_fusion(
                torch.sigmoid(logits_320),
                torch.sigmoid(logits_640),
                torch.sigmoid(dense_logit),
            )

        stages = {
            "e16_encode": _measure(
                encode_short, device=device, warmup=args.warmup, repeats=args.repeats
            ),
            "e48_encode": _measure(
                encode_dense, device=device, warmup=args.warmup, repeats=args.repeats
            ),
            "three_320_heads": _measure(
                short_320_heads, device=device, warmup=args.warmup, repeats=args.repeats
            ),
            "three_640_heads": _measure(
                short_640_heads, device=device, warmup=args.warmup, repeats=args.repeats
            ),
            "dense_head": _measure(
                dense_head, device=device, warmup=args.warmup, repeats=args.repeats
            ),
            "full_classifier_tail": _measure(
                full_tail, device=device, warmup=args.warmup, repeats=args.repeats
            ),
            "parallel_classifier_tail": _measure(
                parallel_tail, device=device, warmup=args.warmup, repeats=args.repeats
            ),
        }
        sequential_output = full_tail()
        parallel_output = parallel_tail()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        output_probability = float(sequential_output.item())
        parallel_max_abs_difference = float(
            torch.max(torch.abs(sequential_output - parallel_output)).item()
        )

    parameters = _parameter_evidence(modules)
    peak_bytes = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    )
    report = {
        "protocol": "edgefall_seven_head_classifier_tail_benchmark_v1",
        "scope": "cached_features_to_probability_only",
        "excluded_from_timing": [
            "video_decode",
            "pose_detection",
            "tracking",
            "YOLO_320_and_640_feature_extraction",
            "window_materialization_and_host_to_device_transfer",
        ],
        "hardware": {
            "platform": platform.platform(),
            "torch": torch.__version__,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
            ),
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
        },
        "warmup": args.warmup,
        "repeats": args.repeats,
        "batch_size": 1,
        "latency": stages,
        "parallel_scheduling": {
            "method": "independent CUDA streams for e16+six short heads and e48+dense head",
            "max_abs_output_difference": parallel_max_abs_difference,
            "exact_output_match": bool(torch.equal(sequential_output, parallel_output)),
            "p95_speedup": (
                stages["full_classifier_tail"]["p95_ms"]
                / stages["parallel_classifier_tail"]["p95_ms"]
            ),
        },
        "peak_allocated_bytes": peak_bytes,
        "peak_allocated_mib": peak_bytes / (1024**2) if peak_bytes is not None else None,
        **parameters,
        "gates": {
            "parameter_count_le_20m": parameters["parameter_count"] <= 20_000_000,
            "fp32_parameters_le_80mb": parameters["fp32_parameter_bytes"] <= 80_000_000,
            "classifier_tail_p95_le_100ms": stages["full_classifier_tail"]["p95_ms"] <= 100.0,
            "parallel_output_exact": bool(torch.equal(sequential_output, parallel_output)),
            "full_pipeline_latency_gate_closed": False,
        },
        "output_probability_smoke": output_probability,
        "inputs": {
            "member_report_sha256": _sha256_file(args.member_report),
            "dense_checkpoint_sha256": _sha256_file(args.dense_checkpoint),
            "short_cache_metadata_sha256": _sha256_file(args.short_cache / "metadata.json"),
            "dense_cache_metadata_sha256": _sha256_file(args.dense_cache / "metadata.json"),
            "roi_320_sha256": _sha256_file(args.roi_320),
            "roi_640_sha256": _sha256_file(args.roi_640),
            "short_checkpoints": [
                {"path": str(path), "sha256": _sha256_file(path)} for path in short_paths
            ],
        },
        "test_accessed": False,
        "limitations": [
            "This is not the report's required end-to-end V100 benchmark.",
            "Cached tensor inputs remove detector, tracker, feature extraction, and transfer costs.",
            "The local RTX measurement can locate bottlenecks but cannot close the V100 latency gate.",
        ],
    }
    _atomic_json(args.output, report)
    print(json.dumps({"output": str(args.output.resolve()), "tail_p95_ms": stages["full_classifier_tail"]["p95_ms"]}))


if __name__ == "__main__":
    main()
