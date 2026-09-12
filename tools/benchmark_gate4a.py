"""Gate 4A 1080P 基准；正式运行默认强制要求 NVIDIA V100。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from eval.tcn_ablation import atomic_json, sha256_file
from pipeline.inference_engine import InferenceEngine, crop_frame, sanitize_fps

PROTOCOL = "gate4a_1080p_v1"
MAX_P95_MS = 100.0
MAX_PARAMETERS = 20_000_000
MAX_FP32_BYTES = 80_000_000
FROZEN_CHECKPOINT_SHA256 = "86d5eb6aa304dc6213dc42fce0af338f608872570950662346c39bdf0caa3d5c"
FROZEN_RUN_JSON_SHA256 = "83a57ba2cfe191f2b24907117d062c63c83d238ee2c9e99808f00d607722dfba"
FROZEN_ABLATION_SHA256 = "383c4307a012e0a43f09bff6486a4fdfb43170bf78fecf8a7a4da21b29e41489"
FROZEN_FIXTURE_SOURCE_SHA256 = "5a6844e3644f472b3859448ddb9a4c8ca9dd761a623660a06c4220ee5267d412"


def letterbox_1080(frame: np.ndarray) -> np.ndarray:
    """保持宽高比缩放到 1920×1080，并以黑边补齐。"""
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame 必须是 HxWx3 ndarray")
    height, width = frame.shape[:2]
    scale = min(1920 / width, 1080 / height)
    resized_w, resized_h = round(width * scale), round(height * scale)
    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    output = np.zeros((1080, 1920, 3), dtype=np.uint8)
    x, y = (1920 - resized_w) // 2, (1080 - resized_h) // 2
    output[y : y + resized_h, x : x + resized_w] = resized
    return output


def make_1080p_fixture(source: Path, output: Path, *, min_frames: int) -> dict[str, Any]:
    if min_frames < 1:
        raise ValueError("min_frames 必须为正")
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"无法打开 fixture 源视频: {source}")
    fps = sanitize_fps(capture.get(cv2.CAP_PROP_FPS))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        rgb_view, _ = crop_frame(frame, "auto")
        frames.append(letterbox_1080(rgb_view))
    capture.release()
    if not frames:
        raise ValueError("fixture 源视频没有可读帧")
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1920, 1080)
    )
    if not writer.isOpened():
        raise ValueError(f"无法创建 1080P fixture: {output}")
    written = 0
    try:
        while written < min_frames:
            writer.write(frames[written % len(frames)])
            written += 1
    finally:
        writer.release()
    return {
        "source": str(source.resolve()),
        "source_sha256": sha256_file(source),
        "output": str(output.resolve()),
        "output_sha256": sha256_file(output),
        "width": 1920,
        "height": 1080,
        "fps": fps,
        "source_frames": len(frames),
        "written_frames": written,
        "construction": "auto RGB crop, aspect-preserving resize, black letterbox, cyclic repeat",
    }


def _sha_payload(value: dict[str, Any]) -> str:
    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _cuda_device_index(device: str) -> int | None:
    if device.isdigit():
        return int(device)
    if device.startswith("cuda:") and device.removeprefix("cuda:").isdigit():
        return int(device.removeprefix("cuda:"))
    return None


def detect_gpu_name(device: str) -> str | None:
    index = _cuda_device_index(device)
    if not torch.cuda.is_available() or index is None:
        return None
    return torch.cuda.get_device_name(index)


def is_nvidia_v100(gpu_name: str | None) -> bool:
    """接受 NVIDIA 驱动常见的 Tesla V100 PCIe/SXM 名称。"""
    return gpu_name is not None and "V100" in gpu_name.upper()


def gate4a_checks(
    *,
    gpu_name: str | None,
    p95_ms: float,
    parameters: int,
    candidate_artifacts_match: bool = True,
) -> dict[str, bool]:
    return {
        "candidate_artifacts_match": candidate_artifacts_match,
        "nvidia_v100": is_nvidia_v100(gpu_name),
        "p95_le_100ms": p95_ms <= MAX_P95_MS,
        "parameters_le_20m": parameters <= MAX_PARAMETERS,
        "fp32_parameters_le_80mb": parameters * 4 <= MAX_FP32_BYTES,
    }


def resolve_model_path(checkpoint: Path, model: Path | None) -> Path:
    """显式解析 YOLO 权重，避免依赖 Ultralytics 的隐式 weights_dir。"""
    resolved = (model or checkpoint.with_name("yolo11n-pose.pt")).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"YOLO 权重不存在: {resolved}")
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-json", type=Path, required=True)
    parser.add_argument("--ablation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--warmup-frames", type=int, default=32)
    parser.add_argument(
        "--allow-non-v100-precheck",
        action="store_true",
        help="仅用于本地预检；输出不会被认定为正式 Gate 4A 通过",
    )
    args = parser.parse_args()
    gpu_name = detect_gpu_name(args.device)
    if not is_nvidia_v100(gpu_name) and not args.allow_non_v100_precheck:
        raise SystemExit(
            "正式 Gate 4A 必须在 NVIDIA V100 上运行；"
            f"当前设备={gpu_name or '无可用 CUDA GPU'}。"
            "如只做本地预检，请显式传入 --allow-non-v100-precheck。"
        )
    artifact_hashes = {
        "checkpoint": sha256_file(args.checkpoint),
        "run_json": sha256_file(args.run_json),
        "ablation": sha256_file(args.ablation),
        "fixture_source": sha256_file(args.source),
    }
    expected_hashes = {
        "checkpoint": FROZEN_CHECKPOINT_SHA256,
        "run_json": FROZEN_RUN_JSON_SHA256,
        "ablation": FROZEN_ABLATION_SHA256,
        "fixture_source": FROZEN_FIXTURE_SOURCE_SHA256,
    }
    candidate_artifacts_match = artifact_hashes == expected_hashes
    if not candidate_artifacts_match:
        raise ValueError(
            "Gate 4A 候选或 validation fixture 哈希不匹配；"
            "必须使用已 review 的 epoch 7/aligned/fall-18-cam0 组合"
        )
    model_path = resolve_model_path(args.checkpoint, args.model)
    fixture = make_1080p_fixture(args.source, args.fixture, min_frames=args.frames)
    ablation = json.loads(args.ablation.read_text(encoding="utf-8"))
    if ablation.get("split") != "val":
        raise ValueError("候选阈值必须来自 validation")
    engine = InferenceEngine(
        model_path=str(model_path),
        device=args.device,
        temporal_mode="tcn",
        tcn_checkpoint=args.checkpoint,
        tcn_run_json=args.run_json,
    )
    engine.analyze(args.fixture, render=False, crop="none", max_frames=args.warmup_frames)
    measured = engine.analyze(args.fixture, render=False, crop="none")
    yolo_parameters = sum(parameter.numel() for parameter in engine.model.model.parameters())
    assert engine.tcn_scorer is not None
    tcn_parameters = sum(
        parameter.numel() for parameter in engine.tcn_scorer.model.parameters()
    )
    total_parameters = yolo_parameters + tcn_parameters
    signature = {
        "protocol": PROTOCOL,
        "fixture_sha256": fixture["output_sha256"],
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "run_json_sha256": sha256_file(args.run_json),
        "ablation_sha256": sha256_file(args.ablation),
        "implementation_sha256": sha256_file(Path(__file__)),
        "engine_signature": engine.cache_signature(),
    }
    stage = measured["stage_latency_ms"]
    checks = gate4a_checks(
        gpu_name=gpu_name,
        p95_ms=stage["frame_end_to_end"]["p95"],
        parameters=total_parameters,
        candidate_artifacts_match=candidate_artifacts_match,
    )
    local_precheck_pass = all(value for key, value in checks.items() if key != "nvidia_v100")
    formal_gate4a_pass = all(checks.values())
    result = {
        "protocol": PROTOCOL,
        "candidate": {
            "temporal_mode": "tcn",
            "checkpoint_epoch": ablation["checkpoint_epoch"],
            "r90_threshold": ablation["tcn_only"]["r90_point"]["threshold"],
            "r95_threshold": ablation["tcn_only"]["r95_point"]["threshold"],
            "full_val_metrics": ablation["tcn_only"],
        },
        "fixture": fixture,
        "hardware": {
            "cuda_available": torch.cuda.is_available(),
            "gpu": gpu_name,
            "device": args.device,
            "v100_measured": checks["nvidia_v100"],
        },
        "measurement": {
            "render": False,
            "processed_frames": measured["processed_frames"],
            "frame_end_to_end_ms": stage["frame_end_to_end"],
            "detector_predict_ms": stage["predict"],
            "cpu_transfer_ms": stage["cpu_transfer"],
            "track_rule_ms": stage["track_rule"],
            "tcn_ms": stage["tcn_fusion"],
            "pose_coverage": measured["pose_coverage"],
        },
        "size": {
            "yolo_parameters": yolo_parameters,
            "tcn_parameters": tcn_parameters,
            "total_parameters": total_parameters,
            "fp32_parameter_mb_decimal": total_parameters * 4 / 1e6,
            "yolo_weight_bytes": model_path.stat().st_size,
            "tcn_checkpoint_bytes": args.checkpoint.stat().st_size,
            "combined_artifact_bytes": (
                model_path.stat().st_size + args.checkpoint.stat().st_size
            ),
        },
        "gate4a_checks": checks,
        "frozen_artifact_sha256": artifact_hashes,
        "local_precheck_pass": local_precheck_pass,
        "formal_gate4a_pass": formal_gate4a_pass,
        "formal_gate4a_blocker": None if formal_gate4a_pass else (
            "formal Gate 4A requires NVIDIA V100 plus all latency/size limits"
        ),
        "signature": signature,
        "signature_sha256": _sha_payload(signature),
        "test_accessed": False,
    }
    atomic_json(args.output, result)
    print(f"wrote={args.output} sha256={sha256_file(args.output)} signature={result['signature_sha256']}")


if __name__ == "__main__":
    main()
