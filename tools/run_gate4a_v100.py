"""从冻结验收包一键运行 Gate 4A，并拒绝任何不完整或非 V100 的结果。"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from tools.benchmark_gate4a import (
    FROZEN_ABLATION_SHA256,
    FROZEN_CHECKPOINT_SHA256,
    FROZEN_FIXTURE_SOURCE_SHA256,
    FROZEN_RUN_JSON_SHA256,
    PROTOCOL,
)


def validate_gate4a_result(result: dict[str, Any], candidate: dict[str, Any]) -> None:
    if result.get("protocol") != PROTOCOL:
        raise ValueError("Gate 4A protocol 不匹配")
    if result.get("formal_gate4a_pass") is not True:
        raise ValueError("formal_gate4a_pass 不是 true")
    if result.get("test_accessed") is not False:
        raise ValueError("Gate 4A 不得访问 test")
    hardware = result.get("hardware", {})
    if hardware.get("v100_measured") is not True or "V100" not in str(
        hardware.get("gpu", "")
    ).upper():
        raise ValueError("结果不是 NVIDIA V100 实测")
    checks = result.get("gate4a_checks", {})
    required_checks = {
        "candidate_artifacts_match",
        "nvidia_v100",
        "p95_le_100ms",
        "parameters_le_20m",
        "fp32_parameters_le_80mb",
    }
    if set(checks) != required_checks or not all(checks.values()):
        raise ValueError("Gate 4A checks 不完整或未全部通过")
    expected_hashes = {
        "checkpoint": FROZEN_CHECKPOINT_SHA256,
        "run_json": FROZEN_RUN_JSON_SHA256,
        "ablation": FROZEN_ABLATION_SHA256,
        "fixture_source": FROZEN_FIXTURE_SOURCE_SHA256,
    }
    if result.get("frozen_artifact_sha256") != expected_hashes:
        raise ValueError("Gate 4A 冻结产物哈希不匹配")
    fixture = result.get("fixture", {})
    if (fixture.get("width"), fixture.get("height"), fixture.get("written_frames")) != (
        1920,
        1080,
        300,
    ):
        raise ValueError("Gate 4A fixture 不是 1920x1080/300 帧")
    measurement = result.get("measurement", {})
    if measurement.get("render") is not False or measurement.get("processed_frames") != 300:
        raise ValueError("Gate 4A measurement 协议不匹配")
    if measurement.get("frame_end_to_end_ms", {}).get("p95", float("inf")) > 100.0:
        raise ValueError("Gate 4A P95 超过 100ms")
    frozen = candidate.get("candidate", {})
    measured = result.get("candidate", {})
    comparable = {
        "temporal_mode": measured.get("temporal_mode"),
        "checkpoint_epoch": measured.get("checkpoint_epoch"),
        "r90_threshold": measured.get("r90_threshold"),
        "r95_threshold": measured.get("r95_threshold"),
    }
    if comparable != {key: frozen.get(key) for key in comparable}:
        raise ValueError("结果候选与 gate4a_candidate.json 不一致")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    output = args.output or root / "gate4a-v100.json"
    if output.exists():
        raise SystemExit(f"输出已存在，拒绝覆盖正式证据: {output}")
    command = [
        sys.executable,
        "-m",
        "tools.benchmark_gate4a",
        "--source",
        str(root / "gate4a_fixture" / "fall-18-cam0.mp4"),
        "--fixture",
        str(root / "gate4a-work" / "fall-18-cam0-rgb-1080p.mp4"),
        "--model",
        str(root / "weights" / "yolo11n-pose.pt"),
        "--checkpoint",
        str(root / "weights" / "fall_tcn_best.pt"),
        "--run-json",
        str(root / "weights" / "fall_tcn_run.json"),
        "--ablation",
        str(root / "configs" / "ofsyn_val_tcn_ablation.json"),
        "--output",
        str(output),
        "--device",
        args.device,
        "--frames",
        "300",
        "--warmup-frames",
        "32",
    ]
    completed = subprocess.run(command, cwd=root, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    result = json.loads(output.read_text(encoding="utf-8"))
    candidate = json.loads(
        (root / "configs" / "gate4a_candidate.json").read_text(encoding="utf-8")
    )
    validate_gate4a_result(result, candidate)
    print(
        "formal_gate4a_pass=true "
        f"gpu={result['hardware']['gpu']} "
        f"p95_ms={result['measurement']['frame_end_to_end_ms']['p95']:.3f}"
    )


if __name__ == "__main__":
    main()
