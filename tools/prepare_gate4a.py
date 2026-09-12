"""冻结 Gate 4A 候选并生成可复现、匿名、确定性的提交 ZIP dry-run。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from models.tcn_inference import OnlineFallTCNScorer, _repository_path
from tools.benchmark_gate4a import (
    FROZEN_ABLATION_SHA256,
    FROZEN_CHECKPOINT_SHA256,
    FROZEN_FIXTURE_SOURCE_SHA256,
    FROZEN_RUN_JSON_SHA256,
    detect_gpu_name,
    is_nvidia_v100,
)

MAX_ZIP_BYTES = 200_000_000
TEXT_SUFFIXES = {".json", ".md", ".py", ".txt"}
ANONYMITY_PATTERNS = (
    b"c:\\users\\",
    b"c:/users/",
    b"/users/",
    b"/home/",
    b"kingmt123",
)

PACKAGE_FILES = {
    "SUBMISSION_README.md": "SUBMISSION_README.md",
    "requirements.txt": "requirements.txt",
    "infer.py": "infer.py",
    "eval/__init__.py": "eval/__init__.py",
    "eval/evaluate_manifest.py": "eval/evaluate_manifest.py",
    "eval/metrics.py": "eval/metrics.py",
    "eval/tcn_ablation.py": "eval/tcn_ablation.py",
    "models/__init__.py": "models/__init__.py",
    "models/tcn.py": "models/tcn.py",
    "models/tcn_dataset.py": "models/tcn_dataset.py",
    "models/tcn_inference.py": "models/tcn_inference.py",
    "models/tcn_window.py": "models/tcn_window.py",
    "pipeline/__init__.py": "pipeline/__init__.py",
    "pipeline/event_aggregator.py": "pipeline/event_aggregator.py",
    "pipeline/fusion.py": "pipeline/fusion.py",
    "pipeline/inference_engine.py": "pipeline/inference_engine.py",
    "pipeline/pose_cache.py": "pipeline/pose_cache.py",
    "pipeline/pose_track.py": "pipeline/pose_track.py",
    "pipeline/rules.py": "pipeline/rules.py",
    "pipeline/video_source.py": "pipeline/video_source.py",
    "tools/benchmark_gate4a.py": "tools/benchmark_gate4a.py",
    "tools/run_gate4a_v100.py": "tools/run_gate4a_v100.py",
    "tools/__init__.py": "tools/__init__.py",
    "tools/train_tcn.py": "tools/train_tcn.py",
    "configs/tcn_gate3.json": "configs/tcn_gate3.json",
}


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )


def _git_revision(project_root: Path) -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _validate_training_signature(project_root: Path, run: dict[str, Any]) -> None:
    for relative_path, expected in run["signature"]["code_sha256"].items():
        source = _repository_path(project_root, relative_path)
        if not source.is_file() or sha256_file(source) != expected:
            raise ValueError(f"训练签名源码不匹配: {relative_path}")


def build_candidate(
    project_root: Path,
    *,
    checkpoint: Path,
    run_json: Path,
    ablation_path: Path,
) -> tuple[dict[str, Any], dict[str, Path]]:
    seal = project_root / "runs" / "eval" / "test_split.seal.json"
    if seal.exists():
        raise ValueError(f"test seal 已存在，拒绝重新冻结 pre-seal 候选: {seal}")

    run = json.loads(run_json.read_text(encoding="utf-8"))
    ablation = json.loads(ablation_path.read_text(encoding="utf-8"))
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    frozen_hashes = {
        "checkpoint": FROZEN_CHECKPOINT_SHA256,
        "run_json": FROZEN_RUN_JSON_SHA256,
        "ablation": FROZEN_ABLATION_SHA256,
    }
    actual_hashes = {
        "checkpoint": sha256_file(checkpoint),
        "run_json": sha256_file(run_json),
        "ablation": sha256_file(ablation_path),
    }
    if actual_hashes != frozen_hashes:
        raise ValueError("Gate 4A 候选哈希不匹配；必须使用已 review 的 aligned 产物")
    if run.get("pilot") is not False:
        raise ValueError("Gate 4A 只接受非 pilot checkpoint")
    if checkpoint_payload.get("run_signature_sha256") != run.get("signature_sha256"):
        raise ValueError("checkpoint 与 run.json 签名不一致")
    if ablation.get("dataset") != "of-syn" or ablation.get("split") != "val":
        raise ValueError("Gate 4A 阈值必须来自 OF-Syn validation")
    checkpoint_epoch = int(checkpoint_payload["epoch"])
    if checkpoint_epoch != int(ablation.get("checkpoint_epoch", -1)):
        raise ValueError("ablation 与 checkpoint epoch 不一致")
    _validate_training_signature(project_root, run)
    OnlineFallTCNScorer.from_artifacts(
        checkpoint, run_json, device=torch.device("cpu")
    )

    package_sources = {
        archive_path: project_root / source_path
        for archive_path, source_path in PACKAGE_FILES.items()
    }
    package_sources.update(
        {
            "weights/yolo11n-pose.pt": project_root / "yolo11n-pose.pt",
            "weights/fall_tcn_best.pt": checkpoint,
            "weights/fall_tcn_run.json": run_json,
            "configs/ofsyn_val_tcn_ablation.json": ablation_path,
        }
    )
    missing = [name for name, path in package_sources.items() if not path.is_file()]
    if missing:
        raise ValueError(f"候选文件缺失: {missing}")

    source_hashes = {
        name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for name, path in sorted(package_sources.items())
    }
    tcn_metrics = ablation["tcn_only"]
    unsigned = {
        "schema": 1,
        "protocol": "gate4a_candidate_freeze_v1",
        "git_base_commit": _git_revision(project_root),
        "candidate": {
            "temporal_mode": "tcn",
            "checkpoint_epoch": checkpoint_epoch,
            "checkpoint_run_signature_sha256": run["signature_sha256"],
            "r90_threshold": tcn_metrics["r90_point"]["threshold"],
            "r95_threshold": tcn_metrics["r95_point"]["threshold"],
            "validation_map": tcn_metrics["map"],
        },
        "files": source_hashes,
        "test_seal": "absent",
    }
    candidate_id = sha256_bytes(_canonical_json(unsigned))
    return {**unsigned, "candidate_id": candidate_id}, package_sources


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    return info


def write_deterministic_zip(
    output: Path, package_sources: dict[str, Path], candidate: dict[str, Any]
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    with zipfile.ZipFile(temporary, "w", allowZip64=True) as archive:
        for name, source in sorted(package_sources.items()):
            archive.writestr(_zip_info(name), source.read_bytes())
        archive.writestr(
            _zip_info("configs/gate4a_candidate.json"), _canonical_json(candidate)
        )
    temporary.replace(output)


def inspect_zip(path: Path) -> dict[str, Any]:
    if path.stat().st_size > MAX_ZIP_BYTES:
        raise ValueError(f"ZIP 超过 200MB: {path.stat().st_size}")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("ZIP 含重复路径")
        forbidden = [
            name
            for name in names
            if name.startswith(("data/", "reports/", "tests/", ".git/"))
            or "test_split" in name
        ]
        if forbidden:
            raise ValueError(f"ZIP 含禁止内容: {forbidden}")
        anonymity_hits: list[str] = []
        for name in names:
            if Path(name).suffix.lower() not in TEXT_SUFFIXES:
                continue
            lowered = archive.read(name).lower()
            if any(pattern in lowered for pattern in ANONYMITY_PATTERNS):
                anonymity_hits.append(name)
        if anonymity_hits:
            raise ValueError(f"ZIP 文本匿名扫描失败: {anonymity_hits}")
    return {
        "entries": len(names),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "anonymous_text_scan": True,
        "forbidden_content_scan": True,
    }


def smoke_verify_zip(path: Path, verify_root: Path) -> dict[str, bool]:
    verify_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="gate4a-package-", dir=verify_root) as temp:
        extracted = Path(temp)
        with zipfile.ZipFile(path) as archive:
            archive.extractall(extracted)
        help_run = subprocess.run(
            [sys.executable, "-m", "eval.evaluate_manifest", "--help"],
            cwd=extracted,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if help_run.returncode != 0:
            raise RuntimeError(f"批量入口 smoke 失败: {help_run.stderr}")
        loader_code = (
            "from pathlib import Path; import torch; "
            "from models.tcn_inference import OnlineFallTCNScorer; "
            "OnlineFallTCNScorer.from_artifacts("
            "Path('weights/fall_tcn_best.pt'), Path('weights/fall_tcn_run.json'), "
            "device=torch.device('cpu'))"
        )
        loader_run = subprocess.run(
            [sys.executable, "-c", loader_code],
            cwd=extracted,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if loader_run.returncode != 0:
            raise RuntimeError(f"checkpoint clean-tree smoke 失败: {loader_run.stderr}")

        smoke_video = extracted / "smoke.mp4"
        writer = cv2.VideoWriter(
            str(smoke_video), cv2.VideoWriter_fourcc(*"mp4v"), 16.0, (64, 64)
        )
        if not writer.isOpened():
            raise RuntimeError("无法创建 clean-tree batch smoke 视频")
        try:
            for _ in range(16):
                writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
        finally:
            writer.release()
        manifest = extracted / "smoke_manifest.csv"
        with manifest.open("w", encoding="utf-8", newline="") as handle:
            csv_writer = csv.DictWriter(
                handle,
                fieldnames=["dataset", "split", "clip_id", "video_path", "has_fall"],
            )
            csv_writer.writeheader()
            csv_writer.writerow(
                {
                    "dataset": "package-smoke",
                    "split": "val",
                    "clip_id": "black-16f",
                    "video_path": "smoke.mp4",
                    "has_fall": "0",
                }
            )
        batch_run = subprocess.run(
            [
                sys.executable,
                "-m",
                "eval.evaluate_manifest",
                "--manifest",
                "smoke_manifest.csv",
                "--dataset",
                "package-smoke",
                "--split",
                "val",
                "--mode",
                "clip",
                "--model",
                "weights/yolo11n-pose.pt",
                "--device",
                "cpu",
                "--imgsz",
                "64",
                "--temporal-mode",
                "tcn",
                "--tcn-checkpoint",
                "weights/fall_tcn_best.pt",
                "--tcn-run-json",
                "weights/fall_tcn_run.json",
                "--output",
                "smoke_predictions.jsonl",
                "--summary",
                "smoke_summary.json",
            ],
            cwd=extracted,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if batch_run.returncode != 0:
            raise RuntimeError(f"clean-tree batch inference 失败: {batch_run.stderr}")
        summary = json.loads((extracted / "smoke_summary.json").read_text(encoding="utf-8"))
        if summary.get("clips_succeeded") != 1:
            raise RuntimeError("clean-tree batch inference 未完成唯一 validation clip")
    return {
        "batch_cli_help": True,
        "checkpoint_load_cpu": True,
        "batch_tcn_validation_clip": True,
    }


def smoke_verify_v100_bundle(path: Path, verify_root: Path) -> dict[str, bool]:
    if is_nvidia_v100(detect_gpu_name("0")):
        raise RuntimeError("非 V100 fail-closed smoke 只能在非 V100 主机执行")
    verify_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="gate4a-v100-bundle-", dir=verify_root) as temp:
        extracted = Path(temp)
        with zipfile.ZipFile(path) as archive:
            archive.extractall(extracted)
        run = subprocess.run(
            [sys.executable, "-m", "tools.run_gate4a_v100", "--device", "0"],
            cwd=extracted,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if run.returncode == 0:
            raise RuntimeError("V100 runner 在非 V100 主机上错误通过")
        if (extracted / "gate4a-v100.json").exists():
            raise RuntimeError("V100 runner 失败后仍生成正式结果")
        if (extracted / "gate4a-work" / "fall-18-cam0-rgb-1080p.mp4").exists():
            raise RuntimeError("V100 runner 未在读取/转换 fixture 前 fail-closed")
    return {
        "non_v100_rejected": True,
        "no_formal_result_created": True,
        "fixture_not_decoded_before_rejection": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-json", type=Path, required=True)
    parser.add_argument("--ablation", type=Path, required=True)
    parser.add_argument("--candidate-output", type=Path, required=True)
    parser.add_argument("--zip-output", type=Path, required=True)
    parser.add_argument("--v100-bundle-output", type=Path, required=True)
    parser.add_argument("--fixture-source", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument(
        "--verify-root", type=Path, default=Path("runs/tmp/gate4a_package_verify")
    )
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parent.parent
    candidate, package_sources = build_candidate(
        project_root,
        checkpoint=args.checkpoint.resolve(),
        run_json=args.run_json.resolve(),
        ablation_path=args.ablation.resolve(),
    )
    args.candidate_output.parent.mkdir(parents=True, exist_ok=True)
    args.candidate_output.write_bytes(_canonical_json(candidate))
    write_deterministic_zip(args.zip_output, package_sources, candidate)
    inspection = inspect_zip(args.zip_output)
    smoke = smoke_verify_zip(args.zip_output, args.verify_root)
    fixture_source = args.fixture_source.resolve()
    if sha256_file(fixture_source) != FROZEN_FIXTURE_SOURCE_SHA256:
        raise ValueError("V100 bundle fixture 不是冻结的 URFD val fall-18-cam0")
    v100_sources = {
        **package_sources,
        "gate4a_fixture/fall-18-cam0.mp4": fixture_source,
    }
    write_deterministic_zip(args.v100_bundle_output, v100_sources, candidate)
    v100_inspection = inspect_zip(args.v100_bundle_output)
    v100_smoke = smoke_verify_v100_bundle(args.v100_bundle_output, args.verify_root)
    report = {
        "protocol": "gate4a_package_dryrun_v1",
        "candidate_id": candidate["candidate_id"],
        "candidate_manifest_sha256": sha256_file(args.candidate_output),
        "zip": inspection,
        "v100_bundle": v100_inspection,
        "smoke": smoke,
        "v100_bundle_smoke": v100_smoke,
        "fixture_split": "urfd/val",
        "ofsyn_test_accessed": False,
    }
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_bytes(_canonical_json(report))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
