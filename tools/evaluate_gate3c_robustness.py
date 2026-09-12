"""在确定性 OF-Syn val 子集上执行 Gate 3C 图像扰动压力测试。"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import cv2
import torch

from eval.robustness import (
    CONDITIONS,
    ROBUSTNESS_PROTOCOL,
    activity_counts,
    frame_transform,
    load_stratified_val_rows,
    robustness_signature,
    summarize_fixed_thresholds,
)
from eval.tcn_ablation import atomic_json, sha256_file
from pipeline.inference_engine import InferenceEngine, crop_frame
from pipeline.video_source import VideoSourceResolver


def _resolve_source(video_path: str, manifest_path: Path) -> str:
    if video_path.startswith("tar://") or Path(video_path).is_absolute():
        return video_path
    root = manifest_path.parent.parent if manifest_path.parent.name == "data" else manifest_path.parent
    return str(root / video_path)


def _thresholds(ablation: dict[str, Any]) -> dict[str, dict[str, float]]:
    if ablation.get("split") != "val":
        raise ValueError("robustness 只允许读取 validation 消融阈值")
    result: dict[str, dict[str, float]] = {}
    for mode in ("rule_only", "tcn_only", "selected_fusion"):
        metrics = ablation[mode]["metrics"] if mode == "selected_fusion" else ablation[mode]
        result[mode] = {
            point: float(metrics[f"{point}_point"]["threshold"])
            for point in ("r90", "r95")
        }
    return result


def _load_progress(path: Path, signature: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"progress 第 {line_number} 行损坏") from exc
        if item.get("signature_sha256") != signature:
            raise ValueError("progress 与本次 robustness 签名不匹配")
        rows.append(item["result"])
    return rows


def _write_grid(path: Path, local_video: Path, seed: int) -> None:
    capture = cv2.VideoCapture(str(local_video))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise ValueError("无法读取 augmentation grid 首帧")
    view, _ = crop_frame(frame, "auto")
    panels = []
    for condition in CONDITIONS:
        panel = frame_transform(condition, seed=seed)(view, 0)
        cv2.putText(panel, condition, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
        panels.append(panel)
    grid = cv2.hconcat(panels)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), grid):
        raise ValueError(f"无法写 augmentation grid: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ablation", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grid-output", type=Path)
    parser.add_argument("--temp-root", type=Path, required=True)
    parser.add_argument("--dataset", default="of-syn")
    parser.add_argument("--per-activity", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--device", default="0")
    parser.add_argument("--model", default="yolo11n-pose.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.10)
    args = parser.parse_args()

    ablation = json.loads(args.ablation.read_text(encoding="utf-8"))
    thresholds = _thresholds(ablation)
    selected = load_stratified_val_rows(
        args.manifest, dataset=args.dataset, per_activity=args.per_activity, seed=args.seed
    )
    signature_sha256, signature = robustness_signature(
        manifest_path=args.manifest, ablation_path=args.ablation,
        checkpoint_path=args.checkpoint, run_path=args.run_json,
        per_activity=args.per_activity, seed=args.seed,
    )
    progress_path = args.output.with_suffix(".progress.jsonl")
    results = _load_progress(progress_path, signature_sha256)
    completed = {(row["clip_id"], row["condition"]) for row in results}
    engine = InferenceEngine(
        model_path=args.model, device=args.device, image_size=args.imgsz,
        confidence=args.confidence, temporal_mode="fusion",
        tcn_checkpoint=args.checkpoint, tcn_run_json=args.run_json,
        tcn_weight=float(ablation["selected_fusion"]["tcn_weight"]),
    )
    started = time.perf_counter()
    total = len(selected) * len(CONDITIONS)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with VideoSourceResolver(args.temp_root) as resolver:
        for clip_position, row in enumerate(selected):
            source = _resolve_source(row["video_path"], args.manifest)
            with resolver.materialize(source) as materialized:
                if clip_position == 0 and args.grid_output is not None and not args.grid_output.exists():
                    clip_seed = int(hashlib.sha256(row["clip_id"].encode()).hexdigest()[:8], 16) ^ args.seed
                    _write_grid(args.grid_output, materialized.local_path, clip_seed)
                for condition in CONDITIONS:
                    key = (row["clip_id"], condition)
                    if key in completed:
                        continue
                    clip_seed = int(hashlib.sha256(f"{row['clip_id']}:{condition}".encode()).hexdigest()[:8], 16) ^ args.seed
                    payload = engine.analyze(
                        materialized.local_path, render=False, crop="auto",
                        frame_transform=frame_transform(condition, seed=clip_seed),
                    )
                    scores = payload["component_clip_scores"]
                    result = {
                        "clip_id": row["clip_id"],
                        "activity": row["clip_id"].split("/", 1)[0],
                        "has_fall": row["has_fall"] == "1",
                        "condition": condition,
                        "rule_score": float(scores["rule"]),
                        "tcn_score": float(scores["tcn"]),
                        "fusion_score": float(scores["fusion"]),
                        "processed_frames": int(payload["processed_frames"]),
                        "pose_coverage": payload["pose_coverage"],
                    }
                    with progress_path.open("a", encoding="utf-8", newline="\n") as handle:
                        handle.write(json.dumps({"signature_sha256": signature_sha256, "result": result}, sort_keys=True) + "\n")
                        handle.flush()
                    results.append(result)
                    completed.add(key)
                    if len(completed) % 10 == 0 or len(completed) == total:
                        print(f"progress={len(completed)}/{total} clip={row['clip_id']} condition={condition}", flush=True)

    if len(completed) != total:
        raise RuntimeError(f"robustness 结果不完整: {len(completed)}/{total}")
    coverage_by_condition = {}
    for condition in CONDITIONS:
        subset = [row for row in results if row["condition"] == condition]
        coverage_by_condition[condition] = {
            "mean_frame_fraction": sum(row["pose_coverage"]["frame_fraction"] for row in subset) / len(subset),
            "frames_with_pose": sum(row["pose_coverage"]["frames_with_pose"] for row in subset),
            "processed_frames": sum(row["processed_frames"] for row in subset),
            "observations": sum(row["pose_coverage"]["observations"] for row in subset),
        }
    output = {
        "protocol": ROBUSTNESS_PROTOCOL,
        "dataset": args.dataset,
        "split": "val",
        "subset_clip_count": len(selected),
        "positive_clips": sum(row["has_fall"] == "1" for row in selected),
        "activity_counts": activity_counts(selected),
        "thresholds_from_full_clean_val": thresholds,
        "summary": summarize_fixed_thresholds(results, thresholds),
        "pose_coverage": coverage_by_condition,
        "results": sorted(results, key=lambda row: (row["clip_id"], CONDITIONS.index(row["condition"]))),
        "signature": {**signature, "engine_signature": engine.cache_signature()},
        "signature_sha256": signature_sha256,
        "runtime": {
            "seconds_this_invocation": time.perf_counter() - started,
            "device": args.device,
            "cuda_device": torch.cuda.get_device_name(int(args.device)) if torch.cuda.is_available() and args.device.isdigit() else None,
        },
        "limitations": [
            "subset MAP is descriptive only; all operating thresholds come from full clean validation",
            "image corruptions are deterministic synthetic stress tests, not estimates of deployment prevalence",
            "test split is neither selected nor decoded",
        ],
    }
    atomic_json(args.output, output)
    print(f"wrote={args.output} sha256={sha256_file(args.output)} signature={signature_sha256}")


if __name__ == "__main__":
    main()
