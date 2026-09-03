"""在完整 validation 上评估冻结 TCN 的固定时间重采样 canary。"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from eval.tcn_ablation import (
    atomic_json,
    load_trained_tcn,
    metric_summary,
    sha256_file,
)
from models.clip_aggregator import aggregate_window_scores_max
from models.tcn_features_v2 import LOCAL_DIM, build_tcn_v2_windows
from models.tcn_window import parse_activity_intervals
from pipeline.pose_cache import pose_cache_path, read_pose_cache
from tools.train_tcn_v2_pilot import _load_audit

PROTOCOL = "fall_tcn_fixed_time_canary_v1"
WINDOW_SIZE = 16


def sample_rate_for_duration(duration_seconds: float) -> float:
    if duration_seconds <= 0:
        raise ValueError("duration 必须为正")
    return (WINDOW_SIZE - 1) / duration_seconds


def _load_rows(manifest: Path, *, dataset: str) -> list[dict[str, str]]:
    with manifest.open(encoding="utf-8", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("dataset") == dataset and row.get("split") == "val"
        ]
    if not rows:
        raise ValueError("manifest 没有 validation 行")
    if len({row["clip_id"] for row in rows}) != len(rows):
        raise ValueError("validation clip_id 重复")
    return sorted(rows, key=lambda row: row["clip_id"])


@torch.inference_mode()
def _predict_duration(
    rows: list[dict[str, str]],
    *,
    duration: float,
    dataset: str,
    pose_cache_root: Path,
    extractor_signature: dict[str, Any],
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, float], int]:
    features: list[np.ndarray] = []
    clip_indices: list[int] = []
    sample_rate_hz = sample_rate_for_duration(duration)
    for clip_index, row in enumerate(rows):
        record = read_pose_cache(
            pose_cache_path(pose_cache_root, dataset, "val", row["clip_id"])
        )
        if record.extractor_signature != extractor_signature:
            raise ValueError(f"pose cache 提取签名不匹配: {row['clip_id']}")
        windows = build_tcn_v2_windows(
            record,
            parse_activity_intervals(row["events_json"]),
            window_size=WINDOW_SIZE,
            sample_rate_hz=sample_rate_hz,
            min_observed_frames=8,
        )
        features.extend(window.features[:, :LOCAL_DIM] for window in windows)
        clip_indices.extend([clip_index] * len(windows))
        if (clip_index + 1) % 200 == 0 or clip_index + 1 == len(rows):
            print(
                json.dumps(
                    {
                        "stage": "windows",
                        "duration": duration,
                        "clips": clip_index + 1,
                        "total": len(rows),
                        "windows": len(features),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if not features:
        raise ValueError(f"duration={duration} 未生成窗口")
    array = np.stack(features).astype(np.float32, copy=False)
    probabilities = []
    model.eval()
    for start in range(0, len(array), batch_size):
        batch = torch.from_numpy(array[start : start + batch_size]).to(device)
        probabilities.append(torch.sigmoid(model(batch)).cpu().numpy())
    window_scores = np.concatenate(probabilities).astype(np.float32, copy=False)
    clip_scores = aggregate_window_scores_max(
        window_scores, np.asarray(clip_indices), len(rows),
        require_all=False, missing_score=0.0,
    )
    return {
        row["clip_id"]: float(clip_scores[index]) for index, row in enumerate(rows)
    }, len(array)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pose-cache-root", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-json", type=Path, required=True)
    parser.add_argument("--baseline-ablation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", default="of-syn")
    parser.add_argument("--durations", type=float, nargs="+", default=[0.75, 1.0, 1.5])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("输出已存在，拒绝覆盖")
    if len(set(args.durations)) != len(args.durations) or args.batch_size < 1:
        raise ValueError("durations 不能重复且 batch-size 必须为正")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("请求 CUDA，但当前不可用")
    rows = _load_rows(args.manifest, dataset=args.dataset)
    labels = {row["clip_id"]: row["has_fall"] == "1" for row in rows}
    extractor_signature = _load_audit(args.audit_report, args.dataset)
    model, run, checkpoint = load_trained_tcn(
        args.checkpoint, args.run_json, device=device
    )
    baseline = json.loads(args.baseline_ablation.read_text(encoding="utf-8"))
    if baseline.get("split") != "val":
        raise ValueError("baseline-ablation 必须来自 validation")
    baseline_metrics = baseline["tcn_only"]
    results = []
    for duration in args.durations:
        scores, window_count = _predict_duration(
            rows,
            duration=duration,
            dataset=args.dataset,
            pose_cache_root=args.pose_cache_root,
            extractor_signature=extractor_signature,
            model=model,
            device=device,
            batch_size=args.batch_size,
        )
        metrics = metric_summary(labels, scores)
        results.append(
            {
                "duration_seconds": duration,
                "sample_rate_hz": sample_rate_for_duration(duration),
                "window_count": window_count,
                "metrics": metrics,
                "delta_map_points": 100.0 * (metrics["map"] - baseline_metrics["map"]),
                "delta_p_at_r95_points": 100.0
                * (metrics["p_at_r95"] - baseline_metrics["p_at_r95"]),
                "predictions": scores,
            }
        )
    selected = max(
        results,
        key=lambda item: (
            min(item["delta_map_points"], item["delta_p_at_r95_points"]),
            item["delta_map_points"] + item["delta_p_at_r95_points"],
        ),
    )
    payload = {
        "protocol": PROTOCOL,
        "pilot": True,
        "dataset": args.dataset,
        "split": "val",
        "clip_count": len(rows),
        "positive_clips": sum(labels.values()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "run_signature_sha256": run["signature_sha256"],
        "baseline": {
            "map": baseline_metrics["map"],
            "p_at_r95": baseline_metrics["p_at_r95"],
        },
        "results": results,
        "selected_duration_seconds": selected["duration_seconds"],
        "proceed_to_fair_training": bool(
            selected["delta_map_points"] > 0.0
            and selected["delta_p_at_r95_points"] > 0.0
        ),
        "decision_rule": (
            "only run fair fixed-time training if one frozen-checkpoint duration improves "
            "both full-val MAP and P@R95 over frame-index preprocessing"
        ),
        "artifact_sha256": {
            "manifest": sha256_file(args.manifest),
            "audit_report": sha256_file(args.audit_report),
            "checkpoint": sha256_file(args.checkpoint),
            "run_json": sha256_file(args.run_json),
            "baseline_ablation": sha256_file(args.baseline_ablation),
            "canary_code": sha256_file(Path(__file__)),
        },
        "test_accessed": False,
    }
    atomic_json(args.output, payload)
    print(
        json.dumps(
            {
                "stage": "complete",
                "output": str(args.output.resolve()),
                "selected_duration_seconds": payload["selected_duration_seconds"],
                "proceed_to_fair_training": payload["proceed_to_fair_training"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
