"""Gate 3D：在确定性分层子集上验证固定时长 + 全局运动 TCN v2。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from eval.metrics import competition_map
from eval.tcn_ablation import atomic_json, sha256_file
from models.clip_aggregator import aggregate_window_scores_max
from models.tcn import FallTCN, count_params
from models.tcn_features_v2 import FEATURE_DIM, LOCAL_DIM, build_tcn_v2_windows
from models.tcn_window import parse_activity_intervals
from pipeline.pose_cache import pose_cache_path, read_pose_cache
from tools.train_tcn import select_training_indices, set_deterministic, train_epoch

PILOT_PROTOCOL = "fall_tcn_global_motion_pilot_v2"


def _rank(clip_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{clip_id}".encode()).hexdigest()


def _select_rows(
    manifest_path: Path, *, dataset: str, split: str, per_activity: int, seed: int
) -> list[dict[str, str]]:
    if split == "test":
        raise ValueError("Gate 3D pilot 禁止读取 test")
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if row.get("dataset") == dataset and row.get("split") == split
        ]
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["clip_id"].split("/", 1)[0]].append(row)
    selected = []
    for activity in sorted(grouped):
        ordered = sorted(grouped[activity], key=lambda row: (_rank(row["clip_id"], seed), row["clip_id"]))
        if len(ordered) < per_activity:
            raise ValueError(f"{split}/{activity} 样本不足")
        selected.extend(ordered[:per_activity])
    return selected


def _load_audit(path: Path, dataset: str) -> dict[str, Any]:
    audit = json.loads(path.read_text(encoding="utf-8"))
    if audit.get("errors") or audit.get("dataset") not in (None, dataset):
        raise ValueError("pose cache audit 未通过")
    signatures = audit.get("extractor_signatures")
    if not isinstance(signatures, list) or len(signatures) != 1:
        raise ValueError("pose cache audit 提取签名无效")
    if not isinstance(signatures[0], dict):
        raise TypeError("pose cache audit 提取签名必须是对象")
    return signatures[0]


def _materialize_windows(
    rows: list[dict[str, str]], *, pose_cache_root: Path, dataset: str,
    split: str, extractor_signature: dict[str, Any], progress_prefix: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[float] = []
    clip_indices: list[int] = []
    for clip_index, row in enumerate(rows):
        record = read_pose_cache(pose_cache_path(pose_cache_root, dataset, split, row["clip_id"]))
        if (record.dataset, record.split, record.clip_id) != (dataset, split, row["clip_id"]):
            raise ValueError(f"pose cache 身份不匹配: {row['clip_id']}")
        if record.extractor_signature != extractor_signature:
            raise ValueError(f"pose cache 提取签名不匹配: {row['clip_id']}")
        windows = build_tcn_v2_windows(record, parse_activity_intervals(row["events_json"]))
        features.extend(window.features for window in windows)
        labels.extend(float(window.label) for window in windows)
        clip_indices.extend([clip_index] * len(windows))
        if (clip_index + 1) % 100 == 0 or clip_index + 1 == len(rows):
            print(
                json.dumps({"stage": progress_prefix, "clips": clip_index + 1, "total": len(rows), "windows": len(features)}),
                flush=True,
            )
    if not features:
        raise ValueError(f"{split} 没有生成 v2 窗口")
    return (
        np.stack(features).astype(np.float32, copy=False),
        np.asarray(labels, dtype=np.float32),
        np.asarray(clip_indices, dtype=np.int64),
    )


@torch.inference_mode()
def _evaluate(
    model: FallTCN, features: torch.Tensor, clip_indices: np.ndarray,
    rows: list[dict[str, str]], *, device: torch.device, batch_size: int,
) -> tuple[dict[str, Any], np.ndarray]:
    model.eval()
    probabilities = []
    for start in range(0, len(features), batch_size):
        probabilities.append(torch.sigmoid(model(features[start : start + batch_size].to(device))).cpu().numpy())
    window_scores = np.concatenate(probabilities).astype(np.float32, copy=False)
    clip_scores = aggregate_window_scores_max(
        window_scores, clip_indices, len(rows), require_all=False, missing_score=0.0
    )
    labels = {row["clip_id"]: row["has_fall"] == "1" for row in rows}
    scores = {row["clip_id"]: float(clip_scores[index]) for index, row in enumerate(rows)}
    metric = competition_map(labels, scores, mode="clip")
    return {
        "p_at_r90": float(metric["p_at_r90"]),
        "p_at_r95": float(metric["p_at_r95"]),
        "map": float(metric["map"]),
        "map_percent": float(metric["map_percent"]),
    }, clip_scores


def _baseline(ablation_path: Path, rows: list[dict[str, str]]) -> dict[str, Any]:
    ablation = json.loads(ablation_path.read_text(encoding="utf-8"))
    if ablation.get("split") != "val":
        raise ValueError("baseline 必须是 val 消融")
    predictions = {item["clip_id"]: item for item in ablation["predictions"]}
    labels = {row["clip_id"]: row["has_fall"] == "1" for row in rows}
    result = {}
    for mode, key in (
        ("rule_only", "rule_score"), ("tcn_v1", "tcn_score"),
        ("fusion_v1", "selected_fusion_score"),
    ):
        metric = competition_map(labels, {clip_id: float(predictions[clip_id][key]) for clip_id in labels}, mode="clip")
        result[mode] = {
            "p_at_r90": float(metric["p_at_r90"]),
            "p_at_r95": float(metric["p_at_r95"]),
            "map": float(metric["map"]),
            "map_percent": float(metric["map_percent"]),
        }
    return result


def _atomic_checkpoint(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    try:
        torch.save(value, temp_name)
        os.replace(temp_name, path)
    except Exception:
        Path(temp_name).unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pose-cache-root", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--ablation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", default="of-syn")
    parser.add_argument("--train-per-activity", type=int, default=100)
    parser.add_argument("--val-per-activity", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs/batch_size 必须为正")
    set_deterministic(args.seed)
    device = torch.device(args.device)
    extractor_signature = _load_audit(args.audit_report, args.dataset)
    train_rows = _select_rows(args.manifest, dataset=args.dataset, split="train", per_activity=args.train_per_activity, seed=args.seed)
    val_rows = _select_rows(args.manifest, dataset=args.dataset, split="val", per_activity=args.val_per_activity, seed=args.seed)
    train_features, train_labels, _ = _materialize_windows(
        train_rows, pose_cache_root=args.pose_cache_root, dataset=args.dataset,
        split="train", extractor_signature=extractor_signature, progress_prefix="train_windows",
    )
    val_features, _val_labels, val_clip_indices = _materialize_windows(
        val_rows, pose_cache_root=args.pose_cache_root, dataset=args.dataset,
        split="val", extractor_signature=extractor_signature, progress_prefix="val_windows",
    )
    indices = select_training_indices(train_labels, negative_ratio=2.0, seed=args.seed)
    train_x = torch.from_numpy(np.asarray(train_features[indices], dtype=np.float32))
    train_y = torch.from_numpy(np.asarray(train_labels[indices], dtype=np.float32))
    val_x = torch.from_numpy(val_features)
    model = FallTCN(in_dim=FEATURE_DIM, channels=(64, 64, 128), kernel=3, dropout=0.2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    positives = float(train_y.sum())
    pos_weight = float((train_y.numel() - positives) / positives)
    history = []
    best_metric: dict[str, Any] | None = None
    best_epoch = -1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        loss = train_epoch(
            model, optimizer, train_x, train_y, device=device,
            batch_size=args.batch_size, pos_weight=pos_weight, grad_clip=5.0,
            seed=args.seed + epoch,
        )
        metric, _clip_scores = _evaluate(
            model, val_x, val_clip_indices, val_rows, device=device, batch_size=args.batch_size
        )
        scheduler.step()
        record = {"epoch": epoch, "train_loss": loss, "learning_rate": optimizer.param_groups[0]["lr"], "val": metric}
        history.append(record)
        print(json.dumps({"stage": "epoch", **record}, sort_keys=True), flush=True)
        if best_metric is None or metric["map"] > best_metric["map"]:
            best_metric, best_epoch = metric, epoch
            _atomic_checkpoint(
                args.output_dir / "tcn_v2_best.pt",
                {"protocol": PILOT_PROTOCOL, "pilot": True, "epoch": epoch, "feature_dim": FEATURE_DIM, "model_state": model.state_dict()},
            )

    v2_history = history
    v2_best_metric = best_metric
    v2_best_epoch = best_epoch
    set_deterministic(args.seed)
    control_train_x = train_x[:, :, :LOCAL_DIM]
    control_val_x = val_x[:, :, :LOCAL_DIM]
    control_model = FallTCN(
        in_dim=LOCAL_DIM, channels=(64, 64, 128), kernel=3, dropout=0.2
    ).to(device)
    control_optimizer = torch.optim.AdamW(
        control_model.parameters(), lr=1e-3, weight_decay=1e-4
    )
    control_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        control_optimizer, T_max=args.epochs
    )
    control_history = []
    control_best_metric: dict[str, Any] | None = None
    control_best_epoch = -1
    for epoch in range(args.epochs):
        loss = train_epoch(
            control_model,
            control_optimizer,
            control_train_x,
            train_y,
            device=device,
            batch_size=args.batch_size,
            pos_weight=pos_weight,
            grad_clip=5.0,
            seed=args.seed + epoch,
        )
        metric, _clip_scores = _evaluate(
            control_model,
            control_val_x,
            val_clip_indices,
            val_rows,
            device=device,
            batch_size=args.batch_size,
        )
        control_scheduler.step()
        record = {
            "epoch": epoch,
            "train_loss": loss,
            "learning_rate": control_optimizer.param_groups[0]["lr"],
            "val": metric,
        }
        control_history.append(record)
        print(
            json.dumps({"stage": "control_epoch", **record}, sort_keys=True),
            flush=True,
        )
        if control_best_metric is None or metric["map"] > control_best_metric["map"]:
            control_best_metric, control_best_epoch = metric, epoch
            _atomic_checkpoint(
                args.output_dir / "local_control_best.pt",
                {
                    "protocol": PILOT_PROTOCOL,
                    "pilot": True,
                    "epoch": epoch,
                    "feature_dim": LOCAL_DIM,
                    "model_state": control_model.state_dict(),
                },
            )

    assert v2_best_metric is not None and control_best_metric is not None
    baseline = _baseline(args.ablation, val_rows)
    signature = {
        "protocol": PILOT_PROTOCOL,
        "manifest_sha256": sha256_file(args.manifest),
        "audit_sha256": sha256_file(args.audit_report),
        "ablation_sha256": sha256_file(args.ablation),
        "feature_code_sha256": sha256_file(Path(__file__).parent.parent / "models" / "tcn_features_v2.py"),
        "training_code_sha256": sha256_file(Path(__file__)),
        "model_code_sha256": sha256_file(Path(__file__).parent.parent / "models" / "tcn.py"),
        "seed": args.seed,
        "train_per_activity": args.train_per_activity,
        "val_per_activity": args.val_per_activity,
        "epochs": args.epochs,
        "sample_rate_hz": 16.0,
        "window_size": 16,
    }
    signature_sha256 = hashlib.sha256(json.dumps(signature, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    output = {
        "protocol": PILOT_PROTOCOL,
        "pilot": True,
        "dataset": args.dataset,
        "split": "val",
        "train_clips": len(train_rows),
        "val_clips": len(val_rows),
        "val_positive_clips": sum(row["has_fall"] == "1" for row in val_rows),
        "train_windows": len(train_features),
        "selected_train_windows": len(indices),
        "val_windows": len(val_features),
        "feature_dim": FEATURE_DIM,
        "parameter_count": {
            "tcn_v2": count_params(model),
            "local_control": count_params(control_model),
        },
        "baseline_same_subset": baseline,
        "best_epoch": {"tcn_v2": v2_best_epoch, "local_control": control_best_epoch},
        "best_tcn_v2": v2_best_metric,
        "best_local_control": control_best_metric,
        "delta_map_points_vs_same_data_control": 100.0
        * (float(v2_best_metric["map"]) - float(control_best_metric["map"])),
        "delta_map_points_vs_full_train_tcn_v1": 100.0
        * (float(v2_best_metric["map"]) - baseline["tcn_v1"]["map"]),
        "history": {"tcn_v2": v2_history, "local_control": control_history},
        "signature": signature,
        "signature_sha256": signature_sha256,
        "decision_rule": "only proceed to full v2 cache/training if v2 beats the same-data local-only control by >=3 MAP points",
        "test_accessed": False,
    }
    atomic_json(args.output_dir / "result.json", output)
    print(f"wrote={args.output_dir / 'result.json'} sha256={sha256_file(args.output_dir / 'result.json')} signature={signature_sha256}")


if __name__ == "__main__":
    main()
