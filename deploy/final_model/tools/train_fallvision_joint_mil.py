"""Pretrain the transferable joint stream on FallVision keypoint CSV bags."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from eval.metrics import competition_map
from models.fallvision_joint_mil import FallVisionJointMIL
from models.tcn_window import normalize_pose
from tools.train_multiscale_multistream_mil import aggregate_topk_logits
from tools.train_tcn import (
    _atomic_checkpoint,
    _atomic_json,
    _sha256_file,
    set_deterministic,
)

KEYPOINT_ORDER = (
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip", "Left Knee",
    "Right Knee", "Left Ankle", "Right Ankle",
)
_CANONICAL_SUFFIX = re.compile(r"_(?:resized_)?anonymized_keypoints$|_resized_keypoints$|_keypoints$")


def _canonical_id(path: Path) -> str:
    return _CANONICAL_SUFFIX.sub("", path.stem)


def _variant_priority(path: Path) -> tuple[int, str]:
    name = path.stem.lower()
    return (0 if "anonymized" in name else 1 if "resized" in name else 2, name)


def discover_clips(root: Path) -> list[tuple[str, Path, bool]]:
    """Deduplicate video variants and infer only directory-level fall labels."""
    chosen: dict[tuple[bool, str], Path] = {}
    for directory in sorted(path for path in Path(root).iterdir() if path.is_dir()):
        positive = directory.name.startswith("f_") and not directory.name.startswith("nf_")
        if not positive and not directory.name.startswith("nf_"):
            continue
        for path in directory.rglob("*.csv"):
            key = (positive, _canonical_id(path))
            if key not in chosen or _variant_priority(path) < _variant_priority(chosen[key]):
                chosen[key] = path
    clips = [(f"{'fall' if label else 'no_fall'}/{clip_id}", path, label) for (label, clip_id), path in chosen.items()]
    if not clips or not any(label for _, _, label in clips) or all(label for _, _, label in clips):
        raise ValueError("FallVision 必须同时包含 fall 与 no-fall CSV")
    return sorted(clips)


def _read_pose_sequence(path: Path) -> tuple[np.ndarray, np.ndarray]:
    by_frame: dict[int, dict[str, tuple[float, float, float]]] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                frame = int(row["Frame"])
                values = (float(row["X"]), float(row["Y"]), float(row["Confidence"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"非法 FallVision CSV: {path}") from exc
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"FallVision CSV 含非有限值: {path}")
            by_frame.setdefault(frame, {})[row["Keypoint"]] = values
    if not by_frame:
        return np.empty((0, 51), np.float32), np.empty(0, np.bool_)
    sequence = np.zeros((max(by_frame) - min(by_frame) + 1, 17, 3), np.float32)
    observed = np.zeros(sequence.shape[0], np.bool_)
    for frame, values in by_frame.items():
        keypoints = np.asarray([values.get(name, (0.0, 0.0, 0.0)) for name in KEYPOINT_ORDER], np.float32)
        visible = keypoints[:, 2] >= 0.05
        if visible.sum() < 3:
            continue
        xy = keypoints[visible, :2]
        extent = np.maximum(xy.max(0) - xy.min(0), 1.0)
        bbox = np.r_[xy.min(0) - 0.05 * extent, xy.max(0) + 0.05 * extent]
        offset = frame - min(by_frame)
        sequence[offset] = normalize_pose(keypoints, bbox)
        observed[offset] = True
    return sequence.reshape(sequence.shape[0], 51), observed


def materialize_windows(
    clips: list[tuple[str, Path, bool]], *, window_size: int = 16, stride: int = 2
) -> tuple[torch.Tensor, list[np.ndarray], np.ndarray, list[str]]:
    windows: list[np.ndarray] = []
    groups: list[np.ndarray] = []
    labels: list[float] = []
    clip_ids: list[str] = []
    for clip_id, path, label in clips:
        sequence, observed = _read_pose_sequence(path)
        start = len(windows)
        for end in range(window_size - 1, sequence.shape[0], stride):
            if observed[end - window_size + 1 : end + 1].sum() >= 8:
                windows.append(sequence[end - window_size + 1 : end + 1])
        if len(windows) == start:
            continue
        groups.append(np.arange(start, len(windows), dtype=np.int64))
        labels.append(float(label))
        clip_ids.append(clip_id)
    if not windows or not any(labels) or all(labels):
        raise ValueError("FallVision 没有形成有效的正负窗口 bags")
    return torch.from_numpy(np.asarray(windows)), groups, np.asarray(labels, np.float32), clip_ids


def _split(clips: list[tuple[str, Path, bool]], seed: int) -> tuple[list[tuple[str, Path, bool]], list[tuple[str, Path, bool]]]:
    train, val = [], []
    for item in clips:
        bucket = int.from_bytes(hashlib.sha256(f"{seed}:{item[0]}".encode()).digest()[:4], "big") % 10
        (val if bucket == 0 else train).append(item)
    if not any(item[2] for item in val) or all(item[2] for item in val):
        raise ValueError("确定性 FallVision val 必须同时含正负 clips")
    return train, val


@torch.inference_mode()
def _evaluate(model: nn.Module, x: torch.Tensor, groups: list[np.ndarray], labels: np.ndarray, ids: list[str], device: torch.device, batch_size: int) -> dict[str, float]:
    model.eval(); logits = []
    for start in range(0, x.shape[0], batch_size):
        logits.append(model(x[start : start + batch_size].to(device)).float().cpu())
    probabilities = torch.sigmoid(torch.cat(logits)).numpy()
    scores = np.asarray([probabilities[group].max() for group in groups], np.float32)
    ground_truth = {
        clip_id: bool(label) for clip_id, label in zip(ids, labels, strict=True)
    }
    predictions = {
        clip_id: float(score) for clip_id, score in zip(ids, scores, strict=True)
    }
    result = competition_map(ground_truth, predictions, mode="clip")
    return {"clip_map": float(result["map"]), "clip_map_percent": float(result["map_percent"]), "p_at_r90": float(result["p_at_r90"]), "p_at_r95": float(result["p_at_r95"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("FallVision pretrain 输出目录非空")
    set_deterministic(args.seed); device = torch.device(args.device)
    train_clips, val_clips = _split(discover_clips(args.csv_root), args.seed)
    train_x, train_groups, train_labels, _ = materialize_windows(train_clips)
    val_x, val_groups, val_labels, val_ids = materialize_windows(val_clips)
    model = FallVisionJointMIL().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((train_labels.size - train_labels.sum()) / train_labels.sum(), device=device))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best = -math.inf; history: list[dict[str, Any]] = []
    for epoch in range(args.epochs):
        model.train(); order = np.random.default_rng(args.seed + epoch).permutation(len(train_groups)); total = 0.0
        for start in range(0, len(order), 8):
            selected = order[start : start + 8]
            indices = np.concatenate([train_groups[int(index)] for index in selected])
            sizes = [train_groups[int(index)].size for index in selected]
            optimizer.zero_grad(set_to_none=True)
            clip_logits = aggregate_topk_logits(model(train_x[indices].to(device)), sizes, 0.2)
            target = torch.from_numpy(train_labels[selected]).to(device)
            loss = criterion(clip_logits, target); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            total += float(loss.detach()) * len(selected)
        metrics = _evaluate(model, val_x, val_groups, val_labels, val_ids, device, args.batch_size)
        scheduler.step(); record = {"epoch": epoch, "train_loss": total / len(train_groups), "val": metrics}; history.append(record)
        checkpoint = {"epoch": epoch, "model_state": model.state_dict(), "encoder_state": model.encoder.state_dict(), "metrics": record}
        _atomic_checkpoint(args.output_dir / "last.pt", checkpoint)
        if metrics["clip_map"] > best:
            best = metrics["clip_map"]; _atomic_checkpoint(args.output_dir / "best.pt", checkpoint)
        print(json.dumps(record, ensure_ascii=False, sort_keys=True), flush=True)
    summary = {"protocol": "fallvision_joint_clip_mil_v1", "source": "Harvard Dataverse doi:10.7910/DVN/75QPKK", "train_clips": len(train_groups), "val_clips": len(val_groups), "train_windows": int(train_x.shape[0]), "val_windows": int(val_x.shape[0]), "best_clip_map": best, "best": torch.load(args.output_dir / "best.pt", map_location="cpu", weights_only=False)["metrics"], "input_archive_sha256": {path.name: _sha256_file(path) for path in sorted(args.csv_root.parent.glob("*.rar")) if not path.name.endswith(".partial")}}
    _atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
