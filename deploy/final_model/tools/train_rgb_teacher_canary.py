"""Train a frozen-R3D RGB teacher head on a train-only fall/transition cohort."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn
from torchvision.models.video import R3D_18_Weights, r3d_18

from eval.metrics import competition_map
from models.multiscale_multistream_tcn import MultiStreamMultiScaleAttentionTCN
from models.tcn_dataset import load_window_cache
from pipeline.video_source import VideoSourceResolver
from tools.build_tcn_multistream_sidecar import load_sidecar
from tools.train_tcn import (
    _atomic_checkpoint,
    _atomic_json,
    _sha256_file,
    set_deterministic,
)

ACTIVITIES = ("fall", "lie_down", "lying", "stand_up", "sit_down")


def _bucket(seed: int, clip_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{clip_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big") % 5


def select_train_cohort(
    manifest: Path, *, seed: int, per_activity: int
) -> list[dict[str, str]]:
    """Select only allowlisted OF-Syn train clips before opening any video."""
    with Path(manifest).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    eligible: dict[str, list[dict[str, str]]] = {name: [] for name in ACTIVITIES}
    for row in rows:
        if row.get("dataset") != "of-syn" or row.get("split") != "train":
            continue
        clip_id = row.get("clip_id", "")
        activity = clip_id.split("/", 1)[0]
        if activity in eligible:
            if not str(row.get("video_path", "")).startswith("tar://"):
                raise ValueError("OF-Syn RGB teacher 只接受审计过的 tar URI")
            eligible[activity].append(row)
    selected: list[dict[str, str]] = []
    for activity, candidates in eligible.items():
        if len(candidates) < per_activity:
            raise ValueError(f"{activity} train clips 不足 {per_activity}")
        order = sorted(
            candidates,
            key=lambda row: hashlib.sha256(
                f"{seed}:sample:{row['clip_id']}".encode()
            ).digest(),
        )
        selected.extend(order[:per_activity])
    if any(row["split"] != "train" for row in selected):
        raise AssertionError("RGB cohort 泄漏非 train split")
    return selected


def decode_uniform_clip(path: Path, *, frames: int = 16, size: int = 112) -> torch.Tensor:
    capture = cv2.VideoCapture(str(path))
    try:
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if count < frames:
            raise ValueError(f"视频帧数不足: {path}")
        positions = np.linspace(0, count - 1, frames).round().astype(int)
        decoded: list[np.ndarray] = []
        for position in positions:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(position))
            ok, frame = capture.read()
            if not ok or frame is None:
                raise ValueError(f"视频解码失败: {path}@{position}")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width = frame.shape[:2]
            scale = 128.0 / min(height, width)
            resized = cv2.resize(
                frame,
                (round(width * scale), round(height * scale)),
                interpolation=cv2.INTER_AREA,
            )
            y = (resized.shape[0] - size) // 2
            x = (resized.shape[1] - size) // 2
            decoded.append(resized[y : y + size, x : x + size])
    finally:
        capture.release()
    tensor = torch.from_numpy(np.asarray(decoded)).float().permute(3, 0, 1, 2) / 255.0
    mean = torch.tensor((0.43216, 0.394666, 0.37645)).view(3, 1, 1, 1)
    std = torch.tensor((0.22803, 0.22145, 0.216989)).view(3, 1, 1, 1)
    return (tensor - mean) / std


@torch.inference_mode()
def extract_rgb_features(
    rows: list[dict[str, str]], *, device: torch.device, temp_root: Path
) -> np.ndarray:
    backbone = r3d_18(weights=R3D_18_Weights.DEFAULT)
    backbone.fc = nn.Identity()
    backbone.eval().to(device)
    features: list[np.ndarray] = []
    pending: list[torch.Tensor] = []
    with VideoSourceResolver(temp_root) as resolver:
        for index, row in enumerate(rows):
            with resolver.materialize(row["video_path"]) as video:
                pending.append(decode_uniform_clip(video.local_path))
            if len(pending) == 8 or index + 1 == len(rows):
                batch = torch.stack(pending).to(device)
                features.append(backbone(batch).float().cpu().numpy())
                pending.clear()
            if (index + 1) % 50 == 0:
                print(json.dumps({"stage": "rgb_features", "complete": index + 1, "total": len(rows)}), flush=True)
    return np.concatenate(features).astype(np.float32, copy=False)


def _metrics(ids: list[str], labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    result = competition_map(
        {clip_id: bool(label) for clip_id, label in zip(ids, labels, strict=True)},
        {clip_id: float(score) for clip_id, score in zip(ids, scores, strict=True)},
        mode="clip",
    )
    return {
        "map": float(result["map"]),
        "p_at_r90": float(result["p_at_r90"]),
        "p_at_r95": float(result["p_at_r95"]),
    }


@torch.inference_mode()
def student_scores(
    rows: list[dict[str, str]],
    *,
    window_cache: Path,
    sidecar_root: Path,
    checkpoint: Path,
    device: torch.device,
) -> np.ndarray:
    cache = load_window_cache(window_cache)
    sidecar = load_sidecar(sidecar_root, cache)
    metadata = {clip["clip_id"]: clip for clip in cache.metadata["clips"]}
    model = MultiStreamMultiScaleAttentionTCN(
        stream_channels=32,
        stream_output_dim=64,
        dropout=0.5,
        use_geometry=True,
        use_rule_features=True,
        use_discriminative_kinematics=True,
        kinematic_feature_dim=15,
    ).to(device)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    pose = cache.array("features")
    scores: list[float] = []
    for row in rows:
        clip = metadata[row["clip_id"]]
        start, count = int(clip["window_start"]), int(clip["window_count"])
        x = torch.from_numpy(np.asarray(pose[start : start + count]).copy()).flatten(2)
        geometry = torch.from_numpy(np.asarray(sidecar[start : start + count]).copy())
        x = torch.cat((x, geometry), dim=2)
        logits = []
        for offset in range(0, count, 512):
            logits.append(model(x[offset : offset + 512].to(device)).cpu())
        value = torch.cat(logits)
        tau = 0.05
        score = tau * (torch.logsumexp(value / tau, dim=0) - math.log(count))
        scores.append(float(torch.sigmoid(score)))
    return np.asarray(scores, np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--window-cache", type=Path, required=True)
    parser.add_argument("--sidecar-root", type=Path, required=True)
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--temp-root", type=Path, required=True)
    parser.add_argument("--per-activity", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("RGB teacher 输出目录必须为空")
    if args.per_activity < 10:
        raise ValueError("per_activity 必须至少为 10")
    set_deterministic(args.seed)
    device = torch.device(args.device)
    rows = select_train_cohort(
        args.manifest, seed=args.seed, per_activity=args.per_activity
    )
    ids = [row["clip_id"] for row in rows]
    labels = np.asarray([row["has_fall"] == "1" for row in rows], np.float32)
    holdout = np.asarray([_bucket(args.seed, clip_id) == 0 for clip_id in ids])
    if not labels[holdout].any() or labels[holdout].all():
        raise ValueError("train-only holdout 必须同时包含正负样本")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    features = extract_rgb_features(rows, device=device, temp_root=args.temp_root)
    np.savez_compressed(
        args.output_dir / "rgb_features.npz",
        features=features,
        labels=labels,
        clip_ids=np.asarray(ids),
        holdout=holdout,
    )
    train = ~holdout
    head = nn.Linear(features.shape[1], 1).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-3)
    x = torch.from_numpy(features).to(device)
    y = torch.from_numpy(labels).to(device)
    best = -math.inf
    for epoch in range(30):
        head.train()
        optimizer.zero_grad(set_to_none=True)
        logits = head(x[train]).squeeze(1)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y[train])
        loss.backward()
        optimizer.step()
        head.eval()
        with torch.inference_mode():
            scores = torch.sigmoid(head(x[holdout]).squeeze(1)).cpu().numpy()
        metrics = _metrics(np.asarray(ids)[holdout].tolist(), labels[holdout], scores)
        checkpoint = {"epoch": epoch, "model_state": head.state_dict(), "metrics": metrics}
        if metrics["map"] > best:
            best = metrics["map"]
            _atomic_checkpoint(args.output_dir / "best.pt", checkpoint)
    best_payload = torch.load(args.output_dir / "best.pt", map_location=device, weights_only=False)
    head.load_state_dict(best_payload["model_state"])
    head.eval()
    with torch.inference_mode():
        teacher = torch.sigmoid(head(x[holdout]).squeeze(1)).cpu().numpy()
    student = student_scores(
        [row for row, chosen in zip(rows, holdout, strict=True) if chosen],
        window_cache=args.window_cache,
        sidecar_root=args.sidecar_root,
        checkpoint=args.student_checkpoint,
        device=device,
    )
    holdout_ids = np.asarray(ids)[holdout].tolist()
    summary: dict[str, Any] = {
        "protocol": "ofsyn_train_only_frozen_r3d18_teacher_canary_v1",
        "manifest_sha256": _sha256_file(args.manifest),
        "student_checkpoint_sha256": _sha256_file(args.student_checkpoint),
        "cohort_size": len(rows),
        "holdout_size": int(holdout.sum()),
        "teacher": _metrics(holdout_ids, labels[holdout], teacher),
        "student": _metrics(holdout_ids, labels[holdout], student),
        "best_epoch": int(best_payload["epoch"]),
    }
    _atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
