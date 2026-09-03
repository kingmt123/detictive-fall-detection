"""Probe event-aligned RGB features on one template-grouped OF-Syn train OOF fold.

This is O3a, deliberately a frozen-backbone representation gate rather than the
expensive O3b fine-tuning experiment.  It never selects a model epoch from OOF
metrics, and it refuses to use a cache other than the OF-Syn training cache.
"""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn
from torchvision.models.video import R3D_18_Weights, r3d_18

from eval.metrics import competition_map
from models.tcn_dataset import SEMANTIC_TO_CODE, WindowMemmapCache, load_window_cache
from pipeline.video_source import VideoSourceResolver
from tools.train_long_context_event_oracle import group_fold, template_group
from tools.train_tcn import (
    _atomic_checkpoint,
    _atomic_json,
    _sha256_file,
    set_deterministic,
)


@dataclass(frozen=True)
class O3aConfig:
    frames: int = 16
    crop_size: int = 112
    context_seconds: float = 1.0
    batch_size: int = 8
    decode_workers: int = 8
    epochs: int = 40
    learning_rate: float = 1e-3
    weight_decay: float = 1e-3
    hidden_dim: int = 128
    dropout: float = 0.2
    folds: int = 5
    seed: int = 20260824

    def validate(self) -> None:
        if self.frames < 4 or self.crop_size < 32 or self.batch_size < 1:
            raise ValueError("frames/crop_size/batch_size 无效")
        if not 1 <= self.decode_workers <= self.batch_size:
            raise ValueError("decode_workers 必须位于 [1, batch_size]")
        if self.epochs < 1 or self.hidden_dim < 4 or self.folds < 2:
            raise ValueError("epochs/hidden_dim/folds 无效")
        if self.context_seconds < 0.0 or not 0.0 <= self.dropout < 1.0:
            raise ValueError("context_seconds/dropout 无效")


def load_train_rows(manifest: Path, cache: WindowMemmapCache) -> list[dict[str, str]]:
    """Return only cache-matching OF-Syn train rows before any video is opened."""
    if cache.metadata.get("dataset") != "of-syn" or cache.metadata.get("split") != "train":
        raise ValueError("O3a 只接受 OF-Syn train cache")
    with Path(manifest).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"dataset", "split", "clip_id", "has_fall", "video_path"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"manifest 缺少字段: {sorted(missing)}")
        rows = {
            row["clip_id"]: row
            for row in reader
            if row.get("dataset") == "of-syn" and row.get("split") == "train"
        }
    clip_ids = [str(clip["clip_id"]) for clip in cache.metadata["clips"]]
    if len(rows) != len(set(rows)):
        raise ValueError("OF-Syn train manifest clip_id 重复")
    if set(rows) != set(clip_ids):
        raise ValueError("manifest 与 train cache 的 clip 集合不一致")
    ordered = [rows[clip_id] for clip_id in clip_ids]
    if any(row.get("has_fall") not in {"0", "1"} for row in ordered):
        raise ValueError("manifest has_fall 必须为 0 或 1")
    if any(not row.get("video_path", "").startswith("tar://") for row in ordered):
        raise ValueError("O3a 只接受审计过的 tar URI")
    return ordered


def event_crop_interval(
    end_times: np.ndarray,
    semantic_codes: np.ndarray,
    *,
    has_fall: bool,
    context_seconds: float,
) -> tuple[float, float] | None:
    """Derive an RGB crop from train-cache event semantics; None means full clip."""
    times = np.asarray(end_times, dtype=np.float64)
    codes = np.asarray(semantic_codes)
    if times.ndim != 1 or codes.ndim != 1 or times.size != codes.size:
        raise ValueError("end_times/semantic_codes 必须为等长一维数组")
    if times.size == 0 or context_seconds < 0.0 or not np.isfinite(times).all():
        raise ValueError("事件时间或上下文无效")
    if has_fall:
        selected = (codes == SEMANTIC_TO_CODE["fall_process"]) | (
            codes == SEMANTIC_TO_CODE["post_fall_state"]
        )
        if not selected.any():
            # Some valid positive clips have no eligible pose windows.  Keeping
            # their full RGB timeline is safer than inventing a fall location.
            return None
    else:
        selected = codes == SEMANTIC_TO_CODE["hard_negative"]
        if not selected.any():
            return None
    event_times = times[selected]
    start = max(0.0, float(event_times.min()) - context_seconds)
    end = float(event_times.max()) + context_seconds
    if end <= start:
        raise ValueError("事件裁剪区间为空")
    return start, end


def crop_intervals(
    cache: WindowMemmapCache, *, context_seconds: float
) -> list[tuple[float, float] | None]:
    """Build cache-aligned event crop metadata without opening video sources."""
    ends = cache.array("end_times")
    semantics = cache.array("semantic_codes")
    result: list[tuple[float, float] | None] = []
    for clip in cache.metadata["clips"]:
        start = int(clip["window_start"])
        count = int(clip["window_count"])
        result.append(
            event_crop_interval(
                np.asarray(ends[start : start + count]),
                np.asarray(semantics[start : start + count]),
                has_fall=bool(clip["has_fall"]),
                context_seconds=context_seconds,
            )
        )
    return result


def decode_uniform_event_clip(
    path: Path,
    interval: tuple[float, float] | None,
    *,
    frames: int,
    size: int,
) -> torch.Tensor:
    """Decode a full clip or an event-aligned interval into R3D's input tensor."""
    capture = cv2.VideoCapture(str(path))
    try:
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if count < 2 or not np.isfinite(fps) or fps <= 0.0:
            raise ValueError(f"视频元数据无效: {path}")
        if interval is None:
            left, right = 0, count - 1
        else:
            start_seconds, end_seconds = interval
            left = max(0, min(count - 1, int(np.floor(start_seconds * fps))))
            right = max(left + 1, min(count - 1, int(np.ceil(end_seconds * fps))))
        positions = np.linspace(left, right, frames).round().astype(np.int64)
        decoded: list[np.ndarray] = []
        # A single seek plus sequential reads avoids repeatedly decoding from a
        # keyframe for every sampled position (especially costly for tar sources).
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(positions[0]))
        current = int(positions[0]) - 1
        for position in positions:
            while current < int(position):
                ok, frame = capture.read()
                current += 1
                if not ok or frame is None:
                    raise ValueError(f"视频解码失败: {path}@{position}")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width = frame.shape[:2]
            if min(height, width) < 2:
                raise ValueError(f"视频尺寸无效: {path}")
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
def extract_features(
    rows: list[dict[str, str]],
    intervals: list[tuple[float, float] | None],
    *,
    device: torch.device,
    temp_root: Path,
    progress_path: Path,
    config: O3aConfig,
) -> np.ndarray:
    if len(rows) != len(intervals):
        raise ValueError("rows/intervals 长度不一致")
    backbone = r3d_18(weights=R3D_18_Weights.DEFAULT)
    backbone.fc = nn.Identity()
    backbone.eval().to(device)
    outputs: list[np.ndarray] = []
    pending: list[torch.Tensor] = []
    with VideoSourceResolver(temp_root) as resolver, ThreadPoolExecutor(
        max_workers=config.decode_workers
    ) as pool:
        for batch_start in range(0, len(rows), config.batch_size):
            batch_rows = rows[batch_start : batch_start + config.batch_size]
            batch_intervals = intervals[batch_start : batch_start + config.batch_size]
            # Tar members are copied in a deterministic order, then their costly
            # video decoding runs concurrently while the temporary files live.
            with ExitStack() as stack:
                local_paths = [
                    stack.enter_context(resolver.materialize(row["video_path"])).local_path
                    for row in batch_rows
                ]
                pending.extend(
                    pool.map(
                        lambda pair: decode_uniform_event_clip(
                            pair[0], pair[1], frames=config.frames, size=config.crop_size
                        ),
                        zip(local_paths, batch_intervals, strict=True),
                    )
                )
            outputs.append(backbone(torch.stack(pending).to(device)).float().cpu().numpy())
            pending.clear()
            complete = batch_start + len(batch_rows)
            if complete % 50 < config.batch_size or complete == len(rows):
                progress = {"stage": "frozen_r3d_features", "complete": complete, "total": len(rows)}
                _atomic_json(progress_path, progress)
                print(json.dumps(progress), flush=True)
    return np.concatenate(outputs).astype(np.float32, copy=False)


class MlpHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x).squeeze(1)


def metric_result(ids: list[str], labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    result = competition_map(
        dict(zip(ids, (bool(value) for value in labels), strict=True)),
        dict(zip(ids, (float(value) for value in scores), strict=True)),
        mode="clip",
    )
    return {
        "clip_map": float(result["map"]),
        "clip_map_percent": float(result["map_percent"]),
        "clip_p_at_r90": float(result["p_at_r90"]),
        "clip_p_at_r95": float(result["p_at_r95"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--temp-root", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()
    config = O3aConfig(epochs=args.epochs, seed=args.seed)
    config.validate()
    if not 0 <= args.fold < config.folds:
        raise ValueError("fold 必须位于 [0, folds)")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("O3a 输出目录必须为空")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("请求 CUDA，但当前不可用")
    cache = load_window_cache(args.cache, verify_hashes=True)
    rows = load_train_rows(args.manifest, cache)
    intervals = crop_intervals(cache, context_seconds=config.context_seconds)
    clip_ids = [row["clip_id"] for row in rows]
    labels = np.asarray([row["has_fall"] == "1" for row in rows], dtype=np.float32)
    groups = [template_group(clip_id) for clip_id in clip_ids]
    fold_ids = np.asarray([group_fold(group, folds=config.folds, seed=config.seed) for group in groups])
    training, held_out = np.flatnonzero(fold_ids != args.fold), np.flatnonzero(fold_ids == args.fold)
    if set(np.asarray(groups)[training]) & set(np.asarray(groups)[held_out]):
        raise AssertionError("template group 跨 OOF fold 泄漏")
    if not labels[training].any() or labels[training].all() or not labels[held_out].any() or labels[held_out].all():
        raise ValueError("训练与 OOF fold 均须有正负样本")
    set_deterministic(config.seed + args.fold)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    features = extract_features(
        rows,
        intervals,
        device=device,
        temp_root=args.temp_root,
        progress_path=args.output_dir / "feature_progress.json",
        config=config,
    )
    mean = features[training].mean(0, dtype=np.float64).astype(np.float32)
    std = features[training].std(0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 1e-6)
    normalized = (features - mean) / std
    np.savez_compressed(
        args.output_dir / "rgb_features.npz", features=features, clip_ids=np.asarray(clip_ids), labels=labels,
        intervals=np.asarray([(-1.0, -1.0) if item is None else item for item in intervals], dtype=np.float32),
    )
    x = torch.from_numpy(normalized).to(device)
    y = torch.from_numpy(labels).to(device)
    head = MlpHead(features.shape[1], config.hidden_dim, config.dropout).to(device)
    positive = float(labels[training].sum())
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((training.size - positive) / positive, device=device))
    optimizer = torch.optim.AdamW(head.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    history: list[dict[str, Any]] = []
    for epoch in range(config.epochs):
        head.train()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(head(x[training]), y[training])
        loss.backward()
        optimizer.step()
        head.eval()
        with torch.inference_mode():
            scores = torch.sigmoid(head(x[held_out])).cpu().numpy()
        record = {"epoch": epoch, "train_loss": float(loss.detach()), "oof": metric_result([clip_ids[index] for index in held_out], labels[held_out], scores)}
        history.append(record)
        print(json.dumps(record, ensure_ascii=False, sort_keys=True), flush=True)
    with torch.inference_mode():
        final_scores = torch.sigmoid(head(x[held_out])).cpu().numpy()
    _atomic_checkpoint(args.output_dir / "last.pt", {
        "epoch": config.epochs - 1, "model_state": head.state_dict(), "mean": mean, "std": std,
        "config": asdict(config), "fold": args.fold, "metrics": history[-1]["oof"],
    })
    np.savez_compressed(
        args.output_dir / "oof_predictions.npz", clip_indices=held_out,
        clip_ids=np.asarray([clip_ids[index] for index in held_out]), labels=labels[held_out], scores=final_scores,
        fold=np.full(held_out.size, args.fold, dtype=np.int64),
    )
    crop_kinds = {"event_aligned": sum(item is not None for item in intervals), "full_clip": sum(item is None for item in intervals)}
    summary = {
        "protocol": "template_grouped_train_only_event_aligned_frozen_r3d18_o3a_v1",
        "manifest_sha256": _sha256_file(args.manifest), "cache_signature_sha256": cache.metadata["signature_sha256"],
        "fold": args.fold, "template_groups_total": len(set(groups)), "train_clips": int(training.size),
        "oof_clips": int(held_out.size), "train_positive": int(labels[training].sum()), "oof_positive": int(labels[held_out].sum()),
        "crop_kinds": crop_kinds, "config": asdict(config), "final_metrics": history[-1]["oof"],
        "checkpoint_sha256": _sha256_file(args.output_dir / "last.pt"),
    }
    _atomic_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "history.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in history), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
