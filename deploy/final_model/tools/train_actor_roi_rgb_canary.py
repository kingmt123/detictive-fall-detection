"""Train a train-only actor/context ROI RGB representation canary.

R0 intentionally probes information rather than deployment cost. It uses the
audited pose track to crop person and 1.5x context tubes, a shared frozen R3D-18
encoder, and one fixed-final-epoch MLP on a template-disjoint OF-Syn train fold.
Project validation and every test split are rejected.
"""

from __future__ import annotations

import argparse
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

from models.tcn_dataset import SEMANTIC_TO_CODE, WindowMemmapCache, load_window_cache
from pipeline.pose_cache import PoseCacheRecord, pose_cache_path, read_pose_cache
from pipeline.video_source import VideoSourceResolver
from tools.evaluate_oof_multistream_fusion import _logit
from tools.train_event_aligned_rgb_oracle import (
    MlpHead,
    crop_intervals,
    load_train_rows,
    metric_result,
)
from tools.train_long_context_event_oracle import group_fold, template_group
from tools.train_tcn import (
    _atomic_checkpoint,
    _atomic_json,
    _sha256_file,
    set_deterministic,
)


@dataclass(frozen=True)
class ActorRoiConfig:
    frames: int = 16
    crop_size: int = 112
    context_scale: float = 1.5
    context_seconds: float = 1.0
    batch_size: int = 4
    decode_workers: int = 4
    epochs: int = 40
    learning_rate: float = 1e-3
    weight_decay: float = 1e-3
    hidden_dim: int = 128
    dropout: float = 0.2
    folds: int = 5
    seed: int = 20260825

    def validate(self) -> None:
        if self.frames < 4 or self.crop_size < 32 or self.batch_size < 1:
            raise ValueError("frames/crop_size/batch_size 无效")
        if not 1 <= self.decode_workers <= self.batch_size:
            raise ValueError("decode_workers 必须位于 [1,batch_size]")
        if self.context_scale <= 1.0 or self.context_seconds < 0.0:
            raise ValueError("context crop 配置无效")
        if self.epochs < 1 or self.hidden_dim < 4 or self.folds < 2:
            raise ValueError("epochs/hidden_dim/folds 无效")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout 无效")


def primary_track_ids(cache: WindowMemmapCache) -> np.ndarray:
    """Choose the track contributing most target-semantic windows per clip."""
    tracks = cache.array("track_ids")
    semantics = cache.array("semantic_codes")
    selected_tracks: list[int] = []
    for clip in cache.metadata["clips"]:
        start = int(clip["window_start"])
        count = int(clip["window_count"])
        clip_tracks = np.asarray(tracks[start : start + count], dtype=np.int64)
        clip_semantics = np.asarray(semantics[start : start + count])
        if bool(clip["has_fall"]):
            target = (clip_semantics == SEMANTIC_TO_CODE["fall_process"]) | (
                clip_semantics == SEMANTIC_TO_CODE["post_fall_state"]
            )
        else:
            target = clip_semantics == SEMANTIC_TO_CODE["hard_negative"]
        candidates = clip_tracks[target] if target.any() else clip_tracks
        if candidates.size == 0 or np.any(candidates < 0):
            raise ValueError(f"clip 没有有效主 track: {clip['clip_id']}")
        values, counts = np.unique(candidates, return_counts=True)
        selected_tracks.append(int(values[np.flatnonzero(counts == counts.max())[0]]))
    return np.asarray(selected_tracks, dtype=np.int64)


def expand_box(
    box: np.ndarray, *, scale: float, frame_height: int, frame_width: int
) -> np.ndarray:
    """Expand an xyxy box around its center and clip it to the frame."""
    values = np.asarray(box, dtype=np.float32)
    if values.shape != (4,) or scale <= 0.0:
        raise ValueError("bbox/scale 无效")
    x1, y1, x2, y2 = map(float, values)
    if x2 <= x1 or y2 <= y1 or frame_height < 2 or frame_width < 2:
        raise ValueError("bbox/frame geometry 无效")
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    half_width = (x2 - x1) * scale * 0.5
    half_height = (y2 - y1) * scale * 0.5
    expanded = np.asarray(
        [
            np.clip(cx - half_width, 0.0, frame_width - 1.0),
            np.clip(cy - half_height, 0.0, frame_height - 1.0),
            np.clip(cx + half_width, 1.0, float(frame_width)),
            np.clip(cy + half_height, 1.0, float(frame_height)),
        ],
        dtype=np.float32,
    )
    if expanded[2] - expanded[0] < 1.0 or expanded[3] - expanded[1] < 1.0:
        raise ValueError("裁剪后的 bbox 为空")
    return expanded


def track_box_trajectory(
    record: PoseCacheRecord, track_id: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return frames, boxes, and keypoint quality for exactly one audited track."""
    frames: list[int] = []
    boxes: list[np.ndarray] = []
    quality: list[float] = []
    for frame_index in range(record.frame_indices.size):
        columns = np.flatnonzero(
            record.valid_mask[frame_index]
            & (record.track_ids[frame_index] == track_id)
        )
        if columns.size > 1:
            raise ValueError("同一帧出现重复 track_id")
        if columns.size == 1:
            column = int(columns[0])
            frames.append(frame_index)
            boxes.append(record.bboxes[frame_index, column])
            quality.append(float(record.keypoints[frame_index, column, :, 2].mean()))
    if not frames:
        raise ValueError(f"pose cache 缺少 track_id={track_id}")
    return (
        np.asarray(frames, dtype=np.int64),
        np.asarray(boxes, dtype=np.float32),
        np.asarray(quality, dtype=np.float32),
    )


def nearest_track_boxes(
    sampled_frames: np.ndarray,
    observed_frames: np.ndarray,
    boxes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Use the nearest same-track observation and report exact-observation flags."""
    sampled = np.asarray(sampled_frames, dtype=np.int64)
    observed = np.asarray(observed_frames, dtype=np.int64)
    if sampled.ndim != 1 or observed.ndim != 1 or observed.size != len(boxes):
        raise ValueError("sampled/observed/boxes 不兼容")
    distances = np.abs(sampled[:, None] - observed[None, :])
    nearest = distances.argmin(axis=1)
    return np.asarray(boxes)[nearest], distances[np.arange(sampled.size), nearest] == 0


def _crop(frame: np.ndarray, box: np.ndarray, size: int) -> np.ndarray:
    x1, y1, x2, y2 = map(int, np.rint(box))
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError("ROI crop 为空")
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def decode_actor_tubes(
    path: Path,
    record: PoseCacheRecord,
    track_id: int,
    interval: tuple[float, float] | None,
    *,
    config: ActorRoiConfig,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Decode aligned person/context tubes and compact track-quality diagnostics."""
    capture = cv2.VideoCapture(str(path))
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if frame_count < 2 or fps <= 0.0 or not np.isfinite(fps):
            raise ValueError(f"视频元数据无效: {path}")
        if abs(frame_count - record.frame_indices.size) > 1:
            raise ValueError("视频与 pose cache 帧数不一致")
        if not np.isclose(fps, record.fps, rtol=0.0, atol=1e-3):
            raise ValueError("视频与 pose cache FPS 不一致")
        if interval is None:
            left, right = 0, frame_count - 1
        else:
            left = max(0, min(frame_count - 1, int(np.floor(interval[0] * fps))))
            right = max(left + 1, min(frame_count - 1, int(np.ceil(interval[1] * fps))))
        positions = np.linspace(left, right, config.frames).round().astype(np.int64)
        observed_frames, observed_boxes, keypoint_quality = track_box_trajectory(
            record, track_id
        )
        sampled_boxes, exact = nearest_track_boxes(
            positions, observed_frames, observed_boxes
        )
        person_frames: list[np.ndarray] = []
        context_frames: list[np.ndarray] = []
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(positions[0]))
        current = int(positions[0]) - 1
        height, width = map(int, record.frame_size)
        for position, box in zip(positions, sampled_boxes, strict=True):
            while current < int(position):
                ok, frame = capture.read()
                current += 1
                if not ok or frame is None:
                    raise ValueError(f"视频解码失败: {path}@{position}")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            person_box = expand_box(
                box, scale=1.0, frame_height=height, frame_width=width
            )
            context_box = expand_box(
                box,
                scale=config.context_scale,
                frame_height=height,
                frame_width=width,
            )
            person_frames.append(_crop(frame, person_box, config.crop_size))
            context_frames.append(_crop(frame, context_box, config.crop_size))
    finally:
        capture.release()

    def normalize(values: list[np.ndarray]) -> torch.Tensor:
        tensor = torch.from_numpy(np.asarray(values)).float().permute(3, 0, 1, 2) / 255.0
        mean = torch.tensor((0.43216, 0.394666, 0.37645)).view(3, 1, 1, 1)
        std = torch.tensor((0.22803, 0.22145, 0.216989)).view(3, 1, 1, 1)
        return (tensor - mean) / std

    areas = (sampled_boxes[:, 2] - sampled_boxes[:, 0]) * (
        sampled_boxes[:, 3] - sampled_boxes[:, 1]
    )
    quality = np.asarray(
        [
            exact.mean(),
            keypoint_quality.mean(),
            keypoint_quality.min(),
            areas.mean() / float(height * width),
            np.abs(np.diff(sampled_boxes[:, :2], axis=0)).mean()
            / max(height, width),
        ],
        dtype=np.float32,
    )
    return normalize(person_frames), normalize(context_frames), quality


@torch.inference_mode()
def extract_actor_features(
    rows: list[dict[str, str]],
    intervals: list[tuple[float, float] | None],
    track_ids: np.ndarray,
    *,
    pose_cache_root: Path,
    device: torch.device,
    temp_root: Path,
    output_dir: Path,
    config: ActorRoiConfig,
) -> tuple[np.ndarray, np.ndarray]:
    if not (len(rows) == len(intervals) == len(track_ids)):
        raise ValueError("rows/intervals/track_ids 长度不一致")
    backbone = r3d_18(weights=R3D_18_Weights.DEFAULT)
    backbone.fc = nn.Identity()
    backbone.eval().to(device)
    features: list[np.ndarray] = []
    qualities: list[np.ndarray] = []
    with VideoSourceResolver(temp_root) as resolver, ThreadPoolExecutor(
        max_workers=config.decode_workers
    ) as pool:
        for batch_start in range(0, len(rows), config.batch_size):
            batch_rows = rows[batch_start : batch_start + config.batch_size]
            batch_intervals = intervals[batch_start : batch_start + config.batch_size]
            batch_tracks = track_ids[batch_start : batch_start + config.batch_size]
            records = [
                read_pose_cache(
                    pose_cache_path(
                        pose_cache_root, "of-syn", "train", row["clip_id"]
                    )
                )
                for row in batch_rows
            ]
            with ExitStack() as stack:
                paths = [
                    stack.enter_context(resolver.materialize(row["video_path"])).local_path
                    for row in batch_rows
                ]
                decoded = list(
                    pool.map(
                        lambda item: decode_actor_tubes(
                            item[0],
                            item[1],
                            int(item[2]),
                            item[3],
                            config=config,
                        ),
                        zip(
                            paths,
                            records,
                            batch_tracks,
                            batch_intervals,
                            strict=True,
                        ),
                    )
                )
            person = torch.stack([item[0] for item in decoded])
            context = torch.stack([item[1] for item in decoded])
            encoded = backbone(torch.cat((person, context)).to(device)).float().cpu().numpy()
            split = len(decoded)
            features.append(np.concatenate((encoded[:split], encoded[split:]), axis=1))
            qualities.extend(item[2] for item in decoded)
            complete = batch_start + split
            if complete % 50 < config.batch_size or complete == len(rows):
                progress = {"stage": "actor_roi_features", "complete": complete, "total": len(rows)}
                _atomic_json(output_dir / "feature_progress.json", progress)
                print(json.dumps(progress), flush=True)
    return (
        np.concatenate(features).astype(np.float32, copy=False),
        np.asarray(qualities, dtype=np.float32),
    )


def comparison_with_skeleton(
    rgb_ids: list[str],
    labels: np.ndarray,
    rgb_scores: np.ndarray,
    skeleton_oof: Path,
) -> dict[str, Any]:
    with np.load(skeleton_oof, allow_pickle=False) as payload:
        skeleton_ids = [str(value) for value in payload["clip_ids"]]
        skeleton_scores = np.asarray(payload["scores"], dtype=np.float64)
        skeleton_labels = np.asarray(payload["labels"], dtype=np.float32)
    index = {clip_id: offset for offset, clip_id in enumerate(skeleton_ids)}
    if set(index) != set(rgb_ids):
        raise ValueError("RGB 与 skeleton OOF clip 集合不一致")
    order = np.asarray([index[clip_id] for clip_id in rgb_ids], dtype=np.int64)
    skeleton_scores = skeleton_scores[order]
    if not np.array_equal(skeleton_labels[order], labels):
        raise ValueError("RGB 与 skeleton OOF 标签不一致")
    rgb_error = rgb_scores - labels
    skeleton_error = skeleton_scores - labels
    fused_logits = 0.5 * _logit(rgb_scores) + 0.5 * _logit(skeleton_scores)
    fused_scores = 1.0 / (1.0 + np.exp(-np.clip(fused_logits, -40.0, 40.0)))
    return {
        "score_pearson": float(np.corrcoef(rgb_scores, skeleton_scores)[0, 1]),
        "error_pearson": float(np.corrcoef(rgb_error, skeleton_error)[0, 1]),
        "skeleton_metrics": metric_result(rgb_ids, labels, skeleton_scores),
        "fixed_equal_logit_fusion": metric_result(rgb_ids, labels, fused_scores),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--pose-cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--temp-root", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--skeleton-oof", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--decode-workers", type=int, default=8)
    args = parser.parse_args()
    config = ActorRoiConfig(
        epochs=args.epochs,
        seed=args.seed,
        batch_size=args.batch_size,
        decode_workers=args.decode_workers,
    )
    config.validate()
    if not 0 <= args.fold < config.folds:
        raise ValueError("fold 必须位于 [0,folds)")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("R0 输出目录必须为空")
    cache = load_window_cache(args.cache, verify_hashes=True)
    if cache.metadata.get("dataset") != "of-syn" or cache.metadata.get("split") != "train":
        raise ValueError("R0 只接受 OF-Syn train cache")
    rows = load_train_rows(args.manifest, cache)
    intervals = crop_intervals(cache, context_seconds=config.context_seconds)
    tracks = primary_track_ids(cache)
    clip_ids = [row["clip_id"] for row in rows]
    labels = np.asarray([row["has_fall"] == "1" for row in rows], dtype=np.float32)
    groups = [template_group(clip_id) for clip_id in clip_ids]
    fold_ids = np.asarray(
        [group_fold(group, folds=config.folds, seed=20260824) for group in groups]
    )
    training = np.flatnonzero(fold_ids != args.fold)
    held_out = np.flatnonzero(fold_ids == args.fold)
    if set(np.asarray(groups)[training]) & set(np.asarray(groups)[held_out]):
        raise AssertionError("template group 跨 OOF fold 泄漏")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("请求 CUDA，但当前不可用")
    set_deterministic(config.seed + args.fold)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    features, quality = extract_actor_features(
        rows,
        intervals,
        tracks,
        pose_cache_root=args.pose_cache_root,
        device=device,
        temp_root=args.temp_root,
        output_dir=args.output_dir,
        config=config,
    )
    np.savez_compressed(
        args.output_dir / "actor_roi_features.npz",
        features=features,
        quality=quality,
        clip_ids=np.asarray(clip_ids),
        labels=labels,
        track_ids=tracks,
    )
    mean = features[training].mean(0, dtype=np.float64).astype(np.float32)
    std = np.maximum(
        features[training].std(0, dtype=np.float64).astype(np.float32), 1e-6
    )
    x = torch.from_numpy((features - mean) / std).to(device)
    y = torch.from_numpy(labels).to(device)
    head = MlpHead(features.shape[1], config.hidden_dim, config.dropout).to(device)
    positive = float(labels[training].sum())
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor((training.size - positive) / positive, device=device)
    )
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
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
        record = {
            "epoch": epoch,
            "train_loss": float(loss.detach()),
            "oof": metric_result(
                [clip_ids[index] for index in held_out], labels[held_out], scores
            ),
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False, sort_keys=True), flush=True)
    final_scores = scores
    oof_ids = [clip_ids[index] for index in held_out]
    comparison = (
        comparison_with_skeleton(
            oof_ids, labels[held_out], final_scores, args.skeleton_oof
        )
        if args.skeleton_oof is not None
        else None
    )
    checkpoint = {
        "epoch": config.epochs - 1,
        "model_state": head.state_dict(),
        "mean": mean,
        "std": std,
        "config": asdict(config),
        "metrics": history[-1]["oof"],
    }
    _atomic_checkpoint(args.output_dir / "last.pt", checkpoint)
    np.savez_compressed(
        args.output_dir / "oof_predictions.npz",
        clip_indices=held_out,
        clip_ids=np.asarray(oof_ids),
        labels=labels[held_out],
        scores=final_scores,
        fold=np.full(held_out.size, args.fold, dtype=np.int64),
    )
    summary = {
        "protocol": "template_grouped_train_only_actor_context_roi_r3d18_r0_v1",
        "manifest_sha256": _sha256_file(args.manifest),
        "cache_signature_sha256": cache.metadata["signature_sha256"],
        "fold": args.fold,
        "train_clips": int(training.size),
        "oof_clips": int(held_out.size),
        "train_positive": int(labels[training].sum()),
        "oof_positive": int(labels[held_out].sum()),
        "config": asdict(config),
        "final_metrics": history[-1]["oof"],
        "quality_mean": quality.mean(0).tolist(),
        "comparison": comparison,
        "checkpoint_sha256": _sha256_file(args.output_dir / "last.pt"),
    }
    _atomic_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "history.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in history
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
