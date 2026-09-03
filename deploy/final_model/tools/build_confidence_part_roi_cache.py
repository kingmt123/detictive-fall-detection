"""Build confidence-constrained local body-part ROI tokens from frozen YOLO features.

The cache preserves the existing 16-frame person token and appends upper-body,
torso, and lower-body tokens.  A part is usable only when the exact sampled
frame contains the selected track and enough joints exceed the fixed confidence
threshold.  Invalid part tokens are zeroed rather than filled from adjacent
frames.
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

from models.tcn_dataset import load_window_cache
from pipeline.pose_cache import pose_cache_path, read_pose_cache
from pipeline.video_source import VideoSourceResolver
from tools.build_yolo_roi_token_cache import (
    SharedYoloRoiEncoder,
    YoloRoiCacheConfig,
    letterbox_frame,
    load_split_rows,
)
from tools.train_actor_roi_rgb_canary import (
    expand_box,
    nearest_track_boxes,
    primary_track_ids,
    track_box_trajectory,
)
from tools.train_event_aligned_rgb_oracle import crop_intervals
from tools.train_tcn import _atomic_json, _sha256_file

PART_NAMES = ("upper", "torso", "lower")
PART_JOINTS = (
    (5, 6, 7, 8, 9, 10),
    (5, 6, 11, 12),
    (11, 12, 13, 14, 15, 16),
)
PART_MIN_VISIBLE = (3, 3, 3)


@dataclass(frozen=True)
class ConfidencePartRoiConfig:
    frames: int = 16
    input_size: int = 320
    roi_size: int = 7
    context_scale: float = 1.5
    context_seconds: float = 1.0
    p3_layer: int = 16
    p4_layer: int = 19
    confidence_threshold: float = 0.35
    part_margin: float = 0.08
    minimum_part_extent: float = 0.15

    def validate(self) -> None:
        YoloRoiCacheConfig(
            frames=self.frames,
            input_size=self.input_size,
            roi_size=self.roi_size,
            context_scale=self.context_scale,
            context_seconds=self.context_seconds,
            p3_layer=self.p3_layer,
            p4_layer=self.p4_layer,
        ).validate()
        if not 0.0 < self.confidence_threshold < 1.0:
            raise ValueError("confidence_threshold 必须位于 (0,1)")
        if not 0.0 <= self.part_margin <= 0.5:
            raise ValueError("part_margin 必须位于 [0,0.5]")
        if not 0.0 < self.minimum_part_extent <= 1.0:
            raise ValueError("minimum_part_extent 必须位于 (0,1]")

    def yolo_config(self) -> YoloRoiCacheConfig:
        return YoloRoiCacheConfig(
            frames=self.frames,
            input_size=self.input_size,
            roi_size=self.roi_size,
            context_scale=self.context_scale,
            context_seconds=self.context_seconds,
            p3_layer=self.p3_layer,
            p4_layer=self.p4_layer,
            token_layout="mean",
        )


def confidence_part_boxes(
    keypoints: np.ndarray | None,
    person_box: np.ndarray,
    *,
    frame_size: tuple[int, int],
    config: ConfidencePartRoiConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return three safe boxes and reliability values for one exact frame."""
    person = np.asarray(person_box, dtype=np.float32)
    if person.shape != (4,) or not np.all(np.isfinite(person)):
        raise ValueError("person_box 必须是有限 xyxy")
    height, width = frame_size
    if min(height, width) < 2:
        raise ValueError("frame_size 无效")
    boxes = np.repeat(person[None, :], len(PART_NAMES), axis=0)
    reliability = np.zeros(len(PART_NAMES), dtype=np.float32)
    if keypoints is None:
        return boxes, reliability
    values = np.asarray(keypoints, dtype=np.float32)
    if values.shape != (17, 3) or not np.all(np.isfinite(values)):
        raise ValueError("keypoints 必须是有限 [17,3]")

    px1, py1, px2, py2 = (float(value) for value in person)
    person_width = max(px2 - px1, 1.0)
    person_height = max(py2 - py1, 1.0)
    for part_index, (joint_ids, minimum_visible) in enumerate(
        zip(PART_JOINTS, PART_MIN_VISIBLE, strict=True)
    ):
        part = values[np.asarray(joint_ids)]
        visible = part[:, 2] >= config.confidence_threshold
        visible_count = int(visible.sum())
        if visible_count < minimum_visible:
            continue
        points = part[visible, :2]
        center = points.mean(axis=0)
        box_width = max(
            float(np.ptp(points[:, 0])) + 2.0 * config.part_margin * person_width,
            config.minimum_part_extent * person_width,
        )
        box_height = max(
            float(np.ptp(points[:, 1])) + 2.0 * config.part_margin * person_height,
            config.minimum_part_extent * person_height,
        )
        x1 = max(0.0, px1, float(center[0]) - box_width * 0.5)
        y1 = max(0.0, py1, float(center[1]) - box_height * 0.5)
        x2 = min(float(width), px2, float(center[0]) + box_width * 0.5)
        y2 = min(float(height), py2, float(center[1]) + box_height * 0.5)
        if x2 - x1 < 1.0 or y2 - y1 < 1.0:
            continue
        boxes[part_index] = (x1, y1, x2, y2)
        coverage = visible_count / len(joint_ids)
        reliability[part_index] = float(part[visible, 2].mean() * coverage)
    return boxes, reliability


def _exact_track_keypoints(record: Any, track_id: int) -> dict[int, np.ndarray]:
    result: dict[int, np.ndarray] = {}
    for frame_index in range(record.frame_indices.size):
        columns = np.flatnonzero(
            record.valid_mask[frame_index]
            & (record.track_ids[frame_index] == track_id)
        )
        if columns.size > 1:
            raise ValueError("同一帧出现重复 track_id")
        if columns.size == 1:
            result[frame_index] = record.keypoints[
                frame_index, int(columns[0])
            ]
    return result


def decode_part_roi_frames(
    path: Path,
    *,
    record: Any,
    track_id: int,
    interval: tuple[float, float] | None,
    config: ConfidencePartRoiConfig,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    """Decode frames plus person/context/three exact-pose local boxes."""
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
            right = max(
                left + 1,
                min(frame_count - 1, int(np.ceil(interval[1] * fps))),
            )
        positions = np.linspace(left, right, config.frames).round().astype(np.int64)
        observed_frames, observed_boxes, keypoint_quality = track_box_trajectory(
            record, track_id
        )
        sampled_boxes, exact_box = nearest_track_boxes(
            positions, observed_frames, observed_boxes
        )
        exact_keypoints = _exact_track_keypoints(record, track_id)
        height, width = map(int, record.frame_size)
        tensors: list[torch.Tensor] = []
        transformed_boxes: list[np.ndarray] = []
        reliabilities: list[np.ndarray] = []
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(positions[0]))
        current = int(positions[0]) - 1
        for position, box in zip(positions, sampled_boxes, strict=True):
            while current < int(position):
                ok, frame = capture.read()
                current += 1
                if not ok or frame is None:
                    raise ValueError(f"视频解码失败: {path}@{position}")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            person = expand_box(
                box, scale=1.0, frame_height=height, frame_width=width
            )
            context = expand_box(
                box,
                scale=config.context_scale,
                frame_height=height,
                frame_width=width,
            )
            part_boxes, reliability = confidence_part_boxes(
                exact_keypoints.get(int(position)),
                person,
                frame_size=(height, width),
                config=config,
            )
            tensor, boxes = letterbox_frame(
                frame,
                np.concatenate(((person, context), part_boxes), axis=0),
                size=config.input_size,
            )
            tensors.append(tensor)
            transformed_boxes.append(boxes)
            reliabilities.append(reliability)
    finally:
        capture.release()
    areas = (sampled_boxes[:, 2] - sampled_boxes[:, 0]) * (
        sampled_boxes[:, 3] - sampled_boxes[:, 1]
    )
    quality = np.asarray(
        [
            exact_box.mean(),
            keypoint_quality.mean(),
            keypoint_quality.min(),
            areas.mean() / float(height * width),
            np.abs(np.diff(sampled_boxes[:, :2], axis=0)).mean()
            / max(height, width),
        ],
        dtype=np.float32,
    )
    return (
        torch.stack(tensors),
        torch.from_numpy(np.stack(transformed_boxes)).float(),
        np.stack(reliabilities).astype(np.float32),
        quality,
    )


def enhanced_part_tokens(
    pooled: np.ndarray, reliability: np.ndarray
) -> np.ndarray:
    """Create actor/context tokens compatible with the existing F1 loader."""
    values = np.asarray(pooled, dtype=np.float32)
    weights = np.asarray(reliability, dtype=np.float32)
    if values.ndim != 4 or values.shape[2] != 5:
        raise ValueError("pooled 必须为 [B,T,5,D]")
    if weights.shape != (*values.shape[:2], 3):
        raise ValueError("part reliability shape 不匹配")
    weighted_parts = values[:, :, 2:] * weights[..., None]
    actor = np.concatenate(
        (values[:, :, 0], weighted_parts.reshape(*values.shape[:2], -1), weights),
        axis=-1,
    )
    context = np.zeros_like(actor)
    context[:, :, : values.shape[-1]] = values[:, :, 1]
    return np.stack((actor, context), axis=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--pose-cache-root", type=Path, required=True)
    parser.add_argument("--yolo-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--temp-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--clip-batch-size", type=int, default=8)
    parser.add_argument("--decode-workers", type=int, default=8)
    parser.add_argument("--input-size", type=int, default=320)
    parser.add_argument("--confidence-threshold", type=float, default=0.35)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int)
    args = parser.parse_args()
    config = ConfidencePartRoiConfig(
        input_size=args.input_size,
        confidence_threshold=args.confidence_threshold,
    )
    config.validate()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Part-ROI cache 输出目录必须为空")
    if args.clip_batch_size < 1 or not 1 <= args.decode_workers <= args.clip_batch_size:
        raise ValueError("decode_workers 必须位于 [1,clip_batch_size]")
    cache = load_window_cache(args.cache, verify_hashes=True)
    if cache.metadata.get("dataset") != "of-syn" or cache.metadata.get("split") != args.split:
        raise ValueError("Part-ROI cache split 与 window cache 不匹配")
    if not args.yolo_checkpoint.is_file():
        raise FileNotFoundError(f"YOLO checkpoint 不存在: {args.yolo_checkpoint}")
    all_rows = load_split_rows(args.manifest, cache, split=args.split)
    all_intervals = crop_intervals(
        cache, context_seconds=config.context_seconds
    )
    all_tracks = primary_track_ids(cache)
    end_index = len(all_rows) if args.end_index is None else args.end_index
    if not 0 <= args.start_index < end_index <= len(all_rows):
        raise ValueError("Part-ROI clip index range 无效")
    rows = all_rows[args.start_index:end_index]
    intervals = all_intervals[args.start_index:end_index]
    tracks = all_tracks[args.start_index:end_index]
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("请求 CUDA，但当前不可用")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    encoder = SharedYoloRoiEncoder(
        args.yolo_checkpoint, device=device, config=config.yolo_config()
    )
    token_batches: list[np.ndarray] = []
    reliability_rows: list[np.ndarray] = []
    quality_rows: list[np.ndarray] = []
    try:
        with VideoSourceResolver(args.temp_root) as resolver, ThreadPoolExecutor(
            max_workers=args.decode_workers
        ) as pool:
            for batch_start in range(0, len(rows), args.clip_batch_size):
                batch_rows = rows[batch_start : batch_start + args.clip_batch_size]
                batch_intervals = intervals[
                    batch_start : batch_start + args.clip_batch_size
                ]
                batch_tracks = tracks[batch_start : batch_start + args.clip_batch_size]
                records = [
                    read_pose_cache(
                        pose_cache_path(
                            args.pose_cache_root,
                            "of-syn",
                            args.split,
                            row["clip_id"],
                        )
                    )
                    for row in batch_rows
                ]
                with ExitStack() as stack:
                    paths = [
                        stack.enter_context(
                            resolver.materialize(row["video_path"])
                        ).local_path
                        for row in batch_rows
                    ]
                    decoded = list(
                        pool.map(
                            lambda item: decode_part_roi_frames(
                                item[0],
                                record=item[1],
                                track_id=int(item[2]),
                                interval=item[3],
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
                pooled = encoder.encode(
                    torch.cat([item[0] for item in decoded]),
                    torch.cat([item[1] for item in decoded]),
                ).reshape(len(decoded), config.frames, 5, -1).numpy()
                reliability = np.stack([item[2] for item in decoded])
                token_batches.append(enhanced_part_tokens(pooled, reliability))
                reliability_rows.extend(reliability)
                quality_rows.extend(item[3] for item in decoded)
                complete = batch_start + len(decoded)
                if complete % 25 < args.clip_batch_size or complete == len(rows):
                    progress = {
                        "stage": "confidence_part_roi_tokens",
                        "complete": args.start_index + complete,
                        "range_start": args.start_index,
                        "range_end": end_index,
                        "total": len(all_rows),
                    }
                    _atomic_json(args.output_dir / "progress.json", progress)
                    print(json.dumps(progress), flush=True)
    finally:
        encoder.close()
    token_array = np.concatenate(token_batches).astype(np.float32, copy=False)
    reliability_array = np.asarray(reliability_rows, dtype=np.float32)
    quality_array = np.asarray(quality_rows, dtype=np.float32)
    token_path = args.output_dir / "roi_tokens.npz"
    np.savez_compressed(
        token_path,
        tokens=token_array,
        quality=quality_array,
        part_reliability=reliability_array,
        clip_ids=np.asarray([row["clip_id"] for row in rows]),
        labels=np.asarray(
            [row["has_fall"] == "1" for row in rows], dtype=np.float32
        ),
        track_ids=tracks,
    )
    summary = {
        "protocol": "ofsyn_train_confidence_constrained_local_part_roi_v1",
        "split": args.split,
        "manifest_sha256": _sha256_file(args.manifest),
        "window_cache_signature_sha256": cache.metadata["signature_sha256"],
        "yolo_checkpoint_sha256": _sha256_file(args.yolo_checkpoint),
        "config": asdict(config),
        "parts": [
            {
                "name": name,
                "joint_ids": list(joints),
                "minimum_visible": minimum,
            }
            for name, joints, minimum in zip(
                PART_NAMES, PART_JOINTS, PART_MIN_VISIBLE, strict=True
            )
        ],
        "clips": len(rows),
        "clip_index_range": [args.start_index, end_index],
        "token_shape": list(token_array.shape),
        "part_valid_fraction": (reliability_array > 0.0)
        .mean(axis=(0, 1))
        .tolist(),
        "part_reliability_mean": reliability_array.mean(axis=(0, 1)).tolist(),
        "token_cache_sha256": _sha256_file(token_path),
        "test_accessed": False,
    }
    _atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
