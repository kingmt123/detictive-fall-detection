"""Cache train/validation person ROI tokens from shared YOLO pose features."""

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
from torchvision.ops import roi_align
from ultralytics import YOLO

from models.event_proposal import PROTOCOL as EVENT_PROPOSAL_PROTOCOL
from models.event_proposal import propose_motion_event
from models.tcn_dataset import WindowMemmapCache, load_window_cache
from pipeline.pose_cache import pose_cache_path, read_pose_cache
from pipeline.video_source import VideoSourceResolver
from tools.train_actor_roi_rgb_canary import (
    expand_box,
    nearest_track_boxes,
    primary_track_ids,
    track_box_trajectory,
)
from tools.train_event_aligned_rgb_oracle import crop_intervals
from tools.train_tcn import _atomic_json, _sha256_file


@dataclass(frozen=True)
class YoloRoiCacheConfig:
    frames: int = 16
    input_size: int = 640
    roi_size: int = 7
    context_scale: float = 1.5
    context_seconds: float = 1.0
    p3_layer: int = 16
    p4_layer: int = 19
    token_layout: str = "mean"

    def validate(self) -> None:
        if self.frames < 4 or self.input_size < 64 or self.roi_size < 1:
            raise ValueError("frames/input_size/roi_size 无效")
        if self.context_scale <= 1.0 or self.context_seconds < 0.0:
            raise ValueError("context 配置无效")
        if self.p3_layer < 0 or self.p4_layer <= self.p3_layer:
            raise ValueError("YOLO feature layer 无效")
        if self.token_layout not in {"mean", "mean_max", "mean_std_xy"}:
            raise ValueError("token_layout 必须为 mean、mean_max 或 mean_std_xy")


def load_split_rows(
    manifest: Path,
    cache: WindowMemmapCache,
    *,
    split: str,
    dataset: str = "of-syn",
) -> list[dict[str, str]]:
    """Return cache-ordered train/val rows and reject every test split."""
    if split not in {"train", "val"}:
        raise ValueError("YOLO ROI cache 只接受 train 或 val")
    if not dataset or cache.metadata.get("dataset") != dataset or cache.metadata.get("split") != split:
        raise ValueError("YOLO ROI cache split 与 window cache 不匹配")
    with Path(manifest).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"dataset", "split", "clip_id", "has_fall", "video_path"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"manifest 缺少字段: {sorted(missing)}")
        selected = [
            row
            for row in reader
            if row.get("dataset") == dataset and row.get("split") == split
        ]
    rows = {row["clip_id"]: row for row in selected}
    if len(rows) != len(selected):
        raise ValueError(f"{dataset} {split} manifest clip_id 重复")
    clip_ids = [str(clip["clip_id"]) for clip in cache.metadata["clips"]]
    if set(rows) != set(clip_ids):
        raise ValueError(f"manifest 与 {split} cache 的 clip 集合不一致")
    ordered = [rows[clip_id] for clip_id in clip_ids]
    if any(row.get("has_fall") not in {"0", "1"} for row in ordered):
        raise ValueError("manifest has_fall 必须为 0 或 1")
    if dataset == "of-syn" and any(
        not row.get("video_path", "").startswith("tar://") for row in ordered
    ):
        raise ValueError("YOLO ROI cache 只接受审计过的 tar URI")
    return ordered


def letterbox_frame(
    frame: np.ndarray, boxes: np.ndarray, *, size: int
) -> tuple[torch.Tensor, np.ndarray]:
    """Letterbox RGB pixels and transform xyxy boxes into model coordinates."""
    image = np.asarray(frame)
    values = np.asarray(boxes, dtype=np.float32)
    if (
        image.ndim != 3
        or image.shape[2] != 3
        or values.ndim != 2
        or values.shape[1] != 4
        or values.shape[0] < 1
    ):
        raise ValueError("frame/boxes shape 无效")
    height, width = image.shape[:2]
    if min(height, width) < 2 or size < 2:
        raise ValueError("frame/input size 无效")
    scale = min(size / width, size / height)
    resized_width = max(1, min(size, round(width * scale)))
    resized_height = max(1, min(size, round(height * scale)))
    resized = cv2.resize(
        image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
    )
    pad_x = (size - resized_width) // 2
    pad_y = (size - resized_height) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
    transformed = values.copy()
    transformed[:, (0, 2)] = transformed[:, (0, 2)] * scale + pad_x
    transformed[:, (1, 3)] = transformed[:, (1, 3)] * scale + pad_y
    transformed[:, (0, 2)] = transformed[:, (0, 2)].clip(0.0, float(size))
    transformed[:, (1, 3)] = transformed[:, (1, 3)].clip(0.0, float(size))
    tensor = torch.from_numpy(canvas).permute(2, 0, 1).float() / 255.0
    return tensor, transformed


def decode_yolo_frames(
    path: Path,
    *,
    record: Any,
    track_id: int,
    interval: tuple[float, float] | None,
    config: YoloRoiCacheConfig,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Decode one clip into letterboxed model frames and two aligned boxes/frame."""
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
        height, width = map(int, record.frame_size)
        tensors: list[torch.Tensor] = []
        transformed_boxes: list[np.ndarray] = []
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
            tensor, boxes = letterbox_frame(
                frame, np.stack((person, context)), size=config.input_size
            )
            tensors.append(tensor)
            transformed_boxes.append(boxes)
    finally:
        capture.release()
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
    return (
        torch.stack(tensors),
        torch.from_numpy(np.stack(transformed_boxes)).float(),
        quality,
    )


def pool_roi_tokens(
    feature_map: torch.Tensor,
    boxes: torch.Tensor,
    *,
    input_size: int,
    output_size: int,
    token_layout: str = "mean",
) -> torch.Tensor:
    """ROIAlign boxes and retain the requested deterministic spatial statistics."""
    if (
        feature_map.ndim != 4
        or boxes.ndim != 3
        or boxes.shape[2] != 4
        or boxes.shape[1] < 1
    ):
        raise ValueError("feature_map/boxes shape 无效")
    if feature_map.shape[0] != boxes.shape[0]:
        raise ValueError("feature_map/boxes batch 不一致")
    boxes = boxes.to(device=feature_map.device, dtype=feature_map.dtype)
    batch_ids = torch.arange(
        boxes.shape[0], device=boxes.device, dtype=boxes.dtype
    )[:, None, None].expand(-1, boxes.shape[1], 1)
    rois = torch.cat((batch_ids, boxes), dim=2).reshape(-1, 5)
    spatial_scale = feature_map.shape[-1] / float(input_size)
    pooled = roi_align(
        feature_map,
        rois,
        output_size=(output_size, output_size),
        spatial_scale=spatial_scale,
        sampling_ratio=2,
        aligned=True,
    )
    mean = pooled.mean(dim=(-1, -2))
    if token_layout == "mean":
        statistics = mean
    elif token_layout == "mean_max":
        statistics = torch.cat((mean, pooled.amax(dim=(-1, -2))), dim=1)
    elif token_layout == "mean_std_xy":
        height, width = pooled.shape[-2:]
        y_weights = torch.linspace(
            -1.0, 1.0, height, device=pooled.device, dtype=pooled.dtype
        ).view(1, 1, height, 1)
        x_weights = torch.linspace(
            -1.0, 1.0, width, device=pooled.device, dtype=pooled.dtype
        ).view(1, 1, 1, width)
        statistics = torch.cat(
            (
                mean,
                pooled.std(dim=(-1, -2), unbiased=False),
                (pooled * x_weights).mean(dim=(-1, -2)),
                (pooled * y_weights).mean(dim=(-1, -2)),
            ),
            dim=1,
        )
    else:
        raise ValueError("未知 ROI token_layout")
    return statistics.reshape(boxes.shape[0], boxes.shape[1], -1)


class SharedYoloRoiEncoder:
    def __init__(
        self,
        checkpoint: Path,
        *,
        device: torch.device,
        config: YoloRoiCacheConfig,
        model: torch.nn.Module | None = None,
    ) -> None:
        self.device = device
        self.config = config
        self.model = (YOLO(str(checkpoint)).model if model is None else model).eval().to(device)
        self.half = device.type == "cuda"
        if self.half:
            self.model.half()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.outputs: dict[int, torch.Tensor] = {}

    def _forward_to_p4(self, frames: torch.Tensor) -> None:
        """Execute the YOLO graph only through P4, skipping P5 and Pose heads."""
        saved: list[torch.Tensor | None] = []
        value = frames
        self.outputs.clear()
        for module in self.model.model:
            if module.f != -1:
                value = (
                    saved[module.f]
                    if isinstance(module.f, int)
                    else [value if index == -1 else saved[index] for index in module.f]
                )
            value = module(value)
            saved.append(value if module.i in self.model.save else None)
            if module.i in {self.config.p3_layer, self.config.p4_layer}:
                self.outputs[module.i] = value
            if module.i == self.config.p4_layer:
                return
        raise RuntimeError("YOLO graph 未到达配置的 P4 层")

    @torch.inference_mode()
    def encode(
        self,
        frames: torch.Tensor,
        boxes: torch.Tensor,
        *,
        config: YoloRoiCacheConfig | None = None,
    ) -> torch.Tensor:
        active_config = self.config if config is None else config
        active_config.validate()
        if (active_config.p3_layer, active_config.p4_layer) != (
            self.config.p3_layer,
            self.config.p4_layer,
        ):
            raise ValueError("共享 YOLO encoder 不允许切换特征层")
        model_frames = frames.to(self.device)
        if self.half:
            model_frames = model_frames.half()
        self._forward_to_p4(model_frames)
        if set(self.outputs) != {self.config.p3_layer, self.config.p4_layer}:
            raise RuntimeError("未捕获完整 YOLO P3/P4 特征")
        boxes = boxes.to(self.device)
        tokens = [
            pool_roi_tokens(
                self.outputs[layer],
                boxes,
                input_size=active_config.input_size,
                output_size=active_config.roi_size,
                token_layout=active_config.token_layout,
            )
            for layer in (self.config.p3_layer, self.config.p4_layer)
        ]
        return torch.cat(tokens, dim=2).float().cpu()

    def close(self) -> None:
        self.outputs.clear()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--pose-cache-root", type=Path, required=True)
    parser.add_argument("--yolo-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--temp-root", type=Path, required=True)
    parser.add_argument("--dataset", default="of-syn")
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--clip-batch-size", type=int, default=8)
    parser.add_argument("--input-size", type=int, default=320)
    parser.add_argument(
        "--token-layout",
        choices=("mean", "mean_max", "mean_std_xy"),
        default="mean",
    )
    parser.add_argument("--decode-workers", type=int, default=8)
    parser.add_argument(
        "--selection-mode",
        choices=("annotation", "motion"),
        default="annotation",
        help="motion 完全依据 pose 轨迹选择人物和固定 4 秒候选区间",
    )
    args = parser.parse_args()
    config = YoloRoiCacheConfig(
        input_size=args.input_size, token_layout=args.token_layout
    )
    config.validate()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("YOLO ROI cache 输出目录必须为空")
    cache = load_window_cache(args.cache, verify_hashes=True)
    if cache.metadata.get("dataset") != args.dataset or cache.metadata.get("split") != args.split:
        raise ValueError("YOLO ROI cache split 与 window cache 不匹配")
    if not args.yolo_checkpoint.is_file():
        raise FileNotFoundError(f"YOLO checkpoint 不存在: {args.yolo_checkpoint}")
    if args.clip_batch_size < 1:
        raise ValueError("clip_batch_size 必须为正")
    if not 1 <= args.decode_workers <= args.clip_batch_size:
        raise ValueError("decode_workers 必须位于 [1,clip_batch_size]")
    rows = load_split_rows(
        args.manifest, cache, split=args.split, dataset=args.dataset
    )
    if args.selection_mode == "annotation":
        intervals = crop_intervals(cache, context_seconds=config.context_seconds)
        tracks = primary_track_ids(cache)
        proposal_rows: list[dict[str, object]] = []
    else:
        intervals = []
        selected_tracks = []
        proposal_rows = []
        for row in rows:
            record = read_pose_cache(
                pose_cache_path(
                    args.pose_cache_root, args.dataset, args.split, row["clip_id"]
                )
            )
            proposal = propose_motion_event(record, context_seconds=2.0)
            intervals.append((proposal.start_time, proposal.end_time))
            selected_tracks.append(proposal.track_id)
            proposal_rows.append(
                {
                    "clip_id": row["clip_id"],
                    "track_id": proposal.track_id,
                    "start_time": proposal.start_time,
                    "peak_time": proposal.peak_time,
                    "end_time": proposal.end_time,
                    "motion_score": proposal.motion_score,
                    "observed_frames": proposal.observed_frames,
                }
            )
        tracks = np.asarray(selected_tracks, dtype=np.int64)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("请求 CUDA，但当前不可用")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    encoder = SharedYoloRoiEncoder(
        args.yolo_checkpoint, device=device, config=config
    )
    tokens: list[np.ndarray] = []
    qualities: list[np.ndarray] = []
    try:
        with VideoSourceResolver(args.temp_root) as resolver, ThreadPoolExecutor(
            max_workers=args.decode_workers
        ) as pool:
            for batch_start in range(0, len(rows), args.clip_batch_size):
                batch_rows = rows[batch_start : batch_start + args.clip_batch_size]
                batch_intervals = intervals[
                    batch_start : batch_start + args.clip_batch_size
                ]
                batch_tracks = tracks[
                    batch_start : batch_start + args.clip_batch_size
                ]
                records = [
                    read_pose_cache(
                        pose_cache_path(
                            args.pose_cache_root,
                            args.dataset,
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
                            lambda item: decode_yolo_frames(
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
                batch_tokens = encoder.encode(
                    torch.cat([item[0] for item in decoded]),
                    torch.cat([item[1] for item in decoded]),
                ).reshape(len(decoded), config.frames, 2, -1)
                tokens.extend(batch_tokens.numpy())
                qualities.extend(item[2] for item in decoded)
                complete = batch_start + len(decoded)
                if (
                    complete % 25 < args.clip_batch_size
                    or complete == len(rows)
                ):
                    progress = {
                        "stage": "shared_yolo_roi_tokens",
                        "complete": complete,
                        "total": len(rows),
                    }
                    _atomic_json(args.output_dir / "progress.json", progress)
                    print(json.dumps(progress), flush=True)
    finally:
        encoder.close()
    token_array = np.asarray(tokens, dtype=np.float32)
    quality_array = np.asarray(qualities, dtype=np.float32)
    np.savez_compressed(
        args.output_dir / "roi_tokens.npz",
        tokens=token_array,
        quality=quality_array,
        clip_ids=np.asarray([row["clip_id"] for row in rows]),
        labels=np.asarray([row["has_fall"] == "1" for row in rows], dtype=np.float32),
        track_ids=tracks,
    )
    summary: dict[str, Any] = {
        "protocol": (
            f"{args.dataset}_{args.split}_shared_yolo_p3p4_actor_context_roi_"
            f"{config.token_layout}_v1"
        ),
        "dataset": args.dataset,
        "split": args.split,
        "manifest_sha256": _sha256_file(args.manifest),
        "window_cache_signature_sha256": cache.metadata["signature_sha256"],
        "yolo_checkpoint_sha256": _sha256_file(args.yolo_checkpoint),
        "config": asdict(config),
        "clips": len(rows),
        "token_shape": list(token_array.shape),
        "quality_mean": quality_array.mean(0).tolist(),
        "token_cache_sha256": _sha256_file(args.output_dir / "roi_tokens.npz"),
        "test_accessed": False,
        "selection_mode": args.selection_mode,
        "selection_is_label_blind": args.selection_mode == "motion",
        "event_proposal_protocol": (
            EVENT_PROPOSAL_PROTOCOL if args.selection_mode == "motion" else None
        ),
    }
    if proposal_rows:
        _atomic_json(args.output_dir / "proposals.json", proposal_rows)
        summary["proposals_sha256"] = _sha256_file(
            args.output_dir / "proposals.json"
        )
    _atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
