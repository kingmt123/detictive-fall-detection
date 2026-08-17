"""把视频提取为可缓存的逐帧多人姿态序列。"""
from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

from pipeline.inference_engine import CropMode, crop_frame, sanitize_fps
from pipeline.pose_cache import CACHE_SCHEMA
from pipeline.pose_track import MultiPoseTracker, TrackedPose

POSE_EXTRACTION_PROTOCOL = "pose_extraction_v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _implementation_sha256() -> str:
    digest = hashlib.sha256()
    source_root = Path(__file__).parent
    for name in ("inference_engine.py", "pose_extractor.py", "pose_track.py"):
        source = (source_root / name).read_text(encoding="utf-8")
        digest.update(name.encode("utf-8"))
        digest.update(source.replace("\r\n", "\n").encode("utf-8"))
    return digest.hexdigest()


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "missing"


@dataclass(frozen=True)
class PoseSequence:
    fps: float
    frame_indices: np.ndarray
    timestamps: np.ndarray
    keypoints: np.ndarray
    bboxes: np.ndarray
    track_ids: np.ndarray
    valid_mask: np.ndarray
    frame_size: np.ndarray


class PoseExtractor:
    """持有一个姿态模型，并为每个视频创建独立 tracker。"""

    def __init__(
        self,
        model_path: str | Path = "yolo11n-pose.pt",
        device: str = "0",
        image_size: int = 640,
        confidence: float = 0.10,
        model_factory: Callable[[str], Any] = YOLO,
        capture_factory: Callable[[str], Any] = cv2.VideoCapture,
    ) -> None:
        self.model_path = str(model_path)
        self.device = device
        self.image_size = image_size
        self.confidence = confidence
        self._model_factory = model_factory
        self._model: Any | None = None
        self.capture_factory = capture_factory

    def _get_model(self) -> Any:
        if self._model is None:
            self._model = self._model_factory(self.model_path)
        return self._model

    def cache_signature(
        self, *, crop: CropMode, max_frames: int | None = None
    ) -> dict[str, Any]:
        model_path = Path(self.model_path)
        if not model_path.is_file():
            raise ValueError(f"模型权重不存在，无法生成 cache 签名: {model_path}")
        return {
            "protocol": POSE_EXTRACTION_PROTOCOL,
            "cache_schema": CACHE_SCHEMA,
            "model_sha256": _sha256_file(model_path),
            "implementation_sha256": _implementation_sha256(),
            "device": self.device,
            "image_size": self.image_size,
            "confidence": self.confidence,
            "crop": crop,
            "max_frames": max_frames,
            "dependencies": {
                name: _package_version(name)
                for name in ("numpy", "opencv-python", "torch", "ultralytics")
            },
        }

    def extract(
        self,
        source: str | Path,
        *,
        crop: CropMode = "auto",
        max_frames: int | None = None,
    ) -> PoseSequence:
        capture = self.capture_factory(str(source))
        if not capture.isOpened():
            capture.release()
            raise ValueError(f"无法打开视频: {source}")
        fps = sanitize_fps(capture.get(cv2.CAP_PROP_FPS))
        model = self._get_model()
        tracker = MultiPoseTracker(max_misses=max(3, round(fps * 0.4)))
        observations: list[list[TrackedPose]] = []
        frame_size: tuple[int, int] | None = None
        try:
            while max_frames is None or len(observations) < max_frames:
                ok, frame = capture.read()
                if not ok:
                    break
                view, _ = crop_frame(frame, crop)
                current_size = view.shape[:2]
                if frame_size is None:
                    frame_size = current_size
                elif current_size != frame_size:
                    raise ValueError("视频帧尺寸在同一 clip 内发生变化")
                result = model.predict(
                    source=view,
                    imgsz=self.image_size,
                    conf=self.confidence,
                    device=self.device,
                    verbose=False,
                )[0]
                has_detections = (
                    result.boxes is not None
                    and len(result.boxes)
                    and result.keypoints is not None
                )
                if has_detections:
                    boxes = result.boxes.xyxy.detach().cpu().numpy()
                    confidences = result.boxes.conf.detach().cpu().numpy()
                    keypoints = result.keypoints.data.detach().cpu().numpy()
                else:
                    boxes = np.empty((0, 4), dtype=np.float32)
                    confidences = np.empty(0, dtype=np.float32)
                    keypoints = np.empty((0, 17, 3), dtype=np.float32)
                observations.append(tracker.update(boxes, confidences, keypoints))
        finally:
            capture.release()

        if not observations or frame_size is None:
            raise ValueError(f"视频没有可读帧: {source}")
        frame_count = len(observations)
        max_people = max(map(len, observations))
        keypoints_out = np.zeros(
            (frame_count, max_people, 17, 3), dtype=np.float32
        )
        bboxes_out = np.zeros((frame_count, max_people, 4), dtype=np.float32)
        track_ids = np.full((frame_count, max_people), -1, dtype=np.int64)
        valid_mask = np.zeros((frame_count, max_people), dtype=np.bool_)
        for frame_index, tracked_poses in enumerate(observations):
            for person_index, tracked in enumerate(tracked_poses):
                keypoints_out[frame_index, person_index] = tracked.keypoints
                bboxes_out[frame_index, person_index] = tracked.box
                track_ids[frame_index, person_index] = tracked.track_id
                valid_mask[frame_index, person_index] = True

        frame_indices = np.arange(frame_count, dtype=np.int64)
        return PoseSequence(
            fps=fps,
            frame_indices=frame_indices,
            timestamps=frame_indices.astype(np.float64) / fps,
            keypoints=keypoints_out,
            bboxes=bboxes_out,
            track_ids=track_ids,
            valid_mask=valid_mask,
            frame_size=np.asarray(frame_size, dtype=np.int64),
        )
