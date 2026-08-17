"""可复用的视频跌倒推理引擎。

模型在 ``InferenceEngine`` 构造时只加载一次，可连续分析多个视频。批量评测默认
``render=False``，避免绘制与编码污染算法时延。
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from ultralytics import YOLO

from pipeline.event_aggregator import FrameScore, aggregate_tracks
from pipeline.fusion import TemporalFallScorer
from pipeline.pose_track import MultiPoseTracker
from pipeline.rules import PoseFeatures, compute_pose_features

CropMode = Literal["auto", "none", "left", "right"]
ALERT_THRESHOLD = 0.50
INFERENCE_PROTOCOL = "pose_motion_rule_baseline_v2"
SKELETON = [
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 5),
    (0, 6),
]


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _inference_source_sha256() -> str:
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in (
        "inference_engine.py",
        "rules.py",
        "fusion.py",
        "pose_track.py",
        "event_aggregator.py",
    ):
        normalized = (root / name).read_text(encoding="utf-8")
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(normalized.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def crop_frame(frame: np.ndarray, mode: CropMode) -> tuple[np.ndarray, tuple[int, int]]:
    """裁出视觉输入；auto 识别 URFD 的 depth|RGB 横向拼接帧。"""
    height, width = frame.shape[:2]
    effective = mode
    if mode == "auto":
        effective = "right" if width / max(height, 1) > 2.2 else "none"
    if effective == "left":
        return frame[:, : width // 2], (0, 0)
    if effective == "right":
        return frame[:, width // 2 :], (width // 2, 0)
    return frame, (0, 0)


def sanitize_fps(raw_fps: float, default: float = 25.0) -> float:
    """把缺失、非有限或非正的视频帧率回退到安全默认值。"""
    fps = float(raw_fps)
    return fps if np.isfinite(fps) and fps > 0.0 else default


def _draw_pose(
    frame: np.ndarray,
    box: np.ndarray,
    keypoints: np.ndarray,
    score: float | None,
    pose_confidence: float,
    track_id: int,
) -> None:
    x1, y1, x2, y2 = map(int, box)
    if score is None:
        color, score_text = (150, 150, 150), "n/a"
    else:
        color = (0, 0, 255) if score >= ALERT_THRESHOLD else (0, 210, 255)
        score_text = f"{score:.2f}"
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    for a, b in SKELETON:
        if keypoints[a, 2] >= 0.2 and keypoints[b, 2] >= 0.2:
            pa = tuple(keypoints[a, :2].astype(int))
            pb = tuple(keypoints[b, :2].astype(int))
            cv2.line(frame, pa, pb, (70, 255, 70), 2, cv2.LINE_AA)
    for x, y, confidence in keypoints:
        if confidence >= 0.2:
            cv2.circle(frame, (int(x), int(y)), 3, (255, 120, 30), -1)
    cv2.putText(
        frame,
        f"id={track_id} fall={score_text} pose={pose_confidence:.2f}",
        (max(4, x1), max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def _write_banner(frame: np.ndarray, timestamp: float, score: float | None) -> None:
    if score is None:
        status, color, score_text = "POSE UNOBSERVED", (150, 150, 150), "n/a"
    else:
        status = "FALL ALERT" if score >= ALERT_THRESHOLD else "MONITORING"
        color = (0, 0, 255) if score >= ALERT_THRESHOLD else (0, 180, 0)
        score_text = f"{score:.2f}"
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 34), (20, 20, 20), -1)
    cv2.putText(
        frame,
        f"{status}  t={timestamp:.2f}s  score={score_text}",
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        color,
        2,
        cv2.LINE_AA,
    )


def _latency_summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)) if values else 0.0,
        "p50": float(np.percentile(values, 50)) if values else 0.0,
        "p95": float(np.percentile(values, 95)) if values else 0.0,
    }


class InferenceEngine:
    """持有单个姿态模型实例并连续分析视频。"""

    def __init__(
        self,
        model_path: str = "yolo11n-pose.pt",
        device: str = "0",
        image_size: int = 640,
        confidence: float = 0.10,
        model_factory: Callable[[str], Any] = YOLO,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.image_size = image_size
        self.confidence = confidence
        self.model = model_factory(model_path)

    def cache_signature(self) -> dict[str, object]:
        """返回绑定算法实现与全部行为参数的稳定缓存签名。"""
        return {
            "protocol": INFERENCE_PROTOCOL,
            "implementation_sha256": _inference_source_sha256(),
            "dependency_versions": {
                package: _package_version(package)
                for package in ("ultralytics", "torch", "numpy", "opencv-python")
            },
            "crop": "auto",
            "tracker": {
                "max_misses": "max(3, round(fps * 0.4))",
                "min_iou": 0.1,
                "max_center_distance": 1.5,
            },
            "temporal_scorer": {
                "window_seconds": 1.2,
                "reset_gap_seconds": 0.5,
            },
            "event_aggregator": {
                "smooth_win": 3,
                "th_hi": ALERT_THRESHOLD,
                "th_lo": 0.25,
                "merge_gap": 1.0,
                "min_dur": 0.10,
            },
        }

    def analyze(
        self,
        source: Path,
        *,
        render: bool = False,
        output_video: Path | None = None,
        output_events: Path | None = None,
        crop: CropMode = "auto",
        max_frames: int | None = None,
    ) -> dict:
        """分析一个视频；无渲染模式不创建 ``VideoWriter``。"""
        source = Path(source)
        if render and output_video is None:
            raise ValueError("render=True 时必须提供 output_video")

        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise ValueError(f"无法打开视频: {source}")
        fps = sanitize_fps(capture.get(cv2.CAP_PROP_FPS))
        source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        ok, first = capture.read()
        if not ok:
            capture.release()
            raise ValueError(f"视频没有可读帧: {source}")
        first_crop, _ = crop_frame(first, crop)
        out_h, out_w = first_crop.shape[:2]
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

        writer = None
        if render:
            assert output_video is not None
            output_video.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(
                str(output_video),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (out_w, out_h),
            )
            if not writer.isOpened():
                capture.release()
                raise ValueError(f"无法创建输出视频: {output_video}")

        tracker = MultiPoseTracker(max_misses=max(3, round(fps * 0.4)))
        scorers: dict[int, TemporalFallScorer] = {}
        frame_scores_by_track: dict[int, list[FrameScore]] = defaultdict(list)
        previous_center_y: dict[int, float] = {}
        previous_detection_t: dict[int, float] = {}
        stage_ms: dict[str, list[float]] = {
            "decode_crop": [],
            "predict": [],
            "cpu_transfer": [],
            "track_rule": [],
            "render_encode": [],
            "frame_end_to_end": [],
            "aggregate": [],
        }
        frame_index = 0

        try:
            while max_frames is None or frame_index < max_frames:
                frame_start = time.perf_counter()
                stage_start = time.perf_counter()
                ok, frame = capture.read()
                if not ok:
                    break
                timestamp = frame_index / fps
                view, _ = crop_frame(frame, crop)
                stage_ms["decode_crop"].append(
                    (time.perf_counter() - stage_start) * 1000.0
                )

                stage_start = time.perf_counter()
                result = self.model.predict(
                    source=view,
                    imgsz=self.image_size,
                    conf=self.confidence,
                    device=self.device,
                    verbose=False,
                )[0]
                stage_ms["predict"].append(
                    (time.perf_counter() - stage_start) * 1000.0
                )

                transfer_start = time.perf_counter()
                has_detections = (
                    result.boxes is not None
                    and len(result.boxes)
                    and result.keypoints is not None
                )
                if has_detections:
                    boxes = result.boxes.xyxy.detach().cpu().numpy()
                    confidences = result.boxes.conf.detach().cpu().numpy()
                    keypoints = result.keypoints.data.detach().cpu().numpy()
                stage_ms["cpu_transfer"].append(
                    (time.perf_counter() - transfer_start) * 1000.0
                )

                stage_start = time.perf_counter()
                frame_score: float | None = None
                tracked_poses = []
                if has_detections:
                    tracked_poses = tracker.update(boxes, confidences, keypoints)
                    for tracked in tracked_poses:
                        track_id = tracked.track_id
                        scorer = scorers.setdefault(
                            track_id,
                            TemporalFallScorer(
                                window_seconds=1.2, reset_gap_seconds=0.5
                            ),
                        )
                        continuous = (
                            not tracked.track_switched
                            and track_id in previous_detection_t
                            and timestamp - previous_detection_t[track_id] <= 0.5
                        )
                        if tracked.track_switched:
                            scorer.reset()
                            previous_center_y.pop(track_id, None)
                            previous_detection_t.pop(track_id, None)
                        features: PoseFeatures = compute_pose_features(
                            tracked.keypoints,
                            tracked.box,
                            frame_height=out_h,
                            previous_center_y=(
                                previous_center_y[track_id] if continuous else None
                            ),
                            delta_seconds=(
                                timestamp - previous_detection_t[track_id]
                                if continuous
                                else None
                            ),
                        )
                        score = scorer.update(timestamp, features)
                        frame_scores_by_track[track_id].append(
                            FrameScore(timestamp, score)
                        )
                        if score is not None:
                            previous_center_y[track_id] = features.center_y
                            previous_detection_t[track_id] = timestamp
                            frame_score = (
                                score
                                if frame_score is None
                                else max(frame_score, score)
                            )
                        if render:
                            _draw_pose(
                                view,
                                tracked.box,
                                tracked.keypoints,
                                score,
                                tracked.confidence,
                                track_id,
                            )
                else:
                    tracked_poses = tracker.update(
                        np.empty((0, 4), dtype=np.float32),
                        np.empty(0, dtype=np.float32),
                        np.empty((0, 17, 3), dtype=np.float32),
                    )
                observed_ids = {tracked.track_id for tracked in tracked_poses}
                for track_id in tracker.active_track_ids - observed_ids:
                    frame_scores_by_track[track_id].append(FrameScore(timestamp, None))
                stage_ms["track_rule"].append(
                    (time.perf_counter() - stage_start) * 1000.0
                )

                if render:
                    stage_start = time.perf_counter()
                    _write_banner(view, timestamp, frame_score)
                    assert writer is not None
                    writer.write(view)
                    stage_ms["render_encode"].append(
                        (time.perf_counter() - stage_start) * 1000.0
                    )
                stage_ms["frame_end_to_end"].append(
                    (time.perf_counter() - frame_start) * 1000.0
                )
                frame_index += 1
        finally:
            capture.release()
            if writer is not None:
                writer.release()

        aggregate_start = time.perf_counter()
        events = aggregate_tracks(
            frame_scores_by_track,
            smooth_win=3,
            th_hi=ALERT_THRESHOLD,
            th_lo=0.25,
            merge_gap=1.0,
            min_dur=0.10,
        )
        stage_ms["aggregate"].append(
            (time.perf_counter() - aggregate_start) * 1000.0
        )
        stage_latency_ms = {
            stage: _latency_summary(values) for stage, values in stage_ms.items()
        }
        payload = {
            "source": str(source.resolve()),
            "model": self.model_path,
            "protocol": INFERENCE_PROTOCOL,
            "crop": crop,
            "fps": fps,
            "source_frames": source_frames,
            "processed_frames": frame_index,
            "clip_score": max(
                (
                    item.score
                    for frame_scores in frame_scores_by_track.values()
                    for item in frame_scores
                    if item.score is not None
                ),
                default=0.0,
            ),
            "events": [asdict(event) for event in events],
            "stage_latency_ms": stage_latency_ms,
            "detector_predict_wallclock_ms": stage_latency_ms["predict"],
            "rendered": render,
            "assumptions": [
                "event matching protocol is provisional until organizer confirmation",
                "stage timings are local wall-clock measurements, not V100 submission results",
                "rule baseline is untrained; trained TCN will replace the primary temporal score",
            ],
        }
        if output_events is not None:
            output_events.parent.mkdir(parents=True, exist_ok=True)
            output_events.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        return payload
