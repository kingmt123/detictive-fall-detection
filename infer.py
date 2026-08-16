"""端到端纯视觉跌倒检测基线。

流程：视频 → YOLO11-pose → 主目标跟踪 → 姿态/累计下坠规则 → 事件聚合
      → 事件 JSON + 可视化 MP4。

当前版本是无需训练即可运行的规则基线；后续训练好的 FallTCN 通过同一关键点
序列接口接入，不改变输入输出协议。
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from ultralytics import YOLO

from pipeline.event_aggregator import FrameScore, aggregate_tracks
from pipeline.fusion import TemporalFallScorer
from pipeline.pose_track import MultiPoseTracker
from pipeline.rules import PoseFeatures, compute_pose_features

CropMode = Literal["auto", "none", "left", "right"]
ALERT_THRESHOLD = 0.50
SKELETON = [
    (5, 6), (5, 11), (6, 12), (11, 12),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 5), (0, 6),
]


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


def sanitize_fps(raw_fps: float, default: float = 25.0) -> float:
    """把缺失、非有限或非正的视频帧率回退到安全默认值。"""
    fps = float(raw_fps)
    return fps if np.isfinite(fps) and fps > 0.0 else default


def analyze_video(
    source: Path,
    output_video: Path,
    output_events: Path,
    model_path: str = "yolo11n-pose.pt",
    device: str = "0",
    image_size: int = 640,
    confidence: float = 0.10,
    crop: CropMode = "auto",
    max_frames: int | None = None,
) -> dict:
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

    output_video.parent.mkdir(parents=True, exist_ok=True)
    output_events.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h)
    )
    if not writer.isOpened():
        capture.release()
        raise ValueError(f"无法创建输出视频: {output_video}")

    model = YOLO(model_path)
    tracker = MultiPoseTracker(max_misses=max(3, round(fps * 0.4)))
    scorers: dict[int, TemporalFallScorer] = {}
    frame_scores_by_track: dict[int, list[FrameScore]] = defaultdict(list)
    inference_ms: list[float] = []
    previous_center_y: dict[int, float] = {}
    previous_detection_t: dict[int, float] = {}
    frame_index = 0

    try:
        while max_frames is None or frame_index < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = frame_index / fps
            view, _ = crop_frame(frame, crop)
            start = time.perf_counter()
            result = model.predict(
                view,
                imgsz=image_size,
                conf=confidence,
                device=device,
                verbose=False,
            )[0]
            inference_ms.append((time.perf_counter() - start) * 1000.0)

            frame_score: float | None = None
            tracked_poses = []
            if result.boxes is not None and len(result.boxes) and result.keypoints is not None:
                boxes = result.boxes.xyxy.detach().cpu().numpy()
                confidences = result.boxes.conf.detach().cpu().numpy()
                keypoints = result.keypoints.data.detach().cpu().numpy()
                tracked_poses = tracker.update(boxes, confidences, keypoints)
                for tracked in tracked_poses:
                    track_id = tracked.track_id
                    scorer = scorers.setdefault(
                        track_id,
                        TemporalFallScorer(window_seconds=1.2, reset_gap_seconds=0.5),
                    )
                    continuous = (
                        not tracked.track_switched
                        and track_id in previous_detection_t
                        and timestamp - previous_detection_t[track_id] <= 0.5
                    )
                    if tracked.track_switched:
                        scorer.reset()
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
                    frame_scores_by_track[track_id].append(FrameScore(timestamp, score))
                    if score is not None:
                        previous_center_y[track_id] = features.center_y
                        previous_detection_t[track_id] = timestamp
                        frame_score = (
                            score if frame_score is None else max(frame_score, score)
                        )
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
            _write_banner(view, timestamp, frame_score)
            writer.write(view)
            frame_index += 1
    finally:
        capture.release()
        writer.release()

    events = aggregate_tracks(
        frame_scores_by_track,
        smooth_win=3,
        th_hi=ALERT_THRESHOLD,
        th_lo=0.25,
        merge_gap=1.0,
        min_dur=0.10,
    )
    payload = {
        "source": str(source.resolve()),
        "model": model_path,
        "protocol": "pose_motion_rule_baseline_v2",
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
        "detector_predict_wallclock_ms": {
            "mean": float(np.mean(inference_ms)) if inference_ms else 0.0,
            "p50": float(np.percentile(inference_ms, 50)) if inference_ms else 0.0,
            "p95": float(np.percentile(inference_ms, 95)) if inference_ms else 0.0,
        },
        "assumptions": [
            "event matching protocol is provisional until organizer confirmation",
            "timing covers model.predict wall clock only, not complete end-to-end latency",
            "rule baseline is untrained; trained TCN will replace the primary temporal score",
        ],
    }
    output_events.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="端到端视觉跌倒检测基线")
    parser.add_argument("source", type=Path)
    parser.add_argument("--output-video", type=Path, default=Path("reports/demo.mp4"))
    parser.add_argument("--output-events", type=Path, default=Path("reports/events.json"))
    parser.add_argument("--model", default="yolo11n-pose.pt")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--crop", choices=["auto", "none", "left", "right"], default="auto")
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    payload = analyze_video(
        source=args.source,
        output_video=args.output_video,
        output_events=args.output_events,
        model_path=args.model,
        device=args.device,
        image_size=args.imgsz,
        confidence=args.conf,
        crop=args.crop,
        max_frames=args.max_frames,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
