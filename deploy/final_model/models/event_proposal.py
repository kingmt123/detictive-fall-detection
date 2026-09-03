"""Label-blind motion event proposals from tracked pose observations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

PROTOCOL = "pose_motion_event_proposal_v1"


@dataclass(frozen=True)
class MotionEventProposal:
    track_id: int
    start_time: float
    peak_time: float
    end_time: float
    motion_score: float
    observed_frames: int


def _track_observations(
    record: Any, track_id: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cells = record.valid_mask & (record.track_ids == int(track_id))
    frames, columns = np.nonzero(cells)
    if frames.size == 0:
        raise ValueError(f"track_id={track_id} 没有姿态观测")
    if np.unique(frames).size != frames.size:
        raise ValueError(f"track_id={track_id} 在同一帧重复")
    boxes = np.asarray(record.bboxes[frames, columns], dtype=np.float64)
    height, width = map(float, record.frame_size)
    if min(height, width) <= 1.0:
        raise ValueError("pose record frame_size 无效")
    center_y = (boxes[:, 1] + boxes[:, 3]) * 0.5 / height
    box_width = np.maximum(boxes[:, 2] - boxes[:, 0], 1.0)
    box_height = np.maximum(boxes[:, 3] - boxes[:, 1], 1.0)
    aspect = box_width / box_height
    return frames.astype(np.int64), center_y, aspect


def _motion_scores(
    frames: np.ndarray,
    center_y: np.ndarray,
    aspect: np.ndarray,
    *,
    fps: float,
) -> np.ndarray:
    """Score each observation from only its preceding 1.25 seconds."""
    lookback = max(1, round(float(fps) * 1.25))
    scores = np.zeros(frames.size, dtype=np.float64)
    for index, frame in enumerate(frames):
        prior = (frames >= frame - lookback) & (frames <= frame)
        descent = max(0.0, float(center_y[index] - center_y[prior].min()))
        aspect_gain = max(0.0, float(aspect[index] - aspect[prior].min()))
        horizontal_posture = max(0.0, float(aspect[index] - 0.60))
        scores[index] = descent + 0.25 * aspect_gain + 0.05 * horizontal_posture
    return scores


def propose_motion_event(
    record: Any, *, context_seconds: float = 2.0
) -> MotionEventProposal:
    """Select one actor and fixed-width event interval without labels or events."""
    if not np.isfinite(record.fps) or float(record.fps) <= 0.0:
        raise ValueError("pose record fps 无效")
    if not np.isfinite(context_seconds) or context_seconds <= 0.0:
        raise ValueError("context_seconds 必须为正")
    track_ids = np.unique(np.asarray(record.track_ids)[record.valid_mask])
    if track_ids.size == 0 or np.any(track_ids < 0):
        raise ValueError("pose record 没有有效轨迹")
    candidates: list[tuple[tuple[float, int, int], int, int, float]] = []
    for value in track_ids:
        track_id = int(value)
        frames, center_y, aspect = _track_observations(record, track_id)
        scores = _motion_scores(frames, center_y, aspect, fps=float(record.fps))
        peak_index = int(np.argmax(scores))
        rank = (float(scores[peak_index]), int(frames.size), -track_id)
        candidates.append((rank, track_id, int(frames[peak_index]), float(scores[peak_index])))
    _, track_id, peak_frame, motion_score = max(candidates, key=lambda row: row[0])
    duration = float(record.timestamps[-1]) if record.timestamps.size > 1 else 0.0
    peak_time = float(peak_frame) / float(record.fps)
    start_time = max(0.0, peak_time - context_seconds)
    end_time = min(duration, peak_time + context_seconds)
    if end_time <= start_time:
        end_time = min(duration, start_time + 1.0 / float(record.fps))
    if end_time <= start_time:
        raise ValueError("无法生成非空事件区间")
    observed = int(np.count_nonzero(record.valid_mask & (record.track_ids == track_id)))
    return MotionEventProposal(
        track_id=track_id,
        start_time=start_time,
        peak_time=peak_time,
        end_time=end_time,
        motion_score=motion_score,
        observed_frames=observed,
    )
