"""Gate 3D：固定时长采样，并保留局部姿态与全局人体运动。"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from models.tcn_window import (
    FALL_INCIDENT_SEMANTICS,
    ActivityInterval,
    label_for_semantics,
    normalize_pose,
    semantics_at_time,
)
from pipeline.pose_cache import PoseCacheRecord

LOCAL_DIM = 17 * 3
GLOBAL_DIM = 9
FEATURE_DIM = LOCAL_DIM + GLOBAL_DIM


@dataclass(frozen=True)
class TCNV2Window:
    clip_id: str
    track_id: int
    end_time: float
    features: np.ndarray
    observed_mask: np.ndarray
    event_semantics: str
    label: np.float32


def _nearest_frame(timestamps: np.ndarray, target: float) -> tuple[int, float]:
    position = int(np.searchsorted(timestamps, target, side="left"))
    candidates = [index for index in (position - 1, position) if 0 <= index < len(timestamps)]
    index = min(candidates, key=lambda item: (abs(float(timestamps[item]) - target), item))
    return index, abs(float(timestamps[index]) - target)


def _observation(
    record: PoseCacheRecord, frame_index: int, track_id: int
) -> tuple[np.ndarray, np.ndarray] | None:
    columns = np.flatnonzero(
        record.valid_mask[frame_index] & (record.track_ids[frame_index] == track_id)
    )
    if columns.size == 0:
        return None
    if columns.size != 1:
        raise ValueError("同一帧出现重复 track_id")
    column = int(columns[0])
    return record.keypoints[frame_index, column], record.bboxes[frame_index, column]


def encode_window(
    observations: list[tuple[np.ndarray, np.ndarray] | None],
    *,
    frame_size: tuple[int, int],
    sample_rate_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """编码 local pose + [cx,cy,w,h,logAR,vx,vy,vAR,observed]。"""
    if sample_rate_hz <= 0 or len(frame_size) != 2:
        raise ValueError("sample_rate_hz/frame_size 无效")
    height, width = map(float, frame_size)
    if height <= 0 or width <= 0:
        raise ValueError("frame_size 必须为正")
    features = np.zeros((len(observations), FEATURE_DIM), dtype=np.float32)
    observed = np.zeros(len(observations), dtype=np.bool_)
    globals_raw: list[tuple[float, float, float, float, float] | None] = []
    for index, item in enumerate(observations):
        if item is None:
            globals_raw.append(None)
            continue
        keypoints, bbox = item
        bbox = np.asarray(bbox, dtype=np.float32)
        features[index, :LOCAL_DIM] = normalize_pose(keypoints, bbox).reshape(-1)
        x1, y1, x2, y2 = map(float, bbox)
        box_w, box_h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
        cx, cy = (x1 + x2) * 0.5 / width, (y1 + y2) * 0.5 / height
        normalized_w, normalized_h = box_w / width, box_h / height
        log_aspect = math.log(box_w / box_h)
        globals_raw.append((cx, cy, normalized_w, normalized_h, log_aspect))
        observed[index] = True

    dt = 1.0 / sample_rate_hz
    for index, raw in enumerate(globals_raw):
        if raw is None:
            continue
        cx, cy, box_w, box_h, log_aspect = raw
        vx = vy = aspect_velocity = 0.0
        if index > 0 and globals_raw[index - 1] is not None:
            previous = globals_raw[index - 1]
            assert previous is not None
            vx = (cx - previous[0]) / dt
            vy = (cy - previous[1]) / dt
            aspect_velocity = (log_aspect - previous[4]) / dt
        features[index, LOCAL_DIM:] = (
            cx,
            cy,
            box_w,
            box_h,
            np.clip(log_aspect / 3.0, -1.0, 1.0),
            np.clip(vx / 2.0, -1.0, 1.0),
            np.clip(vy / 2.0, -1.0, 1.0),
            np.clip(aspect_velocity / 4.0, -1.0, 1.0),
            1.0,
        )
    return features, observed


def build_tcn_v2_windows(
    record: PoseCacheRecord,
    intervals: tuple[ActivityInterval, ...],
    *,
    window_size: int = 16,
    sample_rate_hz: float = 16.0,
    min_observed_frames: int = 8,
    positive_semantics: frozenset[str] = FALL_INCIDENT_SEMANTICS,
) -> list[TCNV2Window]:
    if window_size < 1 or sample_rate_hz <= 0:
        raise ValueError("窗口参数无效")
    if not 1 <= min_observed_frames <= window_size:
        raise ValueError("min_observed_frames 必须位于窗口范围内")
    timestamps = np.asarray(record.timestamps, dtype=np.float64)
    if timestamps.ndim != 1 or not timestamps.size or np.any(np.diff(timestamps) <= 0):
        raise ValueError("pose cache timestamps 必须严格递增")
    horizon = (window_size - 1) / sample_rate_hz
    tolerance = 0.75 / sample_rate_hz
    track_ids = sorted({int(value) for value in record.track_ids[record.valid_mask]})
    windows = []
    for track_id in track_ids:
        for end_frame, end_time_value in enumerate(timestamps):
            end_time = float(end_time_value)
            if end_time + 1e-9 < float(timestamps[0]) + horizon:
                continue
            if _observation(record, end_frame, track_id) is None:
                continue
            target_times = end_time - np.arange(window_size - 1, -1, -1) / sample_rate_hz
            sampled = []
            for target in target_times:
                frame_index, distance = _nearest_frame(timestamps, float(target))
                sampled.append(
                    _observation(record, frame_index, track_id)
                    if distance <= tolerance else None
                )
            features, observed = encode_window(
                sampled, frame_size=record.frame_size, sample_rate_hz=sample_rate_hz
            )
            if int(observed.sum()) < min_observed_frames:
                continue
            semantics = semantics_at_time(end_time, intervals)
            windows.append(
                TCNV2Window(
                    clip_id=record.clip_id,
                    track_id=track_id,
                    end_time=end_time,
                    features=features,
                    observed_mask=observed,
                    event_semantics=semantics,
                    label=label_for_semantics(semantics, positive_semantics),
                )
            )
    return windows
