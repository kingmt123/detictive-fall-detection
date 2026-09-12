"""Pose cache 到因果 FallTCN 窗口的确定性数据契约。

每个窗口只描述一个 ``track_id``。同一人物即使在 dense cache 中换到不同列，
也会按 ID 正确重组；未观测帧保留为全零并通过 ``observed_mask`` 显式标记。
标签取窗口末端时刻，确保在线推理时不借用未来信息；同时保留原始事件语义，
便于分别报告跌倒过程、倒地后状态和 hard negative。
"""
from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pipeline.pose_cache import PoseCacheRecord, read_pose_cache

TORSO_KEYPOINTS = (5, 6, 11, 12)
FALL_INCIDENT_SEMANTICS = frozenset({"fall_process", "post_fall_state"})
UNLABELED_BACKGROUND = "unlabeled_background"
MAX_BOUNDARY_OVERLAP_SECONDS = 1.0 / 30.0


@dataclass(frozen=True)
class ActivityInterval:
    start: float
    end: float
    semantics: str

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.start)
            or not math.isfinite(self.end)
            or self.end <= self.start
        ):
            raise ValueError("事件区间必须满足有限的 start < end")
        if not self.semantics:
            raise ValueError("事件语义必须是非空字符串")


@dataclass(frozen=True)
class TCNWindow:
    clip_id: str
    track_id: int
    start_frame: int
    end_frame: int
    end_time: float
    features: np.ndarray
    observed_mask: np.ndarray
    event_semantics: str
    label: np.float32


def parse_activity_intervals(events_json: str) -> tuple[ActivityInterval, ...]:
    """读取活动区间，并确定性修复不超过一帧的边界舍入冲突。

    同语义重叠直接合并；不同语义仅允许至多 1/30 秒的相邻边界重叠，并把冲突
    边界放在两端点中点。更大的异语义重叠仍视为标注错误。
    """
    try:
        events = json.loads(events_json)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("events_json 不是合法 JSON") from exc
    if not isinstance(events, list):
        raise TypeError("events_json 必须是事件列表")

    intervals: list[ActivityInterval] = []
    for event in events:
        if not isinstance(event, dict):
            raise TypeError("events_json 中的事件必须是对象")
        try:
            start = float(event["start"])
            end = float(event["end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("事件缺少合法 semantics/start/end") from exc
        semantics = event.get("semantics")
        if not isinstance(semantics, str) or not semantics:
            raise ValueError("事件缺少合法 semantics/start/end")
        intervals.append(ActivityInterval(start, end, semantics))
    ordered = sorted(intervals, key=lambda item: (item.start, item.end))
    normalized: list[ActivityInterval] = []
    for current in ordered:
        if not normalized or current.start >= normalized[-1].end:
            normalized.append(current)
            continue
        previous = normalized[-1]
        if current.semantics == previous.semantics:
            normalized[-1] = ActivityInterval(
                previous.start,
                max(previous.end, current.end),
                previous.semantics,
            )
            continue
        overlap = previous.end - current.start
        if overlap > MAX_BOUNDARY_OVERLAP_SECONDS:
            raise ValueError(
                "异语义事件区间重叠超过一帧容差: "
                f"{previous.semantics}[{previous.start},{previous.end}) 与 "
                f"{current.semantics}[{current.start},{current.end})"
            )
        boundary = (previous.end + current.start) * 0.5
        normalized[-1] = ActivityInterval(
            previous.start, boundary, previous.semantics
        )
        normalized.append(ActivityInterval(boundary, current.end, current.semantics))
    return tuple(normalized)


def semantics_at_time(
    timestamp: float, intervals: Iterable[ActivityInterval]
) -> str:
    """采用左闭右开区间；重叠语义视为标注错误而不是静默选一个。"""
    matches = [
        item.semantics for item in intervals if item.start <= timestamp < item.end
    ]
    if len(matches) > 1:
        raise ValueError(f"时间 {timestamp} 落入多个重叠事件区间: {matches}")
    return matches[0] if matches else UNLABELED_BACKGROUND


def label_for_semantics(
    semantics: str,
    positive_semantics: frozenset[str] = FALL_INCIDENT_SEMANTICS,
) -> np.float32:
    return np.float32(semantics in positive_semantics)


def normalize_pose(
    keypoints: np.ndarray,
    bbox: np.ndarray,
    *,
    confidence_threshold: float = 0.05,
) -> np.ndarray:
    """以躯干中心平移，分别以检测框宽高缩放 x/y，保留置信度。"""
    keypoints = np.asarray(keypoints, dtype=np.float32)
    bbox = np.asarray(bbox, dtype=np.float32)
    if keypoints.shape != (17, 3) or bbox.shape != (4,):
        raise ValueError("单人姿态必须是 keypoints[17,3] 和 bbox[4]")
    if not np.all(np.isfinite(keypoints)) or not np.all(np.isfinite(bbox)):
        raise ValueError("姿态和检测框必须是有限数值")

    confidence = np.clip(keypoints[:, 2], 0.0, 1.0)
    torso = keypoints[np.asarray(TORSO_KEYPOINTS)]
    torso_visible = torso[:, 2] >= confidence_threshold
    if np.any(torso_visible):
        weights = torso[torso_visible, 2]
        center = np.average(torso[torso_visible, :2], axis=0, weights=weights)
    else:
        visible = confidence >= confidence_threshold
        if np.any(visible):
            center = np.average(
                keypoints[visible, :2], axis=0, weights=confidence[visible]
            )
        else:
            center = np.array(
                [(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5],
                dtype=np.float32,
            )

    width = max(float(bbox[2] - bbox[0]), 1.0)
    height = max(float(bbox[3] - bbox[1]), 1.0)
    normalized = np.zeros((17, 3), dtype=np.float32)
    normalized[:, 0] = (keypoints[:, 0] - center[0]) / width
    normalized[:, 1] = (keypoints[:, 1] - center[1]) / height
    normalized[:, 2] = confidence
    normalized[confidence < confidence_threshold, :2] = 0.0
    return normalized


def build_tcn_windows(
    record: PoseCacheRecord,
    intervals: Iterable[ActivityInterval],
    *,
    window_size: int = 16,
    stride: int = 1,
    min_observed_frames: int | None = None,
    causal_left_pad: bool = False,
    positive_semantics: frozenset[str] = FALL_INCIDENT_SEMANTICS,
) -> list[TCNWindow]:
    """将 dense 多人 cache 重组为固定长度、逐轨迹、因果窗口。

    ``causal_left_pad`` 允许序列起点不足 ``window_size`` 时在左侧补零；补零位
    仍由 ``observed_mask=False`` 明确标识，不会伪造 pose。默认关闭以保持历史
    cache 契约。
    """
    if window_size < 1 or stride < 1:
        raise ValueError("window_size 和 stride 必须为正整数")
    if min_observed_frames is None:
        min_observed_frames = (window_size + 1) // 2
    if not 1 <= min_observed_frames <= window_size:
        raise ValueError("min_observed_frames 必须位于 [1, window_size]")
    if not positive_semantics or any(not item for item in positive_semantics):
        raise ValueError("positive_semantics 必须包含非空事件语义")
    frame_count = record.frame_indices.size
    if frame_count < window_size and not causal_left_pad:
        return []

    intervals = tuple(intervals)
    track_ids = sorted({int(value) for value in record.track_ids[record.valid_mask]})
    windows: list[TCNWindow] = []
    for track_id in track_ids:
        first_end_frame = 0 if causal_left_pad else window_size - 1
        for end_frame in range(first_end_frame, frame_count, stride):
            # 在线预测只在当前帧实际看见该人物时产生样本。
            endpoint_columns = np.flatnonzero(
                record.valid_mask[end_frame]
                & (record.track_ids[end_frame] == track_id)
            )
            if endpoint_columns.size != 1:
                continue
            logical_start_frame = end_frame - window_size + 1
            source_start_frame = max(0, logical_start_frame)
            left_pad = source_start_frame - logical_start_frame
            features = np.zeros((window_size, 17, 3), dtype=np.float32)
            observed = np.zeros(window_size, dtype=np.bool_)
            for source_offset, frame_index in enumerate(
                range(source_start_frame, end_frame + 1)
            ):
                columns = np.flatnonzero(
                    record.valid_mask[frame_index]
                    & (record.track_ids[frame_index] == track_id)
                )
                if columns.size == 0:
                    continue
                if columns.size != 1:
                    raise ValueError("同一帧出现重复 track_id")
                column = int(columns[0])
                offset = left_pad + source_offset
                features[offset] = normalize_pose(
                    record.keypoints[frame_index, column],
                    record.bboxes[frame_index, column],
                )
                observed[offset] = True
            if int(observed.sum()) < min_observed_frames:
                continue
            end_time = float(record.timestamps[end_frame])
            event_semantics = semantics_at_time(end_time, intervals)
            windows.append(
                TCNWindow(
                    clip_id=record.clip_id,
                    track_id=track_id,
                    start_frame=logical_start_frame,
                    end_frame=end_frame,
                    end_time=end_time,
                    features=features,
                    observed_mask=observed,
                    event_semantics=event_semantics,
                    label=label_for_semantics(
                        event_semantics, positive_semantics=positive_semantics
                    ),
                )
            )
    return windows


def load_tcn_windows(
    cache_path: Path,
    events_json: str,
    *,
    expected_clip_id: str | None = None,
    expected_dataset: str | None = None,
    expected_split: str | None = None,
    **window_kwargs: object,
) -> list[TCNWindow]:
    """从冻结的只读 NPZ 和 manifest 标注直接构造训练窗口。"""
    record = read_pose_cache(cache_path)
    expected_identity = {
        "clip_id": expected_clip_id,
        "dataset": expected_dataset,
        "split": expected_split,
    }
    for field, expected in expected_identity.items():
        if expected is not None and getattr(record, field) != expected:
            raise ValueError(
                f"pose cache {field} 不匹配: expected={expected!r}, "
                f"actual={getattr(record, field)!r}"
            )
    return build_tcn_windows(
        record,
        parse_activity_intervals(events_json),
        **window_kwargs,
    )
