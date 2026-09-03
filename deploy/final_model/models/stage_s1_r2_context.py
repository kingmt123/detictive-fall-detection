"""Construct masked sparse 48-frame context from audited 16-frame windows."""

from __future__ import annotations

import numpy as np


def sparse_context_indices(
    end_times: np.ndarray,
    track_ids: np.ndarray,
    clip_ranges: list[tuple[int, int]],
    *,
    frame_stride: int = 3,
    horizon_frames: int = 48,
) -> np.ndarray:
    """Return source-window indices for sparse context last tokens, or ``-1``.

    The newest six samples are read directly from the current 16-frame window
    by the caller.  Older samples refer only to earlier windows of the same
    clip and track; unavailable history remains masked rather than copied.
    """
    if end_times.ndim != 1 or track_ids.shape != end_times.shape:
        raise ValueError("end_times/track_ids 必须为同形一维数组")
    if frame_stride < 1 or horizon_frames < 16 or horizon_frames % frame_stride:
        raise ValueError("context 时间尺度参数无效")
    count = horizon_frames // frame_stride
    result = np.full((end_times.size, count), -1, dtype=np.int64)
    for start, size in clip_ranges:
        if start < 0 or size < 0 or start + size > end_times.size:
            raise ValueError("clip 范围越界")
        indices = np.arange(start, start + size, dtype=np.int64)
        for track_id in np.unique(track_ids[indices]):
            track = indices[track_ids[indices] == track_id]
            times = end_times[track]
            if times.size > 1 and np.any(np.diff(times) <= 0.0):
                raise ValueError("同一 track 的窗口结束时间必须严格递增")
            step = float(np.median(np.diff(times))) if times.size > 1 else 0.0
            if step <= 0.0:
                continue
            for row, time in zip(track, times, strict=True):
                for column in range(count - 6):
                    offset = horizon_frames - frame_stride * (column + 1)
                    target = time - offset * step
                    location = int(np.searchsorted(times, target))
                    candidates = [value for value in (location - 1, location) if 0 <= value < times.size]
                    if candidates:
                        closest = min(candidates, key=lambda value: abs(times[value] - target))
                        if abs(times[closest] - target) <= step * 0.25:
                            result[row, column] = track[closest]
    return result


def build_sparse_context(
    features: np.ndarray,
    sidecar: np.ndarray,
    source_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build context windows with zero/observed-mask padding for unavailable history."""
    if features.ndim != 4 or features.shape[1:] != (16, 17, 3):
        raise ValueError("features 必须为形状 (N,16,17,3)")
    if sidecar.shape != (features.shape[0], 16, 6) or source_indices.shape != (features.shape[0], 16):
        raise ValueError("sidecar 或 source_indices 形状不匹配")
    context_features = np.zeros_like(features)
    context_sidecar = np.zeros_like(sidecar)
    for row, sources in enumerate(source_indices):
        for column, source in enumerate(sources):
            if source >= 0:
                context_features[row, column] = features[source, -1]
                context_sidecar[row, column] = sidecar[source, -1]
            elif column >= 10:
                local_offset = (column - 10) * 3
                context_features[row, column] = features[row, local_offset]
                context_sidecar[row, column] = sidecar[row, local_offset]
    return context_features, context_sidecar
