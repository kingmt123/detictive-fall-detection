"""可审计、原子写入的姿态序列 NPZ cache。"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

CACHE_SCHEMA = 1
_SAFE_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


@dataclass(frozen=True)
class PoseCacheRecord:
    clip_id: str
    dataset: str
    split: str
    source_identity: dict[str, Any]
    source_content_sha256: str
    extractor_signature: dict[str, Any]
    fps: float
    frame_indices: np.ndarray
    timestamps: np.ndarray
    keypoints: np.ndarray
    bboxes: np.ndarray
    track_ids: np.ndarray
    valid_mask: np.ndarray
    frame_size: np.ndarray


def _canonical_json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def pose_cache_path(root: Path, dataset: str, split: str, clip_id: str) -> Path:
    """用 clip ID 哈希命名，避免 ``/`` 被解释成目录或发生路径穿越。"""
    if _SAFE_COMPONENT.fullmatch(dataset) is None:
        raise ValueError("dataset 不是安全路径组件")
    if _SAFE_COMPONENT.fullmatch(split) is None:
        raise ValueError("split 不是安全路径组件")
    digest = hashlib.sha256(clip_id.encode("utf-8")).hexdigest()
    return Path(root) / dataset / split / f"{digest}.npz"


def _payload(record: PoseCacheRecord) -> dict[str, np.ndarray]:
    return {
        "cache_schema": np.asarray(CACHE_SCHEMA, dtype=np.int64),
        "clip_id": np.asarray(record.clip_id),
        "dataset": np.asarray(record.dataset),
        "split": np.asarray(record.split),
        "source_identity_json": np.asarray(_canonical_json(record.source_identity)),
        "source_content_sha256": np.asarray(record.source_content_sha256),
        "extractor_signature_json": np.asarray(
            _canonical_json(record.extractor_signature)
        ),
        "fps": np.asarray(record.fps, dtype=np.float64),
        "frame_indices": record.frame_indices,
        "timestamps": record.timestamps,
        "keypoints": record.keypoints,
        "bboxes": record.bboxes,
        "track_ids": record.track_ids,
        "valid_mask": record.valid_mask,
        "frame_size": record.frame_size,
    }


def _validate_record(record: PoseCacheRecord) -> None:
    if not record.clip_id or not record.dataset or not record.split:
        raise ValueError("clip_id、dataset 和 split 必须是非空字符串")
    if re.fullmatch(r"[0-9a-f]{64}", record.source_content_sha256) is None:
        raise ValueError("source_content_sha256 必须是 64 位小写 SHA-256")
    if not record.source_identity or not record.extractor_signature:
        raise ValueError("源身份和提取签名不能为空")
    if (
        record.frame_indices.dtype != np.int64
        or record.frame_indices.ndim != 1
        or not record.frame_indices.size
        or not np.array_equal(
            record.frame_indices,
            np.arange(record.frame_indices.size, dtype=np.int64),
        )
    ):
        raise ValueError("frame_indices 必须是从 0 开始的连续 int64 一维数组")
    if not np.isfinite(record.fps) or record.fps <= 0.0:
        raise ValueError("timestamps 需要有限且为正的 fps")
    expected_timestamps = record.frame_indices.astype(np.float64) / record.fps
    if (
        record.timestamps.dtype != np.float64
        or record.timestamps.shape != record.frame_indices.shape
        or not np.all(np.isfinite(record.timestamps))
        or not np.allclose(
            record.timestamps, expected_timestamps, rtol=0.0, atol=1e-12
        )
    ):
        raise ValueError("timestamps 必须是有限且等于 frame_indices/fps 的 float64 数组")
    frame_count = record.frame_indices.size
    if (
        record.keypoints.dtype != np.float32
        or record.keypoints.ndim != 4
        or record.keypoints.shape[0] != frame_count
        or record.keypoints.shape[2:] != (17, 3)
        or not np.all(np.isfinite(record.keypoints))
    ):
        raise ValueError("keypoints 必须是有限的 float32 [T,P,17,3] 数组")
    observation_shape = record.keypoints.shape[:2]
    if (
        record.bboxes.dtype != np.float32
        or record.bboxes.shape != (*observation_shape, 4)
        or not np.all(np.isfinite(record.bboxes))
    ):
        raise ValueError("bboxes 必须具有 [T,P,4] shape")
    if record.track_ids.dtype != np.int64 or record.track_ids.shape != observation_shape:
        raise ValueError("track_ids 必须具有 int64 [T,P] shape")
    if record.valid_mask.dtype != np.bool_ or record.valid_mask.shape != observation_shape:
        raise ValueError("valid_mask 必须具有 bool [T,P] shape")
    if np.any(record.track_ids[~record.valid_mask] != -1):
        raise ValueError("padding observation 的 track_id 必须为 -1")
    if np.any(record.track_ids[record.valid_mask] < 0):
        raise ValueError("有效 observation 的 track_id 必须为非负整数")
    for frame_track_ids, frame_valid_mask in zip(
        record.track_ids, record.valid_mask, strict=True
    ):
        valid_count = int(frame_valid_mask.sum())
        if not np.all(frame_valid_mask[:valid_count]) or np.any(
            frame_valid_mask[valid_count:]
        ):
            raise ValueError("valid observation 必须按 track_ids 打包在每帧前缀")
        valid_track_ids = frame_track_ids[:valid_count]
        if valid_track_ids.size > 1 and np.any(np.diff(valid_track_ids) <= 0):
            raise ValueError("每帧有效 track_ids 必须唯一且严格递增")
    if (
        record.frame_size.dtype != np.int64
        or record.frame_size.shape != (2,)
        or np.any(record.frame_size <= 0)
    ):
        raise ValueError("frame_size 必须是正整数 int64 [height,width]")


def load_pose_cache(
    path: Path,
    *,
    expected_source_identity: dict[str, Any],
    expected_extractor_signature: dict[str, Any],
) -> PoseCacheRecord:
    """加载与当前源、提取实现一致的 cache；不允许 pickle。"""
    try:
        with np.load(path, allow_pickle=False) as payload:
            schema = int(payload["cache_schema"])
            source_identity_json = str(payload["source_identity_json"].item())
            extractor_signature_json = str(
                payload["extractor_signature_json"].item()
            )
            if schema != CACHE_SCHEMA:
                raise ValueError(f"不支持的 pose cache schema: {schema}")
            if source_identity_json != _canonical_json(expected_source_identity):
                raise ValueError("pose cache 源身份不匹配")
            if extractor_signature_json != _canonical_json(expected_extractor_signature):
                raise ValueError("pose cache 提取签名不匹配")
            record = PoseCacheRecord(
                clip_id=str(payload["clip_id"].item()),
                dataset=str(payload["dataset"].item()),
                split=str(payload["split"].item()),
                source_identity=json.loads(source_identity_json),
                source_content_sha256=str(payload["source_content_sha256"].item()),
                extractor_signature=json.loads(extractor_signature_json),
                fps=float(payload["fps"]),
                frame_indices=payload["frame_indices"].copy(),
                timestamps=payload["timestamps"].copy(),
                keypoints=payload["keypoints"].copy(),
                bboxes=payload["bboxes"].copy(),
                track_ids=payload["track_ids"].copy(),
                valid_mask=payload["valid_mask"].copy(),
                frame_size=payload["frame_size"].copy(),
            )
            _validate_record(record)
            return record
    except (OSError, KeyError, TypeError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
        raise ValueError(f"pose cache 损坏或字段不完整: {path}") from exc


def write_pose_cache(path: Path, record: PoseCacheRecord) -> None:
    """同目录写临时 NPZ，自校验后原子替换目标。"""
    _validate_record(record)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary_path = Path(handle.name)
            np.savez_compressed(handle, **_payload(record))
            handle.flush()
            os.fsync(handle.fileno())
        load_pose_cache(
            temporary_path,
            expected_source_identity=record.source_identity,
            expected_extractor_signature=record.extractor_signature,
        )
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
