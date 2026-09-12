"""从只读 pose NPZ 构建可复用、可审计的 FallTCN 窗口 memmap。"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from models.tcn_window import build_tcn_windows, parse_activity_intervals
from pipeline.pose_cache import pose_cache_path, read_pose_cache

WINDOW_CACHE_SCHEMA = 1
SEMANTICS = (
    "background",
    "fall_process",
    "hard_negative",
    "post_fall_state",
    "unlabeled_background",
)
SEMANTIC_TO_CODE = {name: index for index, name in enumerate(SEMANTICS)}
ARRAY_SPECS = {
    "features": ("features.bin", np.dtype("<f4")),
    "labels": ("labels.bin", np.dtype("u1")),
    "clip_indices": ("clip_indices.bin", np.dtype("<i4")),
    "track_ids": ("track_ids.bin", np.dtype("<i8")),
    "end_times": ("end_times.bin", np.dtype("<f4")),
    "semantic_codes": ("semantic_codes.bin", np.dtype("u1")),
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _implementation_sha256() -> str:
    digest = hashlib.sha256()
    root = Path(__file__).parent
    for name in ("tcn_window.py", "tcn_dataset.py"):
        digest.update(name.encode("utf-8"))
        digest.update(
            (root / name)
            .read_text(encoding="utf-8")
            .replace("\r\n", "\n")
            .encode("utf-8")
        )
    return digest.hexdigest()


def _load_rows(
    manifest_path: Path, *, dataset: str, split: str
) -> list[dict[str, str]]:
    if split == "test":
        raise ValueError("TCN 数据构建阶段禁止读取 test split")
    with Path(manifest_path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"dataset", "split", "clip_id", "has_fall", "events_json"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"manifest 缺少字段: {sorted(missing)}")
        rows = [
            row
            for row in reader
            if row["dataset"] == dataset and row["split"] == split
        ]
    if not rows:
        raise ValueError(f"manifest 中没有 dataset={dataset!r}, split={split!r}")
    identities = [(row["dataset"], row["split"], row["clip_id"]) for row in rows]
    if any(not all(identity) for identity in identities):
        raise ValueError("manifest 的 dataset/split/clip_id 不能为空")
    if len(set(identities)) != len(identities):
        raise ValueError("manifest 存在重复 cache 身份")
    if any(row["has_fall"] not in {"0", "1"} for row in rows):
        raise ValueError("manifest has_fall 必须为 0 或 1")
    return rows


def _load_audit(audit_path: Path, manifest_sha256: str) -> dict[str, Any]:
    try:
        audit = json.loads(Path(audit_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("cache audit 不是合法 JSON") from exc
    if audit.get("manifest_sha256") != manifest_sha256:
        raise ValueError("cache audit 与 manifest SHA-256 不匹配")
    if audit.get("errors"):
        raise ValueError(f"cache audit 未通过: {audit['errors']}")
    signatures = audit.get("extractor_signatures")
    if not isinstance(signatures, list) or len(signatures) != 1:
        raise ValueError("cache audit 必须包含恰好一个提取签名")
    if audit.get("valid") != audit.get("selected"):
        raise ValueError("cache audit valid/selected 不一致")
    return audit


def _signature(
    *,
    manifest_path: Path,
    audit_path: Path,
    audit: dict[str, Any],
    dataset: str,
    split: str,
    window_size: int,
    stride: int,
    min_observed_frames: int,
    causal_left_pad: bool,
) -> tuple[str, dict[str, Any]]:
    payload = {
        "window_cache_schema": WINDOW_CACHE_SCHEMA,
        "dataset": dataset,
        "split": split,
        "manifest_sha256": _sha256_file(manifest_path),
        "audit_sha256": _sha256_file(audit_path),
        "extractor_signature": audit["extractor_signatures"][0],
        "implementation_sha256": _implementation_sha256(),
        "window_config": {
            "window_size": window_size,
            "stride": stride,
            "min_observed_frames": min_observed_frames,
            "causal_left_pad": causal_left_pad,
        },
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest(), payload


def _expected_bytes(metadata: dict[str, Any], key: str) -> int:
    count = int(metadata["sample_count"])
    window_size = int(metadata["window_config"]["window_size"])
    _, dtype = ARRAY_SPECS[key]
    elements = count * window_size * 17 * 3 if key == "features" else count
    return elements * dtype.itemsize


@dataclass(frozen=True)
class WindowMemmapCache:
    root: Path
    metadata: dict[str, Any]

    @property
    def sample_count(self) -> int:
        return int(self.metadata["sample_count"])

    def array(self, key: str) -> np.memmap:
        if key not in ARRAY_SPECS:
            raise KeyError(key)
        filename, dtype = ARRAY_SPECS[key]
        shape: tuple[int, ...]
        if key == "features":
            shape = (
                self.sample_count,
                int(self.metadata["window_config"]["window_size"]),
                17,
                3,
            )
        else:
            shape = (self.sample_count,)
        return np.memmap(self.root / filename, mode="r", dtype=dtype, shape=shape)


def load_window_cache(
    root: Path,
    *,
    expected_signature_sha256: str | None = None,
    verify_hashes: bool = False,
) -> WindowMemmapCache:
    root = Path(root)
    try:
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"窗口 cache metadata 无效: {root}") from exc
    if metadata.get("window_cache_schema") != WINDOW_CACHE_SCHEMA:
        raise ValueError("不支持的窗口 cache schema")
    if expected_signature_sha256 is not None and (
        metadata.get("signature_sha256") != expected_signature_sha256
    ):
        raise ValueError("窗口 cache 签名不匹配")
    if int(metadata.get("sample_count", 0)) <= 0:
        raise ValueError("窗口 cache 不能为空")
    clips = metadata.get("clips")
    if not isinstance(clips, list) or not clips:
        raise ValueError("窗口 cache 缺少 clip 元数据")
    for key, (filename, _) in ARRAY_SPECS.items():
        path = root / filename
        if not path.is_file() or path.stat().st_size != _expected_bytes(metadata, key):
            raise ValueError(f"窗口 cache 文件大小不匹配: {filename}")
        if verify_hashes:
            actual_sha256 = _sha256_file(path)
            expected_sha256 = metadata["files"][key]["sha256"]
            if actual_sha256 != expected_sha256:
                raise ValueError(
                    f"窗口 cache SHA-256 不匹配: {filename}; "
                    f"expected={expected_sha256}, actual={actual_sha256}"
                )

    next_start = 0
    seen_clip_ids: set[str] = set()
    for clip in clips:
        clip_id = str(clip["clip_id"])
        if clip_id in seen_clip_ids:
            raise ValueError(f"窗口 cache clip_id 重复: {clip_id}")
        seen_clip_ids.add(clip_id)
        if int(clip["window_start"]) != next_start:
            raise ValueError(f"窗口 cache 范围不连续: {clip_id}")
        next_start += int(clip["window_count"])
    if next_start != int(metadata["sample_count"]):
        raise ValueError("窗口 cache clip 范围与 sample_count 不一致")
    return WindowMemmapCache(root=root, metadata=metadata)


def build_window_cache(
    manifest_path: Path,
    pose_cache_root: Path,
    audit_path: Path,
    output_root: Path,
    *,
    dataset: str,
    split: str,
    window_size: int = 16,
    stride: int = 1,
    min_observed_frames: int = 8,
    causal_left_pad: bool = False,
    progress: Callable[[dict[str, int]], None] | None = None,
) -> WindowMemmapCache:
    """单次流式构建连续窗口特征；同签名产物存在时只校验并复用。"""
    if window_size < 1 or stride < 1:
        raise ValueError("window_size 和 stride 必须为正整数")
    if not 1 <= min_observed_frames <= window_size:
        raise ValueError("min_observed_frames 必须位于 [1, window_size]")
    manifest_path = Path(manifest_path)
    audit_path = Path(audit_path)
    manifest_sha256 = _sha256_file(manifest_path)
    audit = _load_audit(audit_path, manifest_sha256)
    rows = _load_rows(manifest_path, dataset=dataset, split=split)
    signature_sha256, signature = _signature(
        manifest_path=manifest_path,
        audit_path=audit_path,
        audit=audit,
        dataset=dataset,
        split=split,
        window_size=window_size,
        stride=stride,
        min_observed_frames=min_observed_frames,
        causal_left_pad=causal_left_pad,
    )
    target = Path(output_root) / dataset / split / signature_sha256
    if target.exists():
        return load_window_cache(
            target,
            expected_signature_sha256=signature_sha256,
            verify_hashes=True,
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(
        tempfile.mkdtemp(prefix=f".{signature_sha256}.", dir=target.parent)
    )
    handles: dict[str, Any] = {}
    digests = {key: hashlib.sha256() for key in ARRAY_SPECS}
    clips: list[dict[str, Any]] = []
    semantic_counts = {name: 0 for name in SEMANTICS}
    positive_count = 0
    sample_count = 0
    expected_extractor = audit["extractor_signatures"][0]
    try:
        for key, (filename, _) in ARRAY_SPECS.items():
            handles[key] = (temp_root / filename).open("wb")
        for clip_index, row in enumerate(rows):
            cache_path = pose_cache_path(
                Path(pose_cache_root), dataset, split, row["clip_id"]
            )
            record = read_pose_cache(cache_path)
            if (record.dataset, record.split, record.clip_id) != (
                dataset,
                split,
                row["clip_id"],
            ):
                raise ValueError(f"manifest/cache 身份不匹配: {row['clip_id']}")
            if record.extractor_signature != expected_extractor:
                raise ValueError(f"提取签名不匹配: {row['clip_id']}")
            windows = build_tcn_windows(
                record,
                parse_activity_intervals(row["events_json"]),
                window_size=window_size,
                stride=stride,
                min_observed_frames=min_observed_frames,
                causal_left_pad=causal_left_pad,
            )
            clip_start = sample_count
            if windows:
                arrays = {
                    "features": np.stack(
                        [window.features for window in windows]
                    ).astype("<f4", copy=False),
                    "labels": np.asarray(
                        [int(window.label) for window in windows], dtype="u1"
                    ),
                    "clip_indices": np.full(
                        len(windows), clip_index, dtype="<i4"
                    ),
                    "track_ids": np.asarray(
                        [window.track_id for window in windows], dtype="<i8"
                    ),
                    "end_times": np.asarray(
                        [window.end_time for window in windows], dtype="<f4"
                    ),
                    "semantic_codes": np.asarray(
                        [SEMANTIC_TO_CODE[window.event_semantics] for window in windows],
                        dtype="u1",
                    ),
                }
                for key, array in arrays.items():
                    payload = array.tobytes(order="C")
                    handles[key].write(payload)
                    digests[key].update(payload)
                clip_positive = int(arrays["labels"].sum())
                positive_count += clip_positive
                for code, name in enumerate(SEMANTICS):
                    semantic_counts[name] += int(
                        np.count_nonzero(arrays["semantic_codes"] == code)
                    )
                sample_count += len(windows)
            else:
                clip_positive = 0
            clips.append(
                {
                    "clip_id": row["clip_id"],
                    "has_fall": row["has_fall"] == "1",
                    "window_start": clip_start,
                    "window_count": len(windows),
                    "positive_windows": clip_positive,
                }
            )
            if progress is not None and (
                (clip_index + 1) % 100 == 0 or clip_index + 1 == len(rows)
            ):
                progress(
                    {
                        "clips_completed": clip_index + 1,
                        "clips_total": len(rows),
                        "windows": sample_count,
                    }
                )
        for handle in handles.values():
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
        handles.clear()
        metadata = {
            "window_cache_schema": WINDOW_CACHE_SCHEMA,
            "signature_sha256": signature_sha256,
            "signature": signature,
            "dataset": dataset,
            "split": split,
            "window_config": signature["window_config"],
            "sample_count": sample_count,
            "positive_count": positive_count,
            "negative_count": sample_count - positive_count,
            "semantic_counts": semantic_counts,
            "clips": clips,
            "files": {
                key: {
                    "name": filename,
                    "sha256": digests[key].hexdigest(),
                }
                for key, (filename, _) in ARRAY_SPECS.items()
            },
        }
        metadata_path = temp_root / "metadata.json"
        with metadata_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(metadata, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if sample_count <= 0:
            raise ValueError("所选 split 没有可训练窗口")
        load_window_cache(
            temp_root, expected_signature_sha256=signature_sha256
        )
        os.replace(temp_root, target)
    except Exception:
        for handle in handles.values():
            handle.close()
        shutil.rmtree(temp_root, ignore_errors=True)
        raise
    return load_window_cache(target, expected_signature_sha256=signature_sha256)
