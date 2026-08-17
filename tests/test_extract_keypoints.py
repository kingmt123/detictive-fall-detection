"""manifest 驱动的 cache-first 姿态提取测试。"""
import csv
import os
from pathlib import Path

import numpy as np
import pytest

from pipeline.pose_extractor import PoseSequence
from tools.extract_keypoints import extract_manifest


class _FakeExtractor:
    def __init__(self):
        self.calls: list[Path] = []

    def cache_signature(self, *, crop: str, max_frames: int | None):
        return {"protocol": "fake-v1", "crop": crop, "max_frames": max_frames}

    def extract(self, source: Path, *, crop: str, max_frames: int | None):
        self.calls.append(Path(source))
        frame_indices = np.array([0, 1], dtype=np.int64)
        return PoseSequence(
            fps=16.0,
            frame_indices=frame_indices,
            timestamps=frame_indices.astype(np.float64) / 16.0,
            keypoints=np.empty((2, 0, 17, 3), dtype=np.float32),
            bboxes=np.empty((2, 0, 4), dtype=np.float32),
            track_ids=np.empty((2, 0), dtype=np.int64),
            valid_mask=np.empty((2, 0), dtype=np.bool_),
            frame_size=np.array([48, 64], dtype=np.int64),
        )


def _write_manifest(path: Path, source: Path, *, split: str = "val") -> None:
    row = {
        "dataset": "urfd",
        "split": split,
        "clip_id": "fall-a",
        "video_path": str(source),
        "has_fall": "1",
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=row)
        writer.writeheader()
        writer.writerow(row)


def test_extract_manifest_resumes_valid_cache_and_rebuilds_changed_source(
    tmp_path: Path,
):
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"video-v1")
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, source)
    cache_root = tmp_path / "cache"
    temp_root = tmp_path / "temp"
    extractor = _FakeExtractor()

    first = extract_manifest(
        manifest,
        extractor=extractor,
        dataset="urfd",
        split="val",
        cache_root=cache_root,
        temp_root=temp_root,
        crop="auto",
        max_frames=30,
    )
    original_stat = source.stat()
    second = extract_manifest(
        manifest,
        extractor=extractor,
        dataset="urfd",
        split="val",
        cache_root=cache_root,
        temp_root=temp_root,
        crop="auto",
        max_frames=30,
    )
    source.write_bytes(b"video-v2")
    os.utime(
        source,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )
    third = extract_manifest(
        manifest,
        extractor=extractor,
        dataset="urfd",
        split="val",
        cache_root=cache_root,
        temp_root=temp_root,
        crop="auto",
        max_frames=30,
    )

    assert first == {"selected": 1, "processed": 1, "resumed": 0, "rebuilt": 0}
    assert second == {"selected": 1, "processed": 0, "resumed": 1, "rebuilt": 0}
    assert third == {"selected": 1, "processed": 1, "resumed": 0, "rebuilt": 1}
    assert len(extractor.calls) == 2
    assert temp_root.exists() and not list(temp_root.iterdir())


def test_extract_manifest_rejects_test_before_cache_or_inference(tmp_path: Path):
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"video")
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, source, split="test")
    extractor = _FakeExtractor()

    with pytest.raises(ValueError, match="test split"):
        extract_manifest(
            manifest,
            extractor=extractor,
            dataset="urfd",
            split="test",
            cache_root=tmp_path / "cache",
            temp_root=tmp_path / "temp",
        )

    assert extractor.calls == []


def test_extract_manifest_rebuilds_truncated_cache(tmp_path: Path):
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"video")
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, source)
    cache_root = tmp_path / "cache"
    extractor = _FakeExtractor()
    options = {
        "extractor": extractor,
        "dataset": "urfd",
        "split": "val",
        "cache_root": cache_root,
        "temp_root": tmp_path / "temp",
    }
    extract_manifest(manifest, **options)
    cache_path = next(cache_root.rglob("*.npz"))
    cache_path.write_bytes(b"truncated")

    summary = extract_manifest(manifest, **options)

    assert summary == {"selected": 1, "processed": 1, "resumed": 0, "rebuilt": 1}
    assert len(extractor.calls) == 2
