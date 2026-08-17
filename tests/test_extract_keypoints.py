"""manifest 驱动的 cache-first 姿态提取测试。"""
import csv
import os
from pathlib import Path

import numpy as np
import pytest

from pipeline.pose_extractor import PoseSequence
from tools.extract_keypoints import PoseExtractionBatchError, extract_manifest


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

    assert {key: first[key] for key in ("selected", "processed", "resumed", "rebuilt", "failed")} == {
        "selected": 1,
        "processed": 1,
        "resumed": 0,
        "rebuilt": 0,
        "failed": 0,
    }
    assert {key: second[key] for key in ("processed", "resumed", "rebuilt", "failed")} == {
        "processed": 0,
        "resumed": 1,
        "rebuilt": 0,
        "failed": 0,
    }
    assert {key: third[key] for key in ("processed", "resumed", "rebuilt", "failed")} == {
        "processed": 1,
        "resumed": 0,
        "rebuilt": 1,
        "failed": 0,
    }
    processed_clip = first["clips"][0]
    assert processed_clip["status"] == "processed"
    assert processed_clip["frames"] == 2
    assert processed_clip["max_people"] == 0
    assert processed_clip["observations"] == 0
    assert processed_clip["cache_bytes"] > 0
    assert processed_clip["extract_seconds"] >= 0.0
    assert first["elapsed_seconds"] >= 0.0
    assert first["processed_clips_per_second"] > 0.0
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

    assert {key: summary[key] for key in ("selected", "processed", "resumed", "rebuilt", "failed")} == {
        "selected": 1,
        "processed": 1,
        "resumed": 0,
        "rebuilt": 1,
        "failed": 0,
    }
    assert len(extractor.calls) == 2


def test_partial_failure_keeps_successful_cache_and_returns_nonzero_summary(
    tmp_path: Path,
):
    good = tmp_path / "good.mp4"
    bad = tmp_path / "bad.mp4"
    good.write_bytes(b"good")
    bad.write_bytes(b"bad")
    manifest = tmp_path / "manifest.csv"
    rows = [
        {
            "dataset": "urfd",
            "split": "val",
            "clip_id": source.stem,
            "video_path": str(source),
            "has_fall": "1",
        }
        for source in (good, bad)
    ]
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    class FailingExtractor(_FakeExtractor):
        def extract(self, source: Path, *, crop: str, max_frames: int | None):
            if Path(source).name == "bad.mp4":
                raise RuntimeError("synthetic failure")
            return super().extract(source, crop=crop, max_frames=max_frames)

    with pytest.raises(PoseExtractionBatchError) as raised:
        extract_manifest(
            manifest,
            extractor=FailingExtractor(),
            dataset="urfd",
            split="val",
            cache_root=tmp_path / "cache",
            temp_root=tmp_path / "temp",
        )

    summary = raised.value.summary
    assert summary["processed"] == 1
    assert summary["failed"] == 1
    assert [clip["status"] for clip in summary["clips"]] == ["processed", "error"]
    assert len(list((tmp_path / "cache").rglob("*.npz"))) == 1
    assert not list((tmp_path / "temp").iterdir())
