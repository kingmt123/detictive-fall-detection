"""姿态 NPZ cache schema、原子写入和恢复测试。"""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from pipeline.pose_cache import (
    PoseCacheRecord,
    load_pose_cache,
    pose_cache_path,
    write_pose_cache,
)


def _zero_person_record() -> PoseCacheRecord:
    return PoseCacheRecord(
        clip_id="fall/example",
        dataset="of-syn",
        split="val",
        source_identity={"kind": "tar", "member": "./fall/example.mp4"},
        source_content_sha256="a" * 64,
        extractor_signature={"protocol": "pose-cache-v1", "model_sha256": "b" * 64},
        fps=16.0,
        frame_indices=np.array([0, 1], dtype=np.int64),
        timestamps=np.array([0.0, 1 / 16], dtype=np.float64),
        keypoints=np.empty((2, 0, 17, 3), dtype=np.float32),
        bboxes=np.empty((2, 0, 4), dtype=np.float32),
        track_ids=np.empty((2, 0), dtype=np.int64),
        valid_mask=np.empty((2, 0), dtype=np.bool_),
        frame_size=np.array([720, 1280], dtype=np.int64),
    )


def _one_person_record() -> PoseCacheRecord:
    record = _zero_person_record()
    return replace(
        record,
        keypoints=np.zeros((2, 1, 17, 3), dtype=np.float32),
        bboxes=np.array(
            [[[1, 2, 10, 20]], [[2, 3, 11, 21]]], dtype=np.float32
        ),
        track_ids=np.array([[3], [3]], dtype=np.int64),
        valid_mask=np.ones((2, 1), dtype=np.bool_),
    )


def test_valid_zero_person_cache_round_trips_without_pickle(tmp_path: Path):
    record = _zero_person_record()
    path = pose_cache_path(tmp_path, record.dataset, record.split, record.clip_id)

    write_pose_cache(path, record)
    loaded = load_pose_cache(
        path,
        expected_source_identity=record.source_identity,
        expected_extractor_signature=record.extractor_signature,
    )

    assert path.parent == tmp_path / "of-syn" / "val"
    assert path.name.endswith(".npz") and "/" not in path.name
    assert loaded.clip_id == record.clip_id
    np.testing.assert_array_equal(loaded.frame_indices, record.frame_indices)
    assert loaded.keypoints.shape == (2, 0, 17, 3)
    with np.load(path, allow_pickle=False) as payload:
        assert int(payload["cache_schema"]) == 1
        assert payload["valid_mask"].dtype == np.bool_


def test_writer_rejects_invalid_bbox_shape_before_creating_cache(tmp_path: Path):
    record = replace(
        _zero_person_record(),
        bboxes=np.empty((2, 0, 5), dtype=np.float32),
    )
    path = tmp_path / "bad.npz"

    with pytest.raises(ValueError, match="bboxes"):
        write_pose_cache(path, record)

    assert not path.exists()


def test_writer_rejects_noncontiguous_frame_indices(tmp_path: Path):
    record = replace(
        _zero_person_record(),
        frame_indices=np.array([0, 2], dtype=np.int64),
    )

    with pytest.raises(ValueError, match="frame_indices"):
        write_pose_cache(tmp_path / "bad.npz", record)


def test_writer_rejects_nonfinite_timestamps(tmp_path: Path):
    record = replace(
        _zero_person_record(),
        timestamps=np.array([0.0, np.nan], dtype=np.float64),
    )

    with pytest.raises(ValueError, match="timestamps"):
        write_pose_cache(tmp_path / "bad.npz", record)


def test_writer_rejects_invalid_keypoint_shape(tmp_path: Path):
    record = replace(
        _one_person_record(),
        keypoints=np.zeros((2, 1, 16, 3), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="keypoints"):
        write_pose_cache(tmp_path / "bad.npz", record)


def test_cache_path_rejects_dataset_or_split_path_traversal(tmp_path: Path):
    with pytest.raises(ValueError, match="dataset"):
        pose_cache_path(tmp_path, "../escape", "val", "clip")
    with pytest.raises(ValueError, match="split"):
        pose_cache_path(tmp_path, "of-syn", "../escape", "clip")


def test_writer_rejects_nonnegative_track_id_in_padding(tmp_path: Path):
    record = replace(
        _one_person_record(),
        bboxes=np.array(
            [[[0, 0, 0, 0]], [[2, 3, 11, 21]]], dtype=np.float32
        ),
        valid_mask=np.array([[False], [True]], dtype=np.bool_),
        track_ids=np.array([[3], [3]], dtype=np.int64),
    )

    with pytest.raises(ValueError, match="padding"):
        write_pose_cache(tmp_path / "bad.npz", record)


def test_valid_one_person_cache_round_trips(tmp_path: Path):
    record = _one_person_record()
    path = tmp_path / "valid.npz"

    write_pose_cache(path, record)
    loaded = load_pose_cache(
        path,
        expected_source_identity=record.source_identity,
        expected_extractor_signature=record.extractor_signature,
    )

    np.testing.assert_array_equal(loaded.track_ids, np.array([[3], [3]]))
    assert loaded.valid_mask.all()


def test_loader_rejects_stale_source_or_extractor_signature(tmp_path: Path):
    record = _zero_person_record()
    path = tmp_path / "valid.npz"
    write_pose_cache(path, record)

    with pytest.raises(ValueError, match="源身份"):
        load_pose_cache(
            path,
            expected_source_identity={"kind": "tar", "member": "changed.mp4"},
            expected_extractor_signature=record.extractor_signature,
        )
    with pytest.raises(ValueError, match="提取签名"):
        load_pose_cache(
            path,
            expected_source_identity=record.source_identity,
            expected_extractor_signature={"protocol": "changed"},
        )


def test_loader_rejects_truncated_npz(tmp_path: Path):
    path = tmp_path / "truncated.npz"
    path.write_bytes(b"not-an-npz")

    with pytest.raises(ValueError):
        load_pose_cache(
            path,
            expected_source_identity={},
            expected_extractor_signature={},
        )


def test_failed_temp_write_preserves_existing_cache(monkeypatch, tmp_path: Path):
    import pipeline.pose_cache as module

    record = _zero_person_record()
    path = tmp_path / "valid.npz"
    write_pose_cache(path, record)
    original = path.read_bytes()

    def fail_write(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(module.np, "savez_compressed", fail_write)
    with pytest.raises(OSError, match="disk full"):
        write_pose_cache(path, record)

    assert path.read_bytes() == original
    assert not list(tmp_path.glob(".*.tmp"))


def test_writer_rejects_invalid_content_sha256(tmp_path: Path):
    record = replace(_zero_person_record(), source_content_sha256="not-a-sha")

    with pytest.raises(ValueError, match="SHA-256"):
        write_pose_cache(tmp_path / "bad.npz", record)
