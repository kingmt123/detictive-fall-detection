"""数据标签语义与无泄漏划分测试。"""
import csv
import io
import tarfile
from pathlib import Path

import pytest

from tools.build_manifest import (
    assign_group_splits,
    build_ofsyn_manifest,
    classify_activity,
    urfd_group_id,
)
from tools.prepare_omnifall_events import convert


def test_fall_and_fallen_are_distinct_positive_states():
    assert classify_activity(1) == "fall_process"
    assert classify_activity(2) == "post_fall_state"
    assert classify_activity(5) == "hard_negative"
    assert classify_activity(6) == "hard_negative"
    assert classify_activity(8) == "background"


def test_event_conversion_preserves_post_fall_as_incident_not_negative(tmp_path):
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    source = labels_dir / "toy.csv"
    source.write_text(
        "path,label,start,end,subject,cam,dataset\n"
        "clip-a,1,0.0,1.0,1,0,toy\n"
        "clip-a,2,1.0,3.0,1,0,toy\n"
        "clip-b,5,0.0,2.0,2,0,toy\n",
        encoding="utf-8",
    )
    output = tmp_path / "events.csv"
    convert(labels_dir, output)
    rows = list(csv.DictReader(output.open(encoding="utf-8")))
    by_label = {int(row["label_id"]): row for row in rows}
    assert by_label[1]["event_semantics"] == "fall_process"
    assert by_label[1]["is_fall_incident"] == "1"
    assert by_label[2]["event_semantics"] == "post_fall_state"
    assert by_label[2]["is_fall_incident"] == "1"
    assert by_label[2]["is_hard_negative"] == "0"
    assert by_label[5]["event_semantics"] == "hard_negative"
    assert by_label[5]["is_fall_incident"] == "0"


def test_event_conversion_drops_zero_duration_segments(tmp_path):
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    (labels_dir / "toy.csv").write_text(
        "path,label,start,end,subject,cam,dataset\n"
        "valid,1,0.0,1.0,1,0,toy\n"
        "zero,1,5.063,5.063,1,0,toy\n",
        encoding="utf-8",
    )
    output = tmp_path / "events.csv"
    convert(labels_dir, output)
    rows = list(csv.DictReader(output.open(encoding="utf-8")))
    assert [row["clip_id"] for row in rows] == ["valid"]


def test_ofsyn_manifest_uses_existing_dot_prefixed_tar_member(tmp_path):
    labels = tmp_path / "labels.csv"
    labels.write_text(
        "path,label,start,end,subject,cam,dataset\n"
        "fall/example,1,0.0,1.0,-1,-1,of-syn\n",
        encoding="utf-8",
    )
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    for split in ("train", "val", "test"):
        content = "path\nfall/example\n" if split == "train" else "path\n"
        (split_dir / f"{split}.csv").write_text(content, encoding="utf-8")
    archive = tmp_path / "videos.tar"
    with tarfile.open(archive, "w") as handle:
        info = tarfile.TarInfo("./fall/example.mp4")
        payload = b"video"
        info.size = len(payload)
        handle.addfile(info, io.BytesIO(payload))

    rows = build_ofsyn_manifest(labels, archive, split_dir)

    assert len(rows) == 1
    assert rows[0]["video_path"].endswith("!/./fall/example.mp4")


def test_ofsyn_manifest_rejects_clip_with_only_invalid_segments(tmp_path: Path):
    labels_csv = tmp_path / "labels.csv"
    labels_csv.write_text(
        "path,label,start,end,subject,cam,dataset\n"
        "fall/bad,1,1.0,1.0,1,1,of-syn\n",
        encoding="utf-8",
    )
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    for split in ("train", "val", "test"):
        content = "path\nfall/bad\n" if split == "train" else "path\n"
        (split_dir / f"{split}.csv").write_text(content, encoding="utf-8")
    tar_path = tmp_path / "videos.tar"
    with tarfile.open(tar_path, "w") as archive:
        payload = b"video"
        info = tarfile.TarInfo("./fall/bad.mp4")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))

    with pytest.raises(ValueError, match="没有合法事件片段"):
        build_ofsyn_manifest(labels_csv, tar_path, split_dir)


def test_ofsyn_manifest_rejects_split_clip_missing_from_labels(tmp_path: Path):
    labels_csv = tmp_path / "labels.csv"
    labels_csv.write_text(
        "path,label,start,end,subject,cam,dataset\n"
        "walk/known,0,0.0,1.0,1,1,of-syn\n",
        encoding="utf-8",
    )
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    (split_dir / "train.csv").write_text(
        "path\nwalk/known\nwalk/missing\n", encoding="utf-8"
    )
    (split_dir / "val.csv").write_text("path\n", encoding="utf-8")
    (split_dir / "test.csv").write_text("path\n", encoding="utf-8")
    tar_path = tmp_path / "videos.tar"
    with tarfile.open(tar_path, "w") as archive:
        payload = b"video"
        info = tarfile.TarInfo("./walk/known.mp4")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))

    with pytest.raises(ValueError, match="split 与标签路径集合不一致"):
        build_ofsyn_manifest(labels_csv, tar_path, split_dir)


def test_event_conversion_rejects_empty_input_without_truncating_output(tmp_path: Path):
    labels = tmp_path / "labels"
    labels.mkdir()
    output = tmp_path / "events.csv"
    output.write_text("keep-me", encoding="utf-8")

    with pytest.raises(ValueError, match="没有合法事件片段"):
        convert(labels, output)
    assert output.read_text(encoding="utf-8") == "keep-me"


def test_urfd_two_cameras_share_one_group():
    assert urfd_group_id(Path("fall-07-cam0.mp4")) == "fall-07"
    assert urfd_group_id(Path("fall-07-cam1.mp4")) == "fall-07"
    assert urfd_group_id(Path("adl-14-cam0.mp4")) == "adl-14"


def test_group_split_never_leaks_group_across_partitions():
    groups = [f"fall-{i:02d}" for i in range(1, 31)] + [
        f"adl-{i:02d}" for i in range(1, 41)
    ]
    labels = {g: g.startswith("fall") for g in groups}
    split_by_group = assign_group_splits(groups, labels, seed=42)
    assert set(split_by_group) == set(groups)
    assert set(split_by_group.values()) == {"train", "val", "test"}

    # 同一个 group 无论出现几个 camera，都只能映射到同一 split。
    rows = [(group, split_by_group[group]) for group in groups for _ in range(2)]
    for group in groups:
        assert len({split for g, split in rows if g == group}) == 1


def test_group_split_is_deterministic_and_stratified():
    groups = [f"fall-{i:02d}" for i in range(1, 31)] + [
        f"adl-{i:02d}" for i in range(1, 41)
    ]
    labels = {g: g.startswith("fall") for g in groups}
    a = assign_group_splits(groups, labels, seed=2026)
    b = assign_group_splits(list(reversed(groups)), labels, seed=2026)
    assert a == b
    for split in ("train", "val", "test"):
        split_labels = [labels[g] for g, s in a.items() if s == split]
        assert any(split_labels)
        assert not all(split_labels)
