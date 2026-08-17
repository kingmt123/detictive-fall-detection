"""本地 MP4 与 tar 成员统一视频源测试。"""
import hashlib
import io
import tarfile
from pathlib import Path

import pytest

from pipeline.video_source import (
    LocalSource,
    TarSource,
    VideoSourceResolver,
    parse_video_source,
)


def test_parse_local_and_windows_tar_sources_preserves_member_prefix(tmp_path: Path):
    local = parse_video_source(tmp_path / "clip.mp4")
    tar_source = parse_video_source(
        r"tar://D:\datasets\videos.tar!/./fall/example.mp4"
    )

    assert local == LocalSource(tmp_path / "clip.mp4")
    assert tar_source == TarSource(
        Path(r"D:\datasets\videos.tar"), "./fall/example.mp4"
    )


def test_parse_tar_source_rejects_ambiguous_delimiters():
    with pytest.raises(ValueError, match="格式"):
        parse_video_source("tar://videos.tar!/first.mp4!/second.mp4")


def test_materialize_tar_member_uses_explicit_temp_root_and_cleans_up(tmp_path: Path):
    archive = tmp_path / "videos.tar"
    payload = b"fake-video-bytes"
    with tarfile.open(archive, "w") as handle:
        info = tarfile.TarInfo("./fall/example.mp4")
        info.size = len(payload)
        handle.addfile(info, io.BytesIO(payload))
    temp_root = tmp_path / "materialized"
    uri = f"tar://{archive}!/./fall/example.mp4"

    with VideoSourceResolver(temp_root) as resolver:
        with resolver.materialize(uri) as materialized:
            materialized_path = materialized.local_path
            assert materialized_path.parent == temp_root
            assert materialized_path.read_bytes() == payload
            assert materialized.content_sha256 == hashlib.sha256(payload).hexdigest()
        assert not materialized_path.exists()


def test_probe_reuses_one_tar_handle_for_multiple_members(monkeypatch, tmp_path: Path):
    import pipeline.video_source as module

    archive = tmp_path / "videos.tar"
    with tarfile.open(archive, "w") as handle:
        for name in ("./a.mp4", "./b.mp4"):
            info = tarfile.TarInfo(name)
            info.size = 1
            handle.addfile(info, io.BytesIO(b"x"))
    real_open = module.tarfile.open
    open_calls = 0

    def counting_open(*args, **kwargs):
        nonlocal open_calls
        open_calls += 1
        return real_open(*args, **kwargs)

    monkeypatch.setattr(module.tarfile, "open", counting_open)
    with VideoSourceResolver(tmp_path / "temp") as resolver:
        first = resolver.probe(f"tar://{archive}!/./a.mp4")
        second = resolver.probe(f"tar://{archive}!/./b.mp4")

    assert open_calls == 1
    assert first["member"] == "./a.mp4"
    assert second["member"] == "./b.mp4"


def test_materialized_tar_member_is_cleaned_when_consumer_raises(tmp_path: Path):
    archive = tmp_path / "videos.tar"
    with tarfile.open(archive, "w") as handle:
        info = tarfile.TarInfo("./clip.mp4")
        info.size = 1
        handle.addfile(info, io.BytesIO(b"x"))
    temp_root = tmp_path / "materialized"

    with (
        VideoSourceResolver(temp_root) as resolver,
        pytest.raises(RuntimeError, match="model failed"),
        resolver.materialize(f"tar://{archive}!/./clip.mp4") as video,
    ):
        materialized_path = video.local_path
        raise RuntimeError("model failed")

    assert not materialized_path.exists()


def test_materialized_local_source_is_an_immutable_snapshot(tmp_path: Path):
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"video-A")
    temp_root = tmp_path / "materialized"

    with VideoSourceResolver(temp_root) as resolver:
        with resolver.materialize(source) as video:
            materialized_path = video.local_path
            source.write_bytes(b"video-B")
            assert materialized_path != source
            assert materialized_path.read_bytes() == b"video-A"
            assert video.content_sha256 == hashlib.sha256(b"video-A").hexdigest()
        assert not materialized_path.exists()


def test_probe_rejects_missing_tar_member(tmp_path: Path):
    archive = tmp_path / "videos.tar"
    with tarfile.open(archive, "w"):
        pass

    with (
        VideoSourceResolver(tmp_path / "temp") as resolver,
        pytest.raises(ValueError, match="成员不存在"),
    ):
        resolver.probe(f"tar://{archive}!/./missing.mp4")
