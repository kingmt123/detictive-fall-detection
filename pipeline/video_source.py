"""把本地视频路径和 tar 成员 URI 解析为统一源描述。"""
from __future__ import annotations

import hashlib
import tarfile
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self


@dataclass(frozen=True)
class LocalSource:
    path: Path


@dataclass(frozen=True)
class TarSource:
    archive_path: Path
    member: str


VideoSource = LocalSource | TarSource


@dataclass(frozen=True)
class MaterializedVideo:
    local_path: Path
    identity: dict[str, Any]
    content_sha256: str


def parse_video_source(source: str | Path) -> VideoSource:
    """解析本地路径或严格的 ``tar://archive!/member`` URI。"""
    text = str(source)
    if not text.startswith("tar://"):
        return LocalSource(Path(source))
    payload = text.removeprefix("tar://")
    if payload.count("!/") != 1:
        raise ValueError("tar URI 必须使用 tar://archive!/member 格式")
    archive, member = payload.split("!/", 1)
    if not archive or not member:
        raise ValueError("tar URI 的 archive 和 member 均不能为空")
    return TarSource(Path(archive), member)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class VideoSourceResolver:
    """批次内复用 tar handle，并管理成员临时文件的生命周期。"""

    def __init__(self, temp_root: Path) -> None:
        self.temp_root = Path(temp_root)
        self._archives: dict[Path, tarfile.TarFile] = {}

    def __enter__(self) -> Self:
        self.temp_root.mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(self, *_exc: object) -> None:
        for archive in self._archives.values():
            archive.close()
        self._archives.clear()

    def _open_archive(self, path: Path) -> tarfile.TarFile:
        resolved = path.resolve()
        if resolved not in self._archives:
            # 由 resolver.__exit__ 统一关闭，以便批次内复用 9.7GB tar 索引。
            self._archives[resolved] = tarfile.open(resolved)  # noqa: SIM115
        return self._archives[resolved]

    def probe(self, source: str | Path) -> dict[str, Any]:
        """读取足以判定缓存是否陈旧的廉价源身份，不解包视频字节。"""
        parsed = parse_video_source(source)
        if isinstance(parsed, LocalSource):
            path = parsed.path.resolve()
            if not path.is_file():
                raise ValueError(f"本地视频不存在: {path}")
            stat = path.stat()
            return {
                "kind": "local",
                "path": str(path),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }

        archive_path = parsed.archive_path.resolve()
        archive = self._open_archive(archive_path)
        try:
            info = archive.getmember(parsed.member)
        except KeyError as exc:
            raise ValueError(f"tar 成员不存在: {parsed.member}") from exc
        if not info.isfile():
            raise ValueError(f"tar 成员不是普通文件: {parsed.member}")
        archive_stat = archive_path.stat()
        return {
            "kind": "tar",
            "archive_path": str(archive_path),
            "archive_size": archive_stat.st_size,
            "archive_mtime_ns": archive_stat.st_mtime_ns,
            "member": parsed.member,
            "member_size": info.size,
            "member_mtime": info.mtime,
            "member_offset": info.offset_data,
        }

    @contextmanager
    def materialize(self, source: str | Path) -> Iterator[MaterializedVideo]:
        parsed = parse_video_source(source)
        if isinstance(parsed, LocalSource):
            path = parsed.path.resolve()
            if not path.is_file():
                raise ValueError(f"本地视频不存在: {path}")
            stat = path.stat()
            yield MaterializedVideo(
                path,
                {
                    "kind": "local",
                    "path": str(path),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                },
                _sha256(path),
            )
            return

        archive_path = parsed.archive_path.resolve()
        archive = self._open_archive(archive_path)
        try:
            info = archive.getmember(parsed.member)
        except KeyError as exc:
            raise ValueError(f"tar 成员不存在: {parsed.member}") from exc
        if not info.isfile():
            raise ValueError(f"tar 成员不是普通文件: {parsed.member}")
        source_handle = archive.extractfile(info)
        if source_handle is None:
            raise ValueError(f"无法读取 tar 成员: {parsed.member}")

        self.temp_root.mkdir(parents=True, exist_ok=True)
        suffix = Path(parsed.member).suffix or ".video"
        temporary_path: Path | None = None
        digest = hashlib.sha256()
        try:
            with tempfile.NamedTemporaryFile(
                dir=self.temp_root, suffix=suffix, delete=False
            ) as target:
                temporary_path = Path(target.name)
                for chunk in iter(lambda: source_handle.read(1024 * 1024), b""):
                    target.write(chunk)
                    digest.update(chunk)
            archive_stat = archive_path.stat()
            identity = {
                "kind": "tar",
                "archive_path": str(archive_path),
                "archive_size": archive_stat.st_size,
                "archive_mtime_ns": archive_stat.st_mtime_ns,
                "member": parsed.member,
                "member_size": info.size,
                "member_mtime": info.mtime,
                "member_offset": info.offset_data,
            }
            yield MaterializedVideo(temporary_path, identity, digest.hexdigest())
        finally:
            source_handle.close()
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
