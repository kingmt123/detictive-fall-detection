"""从 manifest cache-first 提取逐帧多人姿态 NPZ。"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from pipeline.pose_cache import (
    PoseCacheRecord,
    load_pose_cache,
    pose_cache_path,
    write_pose_cache,
)
from pipeline.pose_extractor import PoseExtractor
from pipeline.video_source import VideoSourceResolver


class PoseExtractionBatchError(RuntimeError):
    """一个或多个 clip 失败；成功 cache 仍可用于下次 resume。"""

    def __init__(self, summary: dict[str, Any]) -> None:
        self.summary = summary
        super().__init__(f"{summary['failed']} 个 pose cache clip 提取失败")


def _resolve_source(video_path: str, manifest_path: Path) -> str:
    if video_path.startswith("tar://"):
        return video_path
    path = Path(video_path)
    if path.is_absolute():
        return str(path)
    project_root = (
        manifest_path.parent.parent
        if manifest_path.parent.name == "data"
        else manifest_path.parent
    )
    return str(project_root / path)


def _select_rows(
    manifest_path: Path,
    *,
    dataset: str,
    split: str,
    clip_ids: set[str] | None,
    limit: int | None,
) -> list[dict[str, str]]:
    if split == "test":
        raise ValueError("pose cache 阶段禁止读取 test split")
    if not dataset or not split:
        raise ValueError("dataset 和 split 必须显式指定")
    if limit is not None and limit <= 0:
        raise ValueError("limit 必须大于 0")

    with manifest_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"dataset", "split", "clip_id", "video_path"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"manifest 缺少字段: {sorted(missing)}")
        selected = [
            row
            for row in reader
            if row.get("dataset") == dataset and row.get("split") == split
        ]
    if not selected:
        raise ValueError(f"manifest 中没有 dataset={dataset!r}, split={split!r}")

    counts = Counter(row.get("clip_id", "") for row in selected)
    if "" in counts:
        raise ValueError("clip_id 必须是非空字符串")
    duplicates = sorted(clip_id for clip_id, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"重复 clip_id: {duplicates}")

    if clip_ids is not None:
        available = set(counts)
        missing_clip_ids = sorted(clip_ids - available)
        if missing_clip_ids:
            raise ValueError(f"指定 clip_id 不在所选 split: {missing_clip_ids}")
        selected = [row for row in selected if row["clip_id"] in clip_ids]
    if limit is not None:
        selected = selected[:limit]
    return selected


def _cache_matches_row(record: PoseCacheRecord, row: dict[str, str]) -> bool:
    return (
        record.clip_id == row["clip_id"]
        and record.dataset == row["dataset"]
        and record.split == row["split"]
    )


def _process_row(
    row: dict[str, str],
    *,
    manifest_path: Path,
    resolver: VideoSourceResolver,
    extractor: Any,
    dataset: str,
    split: str,
    cache_root: Path,
    signature: dict[str, Any],
    crop: str,
    max_frames: int | None,
) -> dict[str, Any]:
    clip_started = time.perf_counter()
    source = _resolve_source(row["video_path"], manifest_path)
    probe_started = time.perf_counter()
    source_identity = resolver.probe(source)
    probe_seconds = time.perf_counter() - probe_started
    cache_path = pose_cache_path(cache_root, dataset, split, row["clip_id"])
    cache_existed = cache_path.exists()
    cache_validation_seconds = 0.0
    if cache_existed:
        validation_started = time.perf_counter()
        try:
            cached = load_pose_cache(
                cache_path,
                expected_source_identity=source_identity,
                expected_extractor_signature=signature,
            )
            if not _cache_matches_row(cached, row):
                raise ValueError("pose cache manifest 身份不匹配")
            if cached.source_content_sha256 != resolver.content_sha256(source):
                raise ValueError("pose cache 源内容 SHA-256 不匹配")
        except ValueError:
            cache_validation_seconds = time.perf_counter() - validation_started
        else:
            cache_validation_seconds = time.perf_counter() - validation_started
            return {
                "dataset": dataset,
                "split": split,
                "clip_id": row["clip_id"],
                "source_kind": source_identity["kind"],
                "status": "resumed",
                "frames": int(cached.frame_indices.size),
                "max_people": int(cached.keypoints.shape[1]),
                "observations": int(cached.valid_mask.sum()),
                "cache_bytes": cache_path.stat().st_size,
                "probe_seconds": probe_seconds,
                "cache_validation_seconds": cache_validation_seconds,
                "materialize_seconds": 0.0,
                "extract_seconds": 0.0,
                "write_seconds": 0.0,
                "total_seconds": time.perf_counter() - clip_started,
            }

    materialize_started = time.perf_counter()
    with resolver.materialize(source) as materialized:
        materialize_seconds = time.perf_counter() - materialize_started
        extract_started = time.perf_counter()
        sequence = extractor.extract(
            materialized.local_path,
            crop=crop,
            max_frames=max_frames,
        )
        extract_seconds = time.perf_counter() - extract_started
        record = PoseCacheRecord(
            clip_id=row["clip_id"],
            dataset=dataset,
            split=split,
            source_identity=materialized.identity,
            source_content_sha256=materialized.content_sha256,
            extractor_signature=signature,
            fps=sequence.fps,
            frame_indices=sequence.frame_indices,
            timestamps=sequence.timestamps,
            keypoints=sequence.keypoints,
            bboxes=sequence.bboxes,
            track_ids=sequence.track_ids,
            valid_mask=sequence.valid_mask,
            frame_size=sequence.frame_size,
        )
        write_started = time.perf_counter()
        write_pose_cache(cache_path, record)
        write_seconds = time.perf_counter() - write_started
    return {
        "dataset": dataset,
        "split": split,
        "clip_id": row["clip_id"],
        "source_kind": source_identity["kind"],
        "status": "rebuilt" if cache_existed else "processed",
        "frames": int(sequence.frame_indices.size),
        "max_people": int(sequence.keypoints.shape[1]),
        "observations": int(sequence.valid_mask.sum()),
        "cache_bytes": cache_path.stat().st_size,
        "probe_seconds": probe_seconds,
        "cache_validation_seconds": cache_validation_seconds,
        "materialize_seconds": materialize_seconds,
        "extract_seconds": extract_seconds,
        "write_seconds": write_seconds,
        "total_seconds": time.perf_counter() - clip_started,
    }


def extract_manifest(
    manifest_path: Path,
    *,
    extractor: Any,
    dataset: str,
    split: str,
    cache_root: Path,
    temp_root: Path,
    crop: str = "auto",
    max_frames: int | None = None,
    clip_ids: set[str] | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """验证已有 cache 后 resume；陈旧/损坏 cache 仅在新写成功后原子替换。"""
    manifest_path = Path(manifest_path)
    rows = _select_rows(
        manifest_path,
        dataset=dataset,
        split=split,
        clip_ids=clip_ids,
        limit=limit,
    )
    signature = extractor.cache_signature(crop=crop, max_frames=max_frames)
    started = time.perf_counter()
    summary: dict[str, Any] = {
        "selected": len(rows),
        "processed": 0,
        "resumed": 0,
        "rebuilt": 0,
        "failed": 0,
        "clips": [],
    }

    with VideoSourceResolver(Path(temp_root)) as resolver:
        for row in rows:
            clip_started = time.perf_counter()
            try:
                clip_result = _process_row(
                    row,
                    manifest_path=manifest_path,
                    resolver=resolver,
                    extractor=extractor,
                    dataset=dataset,
                    split=split,
                    cache_root=Path(cache_root),
                    signature=signature,
                    crop=crop,
                    max_frames=max_frames,
                )
            except (OSError, RuntimeError, ValueError) as exc:
                summary["failed"] += 1
                summary["clips"].append(
                    {
                        "dataset": dataset,
                        "split": split,
                        "clip_id": row["clip_id"],
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "total_seconds": time.perf_counter() - clip_started,
                    }
                )
                continue
            summary["clips"].append(clip_result)
            if clip_result["status"] == "resumed":
                summary["resumed"] += 1
            else:
                summary["processed"] += 1
                if clip_result["status"] == "rebuilt":
                    summary["rebuilt"] += 1
    elapsed_seconds = time.perf_counter() - started
    summary["elapsed_seconds"] = elapsed_seconds
    summary["processed_clips_per_second"] = (
        summary["processed"] / elapsed_seconds if summary["processed"] else 0.0
    )
    summary["cache_bytes"] = sum(
        clip.get("cache_bytes", 0) for clip in summary["clips"]
    )
    if summary["failed"]:
        raise PoseExtractionBatchError(summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--temp-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("yolo11n-pose.pt"))
    parser.add_argument("--device", default="0")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.10)
    parser.add_argument("--crop", choices=("none", "auto", "left", "right"), default="auto")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--clip-id", action="append", dest="clip_ids")
    return parser


def main() -> None:
    args = _parser().parse_args()
    extractor = PoseExtractor(
        model_path=args.model,
        device=args.device,
        image_size=args.image_size,
        confidence=args.confidence,
        model_snapshot_root=args.temp_root,
    )
    try:
        try:
            summary = extract_manifest(
                args.manifest,
                extractor=extractor,
                dataset=args.dataset,
                split=args.split,
                cache_root=args.cache_root,
                temp_root=args.temp_root,
                crop=args.crop,
                max_frames=args.max_frames,
                clip_ids=set(args.clip_ids) if args.clip_ids else None,
                limit=args.limit,
            )
        except PoseExtractionBatchError as exc:
            print(json.dumps(exc.summary, ensure_ascii=False))
            raise SystemExit(1) from exc
    finally:
        extractor.close()
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
