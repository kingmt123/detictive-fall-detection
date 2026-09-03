"""审计 manifest 与 pose cache 的覆盖、身份、签名和 TCN 窗口可用性。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from models.tcn_window import build_tcn_windows, parse_activity_intervals
from pipeline.pose_cache import pose_cache_path, read_pose_cache
from tools.clip_ids import load_clip_ids


class PoseCacheAuditError(RuntimeError):
    """Cache 集合未通过训练前硬性一致性检查。"""

    def __init__(self, summary: dict[str, Any]) -> None:
        self.summary = summary
        super().__init__("pose cache 审计失败: " + "; ".join(summary["errors"]))


def _canonical_json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _select_manifest_rows(
    manifest_path: Path,
    *,
    datasets: set[str],
    splits: set[str],
    clip_ids: set[str] | None,
) -> list[dict[str, str]]:
    if not datasets or not splits:
        raise ValueError("datasets 和 splits 必须显式指定且不能为空")
    if "test" in splits:
        raise ValueError("pose cache 审计阶段禁止读取 test split")

    with Path(manifest_path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"dataset", "split", "clip_id", "has_fall", "events_json"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"manifest 缺少字段: {sorted(missing)}")
        rows = [
            row
            for row in reader
            if row["dataset"] in datasets and row["split"] in splits
        ]

    identities = Counter(
        (row["dataset"], row["split"], row["clip_id"]) for row in rows
    )
    empty = [identity for identity in identities if not all(identity)]
    duplicates = [identity for identity, count in identities.items() if count > 1]
    if empty:
        raise ValueError("manifest 的 dataset/split/clip_id 不能为空")
    if duplicates:
        raise ValueError(f"manifest 存在重复 cache 身份: {sorted(duplicates)}")
    invalid_fall_flags = sorted(
        {
            row["has_fall"]
            for row in rows
            if row["has_fall"] not in {"0", "1"}
        }
    )
    if invalid_fall_flags:
        raise ValueError(f"manifest has_fall 必须为 0 或 1: {invalid_fall_flags}")
    if clip_ids is not None:
        available = {row["clip_id"] for row in rows}
        missing_ids = sorted(clip_ids - available)
        if missing_ids:
            raise ValueError(f"指定 clip_id 不在所选 manifest 范围: {missing_ids}")
        rows = [row for row in rows if row["clip_id"] in clip_ids]
    if not rows:
        raise ValueError("manifest 选择结果为空")
    return rows


def _selected_cache_files(
    cache_root: Path, datasets: Iterable[str], splits: Iterable[str]
) -> set[Path]:
    files: set[Path] = set()
    for dataset in datasets:
        for split in splits:
            directory = Path(cache_root) / dataset / split
            if directory.exists():
                files.update(path.resolve() for path in directory.glob("*.npz"))
    return files


def audit_pose_cache(
    manifest_path: Path,
    cache_root: Path,
    *,
    datasets: set[str],
    splits: set[str],
    clip_ids: set[str] | None = None,
    window_size: int = 16,
    stride: int = 1,
    min_observed_frames: int | None = None,
) -> dict[str, Any]:
    """返回完整统计；缺失、损坏、多余或签名混用时抛出带 summary 的异常。"""
    if min_observed_frames is None:
        min_observed_frames = (window_size + 1) // 2
    rows = _select_manifest_rows(
        Path(manifest_path),
        datasets=datasets,
        splits=splits,
        clip_ids=clip_ids,
    )
    cache_root = Path(cache_root)
    expected = {
        pose_cache_path(cache_root, row["dataset"], row["split"], row["clip_id"])
        .resolve(): row
        for row in rows
    }
    actual = _selected_cache_files(cache_root, datasets, splits)
    missing_paths = sorted(expected.keys() - actual)
    unexpected_paths = sorted(actual - expected.keys())

    summary: dict[str, Any] = {
        "manifest_sha256": _sha256_file(Path(manifest_path)),
        "selection_sha256": hashlib.sha256(
            _canonical_json(
                {
                    "rows": sorted(
                        (
                            row["dataset"],
                            row["split"],
                            row["clip_id"],
                            row["events_json"],
                        )
                        for row in rows
                    )
                }
            ).encode("utf-8")
        ).hexdigest(),
        "window_config": {
            "window_size": window_size,
            "stride": stride,
            "min_observed_frames": min_observed_frames,
        },
        "selected": len(rows),
        "cache_files": len(actual),
        "valid": 0,
        "missing": [str(path) for path in missing_paths],
        "unexpected": [str(path) for path in unexpected_paths],
        "corrupt_or_mismatched": [],
        "extractor_signatures": [],
        "total_cache_bytes": 0,
        "total_frames": 0,
        "frames_with_pose": 0,
        "total_observations": 0,
        "unique_tracks_per_clip": {},
        "clips_without_pose": [],
        "fall_manifest_clips": sum(row["has_fall"] == "1" for row in rows),
        "fall_manifest_clips_without_temporal_annotations": [],
        "windows": 0,
        "positive_windows": 0,
        "window_semantics": {},
        "positive_manifest_clips_without_positive_window": [],
        "errors": [],
    }
    signatures: dict[str, dict[str, Any]] = {}
    semantic_counts: Counter[str] = Counter()

    for path, row in expected.items():
        if path in missing_paths:
            continue
        try:
            record = read_pose_cache(path)
            identity = (record.dataset, record.split, record.clip_id)
            expected_identity = (row["dataset"], row["split"], row["clip_id"])
            if identity != expected_identity:
                raise ValueError(
                    f"manifest/cache 身份不匹配: expected={expected_identity}, actual={identity}"
                )
            intervals = parse_activity_intervals(row["events_json"])
            windows = build_tcn_windows(
                record,
                intervals,
                window_size=window_size,
                stride=stride,
                min_observed_frames=min_observed_frames,
            )
        except ValueError as exc:
            summary["corrupt_or_mismatched"].append(
                {"path": str(path), "error": str(exc)}
            )
            continue

        signature_key = _canonical_json(record.extractor_signature)
        signatures.setdefault(signature_key, record.extractor_signature)
        valid_observations = int(record.valid_mask.sum())
        frames_with_pose = int(record.valid_mask.any(axis=1).sum())
        track_count = len({int(value) for value in record.track_ids[record.valid_mask]})
        positive_windows = sum(float(window.label) == 1.0 for window in windows)
        semantic_counts.update(window.event_semantics for window in windows)

        summary["valid"] += 1
        summary["total_cache_bytes"] += path.stat().st_size
        summary["total_frames"] += int(record.frame_indices.size)
        summary["frames_with_pose"] += frames_with_pose
        summary["total_observations"] += valid_observations
        summary["unique_tracks_per_clip"][record.clip_id] = track_count
        summary["windows"] += len(windows)
        summary["positive_windows"] += positive_windows
        if valid_observations == 0:
            summary["clips_without_pose"].append(record.clip_id)
        manifest_has_positive = any(
            interval.semantics in {"fall_process", "post_fall_state"}
            for interval in intervals
        )
        if manifest_has_positive != (row["has_fall"] == "1"):
            if row["has_fall"] == "1" and not intervals:
                summary["fall_manifest_clips_without_temporal_annotations"].append(
                    record.clip_id
                )
            else:
                summary["corrupt_or_mismatched"].append(
                    {
                        "path": str(path),
                        "error": "has_fall 与 events_json 跌倒语义不一致",
                    }
                )
        if manifest_has_positive and positive_windows == 0:
            summary["positive_manifest_clips_without_positive_window"].append(
                record.clip_id
            )

    summary["extractor_signatures"] = list(signatures.values())
    summary["window_semantics"] = dict(sorted(semantic_counts.items()))
    summary["frame_pose_coverage"] = (
        summary["frames_with_pose"] / summary["total_frames"]
        if summary["total_frames"]
        else 0.0
    )
    if missing_paths:
        summary["errors"].append(f"缺失 {len(missing_paths)} 个 cache")
    if unexpected_paths:
        summary["errors"].append(f"存在 {len(unexpected_paths)} 个反向无法映射的 cache")
    if summary["corrupt_or_mismatched"]:
        summary["errors"].append(
            f"损坏或身份不匹配 {len(summary['corrupt_or_mismatched'])} 个 cache"
        )
    if len(signatures) != 1:
        summary["errors"].append(f"提取签名数量必须为 1，实际为 {len(signatures)}")
    if summary["errors"]:
        raise PoseCacheAuditError(summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--dataset", action="append", dest="datasets", required=True)
    parser.add_argument(
        "--split", action="append", dest="splits", choices=("train", "val"), required=True
    )
    parser.add_argument("--clip-id", action="append", dest="clip_ids")
    parser.add_argument("--clip-id-file", action="append", type=Path, dest="clip_id_files")
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--min-observed-frames", type=int)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        summary = audit_pose_cache(
            args.manifest,
            args.cache_root,
            datasets=set(args.datasets),
            splits=set(args.splits),
            clip_ids=load_clip_ids(args.clip_ids, args.clip_id_files),
            window_size=args.window_size,
            stride=args.stride,
            min_observed_frames=args.min_observed_frames,
        )
    except PoseCacheAuditError as exc:
        summary = exc.summary
        exit_code = 1
    else:
        exit_code = 0
    payload = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
