"""Build a train/validation-only diagnostic sample of high-confidence errors."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from pipeline.pose_cache import pose_cache_path, read_pose_cache

PROTOCOL = "tcn_validation_error_spotcheck_v1"


def _activity(clip_id: str) -> str:
    activity, separator, remainder = clip_id.partition("/")
    if not separator or not activity or not remainder:
        raise ValueError(f"clip_id 不符合 activity/name 格式: {clip_id!r}")
    return activity


def select_spotcheck_predictions(
    predictions: list[dict[str, Any]], *, threshold: float, per_group: int = 20
) -> list[dict[str, Any]]:
    """Pick top-score FP and all available ``fallen`` FN before transparent backfill."""
    if per_group < 1:
        raise ValueError("per_group 必须为正数")
    if len({str(item.get("clip_id")) for item in predictions}) != len(predictions):
        raise ValueError("predictions 的 clip_id 必须唯一")
    required = {"clip_id", "has_fall", "tcn_score"}
    if any(not required.issubset(item) for item in predictions):
        raise ValueError("predictions 缺少必要字段")

    false_positives = sorted(
        (
            item
            for item in predictions
            if not bool(item["has_fall"]) and float(item["tcn_score"]) >= threshold - 1e-12
        ),
        key=lambda item: (-float(item["tcn_score"]), str(item["clip_id"])),
    )[:per_group]
    if len(false_positives) < per_group:
        raise ValueError("高分误报数量不足")

    all_false_negatives = sorted(
        (
            item
            for item in predictions
            if bool(item["has_fall"]) and float(item["tcn_score"]) < threshold - 1e-12
        ),
        key=lambda item: (float(item["tcn_score"]), str(item["clip_id"])),
    )
    fallen = [item for item in all_false_negatives if _activity(str(item["clip_id"])) == "fallen"]
    selected_fn = fallen[:per_group]
    backfill = [item for item in all_false_negatives if item not in selected_fn]
    selected_fn.extend(backfill[: per_group - len(selected_fn)])
    if len(selected_fn) < per_group:
        raise ValueError("漏报数量不足")

    rows: list[dict[str, Any]] = []
    for item in false_positives:
        rows.append({"sample_group": "high_score_false_positive", **item})
    for item in selected_fn:
        sample_group = (
            "fallen_false_negative"
            if _activity(str(item["clip_id"])) == "fallen"
            else "false_negative_backfill"
        )
        rows.append({"sample_group": sample_group, **item})
    return rows


def _metadata_by_clip(path: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            clip_id = row.get("path")
            if not clip_id:
                raise ValueError("metadata 缺少 path")
            existing = result.setdefault(clip_id, row)
            for key in (
                "age_group",
                "gender_presentation",
                "environment_category",
                "camera_shot",
                "speed",
                "camera_elevation",
                "camera_azimuth",
                "camera_distance",
            ):
                if existing.get(key) != row.get(key):
                    raise ValueError(f"同 clip 的 metadata 不一致: {clip_id}")
    return result


def _manifest_by_clip(path: Path, *, dataset: str, split: str) -> dict[str, dict[str, str]]:
    if split == "test":
        raise ValueError("spot-check 禁止读取 test split")
    result: dict[str, dict[str, str]] = {}
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("dataset") != dataset or row.get("split") != split:
                continue
            clip_id = row.get("clip_id")
            if not clip_id or clip_id in result:
                raise ValueError("manifest clip_id 缺失或重复")
            result[clip_id] = row
    if not result:
        raise ValueError("manifest 中没有所选 dataset/split")
    return result


def _label_boundaries(events_json: str) -> list[dict[str, Any]]:
    try:
        events = json.loads(events_json)
    except json.JSONDecodeError as exc:
        raise ValueError("events_json 非法") from exc
    if not isinstance(events, list):
        raise TypeError("events_json 必须为数组")
    return [
        {
            "semantics": event["semantics"],
            "start_seconds": float(event["start"]),
            "end_seconds": float(event["end"]),
        }
        for event in events
    ]


def build_error_spotcheck(
    ablation_path: Path,
    manifest_path: Path,
    metadata_path: Path,
    pose_cache_root: Path,
    *,
    dataset: str,
    split: str,
    threshold: float,
    per_group: int = 20,
) -> dict[str, Any]:
    """Record score, pose availability, track continuity proxy, view and labels."""
    if split == "test":
        raise ValueError("spot-check 禁止读取 test split")
    try:
        ablation = json.loads(Path(ablation_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("ablation 不是合法 JSON") from exc
    if ablation.get("dataset") != dataset or ablation.get("split") != split:
        raise ValueError("ablation dataset/split 不匹配")
    predictions = ablation.get("predictions")
    if not isinstance(predictions, list):
        raise TypeError("ablation 缺少 predictions")
    manifest = _manifest_by_clip(manifest_path, dataset=dataset, split=split)
    metadata = _metadata_by_clip(metadata_path)

    rows = []
    for selected in select_spotcheck_predictions(
        predictions, threshold=threshold, per_group=per_group
    ):
        clip_id = str(selected["clip_id"])
        manifest_row = manifest.get(clip_id)
        metadata_row = metadata.get(clip_id)
        if manifest_row is None or metadata_row is None:
            raise ValueError(f"clip 无法映射 manifest/metadata: {clip_id}")
        record = read_pose_cache(pose_cache_path(pose_cache_root, dataset, split, clip_id))
        if (record.dataset, record.split, record.clip_id) != (dataset, split, clip_id):
            raise ValueError(f"pose cache identity 不一致: {clip_id}")
        per_track = Counter(int(track_id) for track_id in record.track_ids[record.valid_mask])
        frame_count = int(record.frame_indices.size)
        frames_with_pose = int(record.valid_mask.any(axis=1).sum())
        rows.append(
            {
                "sample_group": selected["sample_group"],
                "clip_id": clip_id,
                "activity": _activity(clip_id),
                "has_fall": bool(selected["has_fall"]),
                "tcn_score": float(selected["tcn_score"]),
                "threshold": threshold,
                "score_margin": float(selected["tcn_score"]) - threshold,
                "pose": {
                    "frame_count": frame_count,
                    "frames_without_pose": frame_count - frames_with_pose,
                    "frame_pose_coverage": frames_with_pose / frame_count,
                    "unique_track_count": len(per_track),
                    "dominant_track_observations": max(per_track.values()),
                },
                "view": {
                    key: metadata_row[key]
                    for key in (
                        "age_group",
                        "gender_presentation",
                        "environment_category",
                        "camera_shot",
                        "speed",
                        "camera_elevation",
                        "camera_azimuth",
                        "camera_distance",
                    )
                },
                "label_boundaries": _label_boundaries(manifest_row["events_json"]),
                "review_status": "cache_metadata_checked; visual_semantics_not_inferred",
            }
        )
    return {
        "protocol": PROTOCOL,
        "dataset": dataset,
        "split": split,
        "source_ablation": str(Path(ablation_path)),
        "threshold": threshold,
        "per_group": per_group,
        "rows": rows,
        "selection_notes": [
            "high_score_false_positive is sorted by descending TCN score",
            "all available fallen false negatives are selected before any other false-negative backfill",
            "pose and view fields are cache/metadata diagnostics, not visual semantic judgments",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--pose-cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", default="of-syn")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--per-group", type=int, default=20)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"输出已存在，拒绝覆盖: {args.output}")
    result = build_error_spotcheck(
        args.ablation,
        args.manifest,
        args.metadata,
        args.pose_cache_root,
        dataset=args.dataset,
        split=args.split,
        threshold=args.threshold,
        per_group=args.per_group,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "rows": len(result["rows"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
