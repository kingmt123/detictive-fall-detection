"""Audit whether a no-training top-2 actor replay has enough observable support."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from models.tcn_dataset import WindowMemmapCache, load_window_cache
from pipeline.pose_cache import pose_cache_path, read_pose_cache
from tools.train_actor_roi_rgb_canary import primary_track_ids
from tools.train_tcn import _atomic_json, _sha256_file


def _track_pose_statistics(record: Any, track_id: int) -> dict[str, float | int]:
    presence = np.any(
        record.valid_mask & (record.track_ids == int(track_id)), axis=1
    )
    observed = int(presence.sum())
    transitions = np.diff(np.pad(presence.astype(np.int8), (1, 1)))
    runs = int(np.count_nonzero(transitions == 1))
    valid_cells = record.valid_mask & (record.track_ids == int(track_id))
    confidences = record.keypoints[..., 2][valid_cells]
    return {
        "observed_frames": observed,
        "observation_runs": runs,
        "mean_joint_confidence": (
            float(confidences.mean()) if confidences.size else 0.0
        ),
    }


def audit_clip_tracks(
    cache: WindowMemmapCache,
    *,
    clip_index: int,
    primary_track: int,
    pose_cache_root: Path,
) -> dict[str, Any]:
    clip = cache.metadata["clips"][clip_index]
    start = int(clip["window_start"])
    count = int(clip["window_count"])
    tracks = np.asarray(cache.array("track_ids")[start : start + count], dtype=np.int64)
    if tracks.size == 0 or np.any(tracks < 0):
        raise ValueError("track feasibility 遇到无效窗口 track")
    values, counts = np.unique(tracks, return_counts=True)
    window_counts = {int(track): int(total) for track, total in zip(values, counts, strict=True)}
    record = read_pose_cache(
        pose_cache_path(pose_cache_root, "of-syn", "train", str(clip["clip_id"]))
    )
    candidates = []
    for track_id, window_count in window_counts.items():
        candidates.append(
            {
                "track_id": track_id,
                "window_count": window_count,
                **_track_pose_statistics(record, track_id),
            }
        )
    candidates.sort(
        key=lambda row: (
            row["track_id"] != primary_track,
            -int(row["window_count"]),
            int(row["track_id"]),
        )
    )
    primary = next(
        (row for row in candidates if row["track_id"] == primary_track), None
    )
    if primary is None:
        raise ValueError("主 track 不在窗口 cache 候选中")
    alternatives = [row for row in candidates if row["track_id"] != primary_track]
    alternatives.sort(
        key=lambda row: (
            -int(row["window_count"]),
            -int(row["observed_frames"]),
            int(row["track_id"]),
        )
    )
    second = alternatives[0] if alternatives else None
    ratio = (
        float(second["window_count"]) / float(primary["window_count"])
        if second is not None and int(primary["window_count"]) > 0
        else 0.0
    )
    return {
        "unique_window_tracks": len(candidates),
        "primary": primary,
        "second": second,
        "second_to_primary_window_ratio": ratio,
        "top2_observable": second is not None and int(second["observed_frames"]) >= 24,
        "top2_ambiguous": (
            second is not None
            and int(second["observed_frames"]) >= 24
            and ratio >= 0.5
        ),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"clips": 0}
    observable = sum(bool(row["track_audit"]["top2_observable"]) for row in rows)
    ambiguous = sum(bool(row["track_audit"]["top2_ambiguous"]) for row in rows)
    multi = sum(int(row["track_audit"]["unique_window_tracks"]) >= 2 for row in rows)
    return {
        "clips": len(rows),
        "multi_track_clips": multi,
        "multi_track_rate": multi / len(rows),
        "top2_observable_clips": observable,
        "top2_observable_rate": observable / len(rows),
        "top2_ambiguous_clips": ambiguous,
        "top2_ambiguous_rate": ambiguous / len(rows),
        "suggested_bucket_counts": dict(
            sorted(Counter(str(row["suggested_bucket"]) for row in rows).items())
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--pose-cache-root", type=Path, required=True)
    parser.add_argument("--error-review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("top-2 track audit 输出已存在，拒绝覆盖")
    cache = load_window_cache(args.cache, verify_hashes=True)
    if cache.metadata.get("dataset") != "of-syn" or cache.metadata.get("split") != "train":
        raise ValueError("top-2 track audit 只接受 OF-Syn train cache")
    review = json.loads(args.error_review.read_text(encoding="utf-8"))
    if review.get("protocol") != "edgefall_train_only_oof_error_review_v1":
        raise ValueError("top-2 track audit error review 协议不匹配")
    source_rows = review.get("rows")
    if not isinstance(source_rows, list) or not source_rows:
        raise ValueError("top-2 track audit 缺少 error rows")
    clip_indices = {
        str(clip["clip_id"]): index for index, clip in enumerate(cache.metadata["clips"])
    }
    primary_tracks = primary_track_ids(cache)
    rows = []
    for source in source_rows:
        clip_id = str(source["clip_id"])
        if clip_id not in clip_indices:
            raise ValueError(f"error review clip 不在 cache: {clip_id}")
        index = clip_indices[clip_id]
        rows.append(
            {
                "clip_id": clip_id,
                "review_set": str(source["review_set"]),
                "suggested_bucket": str(source["suggested_bucket"]),
                "score": float(source["score"]),
                "human_review_label": str(source["human_review_label"]),
                "track_audit": audit_clip_tracks(
                    cache,
                    clip_index=index,
                    primary_track=int(primary_tracks[index]),
                    pose_cache_root=args.pose_cache_root,
                ),
            }
        )
    groups = {
        name: summarize([row for row in rows if row["review_set"] == name])
        for name in sorted({str(row["review_set"]) for row in rows})
    }
    report = {
        "protocol": "edgefall_top2_track_observability_audit_v1",
        "qualification": "feasibility_only_no_alternate_actor_classification",
        "source_cache_metadata_sha256": _sha256_file(args.cache / "metadata.json"),
        "source_error_review_sha256": _sha256_file(args.error_review),
        "groups": groups,
        "rows": rows,
        "test_accessed": False,
        "limitations": [
            "top2_observable means at least 24 pose-observed frames, not correct actor identity",
            "top2_ambiguous is an unlabeled coverage heuristic, not an oracle accuracy result",
            "the current ROI cache contains only the primary actor, so alternate seven-head scores are unavailable",
            "human_review_label remains unreviewed and no visual semantics are inferred",
        ],
    }
    _atomic_json(args.output, report)
    print(json.dumps({"output": str(args.output.resolve()), "groups": groups}))


if __name__ == "__main__":
    main()
