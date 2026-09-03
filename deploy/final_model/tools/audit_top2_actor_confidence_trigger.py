"""Scan all OF-Syn train clips for a fixed, label-blind top-2 actor trigger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from models.tcn_dataset import load_window_cache
from tools.audit_top2_track_feasibility import audit_clip_tracks
from tools.evaluate_top2_actor_oracle_replay import _passes_confidence_trigger
from tools.train_actor_roi_rgb_canary import primary_track_ids
from tools.train_tcn import _atomic_json, _sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--pose-cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("全量 confidence trigger audit 输出已存在，拒绝覆盖")
    cache = load_window_cache(args.cache, verify_hashes=True)
    if cache.metadata.get("dataset") != "of-syn" or cache.metadata.get("split") != "train":
        raise ValueError("全量 confidence trigger audit 只接受 OF-Syn train cache")
    primary_tracks = primary_track_ids(cache)
    rows = []
    clips = cache.metadata["clips"]
    for index, clip in enumerate(clips):
        audit = audit_clip_tracks(
            cache,
            clip_index=index,
            primary_track=int(primary_tracks[index]),
            pose_cache_root=args.pose_cache_root,
        )
        if _passes_confidence_trigger({"track_audit": audit}):
            has_fall = bool(clip["has_fall"])
            rows.append(
                {
                    "clip_id": str(clip["clip_id"]),
                    "has_fall": has_fall,
                    "review_set": "global_positive" if has_fall else "global_negative",
                    "track_audit": audit,
                }
            )
        if (index + 1) % 1000 == 0 or index + 1 == len(clips):
            print(json.dumps({"stage": "pose_scan", "complete": index + 1, "total": len(clips)}), flush=True)
    positives = sum(row["has_fall"] for row in rows)
    report = {
        "protocol": "edgefall_top2_actor_confidence_trigger_full_train_audit_v1",
        "selection_is_label_blind": True,
        "trigger": {
            "second_observed_frames_min": 24,
            "second_to_primary_window_ratio_min": 0.75,
            "second_minus_primary_mean_joint_confidence_min": 0.20,
        },
        "source_cache_metadata_sha256": _sha256_file(args.cache / "metadata.json"),
        "summary": {
            "clips_scanned": len(clips),
            "triggered": len(rows),
            "trigger_rate": len(rows) / len(clips),
            "positive": positives,
            "negative": len(rows) - positives,
        },
        "rows": rows,
        "test_accessed": False,
    }
    _atomic_json(args.output, report)
    print(json.dumps(report["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
