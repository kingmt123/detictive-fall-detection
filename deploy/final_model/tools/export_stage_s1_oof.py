"""Export and audit the immutable template-grouped Stage-S1 OOF fold map."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from models.tcn_dataset import load_window_cache
from tools.evaluate_oof_multistream_fusion import _template_grouped_folds
from tools.train_long_context_event_oracle import template_group


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260826)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("OOF fold map 已存在，拒绝覆盖")
    cache = load_window_cache(args.train_cache, verify_hashes=True)
    if cache.metadata.get("split") != "train":
        raise ValueError("OOF fold map 只允许 train cache")
    clips = cache.metadata["clips"]
    assignment = _template_grouped_folds(clips, args.folds, seed=args.seed)
    rows = []
    for index, clip in enumerate(clips):
        clip_id = str(clip["clip_id"])
        rows.append({
            "clip_id": clip_id,
            "group_id": template_group(clip_id),
            "label": int(bool(clip["has_fall"])),
            "fold": int(assignment[index]),
        })
    payload = {
        "protocol": "stage_s1_r2_template_grouped_train_only_oof_v1",
        "train_cache_signature_sha256": cache.metadata["signature_sha256"],
        "folds": args.folds,
        "seed": args.seed,
        "fold_assignment_sha256": hashlib.sha256(assignment.tobytes()).hexdigest(),
        "rows": rows,
        "summary": [
            {
                "fold": fold,
                "clips": int(np.count_nonzero(assignment == fold)),
                "positive_clips": int(sum(row["label"] for row in rows if row["fold"] == fold)),
                "template_groups": len({row["group_id"] for row in rows if row["fold"] == fold}),
            }
            for fold in range(args.folds)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
