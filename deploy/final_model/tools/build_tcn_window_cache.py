"""Build one signed TCN window cache from an existing pose cache and audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from models.tcn_dataset import build_window_cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pose-cache-root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--window-size", type=int, required=True)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--min-observed-frames", type=int, required=True)
    parser.add_argument("--causal-left-pad", action="store_true")
    args = parser.parse_args()
    cache = build_window_cache(
        args.manifest,
        args.pose_cache_root,
        args.audit,
        args.output_root,
        dataset=args.dataset,
        split=args.split,
        window_size=args.window_size,
        stride=args.stride,
        min_observed_frames=args.min_observed_frames,
        causal_left_pad=args.causal_left_pad,
        progress=lambda row: print(json.dumps(row), flush=True),
    )
    print(json.dumps({"cache": str(cache.root.resolve())}))


if __name__ == "__main__":
    main()
