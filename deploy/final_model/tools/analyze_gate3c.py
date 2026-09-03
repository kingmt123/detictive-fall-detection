"""生成 Gate 3C 高召回 FP/FN 行为切片。"""
from __future__ import annotations

import argparse
from pathlib import Path

from eval.error_slices import signed_error_slices
from eval.tcn_ablation import atomic_json, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = signed_error_slices(args.ablation)
    atomic_json(args.output, result)
    print(
        f"wrote={args.output} sha256={sha256_file(args.output)} "
        f"signature={result['signature_sha256']}"
    )


if __name__ == "__main__":
    main()
