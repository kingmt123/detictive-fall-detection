"""Run the condition-compliant label-blind seven-head model on one video."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from models.edgefall_seven_head_runtime import SevenHeadVideoModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--yolo-checkpoint", type=Path, default=Path("yolo11n-pose.pt"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("输出已存在，拒绝覆盖")
    model = SevenHeadVideoModel(
        args.artifact, args.yolo_checkpoint, device=args.device
    )
    try:
        result = model.predict(args.source)
    finally:
        model.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
