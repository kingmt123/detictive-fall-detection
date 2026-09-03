"""Prepare a clip-level GMDCSA24 manifest without running model inference."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from tools.build_manifest import write_manifest
from tools.train_tcn import _atomic_json, _sha256_file


def build_gmdcsa24_manifest(source_root: Path) -> list[dict[str, object]]:
    source_root = Path(source_root)
    rows: list[dict[str, object]] = []
    for subject_dir in sorted(source_root.glob("Subject *")):
        subject_text = subject_dir.name.removeprefix("Subject ")
        if not subject_text.isdigit():
            raise ValueError(f"无法解析 GMDCSA24 subject: {subject_dir.name}")
        subject = int(subject_text)
        for category, has_fall in (("ADL", False), ("Fall", True)):
            csv_path = subject_dir / f"{category}.csv"
            video_dir = subject_dir / category
            if not csv_path.is_file() or not video_dir.is_dir():
                raise FileNotFoundError(f"GMDCSA24 目录不完整: {subject_dir}/{category}")
            with csv_path.open(encoding="utf-8-sig", newline="") as handle:
                annotations = list(csv.DictReader(handle))
            annotation_names = {str(row.get("File Name", "")).strip() for row in annotations}
            videos = sorted(video_dir.glob("*.mp4"))
            video_names = {path.name for path in videos}
            if annotation_names != video_names:
                raise ValueError(
                    f"GMDCSA24 CSV/video 不匹配: subject={subject}, category={category}"
                )
            category_key = category.lower()
            for path in videos:
                stem = path.stem
                rows.append(
                    {
                        "dataset": "gmdcsa24",
                        "video_path": str(path.resolve()),
                        "clip_id": f"{category_key}/{stem}_subject_{subject}",
                        "group_id": f"gmdcsa24:subject:{subject}",
                        "subject": str(subject),
                        "trial": f"{category_key}-{stem}",
                        "camera": "cam0",
                        "split": "val",
                        "has_fall": int(has_fall),
                        "activity_semantics": (
                            "fall_incident" if has_fall else "non_fall"
                        ),
                        "events_json": "[]",
                    }
                )
    if len(rows) != 160:
        raise ValueError(f"GMDCSA24 预期 160 clips，实际 {len(rows)}")
    if len({str(row["clip_id"]) for row in rows}) != len(rows):
        raise ValueError("GMDCSA24 manifest clip_id 重复")
    if sum(int(row["has_fall"]) for row in rows) != 79:
        raise ValueError("GMDCSA24 manifest fall 数量不匹配")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise FileExistsError("GMDCSA24 manifest/report 已存在，拒绝覆盖")
    rows = build_gmdcsa24_manifest(args.source_root)
    write_manifest(rows, args.output)
    report = {
        "protocol": "gmdcsa24_external_validation_manifest_v1",
        "dataset": "gmdcsa24",
        "split": "val",
        "clips": len(rows),
        "positive_clips": sum(int(row["has_fall"]) for row in rows),
        "negative_clips": sum(not int(row["has_fall"]) for row in rows),
        "subjects": sorted({str(row["subject"]) for row in rows}),
        "manifest_sha256": _sha256_file(args.output),
        "model_inference_run": False,
        "metrics_inspected": False,
        "test_accessed": False,
        "limitations": [
            "clip labels use the official ADL/Fall directory assignment",
            "temporal annotations are not used for this clip-level validation",
        ],
    }
    _atomic_json(args.report, report)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
