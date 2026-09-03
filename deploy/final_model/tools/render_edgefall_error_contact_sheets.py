"""Render train-only EdgeFall OOF review contact sheets and a label template."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pipeline.video_source import VideoSourceResolver
from tools.train_tcn import _atomic_json


def select_review_rows(rows: list[dict[str, Any]], *, per_group: int) -> list[dict[str, Any]]:
    if per_group < 1:
        raise ValueError("per_group 必须为正")
    groups = ("high_score_false_positive", "low_score_false_negative")
    selected: list[dict[str, Any]] = []
    for group in groups:
        candidates = [row for row in rows if row.get("review_set") == group]
        if group == "high_score_false_positive":
            candidates.sort(key=lambda row: (-float(row["score"]), str(row["clip_id"])))
        else:
            candidates.sort(key=lambda row: (float(row["score"]), str(row["clip_id"])))
        selected.extend(candidates[:per_group])
    if len({str(row["clip_id"]) for row in selected}) != len(selected):
        raise ValueError("审阅 clip_id 不能重复")
    return selected


def _manifest_rows(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("dataset") != "of-syn" or row.get("split") != "train":
                continue
            clip_id = row.get("clip_id")
            if not clip_id or clip_id in rows:
                raise ValueError("manifest OF-Syn train clip_id 缺失或重复")
            rows[clip_id] = row
    return rows


def _frame_grid(path: Path, *, width: int = 192, height: int = 108) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    try:
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if count < 2:
            raise ValueError(f"视频帧数无效: {path}")
        frames: list[np.ndarray] = []
        for position in np.linspace(0, count - 1, 6).round().astype(int):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(position))
            ok, frame = capture.read()
            if not ok or frame is None:
                raise ValueError(f"无法读取视频帧: {path}@{position}")
            frames.append(cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA))
    finally:
        capture.release()
    return np.vstack((np.hstack(frames[:3]), np.hstack(frames[3:])))


def _contact_sheet(row: dict[str, Any], grid: np.ndarray) -> np.ndarray:
    header = np.full((78, grid.shape[1], 3), 255, dtype=np.uint8)
    lines = (
        f"{row['review_set']} | {row['clip_id']}",
        f"score={float(row['score']):.4f} | suggested={row['suggested_bucket']} | pose_proxy={row['pose_error_proxy']}",
        "Human label: _________________________________________________",
    )
    for index, text in enumerate(lines):
        cv2.putText(header, text, (8, 22 + index * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack((header, grid))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-json", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--temp-root", type=Path, required=True)
    parser.add_argument("--per-group", type=int, default=20)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("联系表输出目录必须为空")
    report = json.loads(args.review_json.read_text(encoding="utf-8"))
    if report.get("protocol") != "edgefall_train_only_oof_error_review_v1":
        raise ValueError("错误审阅协议不匹配")
    rows = report.get("rows")
    if not isinstance(rows, list):
        raise TypeError("错误审阅缺少 rows")
    selected = select_review_rows(rows, per_group=args.per_group)
    manifest = _manifest_rows(args.manifest)
    if any(str(row["clip_id"]) not in manifest for row in selected):
        raise ValueError("错误审阅 clip 不属于 OF-Syn train manifest")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = args.output_dir / "images"
    image_dir.mkdir()
    label_path = args.output_dir / "review_labels.csv"
    rendered: list[dict[str, Any]] = []
    with VideoSourceResolver(args.temp_root) as resolver, label_path.open("x", encoding="utf-8", newline="") as handle:
        fields = ("image", "review_set", "clip_id", "activity", "score", "suggested_bucket", "pose_error_proxy", "human_review_label", "reviewer_notes")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(selected, start=1):
            clip_id = str(row["clip_id"])
            with resolver.materialize(manifest[clip_id]["video_path"]) as source:
                image = _contact_sheet(row, _frame_grid(source.local_path))
            filename = f"{rank:03d}_{row['review_set']}_{clip_id.replace('/', '__')}.jpg"
            relative = Path("images") / filename
            if not cv2.imwrite(str(args.output_dir / relative), image):
                raise RuntimeError(f"无法写入联系表: {relative}")
            label_row = {
                "image": str(relative), "review_set": row["review_set"], "clip_id": clip_id,
                "activity": row["activity"], "score": row["score"],
                "suggested_bucket": row["suggested_bucket"], "pose_error_proxy": row["pose_error_proxy"],
                "human_review_label": "", "reviewer_notes": "",
            }
            writer.writerow(label_row)
            # The renderer can take minutes for remote/archive-backed clips.
            # Persist each completed label so an interrupted run never leaves
            # contact sheets without a matching human-review row.
            handle.flush()
            rendered.append(label_row)
    _atomic_json(args.output_dir / "summary.json", {
        "protocol": "edgefall_train_only_oof_contact_sheet_review_v1",
        "review_json": str(args.review_json.resolve()), "per_group": args.per_group,
        "rendered": len(rendered), "test_accessed": False,
    })
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "rendered": len(rendered)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
