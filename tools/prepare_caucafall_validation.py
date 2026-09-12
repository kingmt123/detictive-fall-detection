"""Reconstruct 100 CAUCAFall source sequences and write a validation manifest."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2

from tools.build_manifest import write_manifest
from tools.train_tcn import _atomic_json, _sha256_file


@dataclass(frozen=True)
class SourcePlan:
    subject: int
    source_path: str
    segments: tuple[dict[str, str], ...]
    has_fall: bool


def build_source_plans(label_csv: Path) -> list[SourcePlan]:
    with Path(label_csv).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"path", "label", "start", "end", "subject", "cam", "dataset"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("CAUCAFall labels 缺少必要字段")
    grouped: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["dataset"] != "caucafall" or row["cam"] != "1":
            raise ValueError("CAUCAFall labels dataset/camera 身份无效")
        subject = int(row["subject"])
        if not 1 <= subject <= 10 or row["label"] not in {str(i) for i in range(10)}:
            raise ValueError("CAUCAFall subject/label 无效")
        if float(row["end"]) <= float(row["start"]):
            raise ValueError("CAUCAFall 时间区间无效")
        grouped[(subject, row["path"])].append(row)
    plans = []
    for (subject, source_path), segments in sorted(
        grouped.items(), key=lambda item: (item[0][0], item[0][1])
    ):
        ordered = tuple(sorted(segments, key=lambda row: float(row["start"])))
        plans.append(
            SourcePlan(
                subject=subject,
                source_path=source_path,
                segments=ordered,
                has_fall=any(row["label"] == "1" for row in ordered),
            )
        )
    if len(plans) != 100 or {plan.subject for plan in plans} != set(range(1, 11)):
        raise ValueError("CAUCAFall 必须包含 10 subjects 和 100 source sequences")
    per_subject = {
        subject: sum(plan.subject == subject for plan in plans)
        for subject in range(1, 11)
    }
    if set(per_subject.values()) != {10} or sum(plan.has_fall for plan in plans) != 50:
        raise ValueError("CAUCAFall 每 subject 10 sequences / 总计 50 falls 不匹配")
    return plans


def _canonical_segment(clips_root: Path, plan: SourcePlan, row: dict[str, str]) -> Path:
    base = plan.source_path.rsplit("/", 1)[-1]
    candidates = sorted(
        (clips_root / f"Subject.{plan.subject}").glob(
            f"{base}_{row['label']}_*.avi"
        )
    )
    matches = []
    for path in candidates:
        parts = path.stem.split("_")
        try:
            start, end = float(parts[-5]), float(parts[-4])
        except (ValueError, IndexError):
            continue
        if abs(start - float(row["start"])) <= 0.002 and abs(
            end - float(row["end"])
        ) <= 0.002:
            matches.append(path)
    if len(matches) != 1:
        raise ValueError(
            f"CAUCAFall canonical segment 匹配失败: {plan.source_path} {row}"
        )
    return matches[0]


def reconstruct_sources(
    source_root: Path, output_root: Path, plans: list[SourcePlan]
) -> list[dict[str, object]]:
    clips_root = Path(source_root) / "clips"
    rows: list[dict[str, object]] = []
    for index, plan in enumerate(plans, start=1):
        category = "fall" if plan.has_fall else "adl"
        base = plan.source_path.rsplit("/", 1)[-1]
        target = Path(output_root) / f"subject_{plan.subject}" / category / f"{base}.mp4"
        if target.exists():
            raise FileExistsError(f"CAUCAFall reconstructed video 已存在: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(target), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (720, 480)
        )
        if not writer.isOpened():
            raise RuntimeError(f"无法创建 CAUCAFall reconstructed video: {target}")
        frames = 0
        try:
            for segment in plan.segments:
                path = _canonical_segment(clips_root, plan, segment)
                capture = cv2.VideoCapture(str(path))
                if not capture.isOpened():
                    raise ValueError(f"无法打开 CAUCAFall segment: {path}")
                if (
                    abs(capture.get(cv2.CAP_PROP_FPS) - 20.0) > 1e-6
                    or int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) != 720
                    or int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) != 480
                ):
                    capture.release()
                    raise ValueError(f"CAUCAFall segment 视频签名不匹配: {path}")
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    writer.write(frame)
                    frames += 1
                capture.release()
        finally:
            writer.release()
        if frames < 20 or not target.is_file() or target.stat().st_size <= 0:
            raise ValueError(f"CAUCAFall reconstructed video 无效: {target}")
        rows.append(
            {
                "dataset": "caucafall_external",
                "video_path": str(target.resolve()),
                "clip_id": f"{category}/{base}_subject_{plan.subject}",
                "group_id": f"caucafall:subject:{plan.subject}",
                "subject": str(plan.subject),
                "trial": plan.source_path,
                "camera": "cam1",
                "split": "val",
                "has_fall": int(plan.has_fall),
                "activity_semantics": "fall_incident" if plan.has_fall else "non_fall",
                "events_json": "[]",
            }
        )
        if index % 20 == 0 or index == len(plans):
            print(json.dumps({"reconstructed": index, "total": len(plans)}), flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--video-output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.manifest.exists() or args.report.exists() or args.video_output_root.exists():
        raise FileExistsError("CAUCAFall reconstruction/manifest/report 已存在，拒绝覆盖")
    plans = build_source_plans(args.source_root / "caucafall.csv")
    rows = reconstruct_sources(args.source_root, args.video_output_root, plans)
    write_manifest(rows, args.manifest)
    report = {
        "protocol": "caucafall_external_source_reconstruction_v1",
        "dataset": "caucafall_external",
        "split": "val",
        "source_segments": sum(len(plan.segments) for plan in plans),
        "source_sequences": len(plans),
        "positive_sequences": sum(plan.has_fall for plan in plans),
        "negative_sequences": sum(not plan.has_fall for plan in plans),
        "subjects": list(range(1, 11)),
        "reconstruction": "chronological concatenation of canonical non-duplicate action segments",
        "manifest_sha256": _sha256_file(args.manifest),
        "model_inference_run": False,
        "metrics_inspected": False,
        "test_accessed": False,
        "limitations": [
            "the downloadable OmniFall-aligned archive contains action segments rather than original source containers",
            "source sequences are reconstructed by chronological frame concatenation and MP4V re-encoding",
            "small annotation boundary overlaps or gaps are not repaired",
        ],
    }
    _atomic_json(args.report, report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
