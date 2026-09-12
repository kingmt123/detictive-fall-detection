"""Render train-only primary/secondary pose-track contact sheets for actor replay."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pipeline.pose_cache import PoseCacheRecord, pose_cache_path, read_pose_cache
from pipeline.video_source import VideoSourceResolver

SKELETON = (
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
)
PRIMARY_COLOR = (40, 210, 40)
SECONDARY_COLOR = (220, 60, 220)
OTHER_COLOR = (150, 150, 150)


def sample_positions(frame_count: int, sample_count: int) -> list[int]:
    """Choose stable, unique positions including both temporal endpoints."""
    if frame_count < 1 or sample_count < 1:
        raise ValueError("frame_count 和 sample_count 必须为正数")
    count = min(frame_count, sample_count)
    return sorted(set(np.linspace(0, frame_count - 1, count).round().astype(int).tolist()))


def _manifest_rows(path: Path, clip_ids: set[str]) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("dataset") != "of-syn" or row.get("split") != "train":
                continue
            clip_id = str(row.get("clip_id", ""))
            if clip_id in clip_ids:
                if clip_id in rows:
                    raise ValueError(f"manifest clip_id 重复: {clip_id}")
                rows[clip_id] = row
    missing = sorted(clip_ids - rows.keys())
    if missing:
        raise ValueError(f"manifest 缺少 train clip: {missing}")
    return rows


def _candidate_rows(path: Path, clip_ids: set[str]) -> dict[str, dict[str, Any]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("protocol") != "edgefall_top2_track_observability_audit_v1":
        raise ValueError("top-2 feasibility 协议不匹配")
    rows = {
        str(row["clip_id"]): row
        for row in report.get("rows", [])
        if str(row.get("clip_id")) in clip_ids
    }
    missing = sorted(clip_ids - rows.keys())
    if missing:
        raise ValueError(f"top-2 feasibility 缺少 clip: {missing}")
    return rows


def _event_at(events: list[dict[str, Any]], timestamp: float) -> str:
    for event in events:
        if float(event["start"]) <= timestamp <= float(event["end"]) + 1e-9:
            return str(event["semantics"])
    return "unlabeled"


def _draw_track(
    frame: np.ndarray,
    record: PoseCacheRecord,
    position: int,
    person: int,
    color: tuple[int, int, int],
    label: str,
) -> None:
    box = record.bboxes[position, person]
    x1, y1, x2, y2 = np.rint(box).astype(int).tolist()
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    cv2.putText(
        frame, label, (max(0, x1), max(22, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX,
        0.72, color, 2, cv2.LINE_AA,
    )
    points = record.keypoints[position, person]
    confident = points[:, 2] >= 0.25
    for first, second in SKELETON:
        if confident[first] and confident[second]:
            p1 = tuple(np.rint(points[first, :2]).astype(int).tolist())
            p2 = tuple(np.rint(points[second, :2]).astype(int).tolist())
            cv2.line(frame, p1, p2, color, 2, cv2.LINE_AA)
    for x, y, _confidence in points[confident]:
        cv2.circle(frame, (round(float(x)), round(float(y))), 3, color, -1, cv2.LINE_AA)


def _annotate(
    frame: np.ndarray,
    record: PoseCacheRecord,
    position: int,
    primary_track: int,
    second_track: int,
    semantics: str,
) -> np.ndarray:
    result = frame.copy()
    for person in np.flatnonzero(record.valid_mask[position]):
        track_id = int(record.track_ids[position, person])
        if track_id == primary_track:
            color, label = PRIMARY_COLOR, f"P:{track_id}"
        elif track_id == second_track:
            color, label = SECONDARY_COLOR, f"S:{track_id}"
        else:
            color, label = OTHER_COLOR, f"T:{track_id}"
        _draw_track(result, record, position, int(person), color, label)
    timestamp = float(record.timestamps[position])
    cv2.rectangle(result, (0, 0), (result.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(
        result,
        f"frame={int(record.frame_indices[position])} t={timestamp:.2f}s {semantics}",
        (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA,
    )
    return result


def _make_sheet(frames: list[np.ndarray], *, tile_width: int = 480) -> np.ndarray:
    tiles = []
    for frame in frames:
        scale = tile_width / frame.shape[1]
        tiles.append(cv2.resize(frame, (tile_width, round(frame.shape[0] * scale))))
    columns = 3
    rows = []
    blank = np.zeros_like(tiles[0])
    for start in range(0, len(tiles), columns):
        group = tiles[start : start + columns]
        rows.append(np.hstack(group + [blank] * (columns - len(group))))
    return np.vstack(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pose-cache-root", type=Path, required=True)
    parser.add_argument("--feasibility-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--clip-id", action="append", required=True)
    parser.add_argument("--samples", type=int, default=9)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖: {args.output_dir}")
    clip_ids = set(args.clip_id)
    if len(clip_ids) != len(args.clip_id):
        raise ValueError("clip-id 不得重复")
    manifest = _manifest_rows(args.manifest, clip_ids)
    candidates = _candidate_rows(args.feasibility_report, clip_ids)
    args.output_dir.mkdir(parents=True)
    rendered = []
    with VideoSourceResolver(args.output_dir / "temp") as resolver:
        for clip_id in sorted(clip_ids):
            record = read_pose_cache(
                pose_cache_path(args.pose_cache_root, "of-syn", "train", clip_id)
            )
            audit = candidates[clip_id]["track_audit"]
            primary_track = int(audit["primary"]["track_id"])
            second_track = int(audit["second"]["track_id"])
            events = json.loads(manifest[clip_id]["events_json"])
            positions = sample_positions(len(record.frame_indices), args.samples)
            wanted = {int(record.frame_indices[position]): position for position in positions}
            frames: list[np.ndarray] = []
            source = manifest[clip_id]["video_path"]
            with resolver.materialize(source) as video:
                capture = cv2.VideoCapture(str(video.local_path))
                if not capture.isOpened():
                    raise ValueError(f"无法打开视频: {clip_id}")
                decoded = 0
                try:
                    while len(frames) < len(wanted):
                        ok, frame = capture.read()
                        if not ok:
                            break
                        position = wanted.get(decoded)
                        if position is not None:
                            frames.append(
                                _annotate(
                                    frame, record, position, primary_track, second_track,
                                    _event_at(events, float(record.timestamps[position])),
                                )
                            )
                        decoded += 1
                finally:
                    capture.release()
            if len(frames) != len(positions):
                raise ValueError(f"视频帧不足: {clip_id}, {len(frames)}/{len(positions)}")
            output = args.output_dir / f"{clip_id.replace('/', '__')}.jpg"
            if not cv2.imwrite(str(output), _make_sheet(frames)):
                raise OSError(f"无法写入接触表: {output}")
            rendered.append(
                {
                    "clip_id": clip_id,
                    "primary_track": primary_track,
                    "second_track": second_track,
                    "positions": positions,
                    "output": str(output),
                }
            )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "protocol": "edgefall_top2_track_visual_replay_v1",
                "dataset": "of-syn",
                "split": "train",
                "test_accessed": False,
                "legend": {"primary": "green", "secondary": "magenta", "other": "gray"},
                "rendered": rendered,
            },
            ensure_ascii=False, indent=2, sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "temp").rmdir()
    print(json.dumps({"output_dir": str(args.output_dir), "clips": len(rendered)}))


if __name__ == "__main__":
    main()
