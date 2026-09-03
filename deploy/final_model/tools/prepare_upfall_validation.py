"""Prepare the untouched UP-Fall segment manifest without model inference."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from tools.build_manifest import write_manifest
from tools.train_tcn import _atomic_json, _sha256_file

LABEL_NAMES = {
    0: "walk",
    1: "fall",
    2: "fallen",
    3: "sit_down",
    4: "sitting",
    5: "lie_down",
    6: "lying",
    7: "stand_up",
    8: "standing",
    9: "other",
}
NAME_PATTERN = re.compile(
    r"^Subject(?P<subject>\d+)Activity(?P<activity>\d+)Trial(?P<trial>\d+)"
    r"Camera(?P<camera>[12])_(?P<label>\d+)_(?P<start>\d+\.\d+)_"
    r"(?P<end>\d+\.\d+)_(?P<subject_copy>\d+)_(?P<camera_copy>[12])_"
    r"up_fall\.avi$"
)
EMPTY_AVI_BYTES = 5762
TRUNCATED_SOURCE_FRAME_COUNTS = {
    "Subject14Activity1Trial3Camera1_2_2.250_9.890_14_1_up_fall.avi": (116, 115),
    "Subject14Activity1Trial3Camera2_2_2.250_9.890_14_2_up_fall.avi": (116, 115),
    "Subject14Activity9Trial3Camera1_9_0.450_3.430_14_1_up_fall.avi": (90, 89),
    "Subject14Activity9Trial3Camera2_9_0.450_3.430_14_2_up_fall.avi": (90, 89),
}


def parse_upfall_filename(
    name: str, *, allow_zero_duration: bool = False
) -> dict[str, str]:
    match = NAME_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(f"UP-Fall filename 无效: {name}")
    values = match.groupdict()
    if (
        values["subject"] != values["subject_copy"]
        or values["camera"] != values["camera_copy"]
        or int(values["label"]) not in LABEL_NAMES
        or float(values["end"]) < float(values["start"])
        or (
            not allow_zero_duration
            and float(values["end"]) == float(values["start"])
        )
    ):
        raise ValueError(f"UP-Fall filename identity 无效: {name}")
    return values


def build_upfall_manifest(
    source_root: Path,
) -> tuple[list[dict[str, object]], Counter[int], list[str]]:
    source_root = Path(source_root)
    clips_root = source_root / "clips"
    annotation_path = source_root / "mmaction2_annotations.txt"
    if not clips_root.is_dir() or not annotation_path.is_file():
        raise FileNotFoundError("UP-Fall source structure 不完整")
    annotations = {}
    for line in annotation_path.read_text(encoding="utf-8").splitlines():
        relative_text, label_text = line.rsplit(" ", 1)
        relative = Path(relative_text.replace("\\", "/"))
        if relative.as_posix() in annotations:
            raise ValueError(f"UP-Fall annotation 重复: {relative}")
        annotations[relative.as_posix()] = int(label_text)
    videos = sorted(clips_root.rglob("*.avi"))
    relative_videos = {path.relative_to(clips_root).as_posix() for path in videos}
    if set(annotations) != relative_videos:
        raise ValueError("UP-Fall annotation/video 集合不匹配")
    parsed = []
    invalid_sync_ids = set()
    for path in videos:
        relative = path.relative_to(clips_root).as_posix()
        values = parse_upfall_filename(path.name, allow_zero_duration=True)
        label = int(values["label"])
        if annotations[relative] != label:
            raise ValueError(f"UP-Fall annotation/filename label 不匹配: {relative}")
        sync_id = (
            f"subject{int(values['subject'])}:activity{int(values['activity'])}:"
            f"trial{int(values['trial'])}:label{label}:"
            f"{values['start']}-{values['end']}"
        )
        is_empty_signature = path.stat().st_size == EMPTY_AVI_BYTES
        if float(values["end"]) == float(values["start"]) and not is_empty_signature:
            raise ValueError(f"UP-Fall zero-duration file signature 无效: {relative}")
        if is_empty_signature or path.name in TRUNCATED_SOURCE_FRAME_COUNTS:
            invalid_sync_ids.add(sync_id)
        parsed.append((path, relative, values, sync_id))

    rows = []
    excluded_empty = []
    synchronized = defaultdict(list)
    label_counts: Counter[int] = Counter()
    for path, relative, values, sync_id in parsed:
        label = int(values["label"])
        if sync_id in invalid_sync_ids:
            excluded_empty.append(relative)
            continue
        subject = int(values["subject"])
        activity = int(values["activity"])
        trial = int(values["trial"])
        camera = int(values["camera"])
        start = values["start"]
        end = values["end"]
        synchronized[sync_id].append(camera)
        label_counts[label] += 1
        rows.append(
            {
                "dataset": "upfall_external",
                "video_path": str(path.resolve()),
                "clip_id": f"subject_{subject}/{path.name}",
                "group_id": f"upfall:subject:{subject}",
                "subject": str(subject),
                "trial": sync_id,
                "camera": f"cam{camera}",
                "split": "val",
                "has_fall": int(label == 1),
                "activity_semantics": LABEL_NAMES[label],
                "events_json": "[]",
            }
        )
    if (
        len(rows) != 2378
        or label_counts != Counter({0: 170, 1: 502, 2: 448, 3: 8, 4: 114, 5: 6, 6: 118, 7: 4, 8: 782, 9: 226})
        or {int(row["subject"]) for row in rows} != set(range(1, 18))
        or any(sorted(cameras) != [1, 2] for cameras in synchronized.values())
        or len(synchronized) != 1189
        or len(invalid_sync_ids) != 24
        or len(excluded_empty) != 48
    ):
        raise ValueError("UP-Fall manifest structural identity 无效")
    if len({str(row["clip_id"]) for row in rows}) != len(rows):
        raise ValueError("UP-Fall manifest clip_id 重复")
    return rows, label_counts, excluded_empty


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise FileExistsError("UP-Fall manifest/report 已存在，拒绝覆盖")
    rows, label_counts, excluded_empty = build_upfall_manifest(args.source_root)
    write_manifest(rows, args.output)
    report = {
        "protocol": "upfall_external_manifest_v3",
        "dataset": "upfall_external",
        "split": "val",
        "clips": len(rows),
        "positive_fall_clips": label_counts[1],
        "negative_nonfall_clips": len(rows) - label_counts[1],
        "label_counts": {LABEL_NAMES[label]: label_counts[label] for label in LABEL_NAMES},
        "subjects": list(range(1, 18)),
        "cameras": ["cam1", "cam2"],
        "synchronized_view_pairs": 1189,
        "excluded_source_defect_synchronized_pair_files": excluded_empty,
        "excluded_source_defect_synchronized_pairs": len(excluded_empty) // 2,
        "zero_frame_detection": (
            "exact 5762-byte empty AVI source signature; both synchronized views are "
            "excluded whenever either view has the signature"
        ),
        "truncated_source_frame_counts": {
            name: {"container_frames": counts[0], "decoded_frames": counts[1]}
            for name, counts in TRUNCATED_SOURCE_FRAME_COUNTS.items()
        },
        "truncated_source_exclusion": (
            "the four files were discovered by fail-closed pose decoding before any "
            "classifier inference; both synchronized views are excluded"
        ),
        "binary_mapping": "OmniFall class 1 fall is positive; every other class is negative",
        "manifest_sha256": _sha256_file(args.output),
        "model_inference_run": False,
        "metrics_inspected": False,
        "test_accessed": False,
        "limitations": [
            "the archive contains action segments rather than original untrimmed recordings",
            "two synchronized views are retained and must be grouped in secondary uncertainty analyses",
            "UP-Fall contains staged falls by young participants",
        ],
    }
    _atomic_json(args.report, report)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
