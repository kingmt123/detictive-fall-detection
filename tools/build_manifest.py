"""构建保留语义与 group_id 的视频级数据清单。

原则：
- ``fall``（跌倒过程）与 ``fallen``（跌倒后状态）均属于跌倒事故，但分开保存；
- 有意躺下/坐下/蹲下等属于 hard negative；
- 同一被试/试次/多机位共享 group_id，并只进入一个 split。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import tarfile
from collections import defaultdict
from pathlib import Path

FALL_PROCESS = 1
POST_FALL_STATE = 2
HARD_NEGATIVES = {3, 4, 5, 6, 10, 11, 12, 13, 14}


def classify_activity(label_id: int) -> str:
    if label_id == FALL_PROCESS:
        return "fall_process"
    if label_id == POST_FALL_STATE:
        return "post_fall_state"
    if label_id in HARD_NEGATIVES:
        return "hard_negative"
    return "background"


def urfd_group_id(path: Path) -> str:
    match = re.fullmatch(r"(fall|adl)-(\d{2})-cam\d+\.mp4", path.name)
    if not match:
        raise ValueError(f"无法解析 URFD 文件名: {path.name}")
    return f"{match.group(1)}-{match.group(2)}"


def _stable_order(groups: list[str], seed: int) -> list[str]:
    return sorted(
        groups,
        key=lambda group: hashlib.sha256(f"{seed}:{group}".encode()).hexdigest(),
    )


def assign_group_splits(
    groups: list[str],
    positive_by_group: dict[str, bool],
    seed: int = 2026,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> dict[str, str]:
    """按标签分层、按 group 划分，保证多机位不泄漏。"""
    unique = sorted(set(groups))
    if set(unique) != set(positive_by_group):
        raise ValueError("groups 与 positive_by_group 的键不一致")
    if not 0 < train_ratio < 1 or not 0 < val_ratio < 1:
        raise ValueError("split 比例必须在 (0, 1) 内")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio 必须小于 1")

    result: dict[str, str] = {}
    for label in (False, True):
        bucket = _stable_order(
            [group for group in unique if positive_by_group[group] is label], seed
        )
        if len(bucket) < 3:
            raise ValueError("每个标签至少需要 3 个 group 才能分到 train/val/test")
        n_train = max(1, round(len(bucket) * train_ratio))
        n_val = max(1, round(len(bucket) * val_ratio))
        if n_train + n_val >= len(bucket):
            n_train = len(bucket) - 2
            n_val = 1
        for index, group in enumerate(bucket):
            result[group] = (
                "train"
                if index < n_train
                else "val"
                if index < n_train + n_val
                else "test"
            )
    return result


def build_urfd_manifest(source_dir: Path, seed: int = 2026) -> list[dict]:
    videos = sorted(source_dir.glob("*.mp4"))
    groups = sorted({urfd_group_id(path) for path in videos})
    positive = {group: group.startswith("fall-") for group in groups}
    splits = assign_group_splits(groups, positive, seed=seed)
    rows = []
    for path in videos:
        group = urfd_group_id(path)
        rows.append(
            {
                "dataset": "urfd",
                "video_path": str(path.resolve()),
                "clip_id": path.stem,
                "group_id": group,
                "subject": "",
                "trial": group,
                "camera": path.stem.rsplit("-", 1)[-1],
                "split": splits[group],
                "has_fall": int(positive[group]),
                "activity_semantics": "fall_incident" if positive[group] else "background",
                "events_json": "[]",
            }
        )
    return rows


def build_ofsyn_manifest(labels_csv: Path, tar_path: Path, split_dir: Path) -> list[dict]:
    with tarfile.open(tar_path) as archive:
        tar_members = set(archive.getnames())
    split_by_path: dict[str, str] = {}
    for split in ("train", "val", "test"):
        with (split_dir / f"{split}.csv").open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                path = row["path"]
                if path in split_by_path:
                    raise ValueError(f"OF-Syn path 跨 split 重复: {path}")
                split_by_path[path] = split

    segments: dict[str, list[dict]] = defaultdict(list)
    metadata: dict[str, dict] = {}
    with labels_csv.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            path = row["path"]
            metadata[path] = row
            label = int(row["label"])
            start = float(row["start"])
            end = float(row["end"])
            if not math.isfinite(start) or not math.isfinite(end) or end <= start:
                continue
            segments[path].append(
                {
                    "label_id": label,
                    "semantics": classify_activity(label),
                    "start": start,
                    "end": end,
                }
            )

    label_paths = set(metadata)
    split_paths = set(split_by_path)
    if label_paths != split_paths:
        only_split = sorted(split_paths - label_paths)[:3]
        only_labels = sorted(label_paths - split_paths)[:3]
        raise ValueError(
            "OF-Syn split 与标签路径集合不一致: "
            f"仅 split={only_split}, 仅 labels={only_labels}"
        )
    invalid_only = sorted(label_paths - set(segments))
    if invalid_only:
        raise ValueError(
            "OF-Syn clip 没有合法事件片段: " + ", ".join(invalid_only[:3])
        )

    rows = []
    for path, event_segments in sorted(segments.items()):
        if path not in split_by_path:
            raise ValueError(f"OF-Syn path 不在 random split 中: {path}")
        row = metadata[path]
        tar_member = f"./{path.lstrip('./')}.mp4"
        if tar_member not in tar_members:
            raise ValueError(f"OF-Syn 视频不在 tar 中: {tar_member}")
        has_fall = any(
            event["semantics"] in {"fall_process", "post_fall_state"}
            for event in event_segments
        )
        rows.append(
            {
                "dataset": "of-syn",
                "video_path": f"tar://{tar_path.resolve()}!/{tar_member}",
                "clip_id": path,
                "group_id": f"of-syn:{path}",
                "subject": row.get("subject", "-1"),
                "trial": path,
                "camera": row.get("cam", "-1"),
                "split": split_by_path[path],
                "has_fall": int(has_fall),
                "activity_semantics": "fall_incident" if has_fall else "non_fall",
                "events_json": json.dumps(event_segments, ensure_ascii=False),
            }
        )
    return rows


def write_manifest(rows: list[dict], output: Path) -> None:
    if not rows:
        raise ValueError("manifest 不能为空")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--urfd-dir", type=Path, default=Path("data/sources/urfd"))
    parser.add_argument("--ofsyn-labels", type=Path, default=Path("data/omnifall/labels/of-syn.csv"))
    parser.add_argument("--ofsyn-tar", type=Path, default=Path("data/omnifall/data_files/omnifall-synthetic_av1.tar"))
    parser.add_argument("--ofsyn-splits", type=Path, default=Path("data/omnifall/splits/syn/random"))
    parser.add_argument("--output", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    rows = build_urfd_manifest(args.urfd_dir, seed=args.seed)
    rows.extend(build_ofsyn_manifest(args.ofsyn_labels, args.ofsyn_tar, args.ofsyn_splits))
    write_manifest(rows, args.output)
    counts = defaultdict(int)
    for row in rows:
        counts[(row["dataset"], row["split"], row["has_fall"])] += 1
    print(f"输出 {len(rows)} 个视频到 {args.output}")
    for key, count in sorted(counts.items()):
        print(f"  dataset={key[0]:6s} split={key[1]:5s} fall={key[2]} count={count}")


if __name__ == "__main__":
    main()
