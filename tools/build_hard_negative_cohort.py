"""Create a deterministic, train-only hard-negative clip cohort for FallTCN."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

COHORT_SCHEMA = 1
PROTOCOL = "tcn_hard_negative_cohort_v1"
DEFAULT_ACTIVITIES = ("lie_down", "lying", "sit_down", "sitting", "stand_up")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build_hard_negative_cohort(
    manifest_path: Path,
    *,
    dataset: str,
    split: str,
    seed: int,
    activities: tuple[str, ...] = DEFAULT_ACTIVITIES,
    per_activity: int = 100,
) -> dict[str, Any]:
    """Select equal-sized hard-negative groups using a stable SHA-256 ranking."""
    if split == "test":
        raise ValueError("困难负例组禁止读取 test split")
    if not dataset or not activities or len(set(activities)) != len(activities):
        raise ValueError("dataset 和 activities 必须非空且互不重复")
    if per_activity < 1:
        raise ValueError("per_activity 必须为正数")

    grouped = {activity: [] for activity in activities}
    excluded_positive_counts = {activity: 0 for activity in activities}
    identities: set[tuple[str, str, str]] = set()
    with Path(manifest_path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"dataset", "split", "clip_id", "has_fall"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"manifest 缺少字段: {sorted(missing)}")
        for row in reader:
            if row["dataset"] != dataset or row["split"] != split:
                continue
            identity = (row["dataset"], row["split"], row["clip_id"])
            if not all(identity) or identity in identities:
                raise ValueError("manifest 身份为空或重复")
            identities.add(identity)
            activity, separator, remainder = row["clip_id"].partition("/")
            if not separator or not remainder:
                raise ValueError(f"clip_id 不符合 activity/name 格式: {row['clip_id']!r}")
            if activity in grouped:
                if row["has_fall"] == "0":
                    grouped[activity].append(row["clip_id"])
                elif row["has_fall"] == "1":
                    excluded_positive_counts[activity] += 1
                else:
                    raise ValueError(f"has_fall 必须为 0 或 1: {row['clip_id']}")

    selected: dict[str, list[str]] = {}
    for activity in activities:
        candidates = grouped[activity]
        if len(candidates) < per_activity:
            raise ValueError(
                f"{activity} 候选不足: {len(candidates)} < {per_activity}"
            )
        selected[activity] = sorted(
            candidates,
            key=lambda clip_id: (
                hashlib.sha256(f"{seed}\0{clip_id}".encode()).hexdigest(),
                clip_id,
            ),
        )[:per_activity]

    payload: dict[str, Any] = {
        "cohort_schema": COHORT_SCHEMA,
        "protocol": PROTOCOL,
        "dataset": dataset,
        "split": split,
        "seed": seed,
        "activities": list(activities),
        "per_activity": per_activity,
        "manifest_sha256": _sha256_file(Path(manifest_path)),
        "candidate_counts": {activity: len(grouped[activity]) for activity in activities},
        "excluded_positive_counts": excluded_positive_counts,
        "clip_count": sum(len(clip_ids) for clip_ids in selected.values()),
        "clips_by_activity": selected,
    }
    payload["signature_sha256"] = hashlib.sha256(
        _canonical_json(payload).encode("utf-8")
    ).hexdigest()
    return payload


def write_hard_negative_cohort(output_path: Path, cohort: dict[str, Any]) -> None:
    """Write once to make cohort identity explicit and resistant to silent replacement."""
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"输出已存在，拒绝覆盖: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(cohort, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", default="of-syn")
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--per-activity", type=int, default=100)
    parser.add_argument("--activity", action="append", dest="activities")
    args = parser.parse_args()
    cohort = build_hard_negative_cohort(
        args.manifest,
        dataset=args.dataset,
        split=args.split,
        seed=args.seed,
        activities=tuple(args.activities or DEFAULT_ACTIVITIES),
        per_activity=args.per_activity,
    )
    write_hard_negative_cohort(args.output, cohort)
    print(json.dumps(cohort, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
