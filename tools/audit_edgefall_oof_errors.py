"""Export a train-only EdgeFall OOF queue for human error review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from eval.metrics import competition_map
from tools.train_edgefall_f3 import pose_error_targets
from tools.train_tcn import _atomic_json

FP_TAXONOMY = (
    "controlled_lie_down", "already_lying", "stand_up", "sit_or_bend",
    "pose_missing", "id_switch", "camera_motion", "scene_bias", "annotation_error",
    "annotation_or_semantic_ambiguity", "controlled_fall_like_action",
    "controlled_sit_down", "controlled_sit_or_recline", "static_lying",
    "sport_or_play_action", "unusual_pose_nonfall", "crawling_nonfall",
    "floor_exercise", "assisted_or_group_action",
)
FN_TAXONOMY = (
    "occluded_fall", "small_person", "slow_fall", "partial_fall", "rapid_recovery",
    "unusual_view", "pose_failure", "annotation_error",
    "annotation_or_semantic_ambiguity", "slow_or_partial_fall",
    "slow_or_subtle_fall", "static_post_fall", "visual_quality_limitation",
)


def suggested_false_positive_bucket(activity: str, pose_error: bool) -> str:
    if pose_error:
        return "pose_missing"
    return {
        "lie_down": "controlled_lie_down", "lying": "already_lying",
        "stand_up": "stand_up", "sit_down": "sit_or_bend",
    }.get(activity, "scene_bias")


def error_review(
    clip_ids: list[str],
    labels: np.ndarray,
    scores: np.ndarray,
    quality: np.ndarray,
    *,
    limit: int,
    target_recall: float = 0.90,
) -> dict[str, Any]:
    if limit < 1:
        raise ValueError("limit 必须为正")
    if not (len(clip_ids) == labels.size == scores.size == quality.shape[0]):
        raise ValueError("OOF 数组长度不一致")
    if not 0.0 < target_recall <= 1.0:
        raise ValueError("target_recall 必须位于 (0,1]")
    metric = competition_map(
        dict(zip(clip_ids, (bool(value) for value in labels), strict=True)),
        dict(zip(clip_ids, (float(value) for value in scores), strict=True)),
        mode="clip",
    )
    feasible = [point for point in metric["curve"] if point.recall >= target_recall]
    if not feasible:
        raise ValueError("分数无法达到目标 recall")
    selected_point = max(feasible, key=lambda item: (item.precision, item.threshold))
    point = {
        "target_recall": target_recall,
        "threshold": float(selected_point.threshold),
        "precision": float(selected_point.precision),
        "recall": float(selected_point.recall),
    }
    pose_error = pose_error_targets(torch.from_numpy(quality)).numpy() > 0.5
    rows: list[dict[str, Any]] = []
    for index, clip_id in enumerate(clip_ids):
        activity = clip_id.split("/", 1)[0]
        label = bool(labels[index])
        score = float(scores[index])
        if not label and score >= point["threshold"]:
            rows.append({
                "review_set": "high_score_false_positive", "clip_id": clip_id,
                "activity": activity, "score": score, "margin": score - point["threshold"],
                "pose_error_proxy": bool(pose_error[index]), "quality": quality[index].tolist(),
                "suggested_bucket": suggested_false_positive_bucket(activity, bool(pose_error[index])),
                "human_review_label": "unreviewed",
            })
        elif label and score < point["threshold"]:
            rows.append({
                "review_set": "low_score_false_negative", "clip_id": clip_id,
                "activity": activity, "score": score, "margin": score - point["threshold"],
                "pose_error_proxy": bool(pose_error[index]), "quality": quality[index].tolist(),
                "suggested_bucket": "pose_failure" if pose_error[index] else "unusual_view",
                "human_review_label": "unreviewed",
            })
    false_positives = sorted(
        (row for row in rows if row["review_set"] == "high_score_false_positive"),
        key=lambda row: (-float(row["score"]), str(row["clip_id"])),
    )[:limit]
    false_negatives = sorted(
        (row for row in rows if row["review_set"] == "low_score_false_negative"),
        key=lambda row: (float(row["score"]), str(row["clip_id"])),
    )[:limit]
    return {
        "protocol": "edgefall_train_only_oof_error_review_v1",
        "operating_point": point,
        "rows": false_positives + false_negatives,
        "taxonomy": {"false_positive": FP_TAXONOMY, "false_negative": FN_TAXONOMY},
        "limitations": [
            "suggested_bucket is a metadata/quality heuristic, not a visual semantic judgment",
            "all rows require human_review_label before use for training or model selection",
            "only train-only grouped OOF data is read; no test split is accessed",
        ],
    }


def quality_from_roi_cache(clip_ids: list[str], roi_cache: Path) -> np.ndarray:
    """Return quality rows aligned exactly to OOF clip IDs from a ROI cache."""
    with np.load(roi_cache, allow_pickle=False) as payload:
        required = {"clip_ids", "quality"}
        if not required.issubset(payload.files):
            raise ValueError("ROI cache 缺少 clip_ids/quality")
        cache_ids = [str(value) for value in payload["clip_ids"]]
        quality = np.asarray(payload["quality"], dtype=np.float32)
    if quality.ndim != 2 or quality.shape[1] != 5 or quality.shape[0] != len(cache_ids):
        raise ValueError("ROI quality 必须为 (N,5) 且与 clip_ids 对齐")
    index = {clip_id: position for position, clip_id in enumerate(cache_ids)}
    if len(index) != len(cache_ids) or any(clip_id not in index for clip_id in clip_ids):
        raise ValueError("ROI cache 缺少 OOF clip 或 clip_id 重复")
    return quality[np.asarray([index[clip_id] for clip_id in clip_ids], dtype=np.int64)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--score-key", default="f3_scores")
    parser.add_argument("--roi-cache", type=Path)
    parser.add_argument("--target-recall", type=float, default=0.90)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("错误审阅输出已存在，拒绝覆盖")
    with np.load(args.oof, allow_pickle=False) as payload:
        clip_key = "clip_ids" if "clip_ids" in payload.files else "clip_id"
        label_key = "labels" if "labels" in payload.files else "label"
        required = {clip_key, label_key, args.score_key}
        if not required.issubset(payload.files):
            raise ValueError("OOF 缺少错误审阅所需字段")
        clip_ids = [str(value) for value in payload[clip_key]]
        quality = (
            np.asarray(payload["quality"], dtype=np.float32)
            if "quality" in payload.files
            else None
        )
        if quality is None:
            if args.roi_cache is None:
                raise ValueError("OOF 未包含 quality，必须提供 --roi-cache")
            quality = quality_from_roi_cache(clip_ids, args.roi_cache)
        report = error_review(
            clip_ids,
            np.asarray(payload[label_key]),
            np.asarray(payload[args.score_key]),
            quality,
            limit=args.limit,
            target_recall=args.target_recall,
        )
    _atomic_json(args.output, report)
    print(json.dumps({"output": str(args.output.resolve()), "rows": len(report["rows"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
