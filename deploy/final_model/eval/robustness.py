"""Gate 3C 图像扰动与固定阈值稳定性评估工具。"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from eval.metrics import competition_map
from eval.tcn_ablation import sha256_file

ROBUSTNESS_PROTOCOL = "gate3c_image_robustness_v1"
CONDITIONS = ("clean", "grayscale", "lowlight", "gaussian_noise", "center_occlusion")


def deterministic_rank(text: str, *, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{text}".encode()).hexdigest()


def load_stratified_val_rows(
    manifest_path: Path, *, dataset: str, per_activity: int, seed: int
) -> list[dict[str, str]]:
    if per_activity < 1:
        raise ValueError("per_activity 必须为正整数")
    with Path(manifest_path).open(encoding="utf-8", newline="") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if row.get("dataset") == dataset and row.get("split") == "val"
        ]
    if not rows:
        raise ValueError("manifest 中没有目标 validation 数据")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    seen: set[str] = set()
    for row in rows:
        clip_id = row.get("clip_id", "")
        if not clip_id or clip_id in seen:
            raise ValueError("validation clip_id 必须非空且唯一")
        seen.add(clip_id)
        activity, separator, _ = clip_id.partition("/")
        if not separator:
            raise ValueError(f"clip_id 不符合 activity/name 格式: {clip_id}")
        grouped[activity].append(row)
    selected = []
    for activity in sorted(grouped):
        ordered = sorted(
            grouped[activity],
            key=lambda row: (
                deterministic_rank(row["clip_id"], seed=seed), row["clip_id"]
            ),
        )
        if len(ordered) < per_activity:
            raise ValueError(f"{activity} 不足 {per_activity} 条")
        selected.extend(ordered[:per_activity])
    return selected


def frame_transform(
    condition: str, *, seed: int
) -> Callable[[np.ndarray, int], np.ndarray]:
    if condition not in CONDITIONS:
        raise ValueError(f"未知 robustness condition: {condition}")

    def transform(frame: np.ndarray, frame_index: int) -> np.ndarray:
        source = np.asarray(frame)
        if condition == "clean":
            return source.copy()
        if condition == "grayscale":
            gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if condition == "lowlight":
            normalized = source.astype(np.float32) / 255.0
            return np.clip(np.power(normalized, 2.2) * 255.0, 0, 255).astype(np.uint8)
        if condition == "gaussian_noise":
            rng = np.random.default_rng(seed + frame_index * 1_000_003)
            noise = rng.normal(0.0, 15.0, source.shape)
            return np.clip(source.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        result = source.copy()
        height, width = result.shape[:2]
        cut_h, cut_w = max(1, round(height * 0.30)), max(1, round(width * 0.30))
        y1, x1 = (height - cut_h) // 2, (width - cut_w) // 2
        result[y1 : y1 + cut_h, x1 : x1 + cut_w] = 0
        return result

    return transform


def summarize_fixed_thresholds(
    rows: Sequence[dict[str, Any]], thresholds: dict[str, dict[str, float]]
) -> dict[str, Any]:
    modes = ("rule_only", "tcn_only", "selected_fusion")
    score_keys = {
        "rule_only": "rule_score",
        "tcn_only": "tcn_score",
        "selected_fusion": "fusion_score",
    }
    clean_scores = {
        (str(row["clip_id"]), mode): float(row[score_keys[mode]])
        for row in rows if row["condition"] == "clean" for mode in modes
    }
    by_condition: dict[str, Any] = {}
    for condition in CONDITIONS:
        subset = [row for row in rows if row["condition"] == condition]
        if not subset:
            raise ValueError(f"缺少 condition={condition} 结果")
        labels = {str(row["clip_id"]): bool(row["has_fall"]) for row in subset}
        condition_result: dict[str, Any] = {}
        for mode in modes:
            key = score_keys[mode]
            scores = {str(row["clip_id"]): float(row[key]) for row in subset}
            metric = competition_map(labels, scores, mode="clip")
            drifts = [scores[clip] - clean_scores[(clip, mode)] for clip in scores]
            mode_result: dict[str, Any] = {
                "subset_map": float(metric["map"]),
                "score_drift": {
                    "mean_absolute": float(np.mean(np.abs(drifts))),
                    "mean_signed": float(np.mean(drifts)),
                    "max_absolute": float(np.max(np.abs(drifts))),
                },
            }
            for point in ("r90", "r95"):
                threshold = float(thresholds[mode][point])
                tp = fp = fn = tn = positive_flips = negative_flips = 0
                for clip_id, score in scores.items():
                    label = labels[clip_id]
                    predicted = score >= threshold
                    clean_predicted = clean_scores[(clip_id, mode)] >= threshold
                    tp += int(label and predicted)
                    fn += int(label and not predicted)
                    fp += int(not label and predicted)
                    tn += int(not label and not predicted)
                    positive_flips += int(label and predicted != clean_predicted)
                    negative_flips += int(not label and predicted != clean_predicted)
                mode_result[point] = {
                    "threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                    "precision": tp / (tp + fp) if tp + fp else None,
                    "recall": tp / (tp + fn) if tp + fn else None,
                    "positive_prediction_flips_vs_clean": positive_flips,
                    "negative_prediction_flips_vs_clean": negative_flips,
                }
            condition_result[mode] = mode_result
        by_condition[condition] = condition_result
    return by_condition


def robustness_signature(
    *, manifest_path: Path, ablation_path: Path, checkpoint_path: Path,
    run_path: Path, per_activity: int, seed: int,
) -> tuple[str, dict[str, Any]]:
    signature = {
        "protocol": ROBUSTNESS_PROTOCOL,
        "manifest_sha256": sha256_file(manifest_path),
        "ablation_sha256": sha256_file(ablation_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "run_json_sha256": sha256_file(run_path),
        "per_activity": per_activity,
        "seed": seed,
        "conditions": list(CONDITIONS),
        "condition_parameters": {
            "lowlight_gamma": 2.2,
            "gaussian_noise_sigma_uint8": 15.0,
            "center_occlusion_fraction_hw": [0.30, 0.30],
        },
        "implementation_sha256": sha256_file(Path(__file__)),
    }
    canonical = json.dumps(signature, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest(), signature


def activity_counts(rows: Sequence[dict[str, str]]) -> dict[str, int]:
    return dict(sorted(Counter(row["clip_id"].split("/", 1)[0] for row in rows).items()))
