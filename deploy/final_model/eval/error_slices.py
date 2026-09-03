"""Gate 3C：在冻结高召回阈值上生成可审计的行为错误切片。"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from eval.tcn_ablation import sha256_file

ERROR_SLICE_PROTOCOL = "gate3c_high_recall_error_slices_v1"
MODE_SPECS = {
    "rule_only": "rule_score",
    "tcn_only": "tcn_score",
    "selected_fusion": "selected_fusion_score",
}


def activity_from_clip_id(clip_id: str) -> str:
    """OF-Syn 的顶层目录就是互斥行为标签。"""
    activity, separator, remainder = clip_id.partition("/")
    if not separator or not activity or not remainder:
        raise ValueError(f"clip_id 不符合 activity/name 格式: {clip_id!r}")
    return activity


def _safe_rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _point_slice(
    predictions: list[dict[str, Any]], *, score_key: str, threshold: float
) -> dict[str, Any]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    errors: dict[str, list[dict[str, Any]]] = {"false_positives": [], "false_negatives": []}
    total = Counter()
    for item in predictions:
        clip_id = str(item["clip_id"])
        activity = activity_from_clip_id(clip_id)
        label = bool(item["has_fall"])
        score = float(item[score_key])
        predicted = score >= threshold
        outcome = "tp" if label and predicted else "fn" if label else "fp" if predicted else "tn"
        counts[activity][outcome] += 1
        total[outcome] += 1
        if outcome == "fp":
            errors["false_positives"].append(
                {"clip_id": clip_id, "activity": activity, "score": score, "margin": score - threshold}
            )
        elif outcome == "fn":
            errors["false_negatives"].append(
                {"clip_id": clip_id, "activity": activity, "score": score, "margin": score - threshold}
            )

    by_activity: dict[str, Any] = {}
    for activity in sorted(counts):
        row = counts[activity]
        tp, fp, fn, tn = (row[key] for key in ("tp", "fp", "fn", "tn"))
        by_activity[activity] = {
            "count": tp + fp + fn + tn,
            "positive": tp + fn,
            "negative": fp + tn,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": _safe_rate(tp, tp + fp),
            "recall": _safe_rate(tp, tp + fn),
            "false_positive_rate": _safe_rate(fp, fp + tn),
        }
    errors["false_positives"].sort(key=lambda item: (-item["score"], item["clip_id"]))
    errors["false_negatives"].sort(key=lambda item: (item["score"], item["clip_id"]))
    tp, fp, fn, tn = (total[key] for key in ("tp", "fp", "fn", "tn"))
    return {
        "threshold": threshold,
        "overall": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": _safe_rate(tp, tp + fp),
            "recall": _safe_rate(tp, tp + fn),
        },
        "by_activity": by_activity,
        **errors,
    }


def _pairwise_transitions(
    predictions: list[dict[str, Any]], *, point_name: str, thresholds: dict[str, float]
) -> dict[str, Any]:
    rows = []
    counts = Counter()
    for item in predictions:
        label = bool(item["has_fall"])
        tcn_positive = float(item["tcn_score"]) >= thresholds["tcn_only"]
        fusion_positive = (
            float(item["selected_fusion_score"]) >= thresholds["selected_fusion"]
        )
        if tcn_positive == fusion_positive:
            continue
        if fusion_positive == label and tcn_positive != label:
            transition = "fusion_corrects_tcn"
        else:
            transition = "fusion_harms_tcn"
        error_type = "positive_recall" if label else "negative_specificity"
        counts[f"{transition}:{error_type}"] += 1
        rows.append(
            {
                "clip_id": item["clip_id"],
                "activity": activity_from_clip_id(str(item["clip_id"])),
                "has_fall": label,
                "tcn_score": float(item["tcn_score"]),
                "fusion_score": float(item["selected_fusion_score"]),
                "transition": transition,
                "axis": error_type,
            }
        )
    rows.sort(key=lambda item: (item["transition"], item["activity"], item["clip_id"]))
    return {"point": point_name, "counts": dict(sorted(counts.items())), "clips": rows}


def build_error_slices(ablation: dict[str, Any]) -> dict[str, Any]:
    if ablation.get("split") != "val":
        raise ValueError("Gate 3C 错误切片只允许冻结 validation 结果")
    predictions = ablation.get("predictions")
    if not isinstance(predictions, list) or not predictions:
        raise ValueError("消融结果缺少 predictions")
    if len({item.get("clip_id") for item in predictions}) != len(predictions):
        raise ValueError("predictions 中 clip_id 必须唯一")

    modes: dict[str, Any] = {}
    threshold_sets: dict[str, dict[str, float]] = {"r90": {}, "r95": {}}
    for mode, score_key in MODE_SPECS.items():
        metric_container = ablation[mode]
        metrics = metric_container["metrics"] if mode == "selected_fusion" else metric_container
        modes[mode] = {}
        for short_name, metric_name in (("r90", "r90_point"), ("r95", "r95_point")):
            point = metrics.get(metric_name)
            if not isinstance(point, dict) or "threshold" not in point:
                raise ValueError(f"{mode} 缺少 {metric_name} 冻结阈值")
            threshold = float(point["threshold"])
            threshold_sets[short_name][mode] = threshold
            sliced = _point_slice(predictions, score_key=score_key, threshold=threshold)
            expected = {key: int(point[key]) for key in ("tp", "fp", "fn")}
            actual = {key: sliced["overall"][key] for key in expected}
            if actual != expected:
                raise ValueError(f"{mode}/{short_name} 重算与消融产物不一致: {actual} != {expected}")
            modes[mode][short_name] = sliced

    return {
        "protocol": ERROR_SLICE_PROTOCOL,
        "dataset": ablation.get("dataset"),
        "split": "val",
        "clip_count": len(predictions),
        "positive_clips": sum(bool(item["has_fall"]) for item in predictions),
        "activity_counts": dict(sorted(Counter(activity_from_clip_id(str(item["clip_id"])) for item in predictions).items())),
        "modes": modes,
        "tcn_vs_fusion": {
            point: _pairwise_transitions(predictions, point_name=point, thresholds=thresholds)
            for point, thresholds in threshold_sets.items()
        },
        "limitations": [
            "thresholds are independently frozen for each model on the same full validation split",
            "activity is inferred from the OF-Syn clip_id top-level directory",
            "test split is neither read nor evaluated",
        ],
    }


def signed_error_slices(input_path: Path) -> dict[str, Any]:
    input_path = Path(input_path)
    try:
        ablation = json.loads(input_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("消融结果不是合法 JSON") from exc
    result = build_error_slices(ablation)
    code_path = Path(__file__)
    signature = {
        "protocol": ERROR_SLICE_PROTOCOL,
        "input_sha256": sha256_file(input_path),
        "input_signature_sha256": ablation.get("signature_sha256"),
        "implementation_sha256": sha256_file(code_path),
    }
    canonical = json.dumps(signature, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    result["signature"] = signature
    result["signature_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return result
