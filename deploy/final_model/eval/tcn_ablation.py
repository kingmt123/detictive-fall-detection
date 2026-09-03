"""在同一 pose cache 上比较 rule-only、TCN-only 与凸组合融合。"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

# 与训练入口一致；必须在首次创建 CUDA/cuBLAS handle 前设置。
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from eval.metrics import competition_map
from models.tcn import FallTCN
from models.tcn_dataset import WindowMemmapCache
from pipeline.fusion import TemporalFallScorer
from pipeline.pose_cache import PoseCacheRecord, pose_cache_path, read_pose_cache
from pipeline.rules import compute_pose_features

ABLATION_PROTOCOL = "fall_tcn_clip_ablation_v1"


def configure_reproducible_inference() -> None:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    torch.use_deterministic_algorithms(True)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def load_trained_tcn(
    checkpoint_path: Path,
    run_path: Path,
    *,
    device: torch.device,
) -> tuple[FallTCN, dict[str, Any], dict[str, Any]]:
    """加载受 run signature 约束的本地可信 checkpoint。"""
    try:
        run = json.loads(Path(run_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("TCN run.json 无效") from exc
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    signature = run.get("signature_sha256")
    if not isinstance(signature, str) or checkpoint.get(
        "run_signature_sha256"
    ) != signature:
        raise ValueError("checkpoint 与 run signature 不匹配")
    if run.get("pilot"):
        raise ValueError("正式消融拒绝 pilot checkpoint")

    project_root = Path(__file__).parent.parent
    code_hashes = run.get("signature", {}).get("code_sha256", {})
    for relative_path, expected_sha256 in code_hashes.items():
        path = project_root / relative_path
        if not path.is_file() or sha256_file(path) != expected_sha256:
            raise ValueError(f"训练代码哈希漂移: {relative_path}")

    config = run["signature"]["config"]
    model = FallTCN(
        channels=tuple(config["channels"]),
        kernel=int(config["kernel"]),
        dropout=float(config["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, run, checkpoint


@torch.inference_mode()
def predict_window_probabilities(
    model: FallTCN,
    cache: WindowMemmapCache,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    if batch_size < 1:
        raise ValueError("batch_size 必须为正整数")
    features = cache.array("features")
    probabilities = np.empty(cache.sample_count, dtype=np.float32)
    for start in range(0, cache.sample_count, batch_size):
        end = min(start + batch_size, cache.sample_count)
        batch = torch.from_numpy(
            np.array(features[start:end], dtype=np.float32, copy=True)
        ).to(device)
        probabilities[start:end] = torch.sigmoid(model(batch)).cpu().numpy()
    return probabilities


def window_scores_to_clip_scores(
    probabilities: np.ndarray,
    clip_indices: np.ndarray,
    clip_count: int,
) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float32)
    clip_indices = np.asarray(clip_indices, dtype=np.int64)
    if probabilities.ndim != 1 or clip_indices.shape != probabilities.shape:
        raise ValueError("窗口概率与 clip index 必须是一维且等长")
    if clip_count < 1 or np.any((clip_indices < 0) | (clip_indices >= clip_count)):
        raise ValueError("clip index 越界")
    scores = np.full(clip_count, -np.inf, dtype=np.float32)
    np.maximum.at(scores, clip_indices, probabilities)
    scores[~np.isfinite(scores)] = 0.0
    return scores


def rule_score_record(record: PoseCacheRecord) -> float:
    """在已跟踪 pose 序列上复现在线规则的 clip 最大分数。"""
    clip_score, _ = rule_score_record_details(record)
    return clip_score


def rule_score_record_details(
    record: PoseCacheRecord,
) -> tuple[float, dict[tuple[int, int], float]]:
    """返回 clip 最大值及每个可用 ``(track_id, frame_index)`` 规则分数。"""
    scorers: dict[int, TemporalFallScorer] = {}
    previous_center_y: dict[int, float] = {}
    previous_detection_t: dict[int, float] = {}
    endpoint_scores: dict[tuple[int, int], float] = {}
    clip_score = 0.0
    frame_height = int(record.frame_size[0])
    for frame_index, timestamp_value in enumerate(record.timestamps):
        timestamp = float(timestamp_value)
        columns = np.flatnonzero(record.valid_mask[frame_index])
        for column_value in columns:
            column = int(column_value)
            track_id = int(record.track_ids[frame_index, column])
            scorer = scorers.setdefault(track_id, TemporalFallScorer())
            continuous = (
                track_id in previous_detection_t
                and timestamp - previous_detection_t[track_id] <= 0.5
            )
            features = compute_pose_features(
                record.keypoints[frame_index, column],
                record.bboxes[frame_index, column],
                frame_height=frame_height,
                previous_center_y=(
                    previous_center_y[track_id] if continuous else None
                ),
                delta_seconds=(
                    timestamp - previous_detection_t[track_id]
                    if continuous
                    else None
                ),
            )
            score = scorer.update(timestamp, features)
            if score is not None:
                previous_center_y[track_id] = features.center_y
                previous_detection_t[track_id] = timestamp
                clip_score = max(clip_score, score)
                endpoint_scores[(track_id, frame_index)] = score
    return clip_score, endpoint_scores


def rule_clip_scores(
    pose_cache_root: Path,
    cache: WindowMemmapCache,
    *,
    progress: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    metadata = cache.metadata
    expected_extractor = metadata["signature"]["extractor_signature"]
    clips = metadata["clips"]
    scores = np.empty(len(clips), dtype=np.float32)
    for index, clip in enumerate(clips):
        path = pose_cache_path(
            pose_cache_root,
            metadata["dataset"],
            metadata["split"],
            clip["clip_id"],
        )
        record = read_pose_cache(path)
        if (
            record.clip_id != clip["clip_id"]
            or record.dataset != metadata["dataset"]
            or record.split != metadata["split"]
        ):
            raise ValueError(f"pose cache 身份不匹配: {clip['clip_id']}")
        if record.extractor_signature != expected_extractor:
            raise ValueError(f"pose cache 提取签名不匹配: {clip['clip_id']}")
        scores[index] = rule_score_record(record)
        if progress is not None and ((index + 1) % 100 == 0 or index + 1 == len(clips)):
            progress(index + 1, len(clips))
    return scores


def aligned_rule_scores(
    pose_cache_root: Path,
    cache: WindowMemmapCache,
    *,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回纯规则 clip 分数、逐 TCN 窗口对齐分数和无 TCN 窗口的回退分数。"""
    metadata = cache.metadata
    expected_extractor = metadata["signature"]["extractor_signature"]
    clips = metadata["clips"]
    track_ids = cache.array("track_ids")
    end_times = cache.array("end_times")
    clip_scores = np.empty(len(clips), dtype=np.float32)
    window_scores = np.full(cache.sample_count, np.nan, dtype=np.float32)
    fallback_scores = np.zeros(len(clips), dtype=np.float32)
    for clip_index, clip in enumerate(clips):
        path = pose_cache_path(
            pose_cache_root,
            metadata["dataset"],
            metadata["split"],
            clip["clip_id"],
        )
        record = read_pose_cache(path)
        if (
            record.clip_id != clip["clip_id"]
            or record.dataset != metadata["dataset"]
            or record.split != metadata["split"]
        ):
            raise ValueError(f"pose cache 身份不匹配: {clip['clip_id']}")
        if record.extractor_signature != expected_extractor:
            raise ValueError(f"pose cache 提取签名不匹配: {clip['clip_id']}")
        clip_score, endpoint_scores = rule_score_record_details(record)
        clip_scores[clip_index] = clip_score
        start = int(clip["window_start"])
        end = start + int(clip["window_count"])
        window_keys: set[tuple[int, int]] = set()
        for window_index in range(start, end):
            frame_index = round(float(end_times[window_index]) * record.fps)
            key = (int(track_ids[window_index]), frame_index)
            window_keys.add(key)
            score = endpoint_scores.get(key)
            if score is not None:
                window_scores[window_index] = score
        fallback_scores[clip_index] = max(
            (
                score
                for key, score in endpoint_scores.items()
                if key not in window_keys
            ),
            default=0.0,
        )
        if progress is not None and (
            (clip_index + 1) % 100 == 0 or clip_index + 1 == len(clips)
        ):
            progress(clip_index + 1, len(clips))
    return clip_scores, window_scores, fallback_scores


def metric_summary(
    labels: dict[str, bool], scores: dict[str, float]
) -> dict[str, Any]:
    result = competition_map(labels, scores, mode="clip")

    def best_point(target: float) -> dict[str, Any] | None:
        feasible = [point for point in result["curve"] if point.recall >= target]
        if not feasible:
            return None
        point = max(feasible, key=lambda item: (item.precision, item.threshold))
        return {
            "threshold": point.threshold,
            "precision": point.precision,
            "recall": point.recall,
            "tp": point.tp,
            "fp": point.fp,
            "fn": point.fn,
        }

    return {
        "p_at_r90": float(result["p_at_r90"]),
        "p_at_r95": float(result["p_at_r95"]),
        "map": float(result["map"]),
        "map_percent": float(result["map_percent"]),
        "r90_point": best_point(0.90),
        "r95_point": best_point(0.95),
    }


def evaluate_score_ablation(
    clip_ids: Sequence[str],
    has_fall: Sequence[bool],
    rule_scores: np.ndarray,
    tcn_scores: np.ndarray,
    *,
    tcn_weights: Sequence[float],
) -> dict[str, Any]:
    if not (len(clip_ids) == len(has_fall) == len(rule_scores) == len(tcn_scores)):
        raise ValueError("clip、标签和两组分数必须等长")
    if len(set(clip_ids)) != len(clip_ids):
        raise ValueError("clip_id 必须唯一")
    weights = sorted({float(weight) for weight in tcn_weights})
    if not weights or any(not 0.0 <= weight <= 1.0 for weight in weights):
        raise ValueError("tcn_weights 必须位于 [0,1]")
    labels = dict(zip(clip_ids, map(bool, has_fall), strict=True))

    def as_dict(values: np.ndarray) -> dict[str, float]:
        return {
            clip_id: float(value)
            for clip_id, value in zip(clip_ids, values, strict=True)
        }

    rule_metrics = metric_summary(labels, as_dict(rule_scores))
    tcn_metrics = metric_summary(labels, as_dict(tcn_scores))
    fusion_grid = []
    for weight in weights:
        fused = (1.0 - weight) * rule_scores + weight * tcn_scores
        fusion_grid.append(
            {
                "tcn_weight": weight,
                "rule_weight": 1.0 - weight,
                "metrics": metric_summary(labels, as_dict(fused)),
            }
        )
    selected = max(
        fusion_grid,
        key=lambda item: (
            item["metrics"]["map"],
            item["metrics"]["p_at_r95"],
            item["metrics"]["p_at_r90"],
            -abs(item["tcn_weight"] - 0.5),
        ),
    )
    return {
        "rule_only": rule_metrics,
        "tcn_only": tcn_metrics,
        "fusion_grid": fusion_grid,
        "selected_fusion": selected,
        "predictions": [
            {
                "clip_id": clip_id,
                "has_fall": bool(label),
                "rule_score": float(rule_score),
                "tcn_score": float(tcn_score),
                "selected_fusion_score": float(
                    (1.0 - selected["tcn_weight"]) * rule_score
                    + selected["tcn_weight"] * tcn_score
                ),
            }
            for clip_id, label, rule_score, tcn_score in zip(
                clip_ids, has_fall, rule_scores, tcn_scores, strict=True
            )
        ],
    }


def evaluate_aligned_ablation(
    clip_ids: Sequence[str],
    has_fall: Sequence[bool],
    rule_clip_scores: np.ndarray,
    rule_window_scores: np.ndarray,
    fallback_rule_scores: np.ndarray,
    tcn_window_scores: np.ndarray,
    clip_indices: np.ndarray,
    *,
    tcn_weights: Sequence[float],
) -> dict[str, Any]:
    """按在线因果端点融合；某一路不可用时回退到另一路。"""
    clip_count = len(clip_ids)
    if not (
        clip_count == len(has_fall) == len(rule_clip_scores)
        == len(fallback_rule_scores)
    ):
        raise ValueError("clip、标签和规则分数必须等长")
    if rule_window_scores.shape != tcn_window_scores.shape:
        raise ValueError("规则与 TCN 窗口分数必须等长")
    tcn_clip_scores = window_scores_to_clip_scores(
        tcn_window_scores, clip_indices, clip_count
    )
    base = evaluate_score_ablation(
        clip_ids,
        has_fall,
        rule_clip_scores,
        tcn_clip_scores,
        tcn_weights=[0.0, 1.0],
    )
    weights = sorted({float(weight) for weight in tcn_weights})
    if not weights or any(not 0.0 <= weight <= 1.0 for weight in weights):
        raise ValueError("tcn_weights 必须位于 [0,1]")
    labels = dict(zip(clip_ids, map(bool, has_fall), strict=True))
    rule_available = np.isfinite(rule_window_scores)
    fusion_grid = []
    fused_by_weight: dict[float, np.ndarray] = {}
    for weight in weights:
        fused_windows = np.array(tcn_window_scores, dtype=np.float32, copy=True)
        fused_windows[rule_available] = (
            (1.0 - weight) * rule_window_scores[rule_available]
            + weight * tcn_window_scores[rule_available]
        )
        fused_clips = window_scores_to_clip_scores(
            fused_windows, clip_indices, clip_count
        )
        fused_clips = np.maximum(fused_clips, fallback_rule_scores)
        fused_by_weight[weight] = fused_clips
        scores = dict(zip(clip_ids, map(float, fused_clips), strict=True))
        fusion_grid.append(
            {
                "tcn_weight": weight,
                "rule_weight": 1.0 - weight,
                "metrics": metric_summary(labels, scores),
            }
        )
    selected = max(
        fusion_grid,
        key=lambda item: (
            item["metrics"]["map"],
            item["metrics"]["p_at_r95"],
            item["metrics"]["p_at_r90"],
            -abs(item["tcn_weight"] - 0.5),
        ),
    )
    selected_scores = fused_by_weight[selected["tcn_weight"]]
    return {
        "rule_only": base["rule_only"],
        "tcn_only": base["tcn_only"],
        "fusion_grid": fusion_grid,
        "selected_fusion": selected,
        "predictions": [
            {
                "clip_id": clip_id,
                "has_fall": bool(label),
                "rule_score": float(rule_score),
                "tcn_score": float(tcn_score),
                "selected_fusion_score": float(fused_score),
            }
            for clip_id, label, rule_score, tcn_score, fused_score in zip(
                clip_ids,
                has_fall,
                rule_clip_scores,
                tcn_clip_scores,
                selected_scores,
                strict=True,
            )
        ],
    }


def evaluation_signature(
    *,
    checkpoint_path: Path,
    run_path: Path,
    cache: WindowMemmapCache,
    weights: Sequence[float],
) -> tuple[str, dict[str, Any]]:
    root = Path(__file__).parent.parent
    payload = {
        "protocol": ABLATION_PROTOCOL,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "run_json_sha256": sha256_file(run_path),
        "window_signature_sha256": cache.metadata["signature_sha256"],
        "tcn_weights": list(weights),
        "code_sha256": {
            str(path.relative_to(root)): sha256_file(path)
            for path in (
                root / "eval" / "tcn_ablation.py",
                root / "eval" / "metrics.py",
                root / "pipeline" / "rules.py",
                root / "pipeline" / "fusion.py",
                root / "models" / "tcn.py",
            )
        },
    }
    return hashlib.sha256(_canonical_json(payload).encode()).hexdigest(), payload
