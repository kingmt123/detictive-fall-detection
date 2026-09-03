"""Replay visually confirmed secondary actors through existing six OOF ROI heads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from models.edgefall_f1 import EdgeFallF1Head
from models.tcn_dataset import load_window_cache
from pipeline.pose_cache import pose_cache_path, read_pose_cache
from pipeline.video_source import VideoSourceResolver
from tools.build_tcn_multistream_sidecar import load_sidecar
from tools.build_yolo_roi_token_cache import (
    SharedYoloRoiEncoder,
    YoloRoiCacheConfig,
    decode_yolo_frames,
    load_split_rows,
)
from tools.train_edgefall_f1 import (
    F1Config,
    build_skeleton_model,
    clip_embeddings,
    metric_result,
)
from tools.train_event_aligned_rgb_oracle import crop_intervals
from tools.train_tcn import _append_sidecar, _atomic_json, _materialize, _sha256_file


def _logit_mean(scores: list[float]) -> float:
    values = np.clip(np.asarray(scores, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    logits = np.log(values / (1.0 - values))
    return float(1.0 / (1.0 + np.exp(-logits.mean())))


def _folds(path: Path, clip_ids: set[str]) -> dict[str, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    result = {
        str(row["clip_id"]): int(row["fold"])
        for row in rows
        if str(row.get("clip_id")) in clip_ids
    }
    if set(result) != clip_ids:
        raise ValueError("fold map 缺少候选 clip")
    return result


def _feasibility_rows(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol") not in {
        "edgefall_top2_track_observability_audit_v1",
        "edgefall_top2_actor_confidence_trigger_full_train_audit_v1",
    }:
        raise ValueError("top-2 feasibility 协议不匹配")
    return {str(row["clip_id"]): row for row in payload.get("rows", [])}


def _passes_confidence_trigger(row: dict[str, Any]) -> bool:
    audit = row["track_audit"]
    second = audit.get("second")
    return bool(
        second is not None
        and int(second["observed_frames"]) >= 24
        and float(audit["second_to_primary_window_ratio"]) >= 0.75
        and float(second["mean_joint_confidence"])
        >= float(audit["primary"]["mean_joint_confidence"]) + 0.20
    )


def _secondary_tracks(
    rows: dict[str, dict[str, Any]], clip_ids: set[str]
) -> dict[str, int]:
    result = {
        clip_id: int(rows[clip_id]["track_audit"]["second"]["track_id"])
        for clip_id in clip_ids
        if clip_id in rows
    }
    if set(result) != clip_ids:
        raise ValueError("top-2 feasibility 缺少候选 clip")
    return result


@torch.inference_mode()
def _alternate_roi_tokens(
    *,
    rows: list[dict[str, str]],
    indices: list[int],
    intervals: list[tuple[float, float] | None],
    tracks: dict[str, int],
    pose_cache_root: Path,
    yolo_checkpoint: Path,
    input_size: int,
    device: torch.device,
    temp_root: Path,
) -> dict[str, np.ndarray]:
    config = YoloRoiCacheConfig(input_size=input_size)
    encoder = SharedYoloRoiEncoder(yolo_checkpoint, device=device, config=config)
    result: dict[str, np.ndarray] = {}
    try:
        with VideoSourceResolver(temp_root) as resolver:
            for index in indices:
                row = rows[index]
                clip_id = row["clip_id"]
                record = read_pose_cache(
                    pose_cache_path(pose_cache_root, "of-syn", "train", clip_id)
                )
                with resolver.materialize(row["video_path"]) as video:
                    frames, boxes, _quality = decode_yolo_frames(
                        video.local_path,
                        record=record,
                        track_id=tracks[clip_id],
                        interval=intervals[index],
                        config=config,
                    )
                tokens = encoder.encode(frames, boxes).reshape(
                    1, config.frames, 2, -1
                )
                result[clip_id] = tokens[0, :, 0].numpy()
    finally:
        encoder.close()
    return result


@torch.inference_mode()
def _skeleton_embeddings(
    *,
    cache: Any,
    sidecar: Any,
    clip_indices: dict[str, int],
    folds: dict[str, int],
    member_report: dict[str, Any],
    device: torch.device,
) -> dict[str, np.ndarray]:
    first_member = member_report["members"][0]
    result: dict[str, np.ndarray] = {}
    for fold in sorted(set(folds.values())):
        ids = sorted(clip_id for clip_id, value in folds.items() if value == fold)
        checkpoint_path = Path(first_member["oof"][fold]["path"]).with_name("last.pt")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = F1Config(**checkpoint["config"])
        model = build_skeleton_model(config).to(device)
        model.load_state_dict(checkpoint["skeleton_model_state"], strict=True)
        windows = []
        for clip_id in ids:
            clip = cache.metadata["clips"][clip_indices[clip_id]]
            start = int(clip["window_start"])
            windows.extend(range(start, start + int(clip["window_count"])))
        selected = np.asarray(windows, dtype=np.int64)
        features, _labels, original_clips = _materialize(cache, selected)
        features = _append_sidecar(features, sidecar, selected)
        pooled_clips, pooled = clip_embeddings(
            model, features, original_clips, device=device, batch_size=2048
        )
        by_index = {int(index): value for index, value in zip(pooled_clips, pooled, strict=True)}
        for clip_id in ids:
            result[clip_id] = by_index[clip_indices[clip_id]]
    return result


def _load_primary_tokens(path: Path, clip_ids: set[str]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        ids = [str(value) for value in payload["clip_ids"]]
        tokens = np.asarray(payload["tokens"], dtype=np.float32)
    index = {clip_id: offset for offset, clip_id in enumerate(ids)}
    if not clip_ids.issubset(index):
        raise ValueError("primary ROI cache 缺少候选 clip")
    return {clip_id: tokens[index[clip_id], :, 0] for clip_id in clip_ids}


@torch.inference_mode()
def _score_head(
    checkpoint_path: Path,
    skeleton: np.ndarray,
    roi: np.ndarray,
    *,
    device: torch.device,
) -> float:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    roi_dim = int(checkpoint.get("roi_token_dim") or roi.shape[1])
    model = EdgeFallF1Head(
        skeleton.shape[0],
        roi_dim=roi_dim,
        roi_token_layout=str(checkpoint.get("roi_token_layout") or "direct"),
        roi_temporal_pooling=str(checkpoint.get("roi_temporal_pooling") or "mean_max"),
    ).to(device)
    model.load_state_dict(checkpoint["fusion_state"], strict=True)
    model.eval()
    logits, _subtype = model(
        torch.from_numpy(skeleton[None]).to(device),
        torch.from_numpy(roi[None]).to(device),
    )
    return float(torch.sigmoid(logits)[0].cpu())


def _oof_score(path: Path, clip_id: str) -> float:
    with np.load(path, allow_pickle=False) as payload:
        ids = [str(value) for value in payload["clip_id"]]
        scores = np.asarray(payload["score"], dtype=np.float64)
    matches = np.flatnonzero(np.asarray(ids) == clip_id)
    if matches.size != 1:
        raise ValueError(f"OOF 中 clip 非唯一: {clip_id}")
    return float(scores[int(matches[0])])


def _global_policy_metrics(
    member_report: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    meta_path = Path(member_report["meta_oof_predictions"]["path"])
    with np.load(meta_path, allow_pickle=False) as payload:
        clip_ids = [str(value) for value in payload["clip_id"]]
        labels = np.asarray(payload["label"], dtype=np.float32)
        expected_baseline = np.asarray(
            payload["fixed_equal_logit_average"], dtype=np.float64
        )
    index = {clip_id: offset for offset, clip_id in enumerate(clip_ids)}
    members = member_report["members"]
    score_matrix = np.empty((len(clip_ids), len(members)), dtype=np.float64)
    for member_index, member in enumerate(members):
        scores: dict[str, float] = {}
        for fold in member["oof"]:
            with np.load(Path(fold["path"]), allow_pickle=False) as payload:
                ids = [str(value) for value in payload["clip_id"]]
                values = np.asarray(payload["score"], dtype=np.float64)
            scores.update(zip(ids, values, strict=True))
        if set(scores) != set(index):
            raise ValueError("member OOF clip 集合不一致")
        score_matrix[:, member_index] = [scores[clip_id] for clip_id in clip_ids]
    clipped = np.clip(score_matrix, 1e-7, 1.0 - 1e-7)
    baseline = 1.0 / (1.0 + np.exp(-np.log(clipped / (1.0 - clipped)).mean(1)))
    baseline_error = float(np.max(np.abs(baseline - expected_baseline)))
    if baseline_error > 3e-6:
        raise ValueError(f"全局六头 baseline 重建不一致: {baseline_error}")
    challenger_matrix = score_matrix.copy()
    for row in rows:
        offset = index[row["clip_id"]]
        challenger_matrix[offset] = [
            member["alternate_actor_score"] for member in row["members"]
        ]
    clipped = np.clip(challenger_matrix, 1e-7, 1.0 - 1e-7)
    challenger = 1.0 / (
        1.0 + np.exp(-np.log(clipped / (1.0 - clipped)).mean(1))
    )
    baseline_metrics = metric_result(clip_ids, labels, baseline)
    challenger_metrics = metric_result(clip_ids, labels, challenger)
    return {
        "baseline": baseline_metrics,
        "challenger": challenger_metrics,
        "delta": {
            key: challenger_metrics[key] - baseline_metrics[key]
            for key in ("clip_map_percent", "clip_p_at_r90", "clip_p_at_r95")
        },
        "baseline_reconstruction_max_abs_error": baseline_error,
        "changed_clips": len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--pose-cache-root", type=Path, required=True)
    parser.add_argument("--yolo-checkpoint", type=Path, required=True)
    parser.add_argument("--roi-320", type=Path, required=True)
    parser.add_argument("--roi-640", type=Path, required=True)
    parser.add_argument("--fold-map", type=Path, required=True)
    parser.add_argument("--feasibility-report", type=Path, required=True)
    parser.add_argument("--member-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--temp-root", type=Path, required=True)
    parser.add_argument("--clip-id", action="append", default=[])
    parser.add_argument(
        "--selection-mode",
        choices=("visual_oracle", "confidence_trigger"),
        default="visual_oracle",
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("oracle replay 输出已存在，拒绝覆盖")
    feasibility_payload = json.loads(args.feasibility_report.read_text(encoding="utf-8"))
    feasibility_rows = _feasibility_rows(args.feasibility_report)
    full_train_trigger = (
        feasibility_payload.get("protocol")
        == "edgefall_top2_actor_confidence_trigger_full_train_audit_v1"
    )
    clip_ids = set(args.clip_id)
    if args.selection_mode == "confidence_trigger" and not clip_ids:
        clip_ids = {
            clip_id
            for clip_id, row in feasibility_rows.items()
            if _passes_confidence_trigger(row)
        }
    if not clip_ids:
        raise ValueError("至少需要一个候选 clip")
    if args.clip_id and len(clip_ids) != len(args.clip_id):
        raise ValueError("clip-id 不得重复")
    if args.selection_mode == "confidence_trigger" and any(
        clip_id not in feasibility_rows
        or not _passes_confidence_trigger(feasibility_rows[clip_id])
        for clip_id in clip_ids
    ):
        raise ValueError("显式 clip 不满足固定 confidence trigger")
    cache = load_window_cache(args.cache, verify_hashes=True)
    if cache.metadata.get("dataset") != "of-syn" or cache.metadata.get("split") != "train":
        raise ValueError("oracle replay 只接受 OF-Syn train cache")
    sidecar = load_sidecar(args.sidecar, cache)
    rows = load_split_rows(args.manifest, cache, split="train")
    clip_indices = {row["clip_id"]: index for index, row in enumerate(rows)}
    if not clip_ids.issubset(clip_indices):
        raise ValueError("候选 clip 不在 train cache")
    folds = _folds(args.fold_map, clip_ids)
    tracks = _secondary_tracks(feasibility_rows, clip_ids)
    member_report = json.loads(args.member_report.read_text(encoding="utf-8"))
    members = member_report.get("members")
    if not isinstance(members, list) or len(members) != 6:
        raise ValueError("member report 必须包含六个 OOF heads")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("请求 CUDA，但当前不可用")
    intervals = crop_intervals(cache, context_seconds=1.0)
    selected_indices = [clip_indices[clip_id] for clip_id in sorted(clip_ids)]
    alternate = {
        320: _alternate_roi_tokens(
            rows=rows, indices=selected_indices, intervals=intervals, tracks=tracks,
            pose_cache_root=args.pose_cache_root, yolo_checkpoint=args.yolo_checkpoint,
            input_size=320, device=device, temp_root=args.temp_root / "roi320",
        ),
        640: _alternate_roi_tokens(
            rows=rows, indices=selected_indices, intervals=intervals, tracks=tracks,
            pose_cache_root=args.pose_cache_root, yolo_checkpoint=args.yolo_checkpoint,
            input_size=640, device=device, temp_root=args.temp_root / "roi640",
        ),
    }
    primary = {
        320: _load_primary_tokens(args.roi_320, clip_ids),
        640: _load_primary_tokens(args.roi_640, clip_ids),
    }
    skeleton = _skeleton_embeddings(
        cache=cache, sidecar=sidecar, clip_indices=clip_indices, folds=folds,
        member_report=member_report, device=device,
    )
    rows_out = []
    maximum_replay_error = 0.0
    for clip_id in sorted(clip_ids):
        baseline_scores = []
        alternate_scores = []
        member_rows = []
        fold = folds[clip_id]
        for member in members:
            name = str(member["name"])
            resolution = 640 if "640" in name else 320
            oof_path = Path(member["oof"][fold]["path"])
            checkpoint_path = oof_path.with_name("last.pt")
            baseline = _score_head(
                checkpoint_path, skeleton[clip_id], primary[resolution][clip_id], device=device
            )
            expected = _oof_score(oof_path, clip_id)
            replay_error = abs(baseline - expected)
            maximum_replay_error = max(maximum_replay_error, replay_error)
            challenger = _score_head(
                checkpoint_path, skeleton[clip_id], alternate[resolution][clip_id], device=device
            )
            baseline_scores.append(expected)
            alternate_scores.append(challenger)
            member_rows.append(
                {
                    "name": name,
                    "resolution": resolution,
                    "baseline_oof_score": expected,
                    "baseline_replay_score": baseline,
                    "baseline_replay_abs_error": replay_error,
                    "alternate_actor_score": challenger,
                    "delta": challenger - expected,
                    "checkpoint_sha256": _sha256_file(checkpoint_path),
                }
            )
        baseline_fused = _logit_mean(baseline_scores)
        alternate_fused = _logit_mean(alternate_scores)
        rows_out.append(
            {
                "clip_id": clip_id,
                "review_set": str(feasibility_rows[clip_id]["review_set"]),
                "fold": fold,
                "secondary_track": tracks[clip_id],
                "members": member_rows,
                "six_head_equal_logit_baseline": baseline_fused,
                "six_head_equal_logit_alternate_actor": alternate_fused,
                "six_head_delta": alternate_fused - baseline_fused,
            }
        )
    if maximum_replay_error > 3e-5:
        raise ValueError(f"primary OOF replay 不一致: max_abs_error={maximum_replay_error}")
    grouped = {}
    for review_set in sorted({row["review_set"] for row in rows_out}):
        selected = [row for row in rows_out if row["review_set"] == review_set]
        grouped[review_set] = {
            "clips": len(selected),
            "positive_deltas": sum(row["six_head_delta"] > 0 for row in selected),
            "mean_delta": float(np.mean([row["six_head_delta"] for row in selected])),
        }
    visual_oracle = args.selection_mode == "visual_oracle"
    report = {
        "protocol": (
            "edgefall_top2_actor_visual_oracle_six_head_replay_v1"
            if visual_oracle
            else "edgefall_top2_actor_confidence_trigger_error_canary_v1"
        ),
        "qualification": (
            "train_oof_visual_oracle_not_deployment_policy"
            if visual_oracle
            else "train_oof_error_set_directional_canary_not_global_policy_evaluation"
        ),
        "selection_mode": args.selection_mode,
        "confidence_trigger": (
            {
                "second_observed_frames_min": 24,
                "second_to_primary_window_ratio_min": 0.75,
                "second_minus_primary_mean_joint_confidence_min": 0.20,
            }
            if not visual_oracle
            else None
        ),
        "rows": rows_out,
        "summary": {
            "clips": len(rows_out),
            "positive_six_head_deltas": sum(row["six_head_delta"] > 0 for row in rows_out),
            "mean_six_head_delta": float(np.mean([row["six_head_delta"] for row in rows_out])),
            "minimum_six_head_delta": float(min(row["six_head_delta"] for row in rows_out)),
            "maximum_primary_replay_abs_error": maximum_replay_error,
            "by_review_set": grouped,
        },
        "test_accessed": False,
        "limitations": [
            (
                "the actors were selected from known low-score false negatives and visually confirmed"
                if visual_oracle
                else "the fixed trigger is evaluated only inside an existing labeled error-review set"
            ),
            "this measures recoverability after correct actor selection, not an automatic selection rule",
            "only the existing six short-window ROI heads are replayed; the dense seventh head is unchanged",
        ],
    }
    if full_train_trigger:
        report["qualification"] = "train_oof_global_fixed_policy_canary"
        report["global_policy_metrics"] = _global_policy_metrics(
            member_report, rows_out
        )
    _atomic_json(args.output, report)
    print(json.dumps(report["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
