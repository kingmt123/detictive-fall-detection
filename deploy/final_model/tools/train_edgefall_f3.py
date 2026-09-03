"""Freeze passed EdgeFall F1 and evaluate quality-gated person-ROI cross attention."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from models.edgefall_f1 import EdgeFallF1Head, EdgeFallF3QualityCrossHead
from models.tcn_dataset import load_window_cache
from tools.build_tcn_multistream_sidecar import load_sidecar
from tools.train_edgefall_f1 import (
    F1Config,
    balanced_focal_loss,
    build_skeleton_model,
    clip_embeddings,
    metric_result,
    subtype_targets,
    template_fold_ids,
)
from tools.train_edgefall_f2 import high_recall_false_positives
from tools.train_multiscale_multistream_tcn import _activity_groups
from tools.train_tcn import (
    _append_sidecar,
    _atomic_checkpoint,
    _atomic_json,
    _materialize,
    _sha256_file,
    select_training_indices,
    set_deterministic,
)


def pose_error_targets(quality: torch.Tensor) -> torch.Tensor:
    """Deterministic proxy for pose/track failure, derived from cached track quality."""
    if quality.ndim != 2 or quality.shape[1] != 5:
        raise ValueError("quality 必须为 (B,5)")
    return (
        (quality[:, 0] < 0.95)
        | (quality[:, 1] < 0.75)
        | (quality[:, 2] < 0.5)
        | (quality[:, 4] > 0.025)
    ).float()


@torch.inference_mode()
def _scores(
    model: nn.Module,
    skeleton: torch.Tensor,
    person: torch.Tensor,
    quality: torch.Tensor | None,
    indices: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    values = []
    for start in range(0, indices.size, batch_size):
        selected = indices[start : start + batch_size]
        if quality is None:
            logits, _ = model(skeleton[selected].to(device), person[selected].to(device))
        else:
            logits, _, _ = model(
                skeleton[selected].to(device),
                person[selected].to(device),
                quality[selected].to(device),
            )
        values.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(values)


def pose_error_false_positives(
    labels: np.ndarray, scores: np.ndarray, quality: np.ndarray, threshold: float
) -> int:
    error = pose_error_targets(torch.from_numpy(quality)).numpy() > 0.5
    return int(((np.asarray(labels) < 0.5) & (np.asarray(scores) >= threshold) & error).sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--roi-cache", type=Path, required=True)
    parser.add_argument("--f1-checkpoint", type=Path, required=True)
    parser.add_argument("--f1-oof", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args()
    if args.epochs < 1:
        raise ValueError("epochs 必须为正")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("F3 输出目录必须为空")
    cache = load_window_cache(args.cache, verify_hashes=True)
    if cache.metadata.get("dataset") != "of-syn" or cache.metadata.get("split") != "train":
        raise ValueError("F3 只接受 OF-Syn train cache")
    sidecar = load_sidecar(args.sidecar, cache)
    checkpoint = torch.load(args.f1_checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError("F1 checkpoint 无效")
    config = F1Config(**checkpoint["config"])
    clips = cache.metadata["clips"]
    fold_ids = template_fold_ids(clips, config)
    if config.fold != 0:
        fold_ids = (fold_ids - config.fold) % config.folds
    all_labels = np.asarray(cache.array("labels"))
    all_clip_indices = np.asarray(cache.array("clip_indices"), dtype=np.int64)
    activities = _activity_groups(cache)
    candidate_windows = np.flatnonzero(fold_ids[all_clip_indices] != 0)
    local = select_training_indices(
        all_labels[candidate_windows],
        negative_ratio=config.negative_ratio,
        seed=config.seed,
        activity_groups=activities[candidate_windows],
        hard_negative_activities=("lie_down", "lying", "stand_up"),
        hard_negative_fraction=0.4,
    )
    selected_windows = candidate_windows[local]
    train_x, _, train_window_clips = _materialize(cache, selected_windows)
    train_x = _append_sidecar(train_x, sidecar, selected_windows)
    heldout_windows = np.flatnonzero(fold_ids[all_clip_indices] == 0)
    heldout_x, _, heldout_window_clips = _materialize(cache, heldout_windows)
    heldout_x = _append_sidecar(heldout_x, sidecar, heldout_windows)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("请求 CUDA，但当前不可用")
    set_deterministic(args.seed)
    skeleton_model = build_skeleton_model(config).to(device)
    skeleton_model.load_state_dict(checkpoint["skeleton_model_state"], strict=True)
    for parameter in skeleton_model.parameters():
        parameter.requires_grad_(False)
    train_clip_indices, train_embeddings = clip_embeddings(
        skeleton_model, train_x, train_window_clips, device=device, batch_size=config.batch_size
    )
    heldout_clip_indices, heldout_embeddings = clip_embeddings(
        skeleton_model, heldout_x, heldout_window_clips, device=device, batch_size=config.batch_size
    )
    selected_clip_indices = np.concatenate((train_clip_indices, heldout_clip_indices))
    skeleton = torch.from_numpy(np.concatenate((train_embeddings, heldout_embeddings)))
    with np.load(args.roi_cache, allow_pickle=False) as payload:
        roi_ids = [str(value) for value in payload["clip_ids"]]
        roi_tokens = np.asarray(payload["tokens"], dtype=np.float32)
        roi_quality = np.asarray(payload["quality"], dtype=np.float32)
    roi_index = {clip_id: index for index, clip_id in enumerate(roi_ids)}
    clip_ids = [str(clips[int(index)]["clip_id"]) for index in selected_clip_indices]
    if any(clip_id not in roi_index for clip_id in clip_ids):
        raise ValueError("ROI cache 缺少 F3 clip")
    person = torch.from_numpy(np.stack([roi_tokens[roi_index[item], :, 0] for item in clip_ids]))
    quality = torch.from_numpy(np.stack([roi_quality[roi_index[item]] for item in clip_ids]))
    labels = torch.tensor([float(clips[int(index)]["has_fall"]) for index in selected_clip_indices])
    subtypes = torch.from_numpy(subtype_targets(clips, selected_clip_indices))
    train_count = train_clip_indices.size
    train_indices = np.arange(train_count, dtype=np.int64)
    heldout_indices = np.arange(train_count, selected_clip_indices.size, dtype=np.int64)
    base = EdgeFallF1Head(int(skeleton.shape[1])).to(device)
    base.load_state_dict(checkpoint["fusion_state"], strict=True)
    f3 = EdgeFallF3QualityCrossHead(base, int(skeleton.shape[1])).to(device)
    subtype_counts = torch.bincount(subtypes[train_indices], minlength=6).float()
    subtype_criterion = nn.CrossEntropyLoss(weight=(subtype_counts.sum() / subtype_counts).to(device))
    quality_criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in f3.parameters() if parameter.requires_grad),
        lr=3e-4, weight_decay=1e-3,
    )
    for epoch in range(args.epochs):
        f3.train()
        order = np.random.default_rng(args.seed + epoch).permutation(train_indices)
        loss_sum = 0.0
        for start in range(0, order.size, config.head_batch_size):
            selected = order[start : start + config.head_batch_size]
            optimizer.zero_grad(set_to_none=True)
            binary, subtype, quality_logit = f3(
                skeleton[selected].to(device), person[selected].to(device), quality[selected].to(device)
            )
            loss = (
                balanced_focal_loss(binary, labels[selected].to(device), gamma=config.focal_gamma)
                + config.subtype_weight * subtype_criterion(subtype, subtypes[selected].to(device))
                + 0.10 * quality_criterion(quality_logit, pose_error_targets(quality[selected]).to(device))
            )
            loss.backward()
            nn.utils.clip_grad_norm_(f3.parameters(), 5.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * selected.size
        print(json.dumps({"stage": "f3_quality_cross", "epoch": epoch, "train_loss": loss_sum / train_indices.size}, sort_keys=True), flush=True)
    f1_scores = _scores(base, skeleton, person, None, heldout_indices, device=device, batch_size=config.head_batch_size)
    f3_scores = _scores(f3, skeleton, person, quality, heldout_indices, device=device, batch_size=config.head_batch_size)
    heldout_ids = clip_ids[train_count:]
    heldout_labels = labels[heldout_indices].numpy()
    heldout_quality = quality[heldout_indices].numpy()
    with np.load(args.f1_oof, allow_pickle=False) as payload:
        expected_ids = [str(value) for value in payload["clip_ids"]]
        expected_scores = np.asarray(payload["fusion_scores"])
    expected_index = {clip_id: index for index, clip_id in enumerate(expected_ids)}
    if set(expected_index) != set(heldout_ids):
        raise ValueError("F1 OOF clip 集合不一致")
    reordered = expected_scores[np.asarray([expected_index[item] for item in heldout_ids])]
    if not np.allclose(f1_scores, reordered, rtol=0.0, atol=1e-6):
        raise ValueError("冻结 F1 预测未严格复现")
    f1_metrics = metric_result(heldout_ids, heldout_labels, f1_scores)
    f3_metrics = metric_result(heldout_ids, heldout_labels, f3_scores)
    f1_r90 = high_recall_false_positives(heldout_ids, heldout_labels, f1_scores)
    f3_r90 = high_recall_false_positives(heldout_ids, heldout_labels, f3_scores)
    f1_pose_fp = pose_error_false_positives(heldout_labels, f1_scores, heldout_quality, float(f1_r90["threshold"]))
    f3_pose_fp = pose_error_false_positives(heldout_labels, f3_scores, heldout_quality, float(f3_r90["threshold"]))
    pose_reduction = (f1_pose_fp - f3_pose_fp) / max(1, f1_pose_fp)
    delta_map = f3_metrics["clip_map_percent"] - f1_metrics["clip_map_percent"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_checkpoint(args.output_dir / "last.pt", {
        "f1_checkpoint_sha256": _sha256_file(args.f1_checkpoint), "f3_state": f3.state_dict(),
        "epochs": args.epochs, "seed": args.seed, "f1_metrics": f1_metrics, "f3_metrics": f3_metrics,
    })
    np.savez_compressed(args.output_dir / "oof_predictions.npz", clip_ids=np.asarray(heldout_ids), labels=heldout_labels, f1_scores=f1_scores, f3_scores=f3_scores, quality=heldout_quality)
    summary: dict[str, Any] = {
        "protocol": "edgefall_f3_frozen_f1_quality_gated_person_roi_cross_attention_v1",
        "f1": f1_metrics, "f3": f3_metrics, "delta_map_percent": delta_map,
        "f1_r90_fp": f1_r90, "f3_r90_fp": f3_r90,
        "f1_pose_error_fp": f1_pose_fp, "f3_pose_error_fp": f3_pose_fp,
        "pose_error_fp_reduction_fraction": pose_reduction,
        "passes_f3_gate": delta_map >= 1.0 and pose_reduction >= 0.25,
        "f1_checkpoint_sha256": _sha256_file(args.f1_checkpoint),
        "roi_cache_sha256": _sha256_file(args.roi_cache),
        "checkpoint_sha256": _sha256_file(args.output_dir / "last.pt"), "test_accessed": False,
    }
    _atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
