"""Train a frozen-anchor, score-band-gated RGB residual on one train-only fold.

This is deliberately an outer-fold canary, not a substitute for the report's
future nested OOF qualification.  It fixes the band from the outer-training
anchor-score distribution, trains only cached RGB features, and evaluates the
held-out template fold once.  OF-Syn test and every other test split are
rejected by construction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from eval.metrics import clip_pr_curve
from models.edgefall_f1 import BandGatedRgbResidual, EdgeFallF1Head
from models.tcn_dataset import load_window_cache
from tools.build_tcn_multistream_sidecar import load_sidecar
from tools.train_edgefall_f1 import (
    F1Config,
    build_skeleton_model,
    clip_embeddings,
    metric_result,
    template_fold_ids,
)
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


def _logit(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float32)
    values = np.clip(values, 1e-5, 1.0 - 1e-5)
    return np.log(values / (1.0 - values))


def score_band_bounds(anchor_logits: np.ndarray) -> tuple[float, float]:
    """Pre-registered broad R95-risk proxy: 80th through 99th score percentile."""
    values = np.asarray(anchor_logits, dtype=np.float64)
    if values.ndim != 1 or values.size < 20 or not np.all(np.isfinite(values)):
        raise ValueError("anchor logits 无法确定分数带")
    low, high = np.quantile(values, (0.80, 0.99))
    if not low < high:
        raise ValueError("anchor logits 无法形成非空分数带")
    return float(low), float(high)


def workpoint_counts(labels: np.ndarray, scores: np.ndarray, recall: float) -> dict[str, float | int]:
    """Return the deterministic best-precision point satisfying target recall."""
    labels = np.asarray(labels, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    ids = {str(index): bool(label) for index, label in enumerate(labels)}
    curve = clip_pr_curve(ids, {str(index): float(score) for index, score in enumerate(scores)})
    feasible = [point for point in curve if point.recall >= recall]
    if not feasible:
        raise ValueError(f"无法达到 R{int(recall * 100)}")
    best = max(feasible, key=lambda point: (point.precision, point.threshold))
    return {
        "threshold": float(best.threshold),
        "precision": float(best.precision),
        "recall": float(best.recall),
        "tp": int(best.tp),
        "fp": int(best.fp),
        "fn": int(best.fn),
    }


@torch.inference_mode()
def _anchor_scores(
    anchor: EdgeFallF1Head,
    skeleton: torch.Tensor,
    person: torch.Tensor,
    indices: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    anchor.eval()
    values = []
    for start in range(0, indices.size, batch_size):
        selected = indices[start : start + batch_size]
        logits, _ = anchor(skeleton[selected].to(device), person[selected].to(device))
        values.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(values).astype(np.float32, copy=False)


@torch.inference_mode()
def _residual_scores(
    model: BandGatedRgbResidual,
    rgb: torch.Tensor,
    anchor_logits: torch.Tensor,
    indices: np.ndarray,
    *,
    low: float,
    high: float,
    temperature: float,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    scores: list[np.ndarray] = []
    gates: list[np.ndarray] = []
    residuals: list[np.ndarray] = []
    for start in range(0, indices.size, batch_size):
        selected = indices[start : start + batch_size]
        logits, gate, residual = model(
            rgb[selected].to(device),
            anchor_logits[selected].to(device),
            low=low,
            high=high,
            temperature=temperature,
        )
        scores.append(torch.sigmoid(logits).cpu().numpy())
        gates.append(gate.cpu().numpy())
        residuals.append(residual.cpu().numpy())
    return np.concatenate(scores), np.concatenate(gates), np.concatenate(residuals)


def _load_rgb_features(path: Path, clip_ids: list[str], labels: np.ndarray) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        required = {"features", "clip_ids", "labels"}
        if not required <= set(payload.files):
            raise ValueError("RGB feature cache 缺少 features/clip_ids/labels")
        feature_ids = [str(value) for value in payload["clip_ids"]]
        features = np.asarray(payload["features"], dtype=np.float32)
        feature_labels = np.asarray(payload["labels"], dtype=np.float32)
    if features.ndim != 2 or len(feature_ids) != features.shape[0]:
        raise ValueError("RGB feature cache 形状无效")
    index = {clip_id: offset for offset, clip_id in enumerate(feature_ids)}
    if len(index) != len(feature_ids) or set(index) != set(clip_ids):
        raise ValueError("RGB feature cache 与 train cache clip 集合不一致")
    ordered = np.asarray([index[clip_id] for clip_id in clip_ids], dtype=np.int64)
    if not np.array_equal(feature_labels[ordered], labels):
        raise ValueError("RGB feature cache 与 train cache 标签不一致")
    return features[ordered]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--roi-cache", type=Path, required=True)
    parser.add_argument("--f1-checkpoint", type=Path, required=True)
    parser.add_argument("--f1-oof", type=Path, required=True)
    parser.add_argument("--rgb-features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument("--correction-cap", type=float, default=1.0)
    args = parser.parse_args()
    if args.epochs < 1 or args.temperature <= 0.0 or args.correction_cap <= 0.0:
        raise ValueError("epochs/temperature/correction-cap 无效")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("输出目录非空，拒绝覆盖")
    cache = load_window_cache(args.cache, verify_hashes=True)
    if cache.metadata.get("dataset") != "of-syn" or cache.metadata.get("split") != "train":
        raise ValueError("R95 expert 只接受 OF-Syn train cache")
    sidecar = load_sidecar(args.sidecar, cache)
    checkpoint = torch.load(args.f1_checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("config"), dict):
        raise TypeError("F1 checkpoint 无效")
    config = F1Config(**checkpoint["config"])
    clips = cache.metadata["clips"]
    fold_ids = template_fold_ids(clips, config)
    if config.fold != 0:
        fold_ids = (fold_ids - config.fold) % config.folds
    all_window_clip_indices = np.asarray(cache.array("clip_indices"), dtype=np.int64)
    all_labels = np.asarray(cache.array("labels"))
    activities = _activity_groups(cache)
    candidate_windows = np.flatnonzero(fold_ids[all_window_clip_indices] != 0)
    selected = select_training_indices(
        all_labels[candidate_windows], negative_ratio=config.negative_ratio,
        seed=config.seed, activity_groups=activities[candidate_windows],
        hard_negative_activities=("lie_down", "lying", "stand_up"), hard_negative_fraction=0.4,
    )
    train_windows = candidate_windows[selected]
    train_x, _, train_window_clips = _materialize(cache, train_windows)
    train_x = _append_sidecar(train_x, sidecar, train_windows)
    heldout_windows = np.flatnonzero(fold_ids[all_window_clip_indices] == 0)
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
    train_embedding_clips, train_embeddings = clip_embeddings(skeleton_model, train_x, train_window_clips, device=device, batch_size=config.batch_size)
    held_embedding_clips, held_embeddings = clip_embeddings(skeleton_model, heldout_x, heldout_window_clips, device=device, batch_size=config.batch_size)
    selected_clips = np.concatenate((train_embedding_clips, held_embedding_clips))
    clip_ids = [str(clips[int(index)]["clip_id"]) for index in selected_clips]
    labels = np.asarray([float(clips[int(index)]["has_fall"]) for index in selected_clips], dtype=np.float32)
    skeleton = torch.from_numpy(np.concatenate((train_embeddings, held_embeddings)))
    with np.load(args.roi_cache, allow_pickle=False) as payload:
        roi_ids = [str(value) for value in payload["clip_ids"]]
        roi_tokens = np.asarray(payload["tokens"], dtype=np.float32)
    roi_index = {clip_id: offset for offset, clip_id in enumerate(roi_ids)}
    if len(roi_index) != len(roi_ids) or any(clip_id not in roi_index for clip_id in clip_ids):
        raise ValueError("ROI cache 缺少或重复 R95 expert clip")
    person = torch.from_numpy(np.stack([roi_tokens[roi_index[clip_id], :, 0] for clip_id in clip_ids]))
    rgb_features = _load_rgb_features(args.rgb_features, clip_ids, labels)
    train_count = train_embedding_clips.size
    train_indices = np.arange(train_count, dtype=np.int64)
    held_indices = np.arange(train_count, selected_clips.size, dtype=np.int64)
    anchor = EdgeFallF1Head(int(skeleton.shape[1])).to(device)
    anchor.load_state_dict(checkpoint["fusion_state"], strict=True)
    for parameter in anchor.parameters():
        parameter.requires_grad_(False)
    anchor_scores = _anchor_scores(anchor, skeleton, person, np.arange(selected_clips.size), device=device, batch_size=config.head_batch_size)
    anchor_logits = _logit(anchor_scores)
    low, high = score_band_bounds(anchor_logits[train_indices])
    rgb_mean = rgb_features[train_indices].mean(axis=0, dtype=np.float64).astype(np.float32)
    rgb_std = np.maximum(rgb_features[train_indices].std(axis=0, dtype=np.float64).astype(np.float32), 1e-6)
    rgb = torch.from_numpy((rgb_features - rgb_mean) / rgb_std)
    anchor_logit_tensor = torch.from_numpy(anchor_logits)
    model = BandGatedRgbResidual(rgb.shape[1], correction_cap=args.correction_cap).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    positive_weight = float((train_count - labels[train_indices].sum()) / labels[train_indices].sum())
    history: list[dict[str, Any]] = []
    for epoch in range(args.epochs):
        model.train()
        order = np.random.default_rng(args.seed + epoch).permutation(train_indices)
        total_loss = 0.0
        seen = 0
        for start in range(0, order.size, config.head_batch_size):
            batch = order[start : start + config.head_batch_size]
            optimizer.zero_grad(set_to_none=True)
            final, gate, bounded = model(rgb[batch].to(device), anchor_logit_tensor[batch].to(device), low=low, high=high, temperature=args.temperature)
            target = torch.from_numpy(labels[batch]).to(device)
            bce = nn.functional.binary_cross_entropy_with_logits(final, target, reduction="none", pos_weight=torch.tensor(positive_weight, device=device))
            # The gate is the local-expert curriculum; it is detached because the
            # band boundaries belong to the frozen anchor protocol.
            weighted = (gate.detach() * bce).sum() / gate.detach().sum().clamp_min(1e-4)
            proposal = anchor_logit_tensor[batch].to(device) + gate * bounded
            proposal_bce = nn.functional.binary_cross_entropy_with_logits(
                proposal, target, reduction="none",
                pos_weight=torch.tensor(positive_weight, device=device),
            )
            # alpha begins at exactly zero as required.  This auxiliary loss
            # learns a bounded local residual before the constrained fusion
            # scale is allowed to activate it.
            proposal_weighted = (
                gate.detach() * proposal_bce
            ).sum() / gate.detach().sum().clamp_min(1e-4)
            suppress_positive = (target * nn.functional.relu(anchor_logit_tensor[batch].to(device) - final)).mean()
            loss = (
                weighted
                + 0.20 * proposal_weighted
                + 0.10 * suppress_positive
                + 0.01 * bounded.square().mean()
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            model.project_alpha_()
            total_loss += float(loss.detach()) * batch.size
            seen += batch.size
        history.append({"epoch": epoch, "train_loss": total_loss / seen, "alpha": float(model.alpha().detach())})
        print(json.dumps({"stage": "r95_rgb_residual", **history[-1]}, sort_keys=True), flush=True)
    final_scores, held_gates, held_residuals = _residual_scores(model, rgb, anchor_logit_tensor, held_indices, low=low, high=high, temperature=args.temperature, device=device, batch_size=config.head_batch_size)
    baseline_scores = anchor_scores[held_indices]
    held_ids = clip_ids[train_count:]
    held_labels = labels[held_indices]
    with np.load(args.f1_oof, allow_pickle=False) as payload:
        expected_ids = [str(value) for value in payload["clip_ids"]]
        expected_scores = np.asarray(payload["fusion_scores"], dtype=np.float32)
    expected_index = {clip_id: index for index, clip_id in enumerate(expected_ids)}
    if set(expected_index) != set(held_ids):
        raise ValueError("F1 OOF clip 集合不一致")
    expected = expected_scores[np.asarray([expected_index[clip_id] for clip_id in held_ids])]
    if not np.allclose(baseline_scores, expected, rtol=0.0, atol=1e-6):
        raise ValueError("冻结 F1 锚点未严格复现")
    anchor_metrics = metric_result(held_ids, held_labels, baseline_scores)
    residual_metrics = metric_result(held_ids, held_labels, final_scores)
    anchor_r95 = workpoint_counts(held_labels, baseline_scores, 0.95)
    residual_r95 = workpoint_counts(held_labels, final_scores, 0.95)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_checkpoint(args.output_dir / "last.pt", {
        "f1_checkpoint_sha256": _sha256_file(args.f1_checkpoint), "model_state": model.state_dict(),
        "rgb_mean": rgb_mean, "rgb_std": rgb_std, "band": {"low": low, "high": high, "temperature": args.temperature},
        "correction_cap": args.correction_cap, "history": history,
    })
    np.savez_compressed(args.output_dir / "oof_predictions.npz", clip_ids=np.asarray(held_ids), labels=held_labels, anchor_scores=baseline_scores, residual_scores=final_scores, gate=held_gates, bounded_residual=held_residuals)
    fp_reduction = int(anchor_r95["fp"]) - int(residual_r95["fp"])
    tp_loss = int(anchor_r95["tp"]) - int(residual_r95["tp"])
    summary = {
        "protocol": "edgefall_r95_local_rgb_residual_outer_fold_canary_v1",
        "qualification": "outer_fold_canary_only; nested_grouped_oof_required_before_production_fusion",
        "anchor": anchor_metrics, "residual": residual_metrics,
        "delta_map_percent": residual_metrics["clip_map_percent"] - anchor_metrics["clip_map_percent"],
        "r95": {"anchor": anchor_r95, "residual": residual_r95, "fp_reduction": fp_reduction, "tp_loss": tp_loss},
        "band": {"low_logit": low, "high_logit": high, "temperature": args.temperature, "held_gate_mean": float(held_gates.mean()), "held_gate_ge_0_05": int((held_gates >= 0.05).sum())},
        "alpha": float(model.alpha().detach()), "correction_cap": args.correction_cap,
        "passes_e95_gate": fp_reduction >= 15 and tp_loss <= 1,
        "f1_checkpoint_sha256": _sha256_file(args.f1_checkpoint), "rgb_features_sha256": _sha256_file(args.rgb_features), "roi_cache_sha256": _sha256_file(args.roi_cache),
        "checkpoint_sha256": _sha256_file(args.output_dir / "last.pt"), "test_accessed": False,
    }
    _atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
