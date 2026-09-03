"""Train one template-OOF fold of the low-dimensional Phase-Motion canary."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from models.phase_motion_expert import (
    PHASE_MOTION_DIM,
    PhaseMotionExpert,
    phase_motion_features,
)
from models.tcn_dataset import load_window_cache
from tools.build_tcn_multistream_sidecar import load_sidecar
from tools.train_edgefall_f1 import (
    SUBTYPES,
    balanced_focal_loss,
    explicit_fold_ids,
    metric_result,
    subtype_targets,
)
from tools.train_multiscale_multistream_tcn import _activity_groups
from tools.train_tcn import _sha256_file, select_training_indices, set_deterministic


@dataclass(frozen=True)
class PhaseMotionConfig:
    fold: int = 0
    folds: int = 5
    seed: int = 20260829
    epochs: int = 12
    batch_size: int = 512
    learning_rate: float = 3e-4
    negative_ratio: float = 2.0
    focal_gamma: float = 2.0
    subtype_weight: float = 0.25
    hidden_dim: int = 32

    def validate(self) -> None:
        if not 0 <= self.fold < self.folds or self.folds < 2:
            raise ValueError("fold 配置无效")
        if min(self.epochs, self.batch_size, self.hidden_dim) < 1:
            raise ValueError("训练配置无效")
        if self.learning_rate <= 0.0 or self.negative_ratio <= 0.0:
            raise ValueError("learning_rate/negative_ratio 无效")


@torch.inference_mode()
def extract_phase_summaries(
    features: np.ndarray,
    sidecar: np.ndarray,
    indices: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """Materialize only the 165D phase summaries, not the large raw windows."""
    outputs: list[torch.Tensor] = []
    for start in range(0, indices.size, batch_size):
        selected = indices[start : start + batch_size]
        pose = torch.from_numpy(np.asarray(features[selected], dtype=np.float32))
        geometry = torch.from_numpy(np.asarray(sidecar[selected], dtype=np.float32))
        batch = torch.cat((pose.flatten(2), geometry), dim=2).to(device)
        outputs.append(phase_motion_features(batch).cpu())
    result = torch.cat(outputs)
    if result.shape != (indices.size, PHASE_MOTION_DIM):
        raise AssertionError("Phase-Motion summary 数量不匹配")
    return result


def standardize_summaries(
    train: torch.Tensor, heldout: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = train.mean(dim=0)
    scale = train.std(dim=0, unbiased=False).clamp_min(1e-5)
    return (train - mean) / scale, (heldout - mean) / scale, mean, scale


@torch.inference_mode()
def _scores(
    model: PhaseMotionExpert,
    summaries: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    values: list[torch.Tensor] = []
    for start in range(0, summaries.shape[0], batch_size):
        logits, _ = model(summaries[start : start + batch_size].to(device))
        values.append(logits.sigmoid().cpu())
    return torch.cat(values).numpy()


def _clip_max(
    window_scores: np.ndarray, window_clips: np.ndarray, clip_indices: np.ndarray
) -> np.ndarray:
    result = np.empty(clip_indices.size, dtype=np.float32)
    for offset, clip_index in enumerate(clip_indices):
        selected = window_clips == clip_index
        if not selected.any():
            raise ValueError("held-out clip 没有 Phase-Motion window")
        result[offset] = float(window_scores[selected].max())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--fold-map", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    config = PhaseMotionConfig(
        fold=args.fold,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    config.validate()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Phase-Motion 输出目录必须为空")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cache = load_window_cache(args.cache, verify_hashes=True)
    if cache.metadata.get("dataset") != "of-syn" or cache.metadata.get("split") != "train":
        raise ValueError("Phase-Motion 只接受 OF-Syn train cache")
    if cache.metadata.get("window_config") != {
        "window_size": 48,
        "stride": 1,
        "min_observed_frames": 24,
        "causal_left_pad": True,
    }:
        raise ValueError("Phase-Motion 需要锁定的 native dense-48 cache")
    sidecar = load_sidecar(args.sidecar, cache, verify_hashes=True)
    clips = cache.metadata["clips"]
    fold_ids, fold_assignment = explicit_fold_ids(
        args.fold_map,
        clips,
        folds=config.folds,
        expected_cache_signature=str(cache.metadata.get("signature_sha256", "")),
    )
    all_window_clips = np.asarray(cache.array("clip_indices"), dtype=np.int64)
    train_clip_mask = fold_ids != config.fold
    heldout_clip_mask = fold_ids == config.fold
    candidate_windows = np.flatnonzero(train_clip_mask[all_window_clips])
    all_labels = np.asarray(cache.array("labels"))
    activities = _activity_groups(cache)
    local = select_training_indices(
        all_labels[candidate_windows],
        negative_ratio=config.negative_ratio,
        seed=config.seed,
        activity_groups=activities[candidate_windows],
        hard_negative_activities=("lie_down", "lying", "stand_up"),
        hard_negative_fraction=0.4,
    )
    train_windows = candidate_windows[local]
    heldout_windows = np.flatnonzero(heldout_clip_mask[all_window_clips])
    train_window_clips = all_window_clips[train_windows]
    heldout_window_clips = all_window_clips[heldout_windows]

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("请求 CUDA，但当前不可用")
    set_deterministic(config.seed)
    started = time.perf_counter()
    train_summaries = extract_phase_summaries(
        cache.array("features"),
        sidecar,
        train_windows,
        device=device,
        batch_size=config.batch_size,
    )
    heldout_summaries = extract_phase_summaries(
        cache.array("features"),
        sidecar,
        heldout_windows,
        device=device,
        batch_size=config.batch_size,
    )
    train_summaries, heldout_summaries, feature_mean, feature_scale = standardize_summaries(
        train_summaries, heldout_summaries
    )
    print(
        json.dumps(
            {
                "stage": "phase_summary_extraction",
                "seconds": time.perf_counter() - started,
                "train_windows": int(train_windows.size),
                "heldout_windows": int(heldout_windows.size),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    train_labels = torch.from_numpy(np.asarray(all_labels[train_windows], dtype=np.float32))
    train_subtypes = torch.from_numpy(subtype_targets(clips, train_window_clips))
    subtype_counts = torch.bincount(train_subtypes, minlength=len(SUBTYPES)).float()
    if torch.any(subtype_counts == 0):
        raise ValueError("训练 fold 必须覆盖全部 subtype")
    subtype_criterion = nn.CrossEntropyLoss(
        weight=(subtype_counts.sum() / subtype_counts).to(device)
    )
    model = PhaseMotionExpert(hidden_dim=config.hidden_dim, subtypes=len(SUBTYPES)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=1e-3
    )
    for epoch in range(config.epochs):
        epoch_started = time.perf_counter()
        model.train()
        order = np.random.default_rng(config.seed + epoch).permutation(train_windows.size)
        total_loss = 0.0
        for start in range(0, order.size, config.batch_size):
            selected = order[start : start + config.batch_size]
            summaries = train_summaries[selected].to(device)
            labels = train_labels[selected].to(device)
            subtypes = train_subtypes[selected].to(device)
            optimizer.zero_grad(set_to_none=True)
            binary, subtype = model(summaries)
            loss = balanced_focal_loss(
                binary, labels, gamma=config.focal_gamma
            ) + config.subtype_weight * subtype_criterion(subtype, subtypes)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach()) * selected.size
        print(
            json.dumps(
                {
                    "stage": "phase_motion_training",
                    "epoch": epoch,
                    "loss": total_loss / order.size,
                    "seconds": time.perf_counter() - epoch_started,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    window_scores = _scores(
        model,
        heldout_summaries,
        device=device,
        batch_size=config.batch_size,
    )
    heldout_clip_indices = np.flatnonzero(heldout_clip_mask)
    clip_scores = _clip_max(window_scores, heldout_window_clips, heldout_clip_indices)
    clip_ids = [str(clips[index]["clip_id"]) for index in heldout_clip_indices]
    clip_labels = np.asarray(
        [bool(clips[index]["has_fall"]) for index in heldout_clip_indices], dtype=np.float32
    )
    np.savez_compressed(
        args.output_dir / "oof_predictions.npz",
        clip_id=np.asarray(clip_ids),
        label=clip_labels,
        score=clip_scores,
        fold=np.full(len(clip_ids), config.fold, dtype=np.int64),
    )
    checkpoint = {
        "protocol": "phase_motion_native48_three_phase_canary_v1",
        "config": asdict(config),
        "fold_assignment": fold_assignment,
        "cache_signature_sha256": cache.metadata.get("signature_sha256"),
        "sidecar_metadata_sha256": _sha256_file(args.sidecar / "metadata.json"),
        "model_state": model.state_dict(),
        "feature_mean": feature_mean,
        "feature_scale": feature_scale,
        "test_accessed": False,
    }
    torch.save(checkpoint, args.output_dir / "checkpoint.pt")
    summary = {
        "protocol": checkpoint["protocol"],
        "config": asdict(config),
        "fold_assignment": fold_assignment,
        "metrics": metric_result(clip_ids, clip_labels, clip_scores),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "train_windows": int(train_windows.size),
        "heldout_windows": int(heldout_windows.size),
        "checkpoint_sha256": _sha256_file(args.output_dir / "checkpoint.pt"),
        "oof_predictions_sha256": _sha256_file(args.output_dir / "oof_predictions.npz"),
        "test_accessed": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
