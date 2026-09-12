"""Train a template-grouped train-only PoseConv3D event oracle fold."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from models.pose_heatmap_event_oracle import PoseHeatmapEventOracle
from models.tcn_dataset import WindowMemmapCache, load_window_cache
from tools.build_tcn_multistream_sidecar import load_sidecar
from tools.train_long_context_event_oracle import (
    _boundary_pos_weight,
    _class_weights,
    _metric_result,
    _smooth_loss,
    boundary_targets,
    group_fold,
    stage_targets,
    template_group,
)
from tools.train_tcn import (
    _atomic_checkpoint,
    _atomic_json,
    _sha256_file,
    set_deterministic,
)


@dataclass(frozen=True)
class PoseOracleConfig:
    max_steps: int = 128
    heatmap_size: int = 32
    sigma: float = 1.5
    coordinate_limit: float = 1.25
    epochs: int = 8
    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 1e-3
    hidden_dim: int = 64
    dropout: float = 0.2
    stage_weight: float = 1.0
    boundary_weight: float = 0.5
    smooth_weight: float = 0.1
    boundary_radius: int = 1
    folds: int = 5
    seed: int = 20260824

    def validate(self) -> None:
        if self.max_steps < 16 or self.heatmap_size < 8 or self.epochs < 1:
            raise ValueError("max_steps/heatmap_size/epochs 无效")
        if self.batch_size < 1 or self.hidden_dim < 8 or self.folds < 2:
            raise ValueError("batch_size/hidden_dim/folds 无效")
        if self.sigma <= 0.0 or self.coordinate_limit <= 0.0:
            raise ValueError("sigma/coordinate_limit 必须为正")
        if self.boundary_radius < 0 or not 0.0 <= self.dropout < 1.0:
            raise ValueError("boundary_radius/dropout 无效")


class PoseOracleDataset(Dataset[tuple[torch.Tensor, ...]]):
    def __init__(
        self,
        poses: torch.Tensor,
        geometry: torch.Tensor,
        lengths: torch.Tensor,
        stages: torch.Tensor,
        boundaries: torch.Tensor,
        labels: torch.Tensor,
        indices: np.ndarray,
    ) -> None:
        self.poses = poses
        self.geometry = geometry
        self.lengths = lengths
        self.stages = stages
        self.boundaries = boundaries
        self.labels = labels
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        selected = int(self.indices[index])
        return (
            self.poses[selected],
            self.geometry[selected],
            self.lengths[selected],
            self.stages[selected],
            self.boundaries[selected],
            self.labels[selected],
            torch.tensor(selected, dtype=torch.long),
        )


def render_pose_heatmaps(
    poses: torch.Tensor,
    lengths: torch.Tensor,
    *,
    size: int,
    sigma: float,
    coordinate_limit: float,
) -> torch.Tensor:
    """Rasterize confidence-weighted normalized joints into Gaussian heatmaps."""
    if poses.ndim != 4 or poses.shape[2:] != (17, 3):
        raise ValueError("poses 必须为 (B,T,17,3)")
    if size < 2 or sigma <= 0.0 or coordinate_limit <= 0.0:
        raise ValueError("heatmap parameters 无效")
    steps = poses.shape[1]
    if lengths.ndim != 1 or torch.any(lengths < 1) or torch.any(lengths > steps):
        raise ValueError("lengths 无效")
    coordinates = poses[..., :2]
    confidence = poses[..., 2].clamp(0.0, 1.0)
    pixels = ((coordinates + coordinate_limit) / (2.0 * coordinate_limit)) * (size - 1)
    pixels = pixels.clamp(0.0, float(size - 1))
    grid = torch.arange(size, device=poses.device, dtype=poses.dtype)
    grid_y, grid_x = torch.meshgrid(grid, grid, indexing="ij")
    delta_x = grid_x[None, None, None] - pixels[..., 0, None, None]
    delta_y = grid_y[None, None, None] - pixels[..., 1, None, None]
    heatmaps = torch.exp(-(delta_x.square() + delta_y.square()) / (2.0 * sigma**2))
    heatmaps = heatmaps * confidence[..., None, None]
    valid = torch.arange(steps, device=poses.device)[None, :] < lengths[:, None]
    return (heatmaps * valid[:, :, None, None, None]).permute(0, 2, 1, 3, 4)


def materialize_pose_context(
    cache: WindowMemmapCache,
    sidecar: np.ndarray,
    *,
    max_steps: int,
    boundary_radius: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if cache.metadata.get("split") != "train":
        raise ValueError("PoseConv3D oracle 只接受 train cache")
    if sidecar.shape != (cache.sample_count, 16, 6):
        raise ValueError("bbox sidecar 与 train cache 不兼容")
    clips = cache.metadata["clips"]
    poses = np.zeros((len(clips), max_steps, 17, 3), dtype=np.float32)
    geometry = np.zeros((len(clips), max_steps, 6), dtype=np.float32)
    stages = np.full((len(clips), max_steps), -100, dtype=np.int64)
    boundaries = np.zeros((len(clips), max_steps, 2), dtype=np.float32)
    lengths = np.zeros(len(clips), dtype=np.int64)
    labels = np.asarray([bool(clip["has_fall"]) for clip in clips], dtype=np.float32)
    feature_array = cache.array("features")
    semantic_codes = cache.array("semantic_codes")
    end_times = cache.array("end_times")
    for clip_index, clip in enumerate(clips):
        start = int(clip["window_start"])
        count = int(clip["window_count"])
        order = np.argsort(np.asarray(end_times[start : start + count]), kind="stable")
        if count > max_steps:
            sampled = np.linspace(0, count - 1, max_steps).round().astype(np.int64)
            order = order[sampled]
        selected = start + order
        length = int(selected.size)
        poses[clip_index, :length] = np.asarray(feature_array[selected, -1])
        geometry[clip_index, :length] = np.asarray(sidecar[selected, -1])
        clip_stages = stage_targets(np.asarray(semantic_codes[selected]))
        stages[clip_index, :length] = clip_stages
        boundaries[clip_index, :length] = boundary_targets(
            clip_stages, radius=boundary_radius
        )
        lengths[clip_index] = length
    return tuple(
        torch.from_numpy(array)
        for array in (poses, geometry, lengths, stages, boundaries, labels)
    )  # type: ignore[return-value]


def train_epoch(
    model: PoseHeatmapEventOracle,
    loader: DataLoader[tuple[torch.Tensor, ...]],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    stage_criterion: nn.Module,
    boundary_criterion: nn.Module,
    clip_criterion: nn.Module,
    config: PoseOracleConfig,
) -> dict[str, float]:
    model.train()
    totals = {name: 0.0 for name in ("loss", "clip", "stage", "boundary", "smooth")}
    seen = 0
    for poses, geometry, lengths, stages, boundaries, labels, _ in loader:
        poses = poses.to(device, non_blocking=True)
        geometry = geometry.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        stages = stages.to(device, non_blocking=True)
        boundaries = boundaries.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        heatmaps = render_pose_heatmaps(
            poses,
            lengths,
            size=config.heatmap_size,
            sigma=config.sigma,
            coordinate_limit=config.coordinate_limit,
        )
        outputs = model(heatmaps, geometry, lengths)
        mask = outputs["mask"]
        clip_loss = clip_criterion(outputs["clip_logit"], labels)
        stage_loss = stage_criterion(outputs["stage_logits"][mask], stages[mask])
        boundary_loss = boundary_criterion(
            outputs["boundary_logits"][mask], boundaries[mask]
        )
        smooth_loss = _smooth_loss(outputs["stage_logits"], mask)
        loss = (
            clip_loss
            + config.stage_weight * stage_loss
            + config.boundary_weight * boundary_loss
            + config.smooth_weight * smooth_loss
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        batch = int(labels.numel())
        seen += batch
        for name, value in (
            ("loss", loss),
            ("clip", clip_loss),
            ("stage", stage_loss),
            ("boundary", boundary_loss),
            ("smooth", smooth_loss),
        ):
            totals[name] += float(value.detach()) * batch
    return {name: value / seen for name, value in totals.items()}


@torch.inference_mode()
def evaluate(
    model: PoseHeatmapEventOracle,
    loader: DataLoader[tuple[torch.Tensor, ...]],
    *,
    device: torch.device,
    clip_ids: list[str],
    config: PoseOracleConfig,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    model.eval()
    scores = np.zeros(len(clip_ids), dtype=np.float32)
    labels_by_index = np.zeros(len(clip_ids), dtype=np.float32)
    stage_correct = stage_total = 0
    boundary_tp = boundary_fp = boundary_fn = 0
    candidate_positive = candidate_recovered = 0
    evaluated: list[int] = []
    for poses, geometry, lengths, stages, boundaries, labels, indices in loader:
        poses = poses.to(device, non_blocking=True)
        geometry = geometry.to(device, non_blocking=True)
        lengths_device = lengths.to(device, non_blocking=True)
        heatmaps = render_pose_heatmaps(
            poses,
            lengths_device,
            size=config.heatmap_size,
            sigma=config.sigma,
            coordinate_limit=config.coordinate_limit,
        )
        outputs = model(heatmaps, geometry, lengths_device)
        mask = outputs["mask"]
        selected = indices.numpy()
        scores[selected] = torch.sigmoid(outputs["clip_logit"]).cpu().numpy()
        labels_by_index[selected] = labels.numpy()
        evaluated.extend(selected.tolist())
        stages_device = stages.to(device)
        stage_predictions = outputs["stage_logits"].argmax(-1)
        stage_correct += int((stage_predictions[mask] == stages_device[mask]).sum())
        stage_total += int(mask.sum())
        boundary_predictions = torch.sigmoid(outputs["boundary_logits"]) >= 0.5
        boundary_truth = boundaries.to(device) >= 0.5
        boundary_tp += int((boundary_predictions & boundary_truth & mask.unsqueeze(-1)).sum())
        boundary_fp += int((boundary_predictions & ~boundary_truth & mask.unsqueeze(-1)).sum())
        boundary_fn += int((~boundary_predictions & boundary_truth & mask.unsqueeze(-1)).sum())
        fall_probability = torch.softmax(outputs["stage_logits"], -1)[..., 2]
        for row, label in enumerate(labels):
            if label > 0.5:
                candidate_positive += 1
                length = int(lengths[row])
                candidate_recovered += int(float(fall_probability[row, :length].max()) >= 0.5)
    unique = np.asarray(sorted(set(evaluated)), dtype=np.int64)
    if unique.size != len(evaluated):
        raise RuntimeError("OOF evaluation clip 重复")
    metrics = _metric_result(
        [clip_ids[index] for index in unique], labels_by_index[unique], scores[unique]
    )
    precision = boundary_tp / max(1, boundary_tp + boundary_fp)
    recall = boundary_tp / max(1, boundary_tp + boundary_fn)
    metrics.update(
        {
            "stage_accuracy": stage_correct / max(1, stage_total),
            "boundary_precision": precision,
            "boundary_recall": recall,
            "boundary_f1": 2 * precision * recall / max(1e-12, precision + recall),
            "candidate_recall_at_stage_0_5": candidate_recovered
            / max(1, candidate_positive),
            "clips": int(unique.size),
        }
    )
    return metrics, unique, scores[unique]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()
    config = PoseOracleConfig(epochs=args.epochs, seed=args.seed)
    config.validate()
    if not 0 <= args.fold < config.folds:
        raise ValueError("fold 必须位于 [0, folds)")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("PoseConv3D oracle 输出目录必须为空")
    cache = load_window_cache(args.cache, verify_hashes=True)
    if cache.metadata.get("split") != "train":
        raise ValueError("PoseConv3D oracle 禁止读取非 train cache")
    sidecar = load_sidecar(args.sidecar, cache)
    set_deterministic(config.seed + args.fold)
    clip_ids = [str(clip["clip_id"]) for clip in cache.metadata["clips"]]
    groups = [template_group(clip_id) for clip_id in clip_ids]
    folds = np.asarray(
        [group_fold(group, folds=config.folds, seed=config.seed) for group in groups]
    )
    held_out = np.flatnonzero(folds == args.fold)
    training = np.flatnonzero(folds != args.fold)
    if set(np.asarray(groups)[held_out]) & set(np.asarray(groups)[training]):
        raise AssertionError("template group 跨 OOF fold 泄漏")
    poses, geometry, lengths, stages, boundaries, labels = materialize_pose_context(
        cache,
        sidecar,
        max_steps=config.max_steps,
        boundary_radius=config.boundary_radius,
    )
    generator = torch.Generator().manual_seed(config.seed + args.fold)
    train_loader = DataLoader(
        PoseOracleDataset(
            poses, geometry, lengths, stages, boundaries, labels, training
        ),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=True,
    )
    evaluation_loader = DataLoader(
        PoseOracleDataset(
            poses, geometry, lengths, stages, boundaries, labels, held_out
        ),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("请求 CUDA，但当前不可用")
    model = PoseHeatmapEventOracle(hidden_dim=config.hidden_dim, dropout=config.dropout).to(device)
    stage_criterion = nn.CrossEntropyLoss(
        weight=_class_weights(stages, lengths, training).to(device)
    )
    boundary_criterion = nn.BCEWithLogitsLoss(
        pos_weight=_boundary_pos_weight(boundaries, lengths, training).to(device)
    )
    positive = float(labels[training].sum())
    clip_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor((training.size - positive) / positive, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    final_indices = final_scores = None
    for epoch in range(config.epochs):
        training_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            stage_criterion=stage_criterion,
            boundary_criterion=boundary_criterion,
            clip_criterion=clip_criterion,
            config=config,
        )
        evaluation, final_indices, final_scores = evaluate(
            model, evaluation_loader, device=device, clip_ids=clip_ids, config=config
        )
        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": training_metrics,
            "oof": evaluation,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False, sort_keys=True), flush=True)
        scheduler.step()
    assert final_indices is not None and final_scores is not None
    checkpoint = {
        "epoch": config.epochs - 1,
        "model_state": model.state_dict(),
        "config": asdict(config),
        "fold": args.fold,
        "metrics": history[-1]["oof"],
    }
    _atomic_checkpoint(args.output_dir / "last.pt", checkpoint)
    np.savez_compressed(
        args.output_dir / "oof_predictions.npz",
        clip_indices=final_indices,
        clip_ids=np.asarray([clip_ids[index] for index in final_indices]),
        labels=labels[final_indices].numpy(),
        scores=final_scores,
        fold=np.full(final_indices.size, args.fold, dtype=np.int64),
    )
    summary = {
        "protocol": "template_grouped_train_only_pose_heatmap_event_oracle_v1",
        "cache_signature_sha256": cache.metadata["signature_sha256"],
        "sidecar_sha256": _sha256_file(args.sidecar / "sidecar.bin"),
        "fold": args.fold,
        "template_groups_total": len(set(groups)),
        "train_clips": int(training.size),
        "oof_clips": int(held_out.size),
        "train_positive": int(labels[training].sum()),
        "oof_positive": int(labels[held_out].sum()),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "config": asdict(config),
        "final_metrics": history[-1]["oof"],
        "checkpoint_sha256": _sha256_file(args.output_dir / "last.pt"),
    }
    _atomic_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "history.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in history),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
