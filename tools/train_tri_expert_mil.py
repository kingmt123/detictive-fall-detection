"""Train the causal TCN + SA-TDGFormer + four-modal InfoGCN experiment."""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from models.multiscale_multistream_tcn import count_params
from models.tcn_dataset import WindowMemmapCache, load_window_cache
from models.tri_expert_skeleton import (
    TriExpertSkeletonFallDetector,
    orthogonal_expert_loss,
)
from tools.build_tcn_multistream_sidecar import load_sidecar
from tools.train_multiscale_multistream_mil import (
    MILConfig,
    _clip_groups,
    aggregate_topk_logits,
    augment_training_features,
)
from tools.train_multiscale_multistream_tcn import _activity_groups
from tools.train_tcn import (
    _append_sidecar,
    _atomic_checkpoint,
    _atomic_json,
    _canonical_json,
    _materialize,
    _run_signature,
    evaluate_model,
    select_training_indices,
    set_deterministic,
)


@dataclass(frozen=True)
class TriExpertConfig(MILConfig):
    tri_channels: int = 32
    expert_dim: int = 64
    attention_heads: int = 4
    infogcn_dynamic_layers: int = 2
    infogcn_knn_k: int = 4
    diversity_loss_weight: float = 0.01
    information_loss_weight: float = 0.0001
    diversity_warmup_epochs: int = 3
    use_amp: bool = True

    @classmethod
    def from_json(cls, path: Path) -> TriExpertConfig:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("三专家配置必须是 JSON 对象")
        unknown = set(payload) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"三专家配置包含未知字段: {sorted(unknown)}")
        if "channels" in payload:
            payload["channels"] = tuple(payload["channels"])
        if "hard_negative_activities" in payload:
            payload["hard_negative_activities"] = tuple(
                payload["hard_negative_activities"]
            )
        config = cls(**payload)
        config.validate()
        return config

    def validate(self) -> None:
        super().validate()
        if self.tri_channels < 4 or self.tri_channels % self.attention_heads:
            raise ValueError("tri_channels 必须能被 attention_heads 整除")
        if self.expert_dim < 4 or self.expert_dim % self.attention_heads:
            raise ValueError("expert_dim 必须能被 attention_heads 整除")
        if self.infogcn_dynamic_layers not in {2, 3}:
            raise ValueError("infogcn_dynamic_layers 必须为 2 或 3")
        if not 1 <= self.infogcn_knn_k <= 17:
            raise ValueError("infogcn_knn_k 必须位于 [1,17]")
        if self.diversity_loss_weight < 0 or self.information_loss_weight < 0:
            raise ValueError("辅助损失权重不能为负")
        if self.diversity_warmup_epochs < 0:
            raise ValueError("diversity_warmup_epochs 不能为负")
        if not isinstance(self.use_amp, bool):
            raise TypeError("use_amp 必须是布尔值")


def _model(config: TriExpertConfig) -> TriExpertSkeletonFallDetector:
    return TriExpertSkeletonFallDetector(
        channels=config.tri_channels,
        expert_dim=config.expert_dim,
        dropout=config.dropout,
        dynamic_layers=config.infogcn_dynamic_layers,
        knn_k=config.infogcn_knn_k,
        heads=config.attention_heads,
    )


def _diversity_scale(config: TriExpertConfig, epoch: int) -> float:
    if config.diversity_warmup_epochs == 0:
        return config.diversity_loss_weight
    progress = min(1.0, (epoch + 1) / config.diversity_warmup_epochs)
    return config.diversity_loss_weight * progress


def train_tri_epoch(
    model: TriExpertSkeletonFallDetector,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    features: torch.Tensor,
    labels: torch.Tensor,
    groups: list[np.ndarray],
    clip_labels: np.ndarray,
    *,
    device: torch.device,
    config: TriExpertConfig,
    epoch: int,
) -> dict[str, float]:
    model.train()
    clip_order = np.random.default_rng(config.seed + epoch).permutation(len(groups))
    window_pos_weight = float((labels.numel() - labels.sum()) / labels.sum())
    clip_pos_weight = float((clip_labels.size - clip_labels.sum()) / clip_labels.sum())
    window_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(window_pos_weight, device=device)
    )
    clip_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(clip_pos_weight, device=device)
    )
    amp_enabled = config.use_amp and device.type == "cuda"
    diversity_weight = _diversity_scale(config, epoch)
    totals = {"loss": 0.0, "classification": 0.0, "diversity": 0.0, "information": 0.0}
    seen_clips = 0
    for start in range(0, len(groups), config.mil_clip_batch_size):
        selected = clip_order[start : start + config.mil_clip_batch_size]
        group_sizes = [int(groups[int(index)].size) for index in selected]
        window_indices = np.concatenate([groups[int(index)] for index in selected])
        batch_x = augment_training_features(
            features[window_indices].to(device),
            config,
            seed=config.seed + epoch * 1_000_003 + start,
        )
        batch_y = labels[window_indices].to(device)
        batch_clip_y = torch.from_numpy(clip_labels[selected]).to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled
        ):
            output = model.forward_detailed(batch_x)
            clip_logits = aggregate_topk_logits(
                output.logits, group_sizes, config.mil_topk_fraction
            )
            classification = window_criterion(
                output.logits, batch_y
            ) + config.mil_loss_weight * clip_criterion(clip_logits, batch_clip_y)
            diversity = orthogonal_expert_loss(output.embeddings)
            information = output.information_loss
            loss = (
                classification
                + diversity_weight * diversity
                + config.information_loss_weight * information
            )
        components = {
            "loss": loss,
            "classification": classification,
            "diversity": diversity,
            "information": information,
        }
        invalid = [
            name for name, value in components.items() if not torch.isfinite(value)
        ]
        if invalid:
            raise FloatingPointError(
                f"epoch={epoch}, clip_batch_start={start} 出现非有限值: {invalid}"
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        clip_count = len(selected)
        totals["loss"] += float(loss.detach()) * clip_count
        totals["classification"] += float(classification.detach()) * clip_count
        totals["diversity"] += float(diversity.detach()) * clip_count
        totals["information"] += float(information.detach()) * clip_count
        seen_clips += clip_count
    return {
        name: value / seen_clips for name, value in totals.items()
    } | {"diversity_weight": diversity_weight}


@torch.inference_mode()
def _expert_diagnostics(
    model: TriExpertSkeletonFallDetector,
    features: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    model.eval()
    weight_sum = np.zeros(3, dtype=np.float64)
    cosine_sum = np.zeros(3, dtype=np.float64)
    seen = 0
    for start in range(0, features.shape[0], batch_size):
        output = model.forward_detailed(features[start : start + batch_size].to(device))
        embeddings = nn.functional.normalize(output.embeddings.float(), dim=-1)
        cosine = torch.stack(
            (
                (embeddings[:, 0] * embeddings[:, 1]).sum(-1),
                (embeddings[:, 0] * embeddings[:, 2]).sum(-1),
                (embeddings[:, 1] * embeddings[:, 2]).sum(-1),
            ),
            dim=-1,
        )
        count = output.logits.numel()
        weight_sum += output.attention_weights.float().sum(0).cpu().numpy()
        cosine_sum += cosine.sum(0).cpu().numpy()
        seen += count
    return {
        "expert_order": ["tcn", "sa_tdgformer", "infogcn"],
        "mean_attention_weights": (weight_sum / seen).tolist(),
        "cosine_pair_order": ["tcn_sa", "tcn_info", "sa_info"],
        "mean_pairwise_cosine": (cosine_sum / seen).tolist(),
    }


def train_tri_expert(
    *,
    config: TriExpertConfig,
    train_cache: WindowMemmapCache,
    val_cache: WindowMemmapCache,
    train_sidecar: np.ndarray,
    val_sidecar: np.ndarray,
    output_dir: Path,
    device_name: str,
) -> dict[str, Any]:
    config.validate()
    if train_cache.metadata.get("split") != "train" or val_cache.metadata.get("split") != "val":
        raise ValueError("三专家实验只允许 train/val cache")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("请求 CUDA 训练，但 CUDA 不可用")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("输出目录非空；三专家训练不支持覆盖")
    set_deterministic(config.seed)
    train_indices = select_training_indices(
        train_cache.array("labels"),
        negative_ratio=config.negative_ratio,
        seed=config.seed,
        activity_groups=_activity_groups(train_cache),
        hard_negative_activities=config.hard_negative_activities,
        hard_negative_fraction=config.hard_negative_fraction,
    )
    train_x, train_y, train_clip_indices = _materialize(train_cache, train_indices)
    val_x, val_y, val_clip_indices = _materialize(val_cache, None)
    train_x = _append_sidecar(train_x, train_sidecar, train_indices)
    val_x = _append_sidecar(val_x, val_sidecar, None)
    groups, clip_labels = _clip_groups(
        train_clip_indices, train_cache.metadata["clips"]
    )
    device = torch.device(device_name)
    model = _model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )
    # BF16 has FP32-like exponent range and does not require loss scaling.
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    architecture = {
        "name": "TriExpertSkeletonFallDetector",
        "experts": ["TCN", "SA-TDGFormer", "InfoGCN"],
        "infogcn_modalities": [
            "joint",
            "bone",
            "joint_motion",
            "bone_motion",
        ],
        "infogcn_topology": {
            "fixed_bottom_layers": 4 - config.infogcn_dynamic_layers,
            "dynamic_knn_last_layers": config.infogcn_dynamic_layers,
            "knn_k": config.infogcn_knn_k,
        },
        "expert_fusion": "multihead_attention",
        "expert_dim": config.expert_dim,
        "channels": config.tri_channels,
        "attention_heads": config.attention_heads,
        "orthogonal_loss": {
            "type": "squared_pairwise_cosine",
            "parameter_overhead": 0,
            "weight": config.diversity_loss_weight,
            "warmup_epochs": config.diversity_warmup_epochs,
        },
        "information_bottleneck_kl_weight": config.information_loss_weight,
        "training_precision": "bfloat16_amp" if config.use_amp else "float32",
        "causal": True,
    }
    root = Path(__file__).parent.parent
    signature_sha256, signature = _run_signature(
        config=config,
        train_cache=train_cache,
        val_cache=val_cache,
        device=device_name,
        max_train_samples=None,
        max_val_samples=None,
        protocol="tri_expert_skeleton_topk_mil_v1",
        code_paths=(
            root / "models" / "tri_expert_skeleton.py",
            root / "models" / "multiscale_multistream_tcn.py",
            root / "tools" / "train_tri_expert_mil.py",
            root / "tools" / "train_multiscale_multistream_mil.py",
            root / "tools" / "train_tcn.py",
        ),
        extra_signature=architecture,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        output_dir / "run.json",
        {
            "signature_sha256": signature_sha256,
            "signature": signature,
            "architecture": architecture,
            "parameter_count": count_params(model),
            "train_samples": int(train_y.numel()),
            "train_positive": int(train_y.sum()),
            "train_clips": len(groups),
            "val_samples": int(val_y.numel()),
            "pilot": False,
        },
    )
    best_map = -math.inf
    history: list[dict[str, Any]] = []
    for epoch in range(config.epochs):
        train_losses = train_tri_epoch(
            model,
            optimizer,
            scaler,
            train_x,
            train_y,
            groups,
            clip_labels,
            device=device,
            config=config,
            epoch=epoch,
        )
        metrics = evaluate_model(
            model,
            val_x,
            val_y,
            val_clip_indices,
            val_cache.metadata["clips"],
            device=device,
            batch_size=config.batch_size,
        )
        scheduler.step()
        record = {
            "epoch": epoch,
            "train": train_losses,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "val": metrics,
        }
        with (output_dir / "history.jsonl").open(
            "a", encoding="utf-8", newline="\n"
        ) as handle:
            handle.write(_canonical_json(record) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        history.append(record)
        improved = metrics["clip_map"] > best_map
        best_map = max(best_map, metrics["clip_map"])
        checkpoint = {
            "epoch": epoch,
            "run_signature_sha256": signature_sha256,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "best_map": best_map,
            "metrics": record,
        }
        _atomic_checkpoint(output_dir / "last.pt", checkpoint)
        if improved:
            _atomic_checkpoint(output_dir / "best.pt", checkpoint)
        print(_canonical_json({"stage": "epoch", **record}), flush=True)
    best_checkpoint = torch.load(
        output_dir / "best.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(best_checkpoint["model_state"])
    diagnostics = _expert_diagnostics(
        model, val_x, device=device, batch_size=config.batch_size
    )
    summary = {
        "run_signature_sha256": signature_sha256,
        "epochs_completed": config.epochs,
        "best_clip_map": best_map,
        "best": best_checkpoint["metrics"],
        "last": history[-1],
        "diagnostics": diagnostics,
        "output_dir": str(output_dir.resolve()),
    }
    _atomic_json(output_dir / "summary.json", summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--train-sidecar", type=Path, required=True)
    parser.add_argument("--val-sidecar", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int)
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = TriExpertConfig.from_json(args.config)
    if args.epochs is not None:
        config = replace(config, epochs=args.epochs)
        config.validate()
    train_cache = load_window_cache(args.train_cache)
    val_cache = load_window_cache(args.val_cache)
    summary = train_tri_expert(
        config=config,
        train_cache=train_cache,
        val_cache=val_cache,
        train_sidecar=load_sidecar(args.train_sidecar, train_cache),
        val_sidecar=load_sidecar(args.val_sidecar, val_cache),
        output_dir=args.output_dir,
        device_name=args.device,
    )
    print(_canonical_json({"stage": "complete", **summary}), flush=True)


if __name__ == "__main__":
    main()
