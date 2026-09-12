"""Train controlled TCN+SA-TDGFormer or TCN+InfoGCN MIL ablations."""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from models.dual_expert_skeleton import (
    DualExpertSkeletonFallDetector,
    DualVariant,
    orthogonal_dual_loss,
)
from models.multiscale_multistream_tcn import count_params
from models.tcn_dataset import WindowMemmapCache, load_window_cache
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
class DualExpertConfig(MILConfig):
    variant: DualVariant = "tcn_sa"
    dual_channels: int = 32
    expert_dim: int = 64
    attention_heads: int = 4
    infogcn_dynamic_layers: int = 2
    infogcn_knn_k: int = 4
    diversity_loss_weight: float = 0.01
    information_loss_weight: float = 0.0001
    diversity_warmup_epochs: int = 3
    use_amp: bool = True

    @classmethod
    def from_json(cls, path: Path) -> DualExpertConfig:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("双专家配置必须是 JSON 对象")
        unknown = set(payload) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"双专家配置包含未知字段: {sorted(unknown)}")
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
        if self.variant not in {"sa_only", "tcn_sa", "tcn_info"}:
            raise ValueError("variant 必须为 sa_only、tcn_sa 或 tcn_info")
        if self.dual_channels < 4 or self.dual_channels % self.attention_heads:
            raise ValueError("dual_channels 必须能被 attention_heads 整除")
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


def _model(config: DualExpertConfig) -> DualExpertSkeletonFallDetector:
    return DualExpertSkeletonFallDetector(
        variant=config.variant,
        channels=config.dual_channels,
        expert_dim=config.expert_dim,
        dropout=config.dropout,
        dynamic_layers=config.infogcn_dynamic_layers,
        knn_k=config.infogcn_knn_k,
        heads=config.attention_heads,
    )


def _diversity_scale(config: DualExpertConfig, epoch: int) -> float:
    if config.diversity_warmup_epochs == 0:
        return config.diversity_loss_weight
    return config.diversity_loss_weight * min(
        1.0, (epoch + 1) / config.diversity_warmup_epochs
    )


def train_dual_epoch(
    model: DualExpertSkeletonFallDetector,
    optimizer: torch.optim.Optimizer,
    features: torch.Tensor,
    labels: torch.Tensor,
    groups: list[np.ndarray],
    clip_labels: np.ndarray,
    *,
    device: torch.device,
    config: DualExpertConfig,
    epoch: int,
) -> dict[str, float]:
    model.train()
    clip_order = np.random.default_rng(config.seed + epoch).permutation(len(groups))
    window_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            float((labels.numel() - labels.sum()) / labels.sum()), device=device
        )
    )
    clip_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            float((clip_labels.size - clip_labels.sum()) / clip_labels.sum()),
            device=device,
        )
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
            diversity = (
                orthogonal_dual_loss(output.embeddings)
                if output.embeddings.shape[1] == 2
                else output.logits.new_zeros(())
            )
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
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        clip_count = len(selected)
        for name, value in components.items():
            totals[name] += float(value.detach()) * clip_count
        seen_clips += clip_count
    return {
        name: value / seen_clips for name, value in totals.items()
    } | {"diversity_weight": diversity_weight}


@torch.inference_mode()
def _diagnostics(
    model: DualExpertSkeletonFallDetector,
    features: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    model.eval()
    expert_count = len(model.expert_names)
    weight_sum = np.zeros(expert_count, dtype=np.float64)
    cosine_sum = 0.0
    seen = 0
    for start in range(0, features.shape[0], batch_size):
        output = model.forward_detailed(features[start : start + batch_size].to(device))
        embeddings = nn.functional.normalize(output.embeddings.float(), dim=-1)
        cosine = (
            (embeddings[:, 0] * embeddings[:, 1]).sum(-1)
            if embeddings.shape[1] == 2
            else torch.zeros(embeddings.shape[0], device=embeddings.device)
        )
        weight_sum += output.attention_weights.float().sum(0).cpu().numpy()
        cosine_sum += float(cosine.sum())
        seen += output.logits.numel()
    return {
        "expert_order": list(model.expert_names),
        "mean_attention_weights": (weight_sum / seen).tolist(),
        "mean_pairwise_cosine": cosine_sum / seen,
    }


def train_dual_expert(
    *,
    config: DualExpertConfig,
    train_cache: WindowMemmapCache,
    val_cache: WindowMemmapCache,
    train_sidecar: np.ndarray,
    val_sidecar: np.ndarray,
    output_dir: Path,
    device_name: str,
) -> dict[str, Any]:
    config.validate()
    if train_cache.metadata.get("split") != "train" or val_cache.metadata.get("split") != "val":
        raise ValueError("双专家实验只允许 train/val cache")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("请求 CUDA 训练，但 CUDA 不可用")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("输出目录非空；双专家训练不支持覆盖")
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
    secondary_name = "SA-TDGFormer" if config.variant in {"sa_only", "tcn_sa"} else "InfoGCN"
    architecture = {
        "name": "DualExpertSkeletonFallDetector",
        "variant": config.variant,
        "experts": ([secondary_name] if config.variant == "sa_only" else ["TCN", secondary_name]),
        "expert_fusion": ("none" if config.variant == "sa_only" else "multihead_attention"),
        "expert_dim": config.expert_dim,
        "channels": config.dual_channels,
        "attention_heads": config.attention_heads,
        "orthogonal_loss": {
            "type": "squared_pairwise_cosine",
            "parameter_overhead": 0,
            "weight": config.diversity_loss_weight,
            "warmup_epochs": config.diversity_warmup_epochs,
        },
        "information_bottleneck_kl_weight": (
            config.information_loss_weight if config.variant == "tcn_info" else 0.0
        ),
        "infogcn_topology": (
            {
                "fixed_bottom_layers": 4 - config.infogcn_dynamic_layers,
                "dynamic_knn_last_layers": config.infogcn_dynamic_layers,
                "knn_k": config.infogcn_knn_k,
            }
            if config.variant == "tcn_info"
            else None
        ),
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
        protocol="dual_expert_skeleton_topk_mil_v1",
        code_paths=(
            root / "models" / "dual_expert_skeleton.py",
            root / "models" / "tri_expert_skeleton.py",
            root / "models" / "multiscale_multistream_tcn.py",
            root / "tools" / "train_dual_expert_mil.py",
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
        train_losses = train_dual_epoch(
            model,
            optimizer,
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
    summary = {
        "run_signature_sha256": signature_sha256,
        "epochs_completed": config.epochs,
        "best_clip_map": best_map,
        "best": best_checkpoint["metrics"],
        "last": history[-1],
        "diagnostics": _diagnostics(
            model, val_x, device=device, batch_size=config.batch_size
        ),
        "output_dir": str(output_dir.resolve()),
    }
    _atomic_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--train-sidecar", type=Path, required=True)
    parser.add_argument("--val-sidecar", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = DualExpertConfig.from_json(args.config)
    train_cache = load_window_cache(args.train_cache)
    val_cache = load_window_cache(args.val_cache)
    summary = train_dual_expert(
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
