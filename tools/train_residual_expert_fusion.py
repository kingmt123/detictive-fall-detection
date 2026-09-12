"""Train graph and Transformer delta experts on a frozen multistream TCN."""

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

from models.multiscale_multistream_tcn import count_params
from models.residual_expert_fusion import (
    ResidualExpertFusion,
    orthogonal_expert_loss,
)
from models.tcn_dataset import WindowMemmapCache, load_window_cache
from tools.build_tcn_multistream_sidecar import load_sidecar
from tools.evaluate_hard_negative_cohort import _model_from_run
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
    _sha256_file,
    evaluate_model,
    select_training_indices,
    set_deterministic,
)


@dataclass(frozen=True)
class ResidualExpertFusionConfig(MILConfig):
    expert_channels: int = 32
    expert_output_dim: int = 64
    transformer_layers: int = 2
    fusion_aux_weight: float = 0.25
    fusion_diversity_weight: float = 0.02
    fusion_correction_penalty: float = 0.01
    fusion_correction_scale: float = 1.5
    fusion_use_global_gate: bool = False
    fusion_global_gate_initial_logit: float = -2.0

    @classmethod
    def from_json(cls, path: Path) -> ResidualExpertFusionConfig:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("Residual expert fusion 配置必须是 JSON 对象")
        unknown = set(payload) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Residual expert fusion 配置包含未知字段: {sorted(unknown)}")
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
        if self.expert_channels < 4 or self.expert_channels % 4:
            raise ValueError("expert_channels 必须是不小于 4 的 4 的倍数")
        if self.expert_output_dim < 1 or self.transformer_layers < 1:
            raise ValueError("expert_output_dim/transformer_layers 无效")
        for name in (
            "fusion_aux_weight",
            "fusion_diversity_weight",
            "fusion_correction_penalty",
            "fusion_correction_scale",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} 必须为正")
        if not isinstance(self.fusion_use_global_gate, bool):
            raise TypeError("fusion_use_global_gate 必须是布尔值")
        if not math.isfinite(self.fusion_global_gate_initial_logit):
            raise ValueError("fusion_global_gate_initial_logit 必须有限")


def train_epoch_residual_fusion(
    model: ResidualExpertFusion,
    optimizer: torch.optim.Optimizer,
    features: torch.Tensor,
    labels: torch.Tensor,
    groups: list[np.ndarray],
    clip_labels: np.ndarray,
    *,
    device: torch.device,
    config: ResidualExpertFusionConfig,
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
    totals = {
        "loss": 0.0,
        "main": 0.0,
        "auxiliary": 0.0,
        "diversity": 0.0,
        "penalty": 0.0,
    }
    seen = 0
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
        fused, auxiliary_logits, corrections, graph, transformer = (
            model.forward_with_experts(batch_x)
        )
        clip_fused = aggregate_topk_logits(
            fused, group_sizes, config.mil_topk_fraction
        )
        main = window_criterion(fused, batch_y) + config.mil_loss_weight * clip_criterion(
            clip_fused, batch_clip_y
        )
        auxiliary_losses = []
        for expert_index in range(auxiliary_logits.shape[1]):
            expert_logits = auxiliary_logits[:, expert_index]
            expert_clip = aggregate_topk_logits(
                expert_logits, group_sizes, config.mil_topk_fraction
            )
            auxiliary_losses.append(
                window_criterion(expert_logits, batch_y)
                + config.mil_loss_weight * clip_criterion(expert_clip, batch_clip_y)
            )
        auxiliary = torch.stack(auxiliary_losses).mean()
        diversity = orthogonal_expert_loss(graph, transformer)
        penalty = corrections.square().mean()
        loss = (
            main
            + config.fusion_aux_weight * auxiliary
            + config.fusion_diversity_weight * diversity
            + config.fusion_correction_penalty * penalty
        )
        loss.backward()
        nn.utils.clip_grad_norm_(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            config.grad_clip,
        )
        optimizer.step()
        batch_clips = len(selected)
        for name, value in (
            ("loss", loss),
            ("main", main),
            ("auxiliary", auxiliary),
            ("diversity", diversity),
            ("penalty", penalty),
        ):
            totals[name] += float(value.detach()) * batch_clips
        seen += batch_clips
    return {name: value / seen for name, value in totals.items()}


def train_residual_expert_fusion(
    *,
    config: ResidualExpertFusionConfig,
    train_cache: WindowMemmapCache,
    val_cache: WindowMemmapCache,
    train_sidecar: np.ndarray,
    val_sidecar: np.ndarray,
    base_run_path: Path,
    base_checkpoint_path: Path,
    output_dir: Path,
    device_name: str,
) -> dict[str, Any]:
    config.validate()
    if train_cache.metadata.get("split") != "train" or val_cache.metadata.get("split") != "val":
        raise ValueError("Residual expert fusion 只允许 train/val，不允许 test")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("输出目录非空，拒绝覆盖")
    set_deterministic(config.seed)
    device = torch.device(device_name)
    base_run = json.loads(Path(base_run_path).read_text(encoding="utf-8"))
    base, _, _ = _model_from_run(
        model_kind="multiscale_multistream_tcn",
        run=base_run,
        checkpoint=base_checkpoint_path,
        device=device,
    )
    model = ResidualExpertFusion(
        base,
        expert_channels=config.expert_channels,
        expert_output_dim=config.expert_output_dim,
        transformer_layers=config.transformer_layers,
        dropout=config.dropout,
        correction_scale=config.fusion_correction_scale,
        max_frames=max(32, config.window_size),
        use_global_gate=config.fusion_use_global_gate,
        global_gate_initial_logit=config.fusion_global_gate_initial_logit,
    ).to(device)
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
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )
    base_architecture = base_run.get("architecture")
    if not isinstance(base_architecture, dict):
        raise TypeError("base run 缺少 architecture")
    architecture = {
        "name": "ResidualExpertFusion",
        "base_architecture": base_architecture,
        "base_checkpoint_sha256": _sha256_file(base_checkpoint_path),
        "base_frozen": True,
        "experts": ["stgcn_joint", "causal_transformer_joint"],
        "expert_channels": config.expert_channels,
        "expert_output_dim": config.expert_output_dim,
        "transformer_layers": config.transformer_layers,
        "correction_scale": config.fusion_correction_scale,
        "use_global_gate": config.fusion_use_global_gate,
        "global_gate_initial_logit": config.fusion_global_gate_initial_logit,
        "fusion": "softmax_weighted_bounded_delta_logit",
        "diversity": "squared_cosine_embedding_loss",
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
        protocol="residual_expert_fusion_v1",
        code_paths=(
            root / "models" / "multiscale_multistream_tcn.py",
            root / "models" / "residual_expert_fusion.py",
            root / "tools" / "train_residual_expert_fusion.py",
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
            "trainable_parameter_count": sum(
                parameter.numel() for parameter in trainable_parameters
            ),
            "train_samples": int(train_y.numel()),
            "train_clips": len(groups),
            "val_samples": int(val_y.numel()),
            "pilot": False,
        },
    )
    best_map = -math.inf
    history: list[dict[str, Any]] = []
    for epoch in range(config.epochs):
        losses = train_epoch_residual_fusion(
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
            "train_loss": losses,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "expert_weights": model.expert_weights().detach().cpu().tolist(),
            "global_gate": float(model.global_gate().detach()),
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
        if improved:
            best_map = metrics["clip_map"]
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
    summary = {
        "run_signature_sha256": signature_sha256,
        "epochs_completed": config.epochs,
        "best_clip_map": best_map,
        "last": history[-1],
        "output_dir": str(output_dir.resolve()),
    }
    _atomic_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "train-cache",
        "val-cache",
        "train-sidecar",
        "val-sidecar",
        "base-run",
        "base-checkpoint",
        "output-dir",
        "config",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    train = load_window_cache(args.train_cache)
    val = load_window_cache(args.val_cache)
    summary = train_residual_expert_fusion(
        config=ResidualExpertFusionConfig.from_json(args.config),
        train_cache=train,
        val_cache=val,
        train_sidecar=load_sidecar(args.train_sidecar, train),
        val_sidecar=load_sidecar(args.val_sidecar, val),
        base_run_path=args.base_run,
        base_checkpoint_path=args.base_checkpoint,
        output_dir=args.output_dir,
        device_name=args.device,
    )
    print(_canonical_json({"stage": "complete", **summary}), flush=True)


if __name__ == "__main__":
    main()
