"""Train a bounded transition residual branch on a frozen fall backbone."""

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
from models.tcn_dataset import WindowMemmapCache, load_window_cache
from models.transition_residual_reranker import (
    TemporalTransitionResidualReranker,
    TransitionResidualReranker,
)
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

FALL_AUX = 0
POSTURE_AUX = 1
SIT_AUX = 2
STAND_UP_AUX = 3
OTHER_AUX = 4
AUXILIARY_NAMES = ("fall", "lie_or_lying", "sit_or_sitting", "stand_up", "other")


@dataclass(frozen=True)
class RerankerConfig(MILConfig):
    reranker_hidden_dim: int = 64
    reranker_aux_weight: float = 0.2
    reranker_correction_penalty: float = 0.01
    reranker_correction_scale: float = 2.0
    reranker_gate_center: float = 0.0
    reranker_gate_temperature: float = 1.0
    temporal_transition_hidden_dim: int = 0
    hard_negative_ranking_weight: float = 0.0
    hard_negative_ranking_margin: float = 0.5

    @classmethod
    def from_json(cls, path: Path) -> RerankerConfig:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("重排配置必须是 JSON 对象")
        unknown = set(payload) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"重排配置包含未知字段: {sorted(unknown)}")
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
        if self.reranker_hidden_dim < 8:
            raise ValueError("reranker_hidden_dim 必须至少为 8")
        for name in (
            "reranker_aux_weight",
            "reranker_correction_penalty",
            "reranker_correction_scale",
            "reranker_gate_temperature",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} 必须为正")
        if self.temporal_transition_hidden_dim not in {0} and self.temporal_transition_hidden_dim < 4:
            raise ValueError("temporal_transition_hidden_dim 必须为 0 或至少为 4")
        if self.hard_negative_ranking_weight < 0.0:
            raise ValueError("hard_negative_ranking_weight 不能为负")
        if self.hard_negative_ranking_margin <= 0.0:
            raise ValueError("hard_negative_ranking_margin 必须为正")


def build_auxiliary_targets(labels: np.ndarray, activities: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    activities = np.asarray(activities, dtype=object)
    if labels.shape != activities.shape or labels.ndim != 1:
        raise ValueError("labels/activities 必须为同形一维数组")
    targets = np.full(labels.shape, OTHER_AUX, dtype=np.int64)
    targets[np.isin(activities, ("lie_down", "lying"))] = POSTURE_AUX
    targets[np.isin(activities, ("sit_down", "sitting"))] = SIT_AUX
    targets[activities == "stand_up"] = STAND_UP_AUX
    targets[labels.astype(bool)] = FALL_AUX
    return targets


def hard_negative_ranking_loss(
    clip_logits: torch.Tensor,
    clip_labels: torch.Tensor,
    activities: np.ndarray,
    *,
    margin: float,
) -> torch.Tensor:
    """Make true-fall clips outrank posture-transition clips in the same batch."""
    if clip_logits.ndim != 1 or clip_labels.shape != clip_logits.shape:
        raise ValueError("clip logits/labels 必须为同形一维张量")
    if len(activities) != clip_logits.numel() or margin <= 0.0:
        raise ValueError("activities 或 margin 无效")
    positive = clip_logits[clip_labels > 0.5]
    hard_mask = torch.from_numpy(
        np.isin(activities, ("lie_down", "lying", "stand_up"))
    ).to(clip_logits.device)
    negative = clip_logits[(clip_labels < 0.5) & hard_mask]
    if not positive.numel() or not negative.numel():
        return clip_logits.new_zeros(())
    return torch.relu(margin - (positive[:, None] - negative[None, :])).mean()


def train_epoch_reranker(
    model: TransitionResidualReranker | TemporalTransitionResidualReranker,
    optimizer: torch.optim.Optimizer,
    features: torch.Tensor,
    labels: torch.Tensor,
    groups: list[np.ndarray],
    clip_labels: np.ndarray,
    auxiliary_targets: torch.Tensor,
    clip_activities: np.ndarray,
    *,
    device: torch.device,
    config: RerankerConfig,
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
    counts = torch.bincount(auxiliary_targets, minlength=len(AUXILIARY_NAMES)).float()
    if torch.any(counts == 0):
        raise ValueError("重排辅助目标必须覆盖全部类别")
    auxiliary_criterion = nn.CrossEntropyLoss(
        weight=(counts.sum() / counts).to(device)
    )
    totals = {"loss": 0.0, "hard": 0.0, "auxiliary": 0.0, "penalty": 0.0, "ranking": 0.0}
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
        logits, auxiliary_logits, correction = model.forward_with_auxiliary(batch_x)
        clip_logits = aggregate_topk_logits(logits, group_sizes, config.mil_topk_fraction)
        hard = window_criterion(logits, batch_y) + config.mil_loss_weight * clip_criterion(
            clip_logits, batch_clip_y
        )
        ranking = hard_negative_ranking_loss(
            clip_logits,
            batch_clip_y,
            clip_activities[selected],
            margin=config.hard_negative_ranking_margin,
        )
        auxiliary = auxiliary_criterion(
            auxiliary_logits, auxiliary_targets[window_indices].to(device)
        )
        penalty = correction.square().mean()
        loss = (
            hard
            + config.reranker_aux_weight * auxiliary
            + config.reranker_correction_penalty * penalty
            + config.hard_negative_ranking_weight * ranking
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
            ("hard", hard),
            ("auxiliary", auxiliary),
            ("penalty", penalty),
            ("ranking", ranking),
        ):
            totals[name] += float(value.detach()) * batch_clips
        seen += batch_clips
    return {name: value / seen for name, value in totals.items()}


def train_reranker(
    *,
    config: RerankerConfig,
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
        raise ValueError("重排训练只允许 train/val，不允许 test")
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
    if not hasattr(base, "encode"):
        raise TypeError("重排主干必须提供 encode")
    reranker_type = (
        TemporalTransitionResidualReranker
        if config.temporal_transition_hidden_dim
        else TransitionResidualReranker
    )
    reranker_kwargs: dict[str, Any] = {
        "hidden_dim": config.reranker_hidden_dim,
        "dropout": config.dropout,
        "correction_scale": config.reranker_correction_scale,
        "gate_center": config.reranker_gate_center,
        "gate_temperature": config.reranker_gate_temperature,
        "auxiliary_classes": len(AUXILIARY_NAMES),
    }
    if config.temporal_transition_hidden_dim:
        reranker_kwargs["temporal_hidden_dim"] = config.temporal_transition_hidden_dim
    model = reranker_type(base, **reranker_kwargs).to(device)
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
    groups, clip_labels = _clip_groups(train_clip_indices, train_cache.metadata["clips"])
    all_activities = _activity_groups(train_cache)
    clip_activities = np.asarray(
        [str(all_activities[group[0]]) for group in groups], dtype=object
    )
    auxiliary_targets = torch.from_numpy(
        build_auxiliary_targets(train_y.numpy(), all_activities[train_indices])
    )
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters, lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    base_architecture = base_run.get("architecture")
    if not isinstance(base_architecture, dict):
        raise TypeError("base run 缺少 architecture")
    architecture = {
        "name": reranker_type.__name__,
        "base_architecture": base_architecture,
        "base_checkpoint_sha256": _sha256_file(base_checkpoint_path),
        "auxiliary_classes": list(AUXILIARY_NAMES),
        "hidden_dim": config.reranker_hidden_dim,
        "dropout": config.dropout,
        "correction_scale": config.reranker_correction_scale,
        "gate_center": config.reranker_gate_center,
        "gate_temperature": config.reranker_gate_temperature,
        "base_frozen": True,
        "temporal_transition_hidden_dim": config.temporal_transition_hidden_dim,
        "hard_negative_ranking_weight": config.hard_negative_ranking_weight,
        "hard_negative_ranking_margin": config.hard_negative_ranking_margin,
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
        protocol="transition_residual_reranker_v1",
        code_paths=(
            root / "models" / "multiscale_multistream_tcn.py",
            root / "models" / "transition_gru.py",
            root / "models" / "transition_residual_reranker.py",
            root / "tools" / "train_transition_residual_reranker.py",
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
            "trainable_parameter_count": sum(p.numel() for p in trainable_parameters),
            "train_samples": int(train_y.numel()),
            "train_clips": len(groups),
            "auxiliary_counts": torch.bincount(
                auxiliary_targets, minlength=len(AUXILIARY_NAMES)
            ).tolist(),
            "val_samples": int(val_y.numel()),
            "pilot": False,
        },
    )
    best_map = -math.inf
    history: list[dict[str, Any]] = []
    for epoch in range(config.epochs):
        losses = train_epoch_reranker(
            model,
            optimizer,
            train_x,
            train_y,
            groups,
            clip_labels,
            auxiliary_targets,
            clip_activities,
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
    summary = train_reranker(
        config=RerankerConfig.from_json(args.config),
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
