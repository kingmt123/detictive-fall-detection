"""Jointly train OF-Syn and FallVision without inventing external event labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from models.multiscale_multistream_tcn import (
    MultiStreamMultiScaleAttentionTCN,
    count_params,
)
from models.tcn_dataset import WindowMemmapCache, load_window_cache
from tools.build_tcn_multistream_sidecar import load_sidecar
from tools.evaluate_hard_negative_cohort import _model_from_run
from tools.train_fallvision_joint_mil import _split, discover_clips, materialize_windows
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
class MixedConfig(MILConfig):
    external_loss_weight: float = 0.2
    external_interval: int = 2
    external_clip_batch_size: int = 8

    @classmethod
    def from_json(cls, path: Path) -> MixedConfig:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("混合训练配置必须是 JSON 对象")
        unknown = set(payload) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"混合训练配置包含未知字段: {sorted(unknown)}")
        if "channels" in payload:
            payload["channels"] = tuple(payload["channels"])
        if "hard_negative_activities" in payload:
            payload["hard_negative_activities"] = tuple(payload["hard_negative_activities"])
        config = cls(**payload)
        config.validate()
        return config

    def validate(self) -> None:
        super().validate()
        if self.external_loss_weight <= 0.0:
            raise ValueError("external_loss_weight 必须为正")
        if self.external_interval < 1 or self.external_clip_batch_size < 1:
            raise ValueError("external_interval/external_clip_batch_size 必须为正")


def _external_loss(
    model: MultiStreamMultiScaleAttentionTCN,
    classifier: nn.Linear,
    windows: torch.Tensor,
    groups: list[np.ndarray],
    labels: np.ndarray,
    selected: np.ndarray,
    *,
    config: MixedConfig,
    device: torch.device,
) -> torch.Tensor:
    indices = np.concatenate([groups[int(index)] for index in selected])
    sizes = [int(groups[int(index)].size) for index in selected]
    joint = windows[indices].to(device)
    logits = classifier(model.streams[0](joint)).squeeze(-1)
    clip_logits = aggregate_topk_logits(logits, sizes, config.mil_topk_fraction)
    selected_labels = torch.from_numpy(labels[selected]).to(device)
    pos_weight = torch.tensor(
        (labels.size - labels.sum()) / labels.sum(), device=device
    )
    return nn.functional.binary_cross_entropy_with_logits(
        clip_logits, selected_labels, pos_weight=pos_weight
    )


def train_epoch_mixed(
    model: MultiStreamMultiScaleAttentionTCN,
    external_classifier: nn.Linear,
    optimizer: torch.optim.Optimizer,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    groups: list[np.ndarray],
    clip_labels: np.ndarray,
    external_x: torch.Tensor,
    external_groups: list[np.ndarray],
    external_labels: np.ndarray,
    *,
    device: torch.device,
    config: MixedConfig,
    epoch: int,
) -> dict[str, float]:
    model.train()
    external_classifier.train()
    clip_order = np.random.default_rng(config.seed + epoch).permutation(len(groups))
    external_order = np.random.default_rng(config.seed + 1_000_000 + epoch).permutation(
        len(external_groups)
    )
    window_pos_weight = float((train_y.numel() - train_y.sum()) / train_y.sum())
    clip_pos_weight = float((clip_labels.size - clip_labels.sum()) / clip_labels.sum())
    window_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(window_pos_weight, device=device)
    )
    clip_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(clip_pos_weight, device=device)
    )
    totals = {"loss": 0.0, "ofsyn": 0.0, "external": 0.0}
    seen = 0
    external_offset = 0
    for batch_number, start in enumerate(range(0, len(groups), config.mil_clip_batch_size)):
        selected = clip_order[start : start + config.mil_clip_batch_size]
        group_sizes = [int(groups[int(index)].size) for index in selected]
        window_indices = np.concatenate([groups[int(index)] for index in selected])
        batch_x = augment_training_features(
            train_x[window_indices].to(device),
            config,
            seed=config.seed + epoch * 1_000_003 + start,
        )
        batch_y = train_y[window_indices].to(device)
        batch_clip_y = torch.from_numpy(clip_labels[selected]).to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_x)
        clip_logits = aggregate_topk_logits(logits, group_sizes, config.mil_topk_fraction)
        ofsyn = window_criterion(logits, batch_y) + config.mil_loss_weight * clip_criterion(
            clip_logits, batch_clip_y
        )
        external = torch.zeros((), device=device)
        if batch_number % config.external_interval == 0:
            positions = (
                np.arange(config.external_clip_batch_size) + external_offset
            ) % external_order.size
            selected_external = external_order[positions]
            external_offset = (
                external_offset + config.external_clip_batch_size
            ) % external_order.size
            external = _external_loss(
                model,
                external_classifier,
                external_x,
                external_groups,
                external_labels,
                selected_external,
                config=config,
                device=device,
            )
        loss = ofsyn + config.external_loss_weight * external
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(external_classifier.parameters()), config.grad_clip
        )
        optimizer.step()
        batch_clips = len(selected)
        totals["loss"] += float(loss.detach()) * batch_clips
        totals["ofsyn"] += float(ofsyn.detach()) * batch_clips
        totals["external"] += float(external.detach()) * batch_clips
        seen += batch_clips
    return {name: value / seen for name, value in totals.items()}


def train_mixed(
    *,
    config: MixedConfig,
    train_cache: WindowMemmapCache,
    val_cache: WindowMemmapCache,
    train_sidecar: np.ndarray,
    val_sidecar: np.ndarray,
    fallvision_root: Path,
    base_run_path: Path,
    base_checkpoint_path: Path,
    output_dir: Path,
    device_name: str,
) -> dict[str, Any]:
    config.validate()
    if train_cache.metadata.get("split") != "train" or val_cache.metadata.get("split") != "val":
        raise ValueError("混合训练只允许 OF-Syn train/val")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("输出目录非空，拒绝覆盖")
    set_deterministic(config.seed)
    device = torch.device(device_name)
    external_train, _ = _split(discover_clips(fallvision_root), config.seed)
    external_x, external_groups, external_labels, external_ids = materialize_windows(
        external_train
    )
    base_run = json.loads(Path(base_run_path).read_text(encoding="utf-8"))
    model, _, _ = _model_from_run(
        model_kind="multiscale_multistream_tcn",
        run=base_run,
        checkpoint=base_checkpoint_path,
        device=device,
    )
    external_classifier = nn.Linear(
        model.streams[0].output_projection.out_features, 1
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
    groups, clip_labels = _clip_groups(train_clip_indices, train_cache.metadata["clips"])
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(external_classifier.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    root = Path(__file__).parent.parent
    architecture = {
        "name": "MixedFallVisionMultiStreamMIL",
        "base_architecture": base_run["architecture"],
        "base_checkpoint_sha256": _sha256_file(base_checkpoint_path),
        "external_source": "FallVision Harvard Dataverse doi:10.7910/DVN/75QPKK",
        "external_protocol": "clip_mil_joint_stream_only_v1",
        "external_train_clips": len(external_groups),
        "external_train_windows": int(external_x.shape[0]),
        "external_clip_ids_sha256": hashlib.sha256(
            "\n".join(external_ids).encode()
        ).hexdigest(),
        "external_loss_weight": config.external_loss_weight,
        "external_interval": config.external_interval,
        "causal": True,
    }
    signature_sha256, signature = _run_signature(
        config=config,
        train_cache=train_cache,
        val_cache=val_cache,
        device=device_name,
        max_train_samples=None,
        max_val_samples=None,
        protocol="mixed_ofsyn_fallvision_joint_mil_v1",
        code_paths=(
            root / "models" / "multiscale_multistream_tcn.py",
            root / "models" / "fallvision_joint_mil.py",
            root / "tools" / "train_fallvision_joint_mil.py",
            root / "tools" / "train_mixed_fallvision_mil.py",
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
            "external_classifier_parameter_count": count_params(external_classifier),
            "train_samples": int(train_y.numel()),
            "train_clips": len(groups),
            "val_samples": int(val_y.numel()),
            "pilot": False,
        },
    )
    best_map = -math.inf
    history: list[dict[str, Any]] = []
    for epoch in range(config.epochs):
        losses = train_epoch_mixed(
            model,
            external_classifier,
            optimizer,
            train_x,
            train_y,
            groups,
            clip_labels,
            external_x,
            external_groups,
            external_labels,
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
        record = {"epoch": epoch, "train_loss": losses, "learning_rate": optimizer.param_groups[0]["lr"], "val": metrics}
        with (output_dir / "history.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
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
            "external_classifier_state": external_classifier.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_map": best_map,
            "metrics": record,
        }
        _atomic_checkpoint(output_dir / "last.pt", checkpoint)
        if improved:
            _atomic_checkpoint(output_dir / "best.pt", checkpoint)
        print(_canonical_json({"stage": "epoch", **record}), flush=True)
    summary = {"run_signature_sha256": signature_sha256, "epochs_completed": config.epochs, "best_clip_map": best_map, "last": history[-1], "output_dir": str(output_dir.resolve())}
    _atomic_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "train-cache", "val-cache", "train-sidecar", "val-sidecar", "fallvision-root",
        "base-run", "base-checkpoint", "output-dir", "config",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    train = load_window_cache(args.train_cache)
    val = load_window_cache(args.val_cache)
    summary = train_mixed(
        config=MixedConfig.from_json(args.config),
        train_cache=train,
        val_cache=val,
        train_sidecar=load_sidecar(args.train_sidecar, train),
        val_sidecar=load_sidecar(args.val_sidecar, val),
        fallvision_root=args.fallvision_root,
        base_run_path=args.base_run,
        base_checkpoint_path=args.base_checkpoint,
        output_dir=args.output_dir,
        device_name=args.device,
    )
    print(_canonical_json({"stage": "complete", **summary}), flush=True)


if __name__ == "__main__":
    main()
