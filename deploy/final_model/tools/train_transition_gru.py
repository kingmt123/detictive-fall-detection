"""Train a three-state causal GRU re-ranker on non-test window caches."""

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
from models.tcn_dataset import SEMANTIC_TO_CODE, WindowMemmapCache, load_window_cache
from models.transition_gru import (
    CONTROLLED_TRANSITION,
    FALL_INCIDENT,
    OTHER_STATE,
    TRANSITION_CLASS_NAMES,
    TransitionGRU,
)
from tools.build_tcn_multistream_sidecar import load_sidecar
from tools.train_tcn import (
    _append_sidecar,
    _atomic_checkpoint,
    _atomic_json,
    _canonical_json,
    _materialize,
    _run_signature,
    evaluate_model,
    set_deterministic,
)


@dataclass(frozen=True)
class TransitionGRUConfig:
    input_features: str = "transition_rule20"
    epochs: int = 12
    batch_size: int = 512
    learning_rate: float = 3e-4
    weight_decay: float = 5e-4
    hidden_size: int = 32
    num_layers: int = 2
    dropout: float = 0.2
    attention_heads: int = 4
    seed: int = 20260817

    @classmethod
    def from_json(cls, path: Path) -> TransitionGRUConfig:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or set(payload) - set(cls.__dataclass_fields__):
            raise ValueError("Transition GRU 配置字段无效")
        config = cls(**payload)
        config.validate()
        return config

    def validate(self) -> None:
        if self.epochs < 1 or self.batch_size < 1 or self.learning_rate <= 0:
            raise ValueError("epochs/batch_size/learning_rate 必须为正")
        if self.hidden_size < self.attention_heads or self.hidden_size % self.attention_heads:
            raise ValueError("hidden_size 必须能被 attention_heads 整除")


def _targets(semantic_codes: np.ndarray) -> np.ndarray:
    codes = np.asarray(semantic_codes)
    targets = np.full(codes.shape, OTHER_STATE, dtype=np.int64)
    targets[codes == SEMANTIC_TO_CODE["hard_negative"]] = CONTROLLED_TRANSITION
    fall_mask = (codes == SEMANTIC_TO_CODE["fall_process"]) | (
        codes == SEMANTIC_TO_CODE["post_fall_state"]
    )
    targets[fall_mask] = FALL_INCIDENT
    return targets


def _balanced_indices(targets: np.ndarray, seed: int) -> np.ndarray:
    """Keep every fall window and an equal deterministic sample of other states."""
    groups = [np.flatnonzero(targets == label) for label in range(3)]
    if any(not group.size for group in groups):
        raise ValueError("三类转变目标必须均非空")
    count = min(group.size for group in groups)
    rng = np.random.default_rng(seed)
    selected = [rng.choice(group, size=count, replace=False) for group in groups]
    return np.sort(np.concatenate(selected))


def _model(config: TransitionGRUConfig) -> TransitionGRU:
    return TransitionGRU(
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
        attention_heads=config.attention_heads,
    )


def _train_epoch(
    model: TransitionGRU,
    optimizer: torch.optim.Optimizer,
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    config: TransitionGRUConfig,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    order = np.random.default_rng(config.seed + epoch).permutation(features.shape[0])
    criterion = nn.CrossEntropyLoss()
    total = 0.0
    for start in range(0, order.size, config.batch_size):
        indices = torch.from_numpy(order[start : start + config.batch_size])
        optimizer.zero_grad(set_to_none=True)
        logits = model.forward_multiclass(features[indices].to(device))
        loss = criterion(logits, targets[indices].to(device))
        if not torch.isfinite(loss):
            raise FloatingPointError("Transition GRU 出现非有限损失")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total += float(loss.detach()) * indices.numel()
    return total / order.size


def train_transition_gru(
    *,
    config: TransitionGRUConfig,
    train_cache: WindowMemmapCache,
    val_cache: WindowMemmapCache,
    train_sidecar: np.ndarray,
    val_sidecar: np.ndarray,
    output_dir: Path,
    device_name: str,
) -> dict[str, Any]:
    config.validate()
    if train_cache.metadata.get("split") != "train" or val_cache.metadata.get("split") != "val":
        raise ValueError("Transition GRU 只允许 train/val，不允许 test")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("输出目录非空，拒绝覆盖")
    set_deterministic(config.seed)
    train_targets = _targets(train_cache.array("semantic_codes"))
    selected = _balanced_indices(train_targets, config.seed)
    train_x, _, _ = _materialize(train_cache, selected)
    train_x = _append_sidecar(train_x, train_sidecar, selected)
    train_y = torch.from_numpy(train_targets[selected])
    val_x, val_binary, val_clip_indices = _materialize(val_cache, None)
    val_x = _append_sidecar(val_x, val_sidecar, None)
    device = torch.device(device_name)
    model = _model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    root = Path(__file__).parent.parent
    architecture = {"name": "TransitionGRU", "classes": list(TRANSITION_CLASS_NAMES), "feature_source": "rule12_plus_transition8", "causal": True}
    signature_sha256, signature = _run_signature(
        config=config, train_cache=train_cache, val_cache=val_cache, device=device_name,
        max_train_samples=None, max_val_samples=None, protocol="transition_gru_v1",
        code_paths=(root / "models" / "transition_gru.py", root / "models" / "multiscale_multistream_tcn.py", root / "tools" / "train_transition_gru.py"),
        extra_signature=architecture,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_dir / "run.json", {"signature_sha256": signature_sha256, "signature": signature, "architecture": architecture, "parameter_count": count_params(model), "train_samples": int(train_y.numel()), "train_class_counts": np.bincount(train_y.numpy(), minlength=3).tolist(), "val_samples": int(val_binary.numel()), "pilot": False})
    best_map = -math.inf
    history: list[dict[str, Any]] = []
    for epoch in range(config.epochs):
        train_loss = _train_epoch(model, optimizer, train_x, train_y, config=config, device=device, epoch=epoch)
        metrics = evaluate_model(model, val_x, val_binary, val_clip_indices, val_cache.metadata["clips"], device=device, batch_size=config.batch_size)
        scheduler.step()
        record = {"epoch": epoch, "train_loss": train_loss, "learning_rate": optimizer.param_groups[0]["lr"], "val": metrics}
        with (output_dir / "history.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(_canonical_json(record) + "\n")
            handle.flush(); os.fsync(handle.fileno())
        history.append(record)
        best_map = max(best_map, metrics["clip_map"])
        checkpoint = {"epoch": epoch, "run_signature_sha256": signature_sha256, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(), "best_map": best_map, "metrics": record}
        _atomic_checkpoint(output_dir / "last.pt", checkpoint)
        if metrics["clip_map"] == best_map:
            _atomic_checkpoint(output_dir / "best.pt", checkpoint)
        print(_canonical_json({"stage": "epoch", **record}), flush=True)
    best = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    summary = {"run_signature_sha256": signature_sha256, "epochs_completed": config.epochs, "best_clip_map": best_map, "best": best["metrics"], "last": history[-1], "output_dir": str(output_dir.resolve())}
    _atomic_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("train-cache", "val-cache", "train-sidecar", "val-sidecar", "output-dir", "config"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    train, val = load_window_cache(args.train_cache), load_window_cache(args.val_cache)
    summary = train_transition_gru(config=TransitionGRUConfig.from_json(args.config), train_cache=train, val_cache=val, train_sidecar=load_sidecar(args.train_sidecar, train), val_sidecar=load_sidecar(args.val_sidecar, val), output_dir=args.output_dir, device_name=args.device)
    print(_canonical_json({"stage": "complete", **summary}), flush=True)


if __name__ == "__main__":
    main()
