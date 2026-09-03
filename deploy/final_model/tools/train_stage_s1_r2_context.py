"""Train the frozen-anchor Stage-S1-R2 M2 context residual."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from models.stage_s1_r2 import StageS1R2Context
from models.stage_s1_r2_context import sparse_context_indices
from models.tcn_dataset import WindowMemmapCache, load_window_cache
from tools.build_tcn_multistream_sidecar import load_sidecar
from tools.train_multiscale_multistream_mil import MILConfig
from tools.train_stage_s1_r2 import _base_model, _groups, _metrics
from tools.train_tcn import (
    _append_sidecar,
    _atomic_checkpoint,
    _atomic_json,
    _sha256_file,
    aggregate_group_logits,
    set_deterministic,
)


def _context_batch(
    features: np.ndarray, sidecar: np.ndarray, rows: np.ndarray, sources: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor]:
    local = np.array(features[rows], dtype=np.float32, copy=True)
    local_sidecar = np.array(sidecar[rows], dtype=np.float32, copy=True)
    context = np.zeros_like(local)
    context_sidecar = np.zeros_like(local_sidecar)
    selected = sources[rows]
    for column in range(16):
        source = selected[:, column]
        available = source >= 0
        if available.any():
            context[available, column] = features[source[available], -1]
            context_sidecar[available, column] = sidecar[source[available], -1]
        if column >= 10:
            local_offset = (column - 10) * 3
            context[~available, column] = local[~available, local_offset]
            context_sidecar[~available, column] = local_sidecar[~available, local_offset]
    return _append_sidecar(torch.from_numpy(local), local_sidecar, None), _append_sidecar(torch.from_numpy(context), context_sidecar, None)


def _sources(cache: WindowMemmapCache) -> np.ndarray:
    ranges = [(int(clip["window_start"]), int(clip["window_count"])) for clip in cache.metadata["clips"]]
    return sparse_context_indices(
        np.asarray(cache.array("end_times"), dtype=np.float32),
        np.asarray(cache.array("track_ids"), dtype=np.int64),
        ranges,
    )


@torch.inference_mode()
def _validation_scores(
    model: StageS1R2Context, cache: WindowMemmapCache, sidecar: np.ndarray, sources: np.ndarray, groups: list[np.ndarray], device: torch.device
) -> np.ndarray:
    features = cache.array("features")
    scores = np.empty(len(groups), dtype=np.float32)
    model.eval()
    for index, group in enumerate(groups):
        local, context = _context_batch(features, sidecar, group, sources)
        logits, _ = model.forward_with_stage(local.to(device), context.to(device))
        scores[index] = float(torch.sigmoid(logits.max()).cpu())
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--train-sidecar", type=Path, required=True)
    parser.add_argument("--val-sidecar", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    settings = json.loads(args.config.read_text(encoding="utf-8"))
    required = {"base_config", "bottleneck_dim", "correction_cap", "epochs", "learning_rate", "seed", "clip_batch_size"}
    if not isinstance(settings, dict) or set(settings) != required:
        raise ValueError("R2 配置字段不完整或包含未知字段")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("输出目录必须为空")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("请求 CUDA 但不可用")
    config = MILConfig.from_json(Path(settings["base_config"]))
    train_cache, val_cache = load_window_cache(args.train_cache, verify_hashes=True), load_window_cache(args.val_cache, verify_hashes=True)
    if train_cache.metadata.get("split") != "train" or val_cache.metadata.get("split") != "val":
        raise ValueError("R2 仅接受 train 和 val cache")
    set_deterministic(int(settings["seed"]))
    device = torch.device(args.device)
    base = _base_model(config, args.base_checkpoint, device)
    model = StageS1R2Context(base, bottleneck_dim=int(settings["bottleneck_dim"]), correction_cap=float(settings["correction_cap"])).to(device)
    train_sidecar, val_sidecar = load_sidecar(args.train_sidecar, train_cache), load_sidecar(args.val_sidecar, val_cache)
    train_sources, val_sources = _sources(train_cache), _sources(val_cache)
    train_groups, clip_labels = _groups(train_cache)
    val_groups, _ = _groups(val_cache)
    features, labels = train_cache.array("features"), np.asarray(train_cache.array("labels"), dtype=np.float32)
    optimizer = torch.optim.AdamW(model.adapter.parameters(), lr=float(settings["learning_rate"]), weight_decay=1e-3)
    window_weight = torch.tensor((labels.size - labels.sum()) / labels.sum(), device=device)
    clip_weight = (
        (len(clip_labels) - clip_labels.sum()) / clip_labels.sum()
    ).to(device)
    window_loss, clip_loss = nn.BCEWithLogitsLoss(pos_weight=window_weight), nn.BCEWithLogitsLoss(pos_weight=clip_weight)
    args.output_dir.mkdir(parents=True)
    status_path = args.output_dir / "status.json"
    history: list[dict[str, float | int]] = []
    batch_size, seed = int(settings["clip_batch_size"]), int(settings["seed"])
    for epoch in range(int(settings["epochs"])):
        _atomic_json(status_path, {"stage": "training", "epoch": epoch})
        model.train()
        losses: list[float] = []
        order = np.random.default_rng(seed + epoch).permutation(len(train_groups))
        for start in range(0, len(order), batch_size):
            selected = order[start : start + batch_size]
            groups = [train_groups[int(index)] for index in selected]
            rows = np.concatenate(groups)
            local, context = _context_batch(features, train_sidecar, rows, train_sources)
            logits, _ = model.forward_with_stage(local.to(device), context.to(device))
            clip_logits = aggregate_group_logits(logits, [len(group) for group in groups], mode="max")
            loss = window_loss(logits, torch.from_numpy(labels[rows]).to(device)) + 0.5 * clip_loss(clip_logits, clip_labels[selected].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.adapter.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        record: dict[str, float | int] = {"epoch": epoch, "train_loss": float(np.mean(losses))}
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
    _atomic_json(status_path, {"stage": "validating_once"})
    scores = _validation_scores(model, val_cache, val_sidecar, val_sources, val_groups, device)
    metrics = _metrics(scores, val_cache.metadata["clips"])
    _atomic_checkpoint(args.output_dir / "final.pt", {"model_state": model.state_dict(), "config": settings, "base_checkpoint_sha256": _sha256_file(args.base_checkpoint), "val_metrics": metrics})
    _atomic_json(args.output_dir / "summary.json", {"stage": "R2", "history": history, "val_metrics": metrics, "base_checkpoint_sha256": _sha256_file(args.base_checkpoint)})
    _atomic_json(status_path, {"stage": "complete"})
    print(json.dumps({"stage": "complete", "val_metrics": metrics}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
