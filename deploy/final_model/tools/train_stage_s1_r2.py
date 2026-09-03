"""Train Stage-S1-R2 R1 evidence pooling without accessing any test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from eval.metrics import competition_map
from models.evidence_pooling import EvidencePoolingResidual
from models.multiscale_multistream_tcn import MultiStreamMultiScaleAttentionTCN
from models.stage_s1_r2 import StageS1R2R1
from models.tcn_dataset import WindowMemmapCache, load_window_cache
from tools.build_tcn_multistream_sidecar import load_sidecar
from tools.train_multiscale_multistream_mil import MILConfig
from tools.train_tcn import (
    _append_sidecar,
    _atomic_checkpoint,
    _atomic_json,
    _sha256_file,
    set_deterministic,
)


def _base_model(config: MILConfig, checkpoint: Path, device: torch.device) -> MultiStreamMultiScaleAttentionTCN:
    model = MultiStreamMultiScaleAttentionTCN(
        stream_channels=config.stream_channels, stream_output_dim=config.stream_output_dim,
        dropout=config.dropout, use_geometry=True, stream_lstm_layers=config.stream_lstm_layers,
        use_rule_features=config.use_rule_features, use_pifr_features=config.use_pifr_features,
        use_stgcn_joint=config.use_stgcn_joint,
        use_stgcn_residual_joint=config.use_stgcn_residual_joint,
        use_transformer_encoder=config.use_transformer_encoder,
        transformer_joint_only=config.transformer_joint_only,
        use_discriminative_kinematics=config.use_discriminative_kinematics,
        kinematic_feature_dim=config.kinematic_feature_dim, use_gated_conv=config.use_gated_conv,
        use_motion_guided_fusion=config.use_motion_guided_fusion,
        motion_validity_mask=config.motion_validity_mask,
        per_joint_mask_correction=config.per_joint_mask_correction,
        use_stream_interaction=config.use_stream_interaction,
        use_stage_auxiliary=config.use_stage_auxiliary,
    ).to(device)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("model_state"), dict):
        raise TypeError("base checkpoint 缺少 model_state")
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    return model


@torch.inference_mode()
def _base_outputs(
    cache: WindowMemmapCache, sidecar: np.ndarray, model: StageS1R2R1, *, device: torch.device, batch_size: int
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if cache.metadata.get("split") == "test":
        raise ValueError("Stage-S1-R2 禁止读取 test split")
    logits: list[torch.Tensor] = []
    stages: list[torch.Tensor] = []
    features = cache.array("features")
    for start in range(0, cache.sample_count, batch_size):
        stop = min(cache.sample_count, start + batch_size)
        raw = torch.from_numpy(np.array(features[start:stop], dtype=np.float32, copy=True))
        x = _append_sidecar(raw, sidecar[start:stop], None).to(device)
        window_logits, stage_logits = model.window_outputs(x)
        logits.append(window_logits.cpu())
        if stage_logits is not None:
            stages.append(stage_logits.cpu())
    return torch.cat(logits), torch.cat(stages) if stages else None


def _groups(cache: WindowMemmapCache) -> tuple[list[np.ndarray], torch.Tensor]:
    groups: list[np.ndarray] = []
    labels: list[float] = []
    for clip in cache.metadata["clips"]:
        start, count = int(clip["window_start"]), int(clip["window_count"])
        if count < 1:
            raise ValueError("clip 缺少窗口")
        groups.append(np.arange(start, start + count, dtype=np.int64))
        labels.append(float(bool(clip["has_fall"])))
    return groups, torch.tensor(labels)


def _metrics(scores: np.ndarray, clips: list[dict[str, object]]) -> dict[str, float]:
    labels = {str(clip["clip_id"]): bool(clip["has_fall"]) for clip in clips}
    predictions = {str(clip["clip_id"]): float(scores[index]) for index, clip in enumerate(clips)}
    result = competition_map(labels, predictions, mode="clip")
    return {"clip_map": float(result["map"]), "clip_p_at_r90": float(result["p_at_r90"]), "clip_p_at_r95": float(result["p_at_r95"])}


def _score(pool: EvidencePoolingResidual, logits: torch.Tensor, stages: torch.Tensor | None, groups: list[np.ndarray], batch_size: int) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    device = next(pool.parameters()).device
    for start in range(0, len(groups), batch_size):
        selected = groups[start : start + batch_size]
        indices = np.concatenate(selected)
        outputs.append(
            pool(
                logits[indices].to(device),
                [len(group) for group in selected],
                stage_logits=None if stages is None else stages[indices].to(device),
            ).cpu()
        )
    return torch.cat(outputs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="R1 JSON configuration")
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--train-sidecar", type=Path, required=True)
    parser.add_argument("--val-sidecar", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--batch-size", type=int, help="complete clips per pooling batch")
    args = parser.parse_args()
    r1_config = json.loads(args.config.read_text(encoding="utf-8"))
    required = {"base_config", "bottleneck_dim", "correction_cap", "peak_delta", "epochs", "learning_rate", "seed", "clip_batch_size"}
    if not isinstance(r1_config, dict) or set(r1_config) != required:
        raise ValueError("R1 配置字段不完整或包含未知字段")
    args.epochs = args.epochs if args.epochs is not None else int(r1_config["epochs"])
    args.learning_rate = args.learning_rate if args.learning_rate is not None else float(r1_config["learning_rate"])
    args.seed = args.seed if args.seed is not None else int(r1_config["seed"])
    args.batch_size = args.batch_size if args.batch_size is not None else int(r1_config["clip_batch_size"])
    if args.epochs < 1 or args.learning_rate <= 0 or args.batch_size < 1:
        raise ValueError("训练参数无效")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("输出目录必须为空")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("请求 CUDA 但不可用")
    config = MILConfig.from_json(Path(r1_config["base_config"]))
    args.output_dir.mkdir(parents=True)
    status_path = args.output_dir / "status.json"
    _atomic_json(status_path, {"stage": "initializing"})
    train_cache, val_cache = load_window_cache(args.train_cache, verify_hashes=True), load_window_cache(args.val_cache, verify_hashes=True)
    if train_cache.metadata.get("split") != "train" or val_cache.metadata.get("split") != "val":
        raise ValueError("R1 仅接受 train cache 和一次 val cache")
    set_deterministic(args.seed)
    device = torch.device(args.device)
    base = _base_model(config, args.base_checkpoint, device)
    wrapper = StageS1R2R1(
        base,
        EvidencePoolingResidual(
            bottleneck_dim=int(r1_config["bottleneck_dim"]),
            correction_cap=float(r1_config["correction_cap"]),
            peak_delta=float(r1_config["peak_delta"]),
        ),
    ).to(device)
    _atomic_json(status_path, {"stage": "precomputing_train_outputs"})
    train_logits, train_stages = _base_outputs(
        train_cache,
        load_sidecar(args.train_sidecar, train_cache),
        wrapper,
        device=device,
        batch_size=512,
    )
    _atomic_json(status_path, {"stage": "precomputing_val_outputs"})
    val_logits, val_stages = _base_outputs(
        val_cache,
        load_sidecar(args.val_sidecar, val_cache),
        wrapper,
        device=device,
        batch_size=512,
    )
    train_groups, train_labels = _groups(train_cache)
    val_groups, _ = _groups(val_cache)
    pool = wrapper.pool
    optimizer = torch.optim.AdamW(pool.parameters(), lr=args.learning_rate, weight_decay=1e-3)
    pos_weight = torch.tensor((len(train_labels) - train_labels.sum()) / train_labels.sum(), device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    history: list[dict[str, float | int]] = []
    for epoch in range(args.epochs):
        _atomic_json(status_path, {"stage": "training", "epoch": epoch})
        pool.train()
        order = np.random.default_rng(args.seed + epoch).permutation(len(train_groups))
        losses: list[float] = []
        for start in range(0, len(order), args.batch_size):
            selected = order[start : start + args.batch_size]
            groups = [train_groups[int(index)] for index in selected]
            indices = np.concatenate(groups)
            scores = pool(train_logits[indices].to(device), [len(group) for group in groups], stage_logits=None if train_stages is None else train_stages[indices].to(device))
            loss = criterion(scores, train_labels[selected].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(pool.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        record: dict[str, float | int] = {"epoch": epoch, "train_clip_loss": float(np.mean(losses))}
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
    pool.eval()
    with torch.inference_mode():
        val_scores = torch.sigmoid(_score(pool, val_logits, val_stages, val_groups, args.batch_size)).numpy()
    metrics = _metrics(val_scores, val_cache.metadata["clips"])
    checkpoint = {"model_state": wrapper.state_dict(), "base_checkpoint_sha256": _sha256_file(args.base_checkpoint), "config": vars(args), "val_metrics": metrics}
    _atomic_checkpoint(args.output_dir / "final.pt", checkpoint)
    _atomic_json(args.output_dir / "summary.json", {"stage": "R1", "history": history, "val_metrics": metrics, "base_checkpoint_sha256": _sha256_file(args.base_checkpoint)})
    _atomic_json(status_path, {"stage": "complete"})
    print(json.dumps({"stage": "complete", "val_metrics": metrics}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
