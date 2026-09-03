"""Challenge B2：多语义辅助头与等预算 control 的完整 val 公平 pilot。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from models.tcn import count_params
from models.tcn_dataset import SEMANTIC_TO_CODE, load_window_cache
from models.tcn_multitask import (
    AUXILIARY_CLASSES,
    MultiTaskFallTCN,
    auxiliary_fall_probability,
    build_auxiliary_targets,
)
from tools.train_tcn import evaluate_model, set_deterministic
from tools.train_tcn_challenge_a import (
    _atomic_checkpoint,
    _atomic_json,
    _load_model,
    _sha256_file,
    aggregate_clip_logits,
    predict_cache,
    select_clip_balanced_indices,
)

PROTOCOL = "fall_tcn_challenge_b2_multisemantic_paired_pilot_v2"


@dataclass(frozen=True)
class PilotConfig:
    epochs: int
    comparison_epoch: int
    clips_per_batch: int
    inference_batch_size: int
    learning_rate: float
    weight_decay: float
    grad_clip: float
    semantic_per_clip: int
    hard_negative_top_k: int
    mil_top_k: int
    clip_loss_weight: float
    aux_loss_weight: float
    aux_class_weight_power: float
    deployment_positive_classes: tuple[str, ...]

    @classmethod
    def from_json(cls, path: Path) -> PilotConfig:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or set(payload) != set(cls.__dataclass_fields__):
            raise ValueError("B2 pilot config 字段不完整或包含未知项")
        payload["deployment_positive_classes"] = tuple(
            payload["deployment_positive_classes"]
        )
        config = cls(**payload)
        if min(
            config.epochs,
            config.comparison_epoch,
            config.clips_per_batch,
            config.inference_batch_size,
            config.semantic_per_clip,
            config.hard_negative_top_k,
            config.mil_top_k,
        ) < 1:
            raise ValueError("B2 pilot 整数配置必须为正")
        if not 0.0 < config.aux_class_weight_power <= 1.0:
            raise ValueError("aux_class_weight_power 必须位于 (0,1]")
        if config.learning_rate <= 0 or config.weight_decay < 0:
            raise ValueError("learning_rate/weight_decay 无效")
        if config.grad_clip <= 0 or config.clip_loss_weight < 0 or config.aux_loss_weight <= 0:
            raise ValueError("loss weight/grad_clip 无效")
        if config.comparison_epoch != config.epochs:
            raise ValueError("paired pilot 必须固定比较最终 epoch")
        if config.deployment_positive_classes != ("fall", "fallen"):
            raise ValueError("部署正类必须固定为 fall/fallen")
        return config


LABEL_SPEC = {
    "classes": list(AUXILIARY_CLASSES),
    "window_rules": {
        "background": "background|unlabeled_background -> background",
        "fall_process": "fall_process -> fall",
        "post_fall_state": "post_fall_state -> fallen",
        "hard_negative": (
            "clip prefix lie_down|lying|stand_up -> same class; "
            "all other hard-negative prefixes -> other"
        ),
    },
    "unknown_semantic_code": "fail_closed",
    "unlisted_hard_negative_prefix": "other",
    "boundary_policy": "cache semantic code is authoritative",
    "deployment_score": "softmax(aux)[fall] + softmax(aux)[fallen]",
}


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    try:
        with open(temporary, "wb") as handle:
            np.save(handle, value, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def _state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


class _AuxiliaryScoreView(nn.Module):
    def __init__(self, model: MultiTaskFallTCN):
        super().__init__()
        self.model = model

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        _, auxiliary_logits = self.model.forward_multitask(features)
        probability = auxiliary_fall_probability(auxiliary_logits)
        return torch.logit(probability.clamp(1e-7, 1.0 - 1e-7))


def balanced_class_weights(targets: np.ndarray, *, power: float) -> np.ndarray:
    targets = np.asarray(targets)
    if targets.ndim != 1 or not np.issubdtype(targets.dtype, np.integer):
        raise ValueError("aux targets 必须是一维整数数组")
    counts = np.bincount(targets, minlength=len(AUXILIARY_CLASSES)).astype(np.float64)
    if np.any(counts == 0) or not 0.0 < power <= 1.0:
        raise ValueError("每个辅助类别必须有样本且 power 必须位于 (0,1]")
    weights = np.power(counts, -power)
    return (weights / weights.mean()).astype(np.float32)


def _load_multitask(checkpoint: Path, device: torch.device) -> MultiTaskFallTCN:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = MultiTaskFallTCN().to(device)
    incompatible = model.load_state_dict(payload["model_state"], strict=False)
    if set(incompatible.missing_keys) != {"aux_head.weight", "aux_head.bias"}:
        raise ValueError(f"初始化 checkpoint 缺失键异常: {incompatible.missing_keys}")
    if incompatible.unexpected_keys:
        raise ValueError(f"初始化 checkpoint 多余键异常: {incompatible.unexpected_keys}")
    return model


def _train_branch(
    *,
    branch: str,
    use_auxiliary: bool,
    seed: int,
    config: PilotConfig,
    initial_checkpoint: Path,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    train_aux_targets: torch.Tensor,
    selected_clips: np.ndarray,
    clip_targets: np.ndarray,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    val_clip_indices: np.ndarray,
    val_clips: list[dict[str, Any]],
    class_weights: np.ndarray,
    epoch_orders: np.ndarray,
    run_signature_sha256: str,
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    set_deterministic(seed)
    model = _load_multitask(initial_checkpoint, device)
    initial_state_sha256 = _state_sha256(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )
    positive = float(train_labels.sum())
    window_loss = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor((train_labels.numel() - positive) / positive, device=device)
    )
    clip_positive = float(np.count_nonzero(clip_targets))
    clip_loss = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor((len(clip_targets) - clip_positive) / clip_positive, device=device)
    )
    auxiliary_loss = nn.CrossEntropyLoss(
        weight=torch.from_numpy(class_weights).to(device)
    )
    history = []
    for epoch in range(config.epochs):
        model.train()
        order = epoch_orders[epoch]
        loss_sum = 0.0
        for offset in range(0, len(order), config.clips_per_batch):
            batch_clips = order[offset : offset + config.clips_per_batch]
            positions = []
            for clip_index in batch_clips:
                left = int(np.searchsorted(selected_clips, clip_index, side="left"))
                right = int(np.searchsorted(selected_clips, clip_index, side="right"))
                positions.append(np.arange(left, right, dtype=np.int64))
            batch_positions = np.concatenate(positions)
            batch_x = train_features[batch_positions].to(device)
            batch_y = train_labels[batch_positions].to(device)
            batch_clip_ids = torch.from_numpy(selected_clips[batch_positions]).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, auxiliary_logits = model.forward_multitask(batch_x)
            binary_window_loss = window_loss(logits, batch_y)
            clip_logits, unique_clips = aggregate_clip_logits(
                logits, batch_clip_ids, top_k=config.mil_top_k
            )
            targets = torch.from_numpy(
                clip_targets[unique_clips.cpu().numpy()].astype(np.float32)
            ).to(device)
            total_loss = binary_window_loss + config.clip_loss_weight * clip_loss(
                clip_logits, targets
            )
            aux_targets = train_aux_targets[batch_positions].to(device)
            aux_term = auxiliary_loss(auxiliary_logits, aux_targets)
            total_loss = total_loss + (
                config.aux_loss_weight if use_auxiliary else 0.0
            ) * aux_term
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            loss_sum += float(total_loss.detach())
        binary_metrics = evaluate_model(
            model,
            val_features,
            val_labels,
            val_clip_indices,
            val_clips,
            device=device,
            batch_size=config.inference_batch_size,
        )
        deployment_metrics = (
            evaluate_model(
                _AuxiliaryScoreView(model),
                val_features,
                val_labels,
                val_clip_indices,
                val_clips,
                device=device,
                batch_size=config.inference_batch_size,
            )
            if use_auxiliary
            else None
        )
        scheduler.step()
        record = {
            "epoch": epoch + 1,
            "train_loss": loss_sum / math.ceil(len(order) / config.clips_per_batch),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "binary_val": binary_metrics,
            "deployment_val": deployment_metrics,
        }
        history.append(record)
        if epoch + 1 == config.comparison_epoch:
            _atomic_checkpoint(
                output_dir / branch / "final.pt",
                {
                    "protocol": PROTOCOL,
                    "pilot": True,
                    "branch": branch,
                    "seed": seed,
                    "epoch": epoch + 1,
                    "comparison_epoch": config.comparison_epoch,
                    "run_signature_sha256": run_signature_sha256,
                    "auxiliary_classes": list(AUXILIARY_CLASSES),
                    "deployment_positive_classes": list(
                        config.deployment_positive_classes
                    ),
                    "model_state": model.state_dict(),
                    "binary_metrics": binary_metrics,
                    "deployment_metrics": deployment_metrics,
                },
            )
        print(json.dumps({"stage": "epoch", "branch": branch, **record}, sort_keys=True), flush=True)
    final = history[config.comparison_epoch - 1]
    _atomic_json(output_dir / branch / "history.json", history)
    return {
        "branch": branch,
        "initial_state_sha256": initial_state_sha256,
        "final": final,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--mining-checkpoint", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--challenge-a-summary", type=Path, required=True)
    parser.add_argument("--current-ablation", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("输出目录必须为空")
    config = PilotConfig.from_json(args.config)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("请求 CUDA，但当前不可用")
    current = json.loads(args.current_ablation.read_text(encoding="utf-8"))
    if current.get("split") != "val":
        raise ValueError("current-ablation 必须来自 validation")
    current_metrics = current["tcn_only"]
    challenge_a_summary = json.loads(
        args.challenge_a_summary.read_text(encoding="utf-8")
    )
    matching_a_results = [
        item
        for item in challenge_a_summary.get("results", [])
        if item.get("seed") == args.seed
    ]
    if len(matching_a_results) != 1:
        raise ValueError("Challenge A summary 必须包含唯一的 paired seed")
    challenge_a_result = matching_a_results[0]
    train_cache = load_window_cache(args.train_cache, verify_hashes=True)
    val_cache = load_window_cache(args.val_cache, verify_hashes=True)
    mining_model = _load_model(args.mining_checkpoint, device)
    mining_scores = predict_cache(
        mining_model,
        train_cache,
        device=device,
        batch_size=config.inference_batch_size,
    )
    del mining_model
    clip_targets = np.asarray(
        [bool(item["has_fall"]) for item in train_cache.metadata["clips"]],
        dtype=np.bool_,
    )
    selected = select_clip_balanced_indices(
        train_cache.array("labels"),
        train_cache.array("clip_indices"),
        train_cache.array("semantic_codes"),
        mining_scores,
        clip_targets,
        semantic_per_clip=config.semantic_per_clip,
        hard_negative_top_k=config.hard_negative_top_k,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_path = args.output_dir / "selected_indices.npy"
    _atomic_npy(selected_path, selected)
    rng = np.random.default_rng(args.seed)
    epoch_orders = np.stack(
        [rng.permutation(len(clip_targets)) for _ in range(config.epochs)]
    ).astype(np.int64, copy=False)
    epoch_orders_path = args.output_dir / "epoch_clip_orders.npy"
    _atomic_npy(epoch_orders_path, epoch_orders)
    selected_clips = np.asarray(train_cache.array("clip_indices")[selected], dtype=np.int64)
    train_features = torch.from_numpy(
        np.array(train_cache.array("features")[selected], dtype=np.float32, copy=True)
    )
    train_labels = torch.from_numpy(
        np.array(train_cache.array("labels")[selected], dtype=np.float32, copy=True)
    )
    aux_targets_np = build_auxiliary_targets(
        np.asarray(train_cache.array("semantic_codes")[selected]),
        selected_clips,
        train_cache.metadata["clips"],
    )
    selected_semantic_codes = np.asarray(
        train_cache.array("semantic_codes")[selected], dtype=np.int64
    )
    selected_semantic_counts = {
        name: int(np.count_nonzero(selected_semantic_codes == code))
        for name, code in SEMANTIC_TO_CODE.items()
    }
    expected_selection = {
        "selected_windows": challenge_a_result["selected_windows"],
        "selected_positive_windows": challenge_a_result["selected_positive_windows"],
        "selected_semantic_counts": challenge_a_result["selected_semantic_counts"],
    }
    actual_selection = {
        "selected_windows": len(selected),
        "selected_positive_windows": int(train_labels.sum()),
        "selected_semantic_counts": selected_semantic_counts,
    }
    if actual_selection != expected_selection:
        raise RuntimeError("重建选窗与 Challenge A seed 记录不一致")
    train_aux_targets = torch.from_numpy(aux_targets_np)
    class_weights = balanced_class_weights(
        aux_targets_np, power=config.aux_class_weight_power
    )
    val_features = torch.from_numpy(
        np.array(val_cache.array("features"), dtype=np.float32, copy=True)
    )
    val_labels = torch.from_numpy(
        np.array(val_cache.array("labels"), dtype=np.float32, copy=True)
    )
    val_clip_indices = np.array(val_cache.array("clip_indices"), dtype=np.int64, copy=True)
    artifact_sha256 = {
        "mining_checkpoint": _sha256_file(args.mining_checkpoint),
        "initial_checkpoint": _sha256_file(args.initial_checkpoint),
        "current_ablation": _sha256_file(args.current_ablation),
        "challenge_a_summary": _sha256_file(args.challenge_a_summary),
        "config": _sha256_file(args.config),
        "train_cache_metadata": _sha256_file(args.train_cache / "metadata.json"),
        "val_cache_metadata": _sha256_file(args.val_cache / "metadata.json"),
        "selected_indices": _sha256_file(selected_path),
        "epoch_clip_orders": _sha256_file(epoch_orders_path),
        "label_spec": _canonical_sha256(LABEL_SPEC),
        "training_code": _sha256_file(Path(__file__)),
        "challenge_a_code": _sha256_file(
            Path(__file__).parent / "train_tcn_challenge_a.py"
        ),
        "model_code": _sha256_file(
            Path(__file__).parent.parent / "models" / "tcn_multitask.py"
        ),
    }
    run_record = {
        "protocol": PROTOCOL,
        "pilot": True,
        "seed": args.seed,
        "config": asdict(config),
        "label_spec": LABEL_SPEC,
        "selection_reconstruction": actual_selection,
        "artifact_sha256": artifact_sha256,
        "comparison_rule": "paired branches at fixed final epoch 15",
        "gate_rule": "both MAP and P@R95 >= +3 points vs control and >= current",
        "test_accessed": False,
    }
    run_signature_sha256 = _canonical_sha256(run_record)
    run_record["run_signature_sha256"] = run_signature_sha256
    _atomic_json(args.output_dir / "run.json", run_record)
    common = {
        "seed": args.seed,
        "config": config,
        "initial_checkpoint": args.initial_checkpoint,
        "train_features": train_features,
        "train_labels": train_labels,
        "train_aux_targets": train_aux_targets,
        "selected_clips": selected_clips,
        "clip_targets": clip_targets,
        "val_features": val_features,
        "val_labels": val_labels,
        "val_clip_indices": val_clip_indices,
        "val_clips": val_cache.metadata["clips"],
        "class_weights": class_weights,
        "epoch_orders": epoch_orders,
        "run_signature_sha256": run_signature_sha256,
        "output_dir": args.output_dir,
        "device": device,
    }
    control = _train_branch(branch="control", use_auxiliary=False, **common)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    auxiliary = _train_branch(branch="auxiliary", use_auxiliary=True, **common)
    if control["initial_state_sha256"] != auxiliary["initial_state_sha256"]:
        raise RuntimeError("paired branches 的初始模型状态不一致")
    control_val = control["final"]["binary_val"]
    auxiliary_val = auxiliary["final"]["deployment_val"]
    delta_control = {
        "map": 100.0 * (auxiliary_val["clip_map"] - control_val["clip_map"]),
        "p_at_r95": 100.0
        * (auxiliary_val["clip_p_at_r95"] - control_val["clip_p_at_r95"]),
    }
    delta_current = {
        "map": 100.0 * (auxiliary_val["clip_map"] - current_metrics["map"]),
        "p_at_r95": 100.0
        * (auxiliary_val["clip_p_at_r95"] - current_metrics["p_at_r95"]),
    }
    summary = {
        "protocol": PROTOCOL,
        "run_signature_sha256": run_signature_sha256,
        "pilot": True,
        "seed": args.seed,
        "config": asdict(config),
        "selected_windows": len(selected),
        "auxiliary_classes": list(AUXILIARY_CLASSES),
        "label_spec": LABEL_SPEC,
        "auxiliary_class_counts": np.bincount(
            aux_targets_np, minlength=len(AUXILIARY_CLASSES)
        ).tolist(),
        "auxiliary_class_weights": class_weights.tolist(),
        "current": {"map": current_metrics["map"], "p_at_r95": current_metrics["p_at_r95"]},
        "control": control,
        "auxiliary": auxiliary,
        "paired_initial_state_sha256": control["initial_state_sha256"],
        "delta_vs_control_points": delta_control,
        "delta_vs_current_points": delta_current,
        "proceed_to_full_training": bool(
            delta_control["map"] >= 3.0
            and delta_control["p_at_r95"] >= 3.0
            and delta_current["map"] >= 0.0
            and delta_current["p_at_r95"] >= 0.0
        ),
        "parameter_count": count_params(MultiTaskFallTCN()),
        "artifact_sha256": artifact_sha256,
        "test_accessed": False,
    }
    _atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps({"stage": "complete", **summary}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
