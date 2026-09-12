"""Challenge B3: paired 123D multi-stream feature canary on full validation."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from models.tcn import FallTCN, count_params
from models.tcn_dataset import WindowMemmapCache, load_window_cache
from models.tcn_multistream import (
    FEATURE_DIM,
    JOINT_DIM,
    SIDECAR_DIM,
    build_multistream_features,
    load_expanded_checkpoint,
)
from tools.train_tcn import evaluate_model, set_deterministic
from tools.train_tcn_challenge_a import (
    _atomic_checkpoint,
    _atomic_json,
    _load_model,
    _sha256_file,
    aggregate_clip_logits,
)

PROTOCOL = "fall_tcn_challenge_b3_multistream_paired_pilot_v1"


@dataclass(frozen=True)
class B3Config:
    epochs: int
    comparison_epoch: int
    clips_per_batch: int
    inference_batch_size: int
    learning_rate: float
    weight_decay: float
    grad_clip: float
    mil_top_k: int
    clip_loss_weight: float

    @classmethod
    def from_json(cls, path: Path) -> B3Config:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or set(payload) != set(cls.__dataclass_fields__):
            raise ValueError("B3 config 字段不完整或包含未知项")
        config = cls(**payload)
        integers = (
            config.epochs,
            config.comparison_epoch,
            config.clips_per_batch,
            config.inference_batch_size,
            config.mil_top_k,
        )
        if any(not isinstance(value, int) or value < 1 for value in integers):
            raise ValueError("B3 整数配置必须为正整数")
        if config.comparison_epoch != config.epochs:
            raise ValueError("paired canary 必须固定比较最终 epoch")
        if config.learning_rate <= 0 or config.weight_decay < 0:
            raise ValueError("learning_rate/weight_decay 无效")
        if config.grad_clip <= 0 or config.clip_loss_weight < 0:
            raise ValueError("grad_clip/clip_loss_weight 无效")
        return config


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def validate_epoch_orders(orders: np.ndarray, *, epochs: int, clips: int) -> None:
    if orders.shape != (epochs, clips) or not np.issubdtype(orders.dtype, np.integer):
        raise ValueError("epoch orders 的形状或 dtype 无效")
    expected = np.arange(clips, dtype=np.int64)
    if any(not np.array_equal(np.sort(row), expected) for row in orders):
        raise ValueError("epoch orders 每行必须是全部 clip 的排列")


def _load_sidecar(
    root: Path, base_cache: WindowMemmapCache, *, verify_hashes: bool
) -> tuple[np.memmap, dict[str, Any]]:
    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_shape = (
        base_cache.sample_count,
        int(base_cache.metadata["window_config"]["window_size"]),
        SIDECAR_DIM,
    )
    if tuple(metadata.get("shape", ())) != expected_shape:
        raise ValueError("sidecar shape 与 base cache 不一致")
    if metadata.get("base_signature_sha256") != base_cache.metadata.get(
        "signature_sha256"
    ):
        raise ValueError("sidecar 与 base cache 签名不一致")
    file_record = metadata.get("files", {}).get("sidecar")
    if not isinstance(file_record, dict):
        raise TypeError("sidecar metadata 缺少文件记录")
    path = root / str(file_record.get("name"))
    if verify_hashes and _sha256_file(path) != file_record.get("sha256"):
        raise ValueError("sidecar SHA-256 不匹配")
    expected_bytes = int(np.prod(expected_shape)) * np.dtype("<f4").itemsize
    if path.stat().st_size != expected_bytes:
        raise ValueError("sidecar 文件大小不匹配")
    return np.memmap(path, mode="r", dtype="<f4", shape=expected_shape), metadata


def _load_multistream(checkpoint: Path, device: torch.device) -> FallTCN:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = FallTCN(in_dim=FEATURE_DIM).to(device)
    load_expanded_checkpoint(model, payload["model_state"])
    return model


class _FeatureView(nn.Module):
    def __init__(self, model: FallTCN, *, include_extra: bool):
        super().__init__()
        self.model = model
        self.include_extra = include_extra

    def forward(self, raw_features: torch.Tensor) -> torch.Tensor:
        joints = raw_features[..., :JOINT_DIM]
        sidecar = raw_features[..., JOINT_DIM:]
        features = build_multistream_features(
            joints, sidecar, include_extra=self.include_extra
        )
        return self.model(features)


def _train_branch(
    *,
    branch: str,
    include_extra: bool,
    seed: int,
    config: B3Config,
    initial_checkpoint: Path,
    train_raw: torch.Tensor,
    train_labels: torch.Tensor,
    selected_clips: np.ndarray,
    clip_targets: np.ndarray,
    val_raw: torch.Tensor,
    val_labels: torch.Tensor,
    val_clip_indices: np.ndarray,
    val_clips: list[dict[str, Any]],
    epoch_orders: np.ndarray,
    output_dir: Path,
    run_signature_sha256: str,
    device: torch.device,
) -> dict[str, Any]:
    set_deterministic(seed)
    model = _load_multistream(initial_checkpoint, device)
    initial_state_sha256 = _state_sha256(model)
    view = _FeatureView(model, include_extra=include_extra)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )
    positives = float(train_labels.sum())
    window_loss = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor((train_labels.numel() - positives) / positives, device=device)
    )
    clip_positives = float(np.count_nonzero(clip_targets))
    clip_loss = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            (len(clip_targets) - clip_positives) / clip_positives, device=device
        )
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
            batch_raw = train_raw[batch_positions].to(device)
            batch_y = train_labels[batch_positions].to(device)
            batch_clip_ids = torch.from_numpy(selected_clips[batch_positions]).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = view(batch_raw)
            clip_logits, unique_clips = aggregate_clip_logits(
                logits, batch_clip_ids, top_k=config.mil_top_k
            )
            targets = torch.from_numpy(
                clip_targets[unique_clips.cpu().numpy()].astype(np.float32)
            ).to(device)
            loss = window_loss(logits, batch_y) + config.clip_loss_weight * clip_loss(
                clip_logits, targets
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            loss_sum += float(loss.detach())
        metrics = evaluate_model(
            view,
            val_raw,
            val_labels,
            val_clip_indices,
            val_clips,
            device=device,
            batch_size=config.inference_batch_size,
        )
        scheduler.step()
        record = {
            "epoch": epoch + 1,
            "train_loss": loss_sum / math.ceil(len(order) / config.clips_per_batch),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "val": metrics,
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
                    "feature_dim": FEATURE_DIM,
                    "include_extra": include_extra,
                    "run_signature_sha256": run_signature_sha256,
                    "model_state": model.state_dict(),
                    "metrics": metrics,
                },
            )
        print(
            json.dumps({"stage": "epoch", "branch": branch, **record}, sort_keys=True),
            flush=True,
        )
    _atomic_json(output_dir / branch / "history.json", history)
    return {
        "branch": branch,
        "initial_state_sha256": initial_state_sha256,
        "final": history[config.comparison_epoch - 1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--train-sidecar", type=Path, required=True)
    parser.add_argument("--val-sidecar", type=Path, required=True)
    parser.add_argument("--selected-indices", type=Path, required=True)
    parser.add_argument("--epoch-orders", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--current-ablation", type=Path, required=True)
    parser.add_argument("--b2-summary", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("输出目录必须为空")
    config = B3Config.from_json(args.config)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("请求 CUDA，但当前不可用")

    current = json.loads(args.current_ablation.read_text(encoding="utf-8"))
    if current.get("split") != "val":
        raise ValueError("current-ablation 必须来自 validation")
    current_metrics = current["tcn_only"]
    b2_summary = json.loads(args.b2_summary.read_text(encoding="utf-8"))
    if b2_summary.get("test_accessed") is not False:
        raise ValueError("B2 provenance 的 test seal 无效")

    train_cache = load_window_cache(args.train_cache, verify_hashes=True)
    val_cache = load_window_cache(args.val_cache, verify_hashes=True)
    train_sidecar, train_sidecar_metadata = _load_sidecar(
        args.train_sidecar, train_cache, verify_hashes=True
    )
    val_sidecar, val_sidecar_metadata = _load_sidecar(
        args.val_sidecar, val_cache, verify_hashes=True
    )
    selected = np.load(args.selected_indices, allow_pickle=False)
    if selected.ndim != 1 or not np.issubdtype(selected.dtype, np.integer):
        raise ValueError("selected indices 必须是一维整数数组")
    if np.any(np.diff(selected) <= 0) or selected[0] < 0 or selected[-1] >= train_cache.sample_count:
        raise ValueError("selected indices 必须严格递增且位于 train cache 范围内")
    epoch_orders = np.load(args.epoch_orders, allow_pickle=False)
    clip_targets = np.asarray(
        [bool(item["has_fall"]) for item in train_cache.metadata["clips"]],
        dtype=np.bool_,
    )
    validate_epoch_orders(
        epoch_orders, epochs=config.epochs, clips=len(clip_targets)
    )
    if _sha256_file(args.selected_indices) != b2_summary["artifact_sha256"][
        "selected_indices"
    ]:
        raise ValueError("selected indices 与 B2 冻结哈希不一致")
    if _sha256_file(args.epoch_orders) != b2_summary["artifact_sha256"][
        "epoch_clip_orders"
    ]:
        raise ValueError("epoch orders 与 B2 冻结哈希不一致")

    train_joint = np.array(
        train_cache.array("features")[selected], dtype=np.float32, copy=True
    ).reshape(len(selected), -1, JOINT_DIM)
    train_extra = np.array(train_sidecar[selected], dtype=np.float32, copy=True)
    train_raw = torch.from_numpy(np.concatenate((train_joint, train_extra), axis=2))
    train_labels = torch.from_numpy(
        np.array(train_cache.array("labels")[selected], dtype=np.float32, copy=True)
    )
    selected_clips = np.asarray(
        train_cache.array("clip_indices")[selected], dtype=np.int64
    )
    val_joint = np.array(
        val_cache.array("features"), dtype=np.float32, copy=True
    ).reshape(val_cache.sample_count, -1, JOINT_DIM)
    val_extra = np.array(val_sidecar, dtype=np.float32, copy=True)
    val_raw = torch.from_numpy(np.concatenate((val_joint, val_extra), axis=2))
    val_labels = torch.from_numpy(
        np.array(val_cache.array("labels"), dtype=np.float32, copy=True)
    )
    val_clip_indices = np.array(
        val_cache.array("clip_indices"), dtype=np.int64, copy=True
    )

    # Exact equivalence is checked on CPU because CUDA may choose a different
    # convolution kernel for 51 versus 123 input channels and introduce small
    # floating-point accumulation differences despite identical mapped weights.
    equivalence_device = torch.device("cpu")
    base_model = _load_model(args.initial_checkpoint, equivalence_device).eval()
    expanded_model = _load_multistream(args.initial_checkpoint, equivalence_device).eval()
    sample = val_raw[: min(2048, len(val_raw))].to(equivalence_device)
    with torch.inference_mode():
        base_logits = base_model(sample[..., :JOINT_DIM])
        expanded_logits = _FeatureView(
            expanded_model, include_extra=True
        )(sample)
    max_initial_logit_error = float((base_logits - expanded_logits).abs().max().cpu())
    del base_model, expanded_model, sample, base_logits, expanded_logits
    if max_initial_logit_error > 1e-6:
        raise RuntimeError("123D 扩展未保持 Challenge A 初始 logits")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact_sha256 = {
        "initial_checkpoint": _sha256_file(args.initial_checkpoint),
        "current_ablation": _sha256_file(args.current_ablation),
        "b2_summary": _sha256_file(args.b2_summary),
        "config": _sha256_file(args.config),
        "selected_indices": _sha256_file(args.selected_indices),
        "epoch_orders": _sha256_file(args.epoch_orders),
        "train_cache_metadata": _sha256_file(args.train_cache / "metadata.json"),
        "val_cache_metadata": _sha256_file(args.val_cache / "metadata.json"),
        "train_sidecar_metadata": _sha256_file(args.train_sidecar / "metadata.json"),
        "val_sidecar_metadata": _sha256_file(args.val_sidecar / "metadata.json"),
        "training_code": _sha256_file(Path(__file__)),
        "feature_code": _sha256_file(
            Path(__file__).parent.parent / "models" / "tcn_multistream.py"
        ),
    }
    run_record = {
        "protocol": PROTOCOL,
        "pilot": True,
        "seed": args.seed,
        "config": asdict(config),
        "feature_dim": FEATURE_DIM,
        "feature_layout": "joint51+bone32+joint_motion34+bbox5+observed1",
        "sidecar_signatures": {
            "train": train_sidecar_metadata.get("signature_sha256"),
            "val": val_sidecar_metadata.get("signature_sha256"),
        },
        "max_initial_logit_error": max_initial_logit_error,
        "initial_equivalence_device": str(equivalence_device),
        "comparison_rule": "paired branches at fixed final epoch 15",
        "gate_rule": "both MAP and P@R95 >= +3 points vs control and >= current",
        "artifact_sha256": artifact_sha256,
        "test_accessed": False,
    }
    run_signature_sha256 = _canonical_sha256(run_record)
    run_record["run_signature_sha256"] = run_signature_sha256
    _atomic_json(args.output_dir / "run.json", run_record)

    common = {
        "seed": args.seed,
        "config": config,
        "initial_checkpoint": args.initial_checkpoint,
        "train_raw": train_raw,
        "train_labels": train_labels,
        "selected_clips": selected_clips,
        "clip_targets": clip_targets,
        "val_raw": val_raw,
        "val_labels": val_labels,
        "val_clip_indices": val_clip_indices,
        "val_clips": val_cache.metadata["clips"],
        "epoch_orders": epoch_orders,
        "output_dir": args.output_dir,
        "run_signature_sha256": run_signature_sha256,
        "device": device,
    }
    control = _train_branch(
        branch="control", include_extra=False, **common
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
    multistream = _train_branch(
        branch="multistream", include_extra=True, **common
    )
    if control["initial_state_sha256"] != multistream["initial_state_sha256"]:
        raise RuntimeError("paired branches 的初始模型状态不一致")

    control_val = control["final"]["val"]
    multistream_val = multistream["final"]["val"]
    delta_control = {
        "map": 100.0 * (multistream_val["clip_map"] - control_val["clip_map"]),
        "p_at_r95": 100.0
        * (multistream_val["clip_p_at_r95"] - control_val["clip_p_at_r95"]),
    }
    delta_current = {
        "map": 100.0 * (multistream_val["clip_map"] - current_metrics["map"]),
        "p_at_r95": 100.0
        * (multistream_val["clip_p_at_r95"] - current_metrics["p_at_r95"]),
    }
    summary = {
        "protocol": PROTOCOL,
        "pilot": True,
        "run_signature_sha256": run_signature_sha256,
        "seed": args.seed,
        "config": asdict(config),
        "feature_dim": FEATURE_DIM,
        "selected_windows": len(selected),
        "max_initial_logit_error": max_initial_logit_error,
        "paired_initial_state_sha256": control["initial_state_sha256"],
        "current": {
            "map": current_metrics["map"],
            "p_at_r95": current_metrics["p_at_r95"],
        },
        "control": control,
        "multistream": multistream,
        "delta_vs_control_points": delta_control,
        "delta_vs_current_points": delta_current,
        "proceed_to_full_training": bool(
            delta_control["map"] >= 3.0
            and delta_control["p_at_r95"] >= 3.0
            and delta_current["map"] >= 0.0
            and delta_current["p_at_r95"] >= 0.0
        ),
        "parameter_count": count_params(FallTCN(in_dim=FEATURE_DIM)),
        "artifact_sha256": artifact_sha256,
        "test_accessed": False,
    }
    _atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps({"stage": "complete", **summary}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
