"""Train a paired template-OOF skeleton control and shared-YOLO person-ROI F1."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from eval.metrics import competition_map
from models.edgefall_f1 import EdgeFallF1Head, SkeletonControlHead
from models.multiscale_multistream_tcn import MultiStreamMultiScaleAttentionTCN
from models.tcn_dataset import WindowMemmapCache, load_window_cache
from tools.build_tcn_multistream_sidecar import load_sidecar
from tools.train_long_context_event_oracle import group_fold, template_group
from tools.train_multiscale_multistream_mil import (
    MILConfig,
    _clip_groups,
    train_epoch_mil,
)
from tools.train_multiscale_multistream_tcn import _activity_groups
from tools.train_tcn import (
    _append_sidecar,
    _atomic_checkpoint,
    _atomic_json,
    _materialize,
    _sha256_file,
    select_training_indices,
    set_deterministic,
)

SUBTYPES = ("fall", "lie_down", "lying", "stand_up", "sit_down", "other")
EMBEDDING_CACHE_PROTOCOL = "edgefall_f1_clip_embeddings_v1"


@dataclass(frozen=True)
class F1Config:
    fold: int = 0
    folds: int = 5
    fold_seed: int = 20260824
    seed: int = 20260825
    backbone_epochs: int = 8
    head_epochs: int = 12
    backbone_learning_rate: float = 1e-4
    head_learning_rate: float = 3e-4
    batch_size: int = 512
    clip_batch_size: int = 8
    head_batch_size: int = 128
    negative_ratio: float = 2.0
    subtype_weight: float = 0.25
    focal_gamma: float = 2.0
    use_transformer_encoder: bool = False
    transformer_joint_only: bool = False

    def validate(self) -> None:
        if not 0 <= self.fold < self.folds or self.folds < 2:
            raise ValueError("fold 配置无效")
        if min(self.backbone_epochs, self.head_epochs, self.head_batch_size) < 1:
            raise ValueError("epoch/batch 配置无效")
        if self.negative_ratio <= 0.0 or self.subtype_weight < 0.0:
            raise ValueError("loss/sampling 配置无效")
        if self.focal_gamma < 0.0:
            raise ValueError("focal_gamma 无效")
        if self.transformer_joint_only and not self.use_transformer_encoder:
            raise ValueError("transformer_joint_only 需要 use_transformer_encoder")


def build_skeleton_model(config: F1Config) -> MultiStreamMultiScaleAttentionTCN:
    """Build the frozen pose backbone selected for a leak-free EdgeFall fold.

    The initial F1 run used the six-stream TCN control.  Keeping this factory
    alongside the persisted configuration lets a challenger use the current
    transformer-joint architecture while F2/F3 can still reconstruct the exact
    frozen F1 backbone from its checkpoint.
    """
    return MultiStreamMultiScaleAttentionTCN(
        stream_channels=32,
        stream_output_dim=64,
        dropout=0.5,
        use_geometry=True,
        use_rule_features=True,
        use_discriminative_kinematics=True,
        kinematic_feature_dim=15,
        use_transformer_encoder=config.use_transformer_encoder,
        transformer_joint_only=config.transformer_joint_only,
    )


def load_joint_pretrain(
    model: MultiStreamMultiScaleAttentionTCN,
    encoder_state: dict[str, torch.Tensor],
    config: F1Config,
) -> tuple[str, ...]:
    """Load FallVision joint weights, allowing only newly added Transformer layers.

    FallVision pretraining predates the Joint-only Transformer adapter.  Its
    checkpoint remains a valid initialization for the shared TCN path, but no
    other missing or unexpected parameter is acceptable: silently accepting a
    shape or naming mismatch would invalidate the paired OOF experiment.
    """
    result = model.streams[0].load_state_dict(encoder_state, strict=False)
    expected_missing = (
        {
            "transformer_feedforward.0.weight",
            "transformer_feedforward.0.bias",
            "transformer_feedforward.3.weight",
            "transformer_feedforward.3.bias",
            "transformer_norm.weight",
            "transformer_norm.bias",
        }
        if config.use_transformer_encoder
        else set()
    )
    missing = set(result.missing_keys)
    if missing != expected_missing or result.unexpected_keys:
        raise ValueError(
            "FallVision Joint 预训练与 F1 骨架不兼容: "
            f"missing={sorted(missing)}, unexpected={sorted(result.unexpected_keys)}"
        )
    return tuple(sorted(missing))


def template_fold_ids(clips: list[dict[str, Any]], config: F1Config) -> np.ndarray:
    return np.asarray(
        [
            group_fold(
                template_group(str(clip["clip_id"])),
                folds=config.folds,
                seed=config.fold_seed,
            )
            for clip in clips
        ],
        dtype=np.int64,
    )


def explicit_fold_ids(
    path: Path,
    clips: list[dict[str, Any]],
    *,
    folds: int,
    expected_cache_signature: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load a persisted, cache-ordered OOF map without silently re-folding it.

    F1 can only become a C5 candidate when every member holds out exactly the
    same clips.  The legacy F1 protocol derives folds from ``fold_seed``; this
    loader is deliberately strict so a map made for another cache, template
    parser, or label assignment cannot be accepted by accident.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    map_cache_signature = (
        payload.get("train_cache_signature_sha256")
        if isinstance(payload, dict)
        else None
    )
    if (
        expected_cache_signature is not None
        and map_cache_signature != expected_cache_signature
    ):
        raise ValueError("F1 OOF fold map train cache signature 不匹配")
    rows = payload.get("rows") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or len(rows) != len(clips):
        raise ValueError("F1 OOF fold map 与 cache clips 不匹配")
    assignments: list[int] = []
    groups: dict[str, int] = {}
    for expected, row in zip(clips, rows, strict=True):
        if not isinstance(row, dict) or row.get("clip_id") != expected.get("clip_id"):
            raise ValueError("F1 OOF fold map clip 顺序不匹配")
        group_id = template_group(str(expected["clip_id"]))
        if row.get("group_id") != group_id:
            raise ValueError("F1 OOF fold map template group 不匹配")
        if bool(row.get("label")) != bool(expected.get("has_fall")):
            raise ValueError("F1 OOF fold map 标签不匹配")
        fold = int(row.get("fold", -1))
        if not 0 <= fold < folds:
            raise ValueError("F1 OOF fold map fold 超出范围")
        if group_id in groups and groups[group_id] != fold:
            raise ValueError("F1 OOF fold map template group 跨 fold")
        groups[group_id] = fold
        assignments.append(fold)
    return np.asarray(assignments, dtype=np.int64), {
        "mode": "explicit_map",
        "path": str(path),
        "sha256": _sha256_file(path),
        "protocol": payload.get("protocol"),
    }


def subtype_targets(clips: list[dict[str, Any]], indices: np.ndarray) -> np.ndarray:
    mapping = {name: index for index, name in enumerate(SUBTYPES)}
    values = []
    for index in np.asarray(indices, dtype=np.int64):
        clip = clips[int(index)]
        activity = str(clip["clip_id"]).split("/", 1)[0]
        if bool(clip["has_fall"]):
            activity = "fall"
        values.append(mapping.get(activity, mapping["other"]))
    return np.asarray(values, dtype=np.int64)


def balanced_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    gamma: float,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if logits.shape != targets.shape or logits.ndim != 1:
        raise ValueError("binary logits/targets 必须为同形一维")
    positive_fraction = targets.mean().clamp(1e-6, 1.0 - 1e-6)
    probabilities = torch.sigmoid(logits)
    true_probability = torch.where(targets > 0.5, probabilities, 1.0 - probabilities)
    alpha = torch.where(targets > 0.5, 1.0 - positive_fraction, positive_fraction)
    bce = nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    losses = alpha * (1.0 - true_probability).pow(gamma) * bce
    if sample_weights is None:
        return losses.mean()
    if sample_weights.shape != targets.shape:
        raise ValueError("sample_weights 必须与 targets 同形")
    if torch.any(sample_weights <= 0.0) or not torch.isfinite(sample_weights).all():
        raise ValueError("sample_weights 必须为有限正数")
    # Normalize by total weight so changing one semantic cost does not also
    # silently change the effective learning rate.
    return (losses * sample_weights).sum() / sample_weights.sum()


def semantic_hard_negative_weights(
    labels: torch.Tensor,
    subtypes: torch.Tensor,
    *,
    weight: float,
) -> torch.Tensor:
    """Upweight only controlled/static lying negatives in the ROI-head loss."""
    if labels.ndim != 1 or subtypes.shape != labels.shape:
        raise ValueError("labels/subtypes 必须为同形一维")
    if not np.isfinite(weight) or weight < 1.0:
        raise ValueError("semantic hard-negative weight 必须至少为 1")
    lie_down = SUBTYPES.index("lie_down")
    lying = SUBTYPES.index("lying")
    hard_negative = (labels < 0.5) & ((subtypes == lie_down) | (subtypes == lying))
    return torch.where(
        hard_negative,
        torch.full_like(labels, float(weight)),
        torch.ones_like(labels),
    )


def _fold_window_indices(
    cache: WindowMemmapCache,
    fold_ids: np.ndarray,
    *,
    held_out: bool,
) -> np.ndarray:
    clip_indices = np.asarray(cache.array("clip_indices"), dtype=np.int64)
    selected_clips = fold_ids == 0 if held_out else fold_ids != 0
    return np.flatnonzero(selected_clips[clip_indices]).astype(np.int64)


def nested_fold_masks(
    fold_ids: np.ndarray,
    *,
    heldout_fold: int,
    excluded_folds: tuple[int, ...] = (),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return train, held-out, and excluded clip masks for nested base training."""
    values = np.asarray(fold_ids, dtype=np.int64)
    if values.ndim != 1 or not values.size:
        raise ValueError("fold_ids 必须为非空一维")
    known = set(values.tolist())
    excluded = tuple(int(item) for item in excluded_folds)
    if heldout_fold not in known:
        raise ValueError("heldout fold 不存在")
    if len(set(excluded)) != len(excluded) or heldout_fold in excluded:
        raise ValueError("excluded folds 必须唯一且不能包含 heldout fold")
    if any(item not in known for item in excluded):
        raise ValueError("excluded fold 不存在")
    heldout_mask = values == heldout_fold
    excluded_mask = np.isin(values, excluded) if excluded else np.zeros_like(values, dtype=bool)
    train_mask = ~(heldout_mask | excluded_mask)
    if not train_mask.any() or not heldout_mask.any():
        raise ValueError("nested fold 划分产生空 train/heldout")
    return train_mask, heldout_mask, excluded_mask


def _remap_clip_indices(
    original: np.ndarray, selected_clips: np.ndarray
) -> np.ndarray:
    mapping = {int(value): index for index, value in enumerate(selected_clips)}
    try:
        return np.asarray([mapping[int(value)] for value in original], dtype=np.int64)
    except KeyError as exc:
        raise ValueError("window clip index 不在选定 clip 集合") from exc


@torch.inference_mode()
def clip_embeddings(
    model: MultiStreamMultiScaleAttentionTCN,
    features: torch.Tensor,
    clip_indices: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    outputs = []
    for start in range(0, features.shape[0], batch_size):
        batch = features[start : start + batch_size]
        if device.type == "cuda" and batch.device.type == "cpu":
            batch = batch.pin_memory()
        outputs.append(model.encode(batch.to(device, non_blocking=True)).cpu())
    encoded = torch.cat(outputs).numpy()
    clips = np.unique(clip_indices)
    pooled = np.stack([encoded[clip_indices == index].max(0) for index in clips])
    return clips.astype(np.int64), pooled.astype(np.float32, copy=False)


def _array_sha256(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _state_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def save_embedding_cache(
    path: Path,
    *,
    metadata: dict[str, Any],
    train_clip_indices: np.ndarray,
    train_embeddings: np.ndarray,
    heldout_clip_indices: np.ndarray,
    heldout_embeddings: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
            train_clip_indices=np.asarray(train_clip_indices, dtype=np.int64),
            train_embeddings=np.asarray(train_embeddings, dtype=np.float32),
            heldout_clip_indices=np.asarray(heldout_clip_indices, dtype=np.int64),
            heldout_embeddings=np.asarray(heldout_embeddings, dtype=np.float32),
        )
    os.replace(temporary, path)


def load_embedding_cache(
    path: Path,
    *,
    expected_metadata: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata"].item()))
        if metadata != expected_metadata:
            raise ValueError("embedding cache 签名不匹配")
        arrays = tuple(
            np.asarray(payload[name]).copy()
            for name in (
                "train_clip_indices",
                "train_embeddings",
                "heldout_clip_indices",
                "heldout_embeddings",
            )
        )
    train_clips, train_embeddings, heldout_clips, heldout_embeddings = arrays
    if train_clips.ndim != 1 or heldout_clips.ndim != 1:
        raise ValueError("embedding cache clip index shape 无效")
    if (
        train_embeddings.ndim != 2
        or heldout_embeddings.ndim != 2
        or train_embeddings.shape[0] != train_clips.size
        or heldout_embeddings.shape[0] != heldout_clips.size
        or train_embeddings.shape[1:] != heldout_embeddings.shape[1:]
    ):
        raise ValueError("embedding cache embedding shape 无效")
    return train_clips, train_embeddings, heldout_clips, heldout_embeddings


def metric_result(
    clip_ids: list[str], labels: np.ndarray, scores: np.ndarray
) -> dict[str, float]:
    result = competition_map(
        dict(zip(clip_ids, (bool(value) for value in labels), strict=True)),
        dict(zip(clip_ids, (float(value) for value in scores), strict=True)),
        mode="clip",
    )
    return {
        "clip_map": float(result["map"]),
        "clip_map_percent": float(result["map_percent"]),
        "clip_p_at_r90": float(result["p_at_r90"]),
        "clip_p_at_r95": float(result["p_at_r95"]),
    }


def equal_logit_scores(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return fixed 50/50 logit fusion with stable clipping."""
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.shape != second.shape:
        raise ValueError("双分辨率 score shape 不匹配")
    epsilon = np.finfo(np.float32).eps
    first = np.clip(first, epsilon, 1.0 - epsilon)
    second = np.clip(second, epsilon, 1.0 - epsilon)
    mean_logit = 0.5 * (
        np.log(first / (1.0 - first)) + np.log(second / (1.0 - second))
    )
    return (1.0 / (1.0 + np.exp(-mean_logit))).astype(np.float32)


def _train_heads(
    control: SkeletonControlHead,
    fusion: EdgeFallF1Head,
    skeleton: torch.Tensor,
    person_roi: torch.Tensor,
    labels: torch.Tensor,
    subtypes: torch.Tensor,
    indices: np.ndarray,
    *,
    config: F1Config,
    device: torch.device,
    semantic_hard_negative_weight: float = 1.0,
    swa_start_epoch: int | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    control.train()
    fusion.train()
    control_optimizer = torch.optim.AdamW(
        control.parameters(), lr=config.head_learning_rate, weight_decay=1e-3
    )
    fusion_parameters = [parameter for parameter in fusion.parameters() if parameter.requires_grad]
    if not fusion_parameters:
        raise ValueError("fusion head 没有可训练参数")
    fusion_optimizer = torch.optim.AdamW(
        fusion_parameters, lr=config.head_learning_rate, weight_decay=1e-3
    )
    subtype_counts = torch.bincount(subtypes[indices], minlength=len(SUBTYPES)).float()
    if torch.any(subtype_counts == 0):
        raise ValueError("训练 fold 必须覆盖所有 subtype")
    subtype_criterion = nn.CrossEntropyLoss(
        weight=(subtype_counts.sum() / subtype_counts).to(device)
    )
    final_control: dict[str, float] = {}
    final_fusion: dict[str, float] = {}
    control_swa_states: list[dict[str, torch.Tensor]] = []
    fusion_swa_states: list[dict[str, torch.Tensor]] = []
    for epoch in range(config.head_epochs):
        epoch_started = time.perf_counter()
        order = np.random.default_rng(config.seed + epoch).permutation(indices)
        totals = {"control": 0.0, "fusion": 0.0}
        seen = 0
        for start in range(0, order.size, config.head_batch_size):
            selected = order[start : start + config.head_batch_size]
            batch_skeleton = skeleton[selected].to(device)
            batch_roi = person_roi[selected].to(device)
            batch_labels = labels[selected].to(device)
            batch_subtypes = subtypes[selected].to(device)
            control_optimizer.zero_grad(set_to_none=True)
            control_binary, control_subtype = control(batch_skeleton)
            control_loss = balanced_focal_loss(
                control_binary, batch_labels, gamma=config.focal_gamma
            ) + config.subtype_weight * subtype_criterion(
                control_subtype, batch_subtypes
            )
            control_loss.backward()
            nn.utils.clip_grad_norm_(control.parameters(), 5.0)
            control_optimizer.step()
            fusion_optimizer.zero_grad(set_to_none=True)
            fusion_binary, fusion_subtype = fusion(batch_skeleton, batch_roi)
            fusion_sample_weights = semantic_hard_negative_weights(
                batch_labels,
                batch_subtypes,
                weight=semantic_hard_negative_weight,
            )
            fusion_loss = balanced_focal_loss(
                fusion_binary,
                batch_labels,
                gamma=config.focal_gamma,
                sample_weights=fusion_sample_weights,
            ) + config.subtype_weight * subtype_criterion(
                fusion_subtype, batch_subtypes
            )
            fusion_loss.backward()
            nn.utils.clip_grad_norm_(fusion_parameters, 5.0)
            fusion_optimizer.step()
            batch = int(selected.size)
            totals["control"] += float(control_loss.detach()) * batch
            totals["fusion"] += float(fusion_loss.detach()) * batch
            seen += batch
        final_control = {"loss": totals["control"] / seen}
        final_fusion = {"loss": totals["fusion"] / seen}
        print(
            json.dumps(
                {
                    "stage": "paired_heads",
                    "epoch": epoch,
                    "control_train_loss": final_control["loss"],
                    "fusion_train_loss": final_fusion["loss"],
                    "epoch_seconds": time.perf_counter() - epoch_started,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if swa_start_epoch is not None and epoch >= swa_start_epoch:
            control_swa_states.append(
                {name: value.detach().cpu().clone() for name, value in control.state_dict().items()}
            )
            fusion_swa_states.append(
                {name: value.detach().cpu().clone() for name, value in fusion.state_dict().items()}
            )
    if swa_start_epoch is not None:
        if not control_swa_states or not fusion_swa_states:
            raise AssertionError("SWA 未收集任何 head checkpoint")
        control.load_state_dict(_average_state_dicts(control_swa_states), strict=True)
        fusion.load_state_dict(_average_state_dicts(fusion_swa_states), strict=True)
    return final_control, final_fusion


def _average_state_dicts(
    states: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Average floating tensors and require non-floating state to stay identical."""
    if not states:
        raise ValueError("SWA state 列表不能为空")
    keys = set(states[0])
    if any(set(state) != keys for state in states[1:]):
        raise ValueError("SWA state schema 不一致")
    averaged: dict[str, torch.Tensor] = {}
    for name, first in states[0].items():
        values = [state[name] for state in states]
        if torch.is_floating_point(first):
            accumulator = torch.zeros_like(first, dtype=torch.float64)
            for value in values:
                accumulator.add_(value.to(dtype=torch.float64))
            averaged[name] = (accumulator / len(values)).to(dtype=first.dtype)
        else:
            if any(not torch.equal(first, value) for value in values[1:]):
                raise ValueError(f"SWA non-floating state 发生变化: {name}")
            averaged[name] = first.clone()
    return averaged


@torch.inference_mode()
def _head_scores(
    model: nn.Module,
    skeleton: torch.Tensor,
    person_roi: torch.Tensor | None,
    indices: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    values = []
    for start in range(0, indices.size, batch_size):
        selected = indices[start : start + batch_size]
        if person_roi is None:
            logits, _ = model(skeleton[selected].to(device))
        else:
            logits, _ = model(
                skeleton[selected].to(device), person_roi[selected].to(device)
            )
        values.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--roi-cache", type=Path, required=True)
    parser.add_argument(
        "--roi-cache-secondary",
        type=Path,
        help="dual_resolution_compact 的第二个同身份 ROI cache",
    )
    parser.add_argument("--joint-pretrain", type=Path, required=True)
    parser.add_argument(
        "--reuse-backbone-checkpoint",
        type=Path,
        help="严格匹配配置/fold 后复用已有 F1 skeleton backbone",
    )
    parser.add_argument(
        "--save-embedding-cache",
        type=Path,
        help="保存与当前骨干/fold/抽样窗口严格绑定的 clip embedding",
    )
    parser.add_argument(
        "--reuse-embedding-cache",
        type=Path,
        help="跳过窗口加载和骨干编码，复用严格签名匹配的 clip embedding",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=512,
        help="仅用于骨干 embedding 提取，不改变训练 batch 配置",
    )
    parser.add_argument(
        "--reuse-fusion-head-checkpoint",
        type=Path,
        help="从同 fold 的 mean_max F1 head 初始化残差注意力实验",
    )
    parser.add_argument(
        "--freeze-reused-fusion-base",
        action="store_true",
        help="只训练 residual_attention_max 新增的 257 个参数",
    )
    parser.add_argument(
        "--fold-map",
        type=Path,
        help="可选的 cache-ordered OOF fold map；C5 兼容运行必须提供",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument(
        "--exclude-fold",
        type=int,
        action="append",
        default=[],
        help="nested inner 训练时完全排除的 outer fold；仅支持显式 fold map",
    )
    parser.add_argument("--backbone-epochs", type=int, default=8)
    parser.add_argument("--head-epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument(
        "--head-seed",
        type=int,
        help="在 head 初始化前显式重置 RNG，供复用 backbone 的 matched runs 使用",
    )
    parser.add_argument(
        "--head-swa-start-epoch",
        type=int,
        help="在固定 head 训练预算内，从该 epoch 起平均每轮最终权重",
    )
    parser.add_argument(
        "--roi-token-layout",
        choices=(
            "direct",
            "dual_resolution_compact",
            "resolution_stochastic",
            "mean_max_compact",
            "mean_std_xy_compact",
        ),
        default="direct",
    )
    parser.add_argument("--use-transformer-encoder", action="store_true")
    parser.add_argument("--transformer-joint-only", action="store_true")
    parser.add_argument(
        "--roi-temporal-pooling",
        choices=(
            "mean_max",
            "attention_max",
            "residual_attention_max",
            "residual_transition_max",
        ),
        default="mean_max",
    )
    parser.add_argument(
        "--semantic-hard-negative-weight",
        type=float,
        default=1.0,
        help="ROI 二分类损失中 lie_down/lying 负样本的 fold-local 代价权重",
    )
    parser.add_argument("--auprc-surrogate-weight", type=float, default=0.0)
    parser.add_argument("--high-recall-positive-fraction", type=float, default=0.25)
    parser.add_argument("--hard-negative-memory-size", type=int, default=256)
    args = parser.parse_args()
    config = F1Config(
        fold=args.fold,
        backbone_epochs=args.backbone_epochs,
        head_epochs=args.head_epochs,
        seed=args.seed,
        use_transformer_encoder=args.use_transformer_encoder,
        transformer_joint_only=args.transformer_joint_only,
    )
    config.validate()
    if args.head_swa_start_epoch is not None and not (
        0 <= args.head_swa_start_epoch < config.head_epochs
    ):
        raise ValueError("head SWA 起始 epoch 必须落在训练预算内")
    if args.embedding_batch_size < 1:
        raise ValueError("embedding batch size 必须为正数")
    if args.auprc_surrogate_weight < 0.0:
        raise ValueError("AUPRC surrogate 权重不能为负")
    if not 0.0 < args.high_recall_positive_fraction <= 1.0:
        raise ValueError("高召回正例比例必须位于 (0,1]")
    if args.hard_negative_memory_size < 1:
        raise ValueError("困难负例 memory 容量必须为正")
    if args.save_embedding_cache is not None and args.reuse_embedding_cache is not None:
        raise ValueError("不能同时保存和复用 embedding cache")
    if args.reuse_embedding_cache is not None and args.reuse_backbone_checkpoint is None:
        raise ValueError("复用 embedding cache 必须同时复用对应 backbone checkpoint")
    if not np.isfinite(args.semantic_hard_negative_weight) or (
        args.semantic_hard_negative_weight < 1.0
    ):
        raise ValueError("semantic hard-negative weight 必须至少为 1")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("F1 输出目录必须为空")
    if args.exclude_fold and args.fold_map is None:
        raise ValueError("--exclude-fold 需要 --fold-map")
    cache = load_window_cache(args.cache, verify_hashes=True)
    if cache.metadata.get("dataset") != "of-syn" or cache.metadata.get("split") != "train":
        raise ValueError("F1 只接受 OF-Syn train cache")
    sidecar = load_sidecar(args.sidecar, cache)
    clips = cache.metadata["clips"]
    if args.fold_map is None:
        fold_ids = template_fold_ids(clips, config)
        # Preserve the legacy convention: the selected external fold is moved
        # to local fold zero for F1/F2/F3 checkpoint compatibility.
        if config.fold != 0:
            fold_ids = (fold_ids - config.fold) % config.folds
        fold_assignment = {
            "mode": "legacy_template_seed",
            "fold_seed": config.fold_seed,
            "held_out_fold": config.fold,
        }
    else:
        fold_ids, fold_assignment = explicit_fold_ids(
            args.fold_map,
            clips,
            folds=config.folds,
            expected_cache_signature=str(cache.metadata.get("signature_sha256", "")),
        )
        if config.fold not in fold_ids:
            raise ValueError("请求的 F1 OOF fold 为空")
    heldout_fold = config.fold if args.fold_map is not None else 0
    train_mask, heldout_mask, excluded_mask = nested_fold_masks(
        fold_ids,
        heldout_fold=heldout_fold,
        excluded_folds=tuple(args.exclude_fold),
    )
    if args.exclude_fold:
        fold_assignment = {
            **fold_assignment,
            "nested_outer_excluded_folds": sorted(args.exclude_fold),
        }
    train_clip_indices = np.flatnonzero(train_mask)
    heldout_clip_indices = np.flatnonzero(heldout_mask)
    train_groups = {template_group(str(clips[index]["clip_id"])) for index in train_clip_indices}
    heldout_groups = {template_group(str(clips[index]["clip_id"])) for index in heldout_clip_indices}
    if train_groups & heldout_groups:
        raise AssertionError("template group 跨 F1 fold 泄漏")
    excluded_groups = {
        template_group(str(clips[index]["clip_id"]))
        for index in np.flatnonzero(excluded_mask)
    }
    if excluded_groups & (train_groups | heldout_groups):
        raise AssertionError("nested excluded template group 泄漏")
    all_labels = np.asarray(cache.array("labels"))
    all_clip_indices = np.asarray(cache.array("clip_indices"), dtype=np.int64)
    candidate_windows = np.flatnonzero(train_mask[all_clip_indices])
    activities = _activity_groups(cache)
    local = select_training_indices(
        all_labels[candidate_windows],
        negative_ratio=config.negative_ratio,
        seed=config.seed,
        activity_groups=activities[candidate_windows],
        hard_negative_activities=("lie_down", "lying", "stand_up"),
        hard_negative_fraction=0.4,
    )
    selected_windows = candidate_windows[local]
    train_x: torch.Tensor | None = None
    train_y: torch.Tensor | None = None
    train_window_clips = all_clip_indices[selected_windows]
    groups: list[np.ndarray] | None = None
    group_labels: np.ndarray | None = None
    group_activities: np.ndarray | None = None
    if args.reuse_embedding_cache is None:
        train_x, train_y, train_window_clips = _materialize(cache, selected_windows)
        train_x = _append_sidecar(train_x, sidecar, selected_windows)
        groups, group_labels = _clip_groups(train_window_clips, clips)
        group_clip_indices = np.unique(train_window_clips)
        group_activities = np.asarray(
            [
                str(clips[int(index)]["clip_id"]).split("/", 1)[0]
                for index in group_clip_indices
            ],
            dtype=object,
        )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("请求 CUDA，但当前不可用")
    set_deterministic(config.seed)
    model = build_skeleton_model(config).to(device)
    pretrain = torch.load(args.joint_pretrain, map_location="cpu", weights_only=False)
    if not isinstance(pretrain, dict) or not isinstance(pretrain.get("encoder_state"), dict):
        raise TypeError("FallVision joint pretrain 缺少 encoder_state")
    joint_pretrain_missing_keys = load_joint_pretrain(
        model, pretrain["encoder_state"], config
    )
    reused_backbone_sha256: str | None = None
    if args.reuse_backbone_checkpoint is not None:
        reused = torch.load(
            args.reuse_backbone_checkpoint, map_location="cpu", weights_only=False
        )
        if not isinstance(reused, dict) or not isinstance(
            reused.get("skeleton_model_state"), dict
        ):
            raise TypeError("复用 checkpoint 缺少 skeleton_model_state")
        if reused.get("config") != asdict(config):
            raise ValueError("复用 backbone 的训练配置不匹配")
        if reused.get("fold_assignment") != fold_assignment:
            raise ValueError("复用 backbone 的 fold assignment 不匹配")
        model.load_state_dict(reused["skeleton_model_state"], strict=True)
        reused_backbone_sha256 = _sha256_file(args.reuse_backbone_checkpoint)
        print(
            json.dumps(
                {
                    "stage": "skeleton_backbone_reused",
                    "checkpoint_sha256": reused_backbone_sha256,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    else:
        assert train_x is not None and train_y is not None
        assert groups is not None and group_labels is not None
        assert group_activities is not None
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.backbone_learning_rate, weight_decay=1e-3
        )
        mil_config = MILConfig(
            epochs=config.backbone_epochs,
            seed=config.seed,
            use_rule_features=True,
            use_discriminative_kinematics=True,
            kinematic_feature_dim=15,
            horizontal_flip_probability=0.5,
            mil_clip_batch_size=config.clip_batch_size,
            hard_negative_activities=("lie_down", "lying", "stand_up"),
            hard_negative_fraction=0.4,
            auprc_surrogate_weight=args.auprc_surrogate_weight,
            high_recall_positive_fraction=args.high_recall_positive_fraction,
            hard_negative_memory_size=args.hard_negative_memory_size,
        )
        for epoch in range(config.backbone_epochs):
            epoch_started = time.perf_counter()
            loss, diagnostics = train_epoch_mil(
                model,
                optimizer,
                train_x,
                train_y,
                groups,
                group_labels,
                clip_activities=group_activities,
                device=device,
                config=mil_config,
                epoch=epoch,
            )
            print(
                json.dumps(
                    {
                        "stage": "skeleton_backbone",
                        "epoch": epoch,
                        "epoch_seconds": time.perf_counter() - epoch_started,
                        "loss": loss,
                        **diagnostics,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    heldout_windows = np.flatnonzero(heldout_mask[all_clip_indices])
    embedding_metadata = {
        "protocol": EMBEDDING_CACHE_PROTOCOL,
        "config": asdict(config),
        "fold_assignment": fold_assignment,
        "cache_signature_sha256": str(cache.metadata.get("signature_sha256", "")),
        "sidecar_metadata_sha256": _sha256_file(args.sidecar / "metadata.json"),
        "selected_windows_sha256": _array_sha256(selected_windows),
        "heldout_windows_sha256": _array_sha256(heldout_windows),
        "skeleton_state_sha256": _state_dict_sha256(model.state_dict()),
    }
    if args.reuse_embedding_cache is not None:
        (
            train_embedding_clips,
            train_embeddings,
            heldout_embedding_clips,
            heldout_embeddings,
        ) = load_embedding_cache(
            args.reuse_embedding_cache,
            expected_metadata=embedding_metadata,
        )
        print(
            json.dumps(
                {
                    "stage": "clip_embeddings_reused",
                    "path": str(args.reuse_embedding_cache),
                    "sha256": _sha256_file(args.reuse_embedding_cache),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    else:
        assert train_x is not None
        heldout_x, _, heldout_original_clips = _materialize(cache, heldout_windows)
        heldout_x = _append_sidecar(heldout_x, sidecar, heldout_windows)
        train_embedding_clips, train_embeddings = clip_embeddings(
            model,
            train_x,
            train_window_clips,
            device=device,
            batch_size=args.embedding_batch_size,
        )
        heldout_embedding_clips, heldout_embeddings = clip_embeddings(
            model,
            heldout_x,
            heldout_original_clips,
            device=device,
            batch_size=args.embedding_batch_size,
        )
        if args.save_embedding_cache is not None:
            save_embedding_cache(
                args.save_embedding_cache,
                metadata=embedding_metadata,
                train_clip_indices=train_embedding_clips,
                train_embeddings=train_embeddings,
                heldout_clip_indices=heldout_embedding_clips,
                heldout_embeddings=heldout_embeddings,
            )
            print(
                json.dumps(
                    {
                        "stage": "clip_embeddings_saved",
                        "path": str(args.save_embedding_cache),
                        "sha256": _sha256_file(args.save_embedding_cache),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    selected_clip_indices = np.concatenate(
        (train_embedding_clips, heldout_embedding_clips)
    )
    skeleton_embeddings = torch.from_numpy(
        np.concatenate((train_embeddings, heldout_embeddings))
    )
    with np.load(args.roi_cache, allow_pickle=False) as payload:
        roi_ids = [str(value) for value in payload["clip_ids"]]
        roi_tokens = np.asarray(payload["tokens"], dtype=np.float32)
        primary_identity = {
            key: np.asarray(payload[key])
            for key in ("labels", "track_ids", "quality")
            if key in payload.files
        }
    if args.roi_token_layout in (
        "dual_resolution_compact",
        "resolution_stochastic",
    ):
        if args.roi_cache_secondary is None:
            raise ValueError("双分辨率 layout 必须提供 --roi-cache-secondary")
        with np.load(args.roi_cache_secondary, allow_pickle=False) as payload:
            secondary_ids = [str(value) for value in payload["clip_ids"]]
            secondary_tokens = np.asarray(payload["tokens"], dtype=np.float32)
            secondary_identity = {
                key: np.asarray(payload[key])
                for key in ("labels", "track_ids", "quality")
                if key in payload.files
            }
        if roi_ids != secondary_ids or roi_tokens.shape != secondary_tokens.shape:
            raise ValueError("双分辨率 ROI cache 身份或 token shape 不匹配")
        if set(primary_identity) != set(secondary_identity) or any(
            not np.array_equal(primary_identity[key], secondary_identity[key])
            for key in primary_identity
        ):
            raise ValueError("双分辨率 ROI cache 标签/track/quality 不匹配")
        roi_tokens = np.concatenate((roi_tokens, secondary_tokens), axis=-1)
    elif args.roi_cache_secondary is not None:
        raise ValueError("只有 dual_resolution_compact 可提供第二 ROI cache")
    roi_index = {clip_id: index for index, clip_id in enumerate(roi_ids)}
    selected_ids = [str(clips[int(index)]["clip_id"]) for index in selected_clip_indices]
    if any(clip_id not in roi_index for clip_id in selected_ids):
        raise ValueError("ROI cache 缺少 F1 clip")
    # F1 uses only person ROI (index 0); context index 1 remains frozen for F2.
    person_roi = torch.from_numpy(
        np.stack([roi_tokens[roi_index[clip_id], :, 0] for clip_id in selected_ids])
    )
    labels = torch.tensor(
        [float(clips[int(index)]["has_fall"]) for index in selected_clip_indices]
    )
    subtypes = torch.from_numpy(subtype_targets(clips, selected_clip_indices))
    train_count = train_embedding_clips.size
    train_indices = np.arange(train_count, dtype=np.int64)
    heldout_indices = np.arange(train_count, selected_clip_indices.size, dtype=np.int64)
    skeleton_dim = int(skeleton_embeddings.shape[1])
    roi_token_dim = int(person_roi.shape[2])
    if args.head_seed is not None:
        set_deterministic(args.head_seed)
    control = SkeletonControlHead(skeleton_dim).to(device)
    fusion = EdgeFallF1Head(
        skeleton_dim,
        roi_dim=roi_token_dim,
        roi_token_layout=args.roi_token_layout,
        roi_temporal_pooling=args.roi_temporal_pooling,
    ).to(device)
    reused_fusion_sha256: str | None = None
    if args.reuse_fusion_head_checkpoint is not None:
        if args.roi_temporal_pooling not in {
            "residual_attention_max",
            "residual_transition_max",
        }:
            raise ValueError("复用 fusion head 只允许零初始化 residual pooling")
        reused_fusion = torch.load(
            args.reuse_fusion_head_checkpoint, map_location="cpu", weights_only=False
        )
        if not isinstance(reused_fusion, dict) or not isinstance(
            reused_fusion.get("fusion_state"), dict
        ):
            raise TypeError("复用 checkpoint 缺少 fusion_state")
        if reused_fusion.get("config") != asdict(config):
            raise ValueError("复用 fusion head 的训练配置不匹配")
        if reused_fusion.get("fold_assignment") != fold_assignment:
            raise ValueError("复用 fusion head 的 fold assignment 不匹配")
        if reused_fusion.get("roi_token_dim") != roi_token_dim:
            raise ValueError("复用 fusion head 的 ROI token dim 不匹配")
        if reused_fusion.get("roi_token_layout", "direct") != args.roi_token_layout:
            raise ValueError("复用 fusion head 的 ROI token layout 不匹配")
        load_result = fusion.load_state_dict(reused_fusion["fusion_state"], strict=False)
        expected_missing = (
            {
                "roi_encoder.temporal_attention.weight",
                "roi_encoder.temporal_attention.bias",
                "roi_encoder.attention_residual_weights",
            }
            if args.roi_temporal_pooling == "residual_attention_max"
            else {
                "roi_encoder.transition_residual.weight",
                "roi_encoder.transition_residual.bias",
            }
        )
        if set(load_result.missing_keys) != expected_missing or load_result.unexpected_keys:
            raise ValueError("复用 fusion head 的 state schema 不匹配")
        reused_fusion_sha256 = _sha256_file(args.reuse_fusion_head_checkpoint)
        if args.freeze_reused_fusion_base:
            for parameter in fusion.parameters():
                parameter.requires_grad_(False)
            if args.roi_temporal_pooling == "residual_attention_max":
                assert fusion.roi_encoder.temporal_attention is not None
                fusion.roi_encoder.temporal_attention.requires_grad_(True)
                assert fusion.roi_encoder.attention_residual_weights is not None
                fusion.roi_encoder.attention_residual_weights.requires_grad_(True)
            else:
                assert fusion.roi_encoder.transition_residual is not None
                fusion.roi_encoder.transition_residual.requires_grad_(True)
    elif args.freeze_reused_fusion_base:
        raise ValueError("冻结 fusion base 必须提供复用 checkpoint")
    initial_control = copy.deepcopy(control.state_dict())
    _train_heads(
        control,
        fusion,
        skeleton_embeddings,
        person_roi,
        labels,
        subtypes,
        train_indices,
        config=config,
        device=device,
        semantic_hard_negative_weight=args.semantic_hard_negative_weight,
        swa_start_epoch=args.head_swa_start_epoch,
    )
    control_scores = _head_scores(
        control,
        skeleton_embeddings,
        None,
        heldout_indices,
        device=device,
        batch_size=config.head_batch_size,
    )
    resolution_scores: dict[str, np.ndarray] = {}
    if args.roi_token_layout == "resolution_stochastic":
        primary_roi, secondary_roi = person_roi.split(roi_token_dim // 2, dim=2)
        primary_eval_roi = torch.cat((primary_roi, primary_roi), dim=2)
        secondary_eval_roi = torch.cat((secondary_roi, secondary_roi), dim=2)
        resolution_scores = {
            "primary": _head_scores(
                fusion,
                skeleton_embeddings,
                primary_eval_roi,
                heldout_indices,
                device=device,
                batch_size=config.head_batch_size,
            ),
            "secondary": _head_scores(
                fusion,
                skeleton_embeddings,
                secondary_eval_roi,
                heldout_indices,
                device=device,
                batch_size=config.head_batch_size,
            ),
        }
        fusion_scores = equal_logit_scores(
            resolution_scores["primary"], resolution_scores["secondary"]
        )
    else:
        fusion_scores = _head_scores(
            fusion,
            skeleton_embeddings,
            person_roi,
            heldout_indices,
            device=device,
            batch_size=config.head_batch_size,
        )
    heldout_ids = selected_ids[train_count:]
    heldout_labels = labels[heldout_indices].numpy()
    control_metrics = metric_result(heldout_ids, heldout_labels, control_scores)
    fusion_metrics = metric_result(heldout_ids, heldout_labels, fusion_scores)
    delta = {
        key: fusion_metrics[key] - control_metrics[key]
        for key in ("clip_map", "clip_map_percent", "clip_p_at_r90", "clip_p_at_r95")
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "config": asdict(config),
        "roi_token_dim": roi_token_dim,
        "roi_token_layout": args.roi_token_layout,
        "roi_temporal_pooling": args.roi_temporal_pooling,
        "head_seed": args.head_seed,
        "head_swa_start_epoch": args.head_swa_start_epoch,
        "head_swa_checkpoints": (
            config.head_epochs - args.head_swa_start_epoch
            if args.head_swa_start_epoch is not None
            else 0
        ),
        "semantic_hard_negative_weight": args.semantic_hard_negative_weight,
        "auprc_surrogate_weight": args.auprc_surrogate_weight,
        "high_recall_positive_fraction": args.high_recall_positive_fraction,
        "hard_negative_memory_size": args.hard_negative_memory_size,
        "fold_assignment": fold_assignment,
        "skeleton_model_state": model.state_dict(),
        "control_state": control.state_dict(),
        "fusion_state": fusion.state_dict(),
        "initial_control_state": initial_control,
        "control_metrics": control_metrics,
        "fusion_metrics": fusion_metrics,
    }
    _atomic_checkpoint(args.output_dir / "last.pt", checkpoint)
    np.savez_compressed(
        args.output_dir / "oof_predictions.npz",
        # Standard Stage-S1/C5 member schema.  Keep the historical F1 arrays
        # below so F2/F3 audit readers remain backward-compatible.
        clip_id=np.asarray(heldout_ids),
        group_id=np.asarray([template_group(clip_id) for clip_id in heldout_ids]),
        label=heldout_labels,
        score=fusion_scores,
        clip_ids=np.asarray(heldout_ids),
        labels=heldout_labels,
        control_scores=control_scores,
        fusion_scores=fusion_scores,
        **{
            f"{name}_scores": scores
            for name, scores in resolution_scores.items()
        },
        fold=np.full(len(heldout_ids), config.fold, dtype=np.int64),
    )
    summary = {
        "protocol": "edgefall_f1_template_grouped_train_only_shared_yolo_person_roi_v1",
        "config": asdict(config),
        "fold_assignment": fold_assignment,
        "train_clips": int(train_count),
        "oof_clips": int(heldout_indices.size),
        "excluded_clips": int(excluded_mask.sum()),
        "control": control_metrics,
        "fusion": fusion_metrics,
        "delta": delta,
        "passes_plus_3_map_gate": delta["clip_map_percent"] >= 3.0,
        "joint_pretrain_sha256": _sha256_file(args.joint_pretrain),
        "joint_pretrain_missing_keys": list(joint_pretrain_missing_keys),
        "reused_backbone_checkpoint_sha256": reused_backbone_sha256,
        "embedding_batch_size": args.embedding_batch_size,
        "embedding_cache_sha256": (
            _sha256_file(args.reuse_embedding_cache)
            if args.reuse_embedding_cache is not None
            else (
                _sha256_file(args.save_embedding_cache)
                if args.save_embedding_cache is not None
                else None
            )
        ),
        "reused_embedding_cache": args.reuse_embedding_cache is not None,
        "reused_fusion_head_checkpoint_sha256": reused_fusion_sha256,
        "freeze_reused_fusion_base": args.freeze_reused_fusion_base,
        "roi_cache_sha256": _sha256_file(args.roi_cache),
        "roi_cache_secondary_sha256": (
            _sha256_file(args.roi_cache_secondary)
            if args.roi_cache_secondary is not None
            else None
        ),
        "roi_token_dim": roi_token_dim,
        "roi_token_layout": args.roi_token_layout,
        "roi_temporal_pooling": args.roi_temporal_pooling,
        "head_seed": args.head_seed,
        "semantic_hard_negative_weight": args.semantic_hard_negative_weight,
        "auprc_surrogate_weight": args.auprc_surrogate_weight,
        "high_recall_positive_fraction": args.high_recall_positive_fraction,
        "hard_negative_memory_size": args.hard_negative_memory_size,
        "checkpoint_sha256": _sha256_file(args.output_dir / "last.pt"),
        "test_accessed": False,
    }
    _atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
