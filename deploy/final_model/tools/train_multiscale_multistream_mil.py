"""以 clip-batched Top-k MIL 辅助损失训练当前多流因果 TCN。

此实验保留逐窗口 BCE、特征和模型不变，只加入每个视频窗口 logit 的
Top-k 均值 clip 损失，避免训练目标与最终 clip-max 评分完全脱节。
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from models.clip_aggregator import aggregate_clip_logits
from models.multiscale_multistream_tcn import (
    MultiStreamMultiScaleAttentionTCN,
    count_params,
)
from models.real_distortion import RealDistortionBank, replay_real_distortion
from models.tcn_dataset import SEMANTIC_TO_CODE, WindowMemmapCache, build_window_cache
from tools.build_tcn_multistream_sidecar import load_sidecar
from tools.train_multiscale_multistream_tcn import MultiStreamConfig, _activity_groups
from tools.train_tcn import (
    _append_sidecar,
    _atomic_checkpoint,
    _atomic_json,
    _canonical_json,
    _materialize,
    _run_signature,
    _sha256_file,
    aggregate_group_logits,
    evaluate_model,
    select_training_indices,
    set_deterministic,
)

_COCO_FLIP_INDEX = torch.tensor(
    [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15],
    dtype=torch.long,
)


def horizontal_flip_features(features: torch.Tensor) -> torch.Tensor:
    """Mirror normalized pose/bbox features and exchange left/right joints."""
    if features.ndim != 3 or features.shape[-1] < 51:
        raise ValueError("水平翻转需要形状 (B, T, >=51) 的特征")
    flipped = features.clone()
    pose = flipped[..., :51].reshape(*flipped.shape[:2], 17, 3)
    mirrored = pose[:, :, _COCO_FLIP_INDEX.to(features.device)].clone()
    mirrored[..., 0].neg_()
    pose.copy_(mirrored)
    if flipped.shape[-1] >= 57:
        flipped[..., 51] = 1.0 - flipped[..., 51]
    return flipped


def augment_training_features(
    features: torch.Tensor, config: MILConfig, *, seed: int
) -> torch.Tensor:
    """Apply deterministic, label-preserving pose perturbations to a train batch."""
    if features.ndim != 3 or features.shape[-1] < 51:
        raise ValueError("训练增强需要形状 (B, T, >=51) 的特征")
    if not any(
        (
            config.pose_jitter_std,
            config.joint_dropout_probability,
            config.horizontal_flip_probability,
            config.temporal_augmentation_probability,
            config.joint_occlusion_probability,
            config.bbox_jitter_std,
            config.track_break_probability,
            config.temporal_speed_probability,
        )
    ):
        return features
    generator = torch.Generator(device=features.device)
    generator.manual_seed(seed)
    augmented = features.clone()
    poses = augmented[..., :51].reshape(*augmented.shape[:2], 17, 3)
    observed = poses[..., 2] > 0
    if config.pose_jitter_std:
        noise = torch.randn(
            poses[..., :2].shape, device=features.device, generator=generator
        )
        # Low-confidence keypoints are less reliable, so they receive a larger
        # (but bounded) coordinate perturbation. Missing joints stay exactly zero.
        confidence_scale = (1.5 - poses[..., 2]).clamp(0.5, 1.5)
        poses[..., :2].add_(
            noise
            * config.pose_jitter_std
            * confidence_scale.unsqueeze(-1)
            * observed.unsqueeze(-1)
        )
    if config.joint_dropout_probability:
        missing = torch.rand(
            observed.shape, device=features.device, generator=generator
        )
        poses.masked_fill_(
            (observed & (missing < config.joint_dropout_probability)).unsqueeze(-1), 0.0
        )
    if config.horizontal_flip_probability:
        selected = (
            torch.rand(features.shape[0], device=features.device, generator=generator)
            < config.horizontal_flip_probability
        )
        if selected.any():
            augmented[selected] = horizontal_flip_features(augmented[selected])
    if config.temporal_augmentation_probability:
        selected = (
            torch.rand(features.shape[0], device=features.device, generator=generator)
            < config.temporal_augmentation_probability
        )
        count = max(1, int(features.shape[1] * config.temporal_max_frame_fraction))
        priorities = torch.rand(
            (features.shape[0], features.shape[1] - 1),
            device=features.device,
            generator=generator,
        )
        positions = torch.topk(priorities, count, dim=1).indices + 1
        repeat_mask = torch.zeros(
            features.shape[:2], dtype=torch.bool, device=features.device
        )
        repeat_mask.scatter_(1, positions, True)
        repeat_mask &= selected.unsqueeze(1)
        previous = torch.cat((augmented[:, :1], augmented[:, :-1]), dim=1)
        augmented = torch.where(repeat_mask.unsqueeze(-1), previous, augmented)
        poses = augmented[..., :51].reshape(*augmented.shape[:2], 17, 3)
    if config.joint_occlusion_probability:
        selected = (
            torch.rand(features.shape[0], device=features.device, generator=generator)
            < config.joint_occlusion_probability
        )
        spans = torch.randint(
            2,
            5,
            (features.shape[0],),
            device=features.device,
            generator=generator,
        )
        starts = (
            torch.rand(features.shape[0], device=features.device, generator=generator)
            * (features.shape[1] - spans + 1)
        ).floor().long()
        timeline = torch.arange(features.shape[1], device=features.device)
        time_mask = (timeline >= starts[:, None]) & (
            timeline < (starts + spans)[:, None]
        )
        limb_masks = torch.zeros((4, 17), dtype=torch.bool, device=features.device)
        limb_masks[0, (5, 7, 9)] = True
        limb_masks[1, (6, 8, 10)] = True
        limb_masks[2, (11, 13, 15)] = True
        limb_masks[3, (12, 14, 16)] = True
        chosen = torch.randint(
            0,
            4,
            (features.shape[0],),
            device=features.device,
            generator=generator,
        )
        occluded = selected[:, None, None] & time_mask[:, :, None] & limb_masks[chosen, None]
        poses.masked_fill_(occluded.unsqueeze(-1), 0.0)
    if config.bbox_jitter_std:
        if augmented.shape[-1] < 57:
            raise ValueError("bbox 抖动需要 Geometry sidecar 特征")
        geometry = augmented[..., 51:57]
        bbox_observed = geometry[..., 5] > 0
        centre_noise = torch.randn(
            geometry[..., :2].shape,
            device=features.device,
            generator=generator,
        ) * config.bbox_jitter_std
        scale_noise = torch.randn(
            geometry[..., 2:4].shape,
            device=features.device,
            generator=generator,
        ) * config.bbox_jitter_std
        geometry[..., :2] = torch.where(
            bbox_observed.unsqueeze(-1),
            (geometry[..., :2] + centre_noise).clamp(0.0, 1.0),
            geometry[..., :2],
        )
        sizes = geometry[..., 2:4] * torch.exp(scale_noise)
        geometry[..., 2:4] = torch.where(
            bbox_observed.unsqueeze(-1), sizes.clamp_min(1e-4), geometry[..., 2:4]
        )
        aspect = torch.log(
            geometry[..., 2].clamp_min(1e-4)
            / geometry[..., 3].clamp_min(1e-4)
        ).div(3.0).clamp(-1.0, 1.0)
        geometry[..., 4] = torch.where(bbox_observed, aspect, geometry[..., 4])
    if config.track_break_probability:
        selected = (
            torch.rand(features.shape[0], device=features.device, generator=generator)
            < config.track_break_probability
        )
        maximum = max(1, math.ceil(features.shape[1] * config.track_break_max_fraction))
        spans = torch.randint(
            1,
            maximum + 1,
            (features.shape[0],),
            device=features.device,
            generator=generator,
        )
        starts = (
            torch.rand(features.shape[0], device=features.device, generator=generator)
            * (features.shape[1] - spans + 1)
        ).floor().long()
        timeline = torch.arange(features.shape[1], device=features.device)
        missing = selected[:, None] & (timeline >= starts[:, None]) & (
            timeline < (starts + spans)[:, None]
        )
        corruption_dim = 57 if augmented.shape[-1] >= 57 else 51
        augmented[..., :corruption_dim].masked_fill_(missing.unsqueeze(-1), 0.0)
    if config.temporal_speed_probability:
        selected = (
            torch.rand(features.shape[0], device=features.device, generator=generator)
            < config.temporal_speed_probability
        )
        endpoint = features.shape[1] - 1
        timeline = torch.arange(features.shape[1], device=features.device)
        rates = config.temporal_speed_min + torch.rand(
            features.shape[0], device=features.device, generator=generator
        ) * (config.temporal_speed_max - config.temporal_speed_min)
        source = (
            endpoint - (endpoint - timeline[None, :]) * rates[:, None]
        ).round().long().clamp_(0, endpoint)
        warped = torch.gather(
            augmented,
            1,
            source.unsqueeze(-1).expand(-1, -1, augmented.shape[-1]),
        )
        augmented = torch.where(selected[:, None, None], warped, augmented)
    return augmented


@dataclass(frozen=True)
class MILConfig(MultiStreamConfig):
    mil_topk_fraction: float = 0.2
    mil_loss_weight: float = 0.5
    mil_clip_batch_size: int = 8
    transition_aux_weight: float = 0.5
    hard_negative_ranking_weight: float = 0.0
    hard_negative_ranking_margin: float = 0.5
    stage_aux_weight: float = 0.2
    physics_aux_weight: float = 0.1
    backbone_learning_rate: float | None = None
    new_head_learning_rate: float | None = None
    warmup_epochs: int = 0
    clip_aggregator: str = "legacy_topk_train_max_eval"
    smooth_max_temperature_start: float = 0.5
    smooth_max_temperature_end: float = 0.05
    use_conditional_verifier: bool = False
    candidate_aux_weight: float = 0.25
    verifier_aux_weight: float = 0.2
    auprc_surrogate_weight: float = 0.0
    negative_spike_weight: float = 0.0
    hard_negative_memory_size: int = 256
    high_recall_positive_fraction: float = 0.25
    verifier_gate_max: float = 1.0
    candidate_anchor_weight: float = 0.0
    verifier_background_fraction: float = 1.0
    verifier_head_only_epochs: int = 0
    use_event_verifier: bool = False
    event_hidden_dim: int = 32
    event_gate_max: float = 1.0
    event_head_only_epochs: int = 0
    event_aux_weight: float = 0.5
    event_suppression_only: bool = False
    event_penalty_cap: float = 0.25
    event_positive_penalty_weight: float = 1.0
    symmetry_consistency_weight: float = 0.0
    real_distortion_mode: str | None = None
    real_distortion_probability: float = 0.0

    @classmethod
    def from_json(cls, path: Path) -> MILConfig:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("MIL 配置必须是 JSON 对象")
        unknown = set(payload) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"MIL 配置包含未知字段: {sorted(unknown)}")
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
        if not 0.0 < self.mil_topk_fraction <= 1.0:
            raise ValueError("mil_topk_fraction 必须位于 (0, 1]")
        if self.mil_loss_weight <= 0.0:
            raise ValueError("mil_loss_weight 必须为正")
        if self.mil_clip_batch_size < 1:
            raise ValueError("mil_clip_batch_size 必须为正")
        if self.transition_aux_weight <= 0.0:
            raise ValueError("transition_aux_weight 必须为正")
        if self.hard_negative_ranking_weight < 0.0:
            raise ValueError("hard_negative_ranking_weight 不能为负")
        if self.hard_negative_ranking_margin <= 0.0:
            raise ValueError("hard_negative_ranking_margin 必须为正")
        if self.stage_aux_weight <= 0.0:
            raise ValueError("stage_aux_weight 必须为正")
        if self.physics_aux_weight <= 0.0:
            raise ValueError("physics_aux_weight 必须为正")
        rates = (self.backbone_learning_rate, self.new_head_learning_rate)
        if any(rate is not None for rate in rates) and any(
            rate is None or rate <= 0.0 for rate in rates
        ):
            raise ValueError("分层学习率必须同时提供两个正值")
        if not 0 <= self.warmup_epochs < self.epochs:
            raise ValueError("warmup_epochs 必须位于 [0, epochs)")
        if self.clip_aggregator not in {
            "legacy_topk_train_max_eval",
            "max",
            "topk_mean",
            "smooth_max",
        }:
            raise ValueError("clip_aggregator 无效")
        if not (
            self.smooth_max_temperature_start > 0.0
            and self.smooth_max_temperature_end > 0.0
            and self.smooth_max_temperature_start
            >= self.smooth_max_temperature_end
        ):
            raise ValueError("smooth-max 温度必须为正且从高到低退火")
        if not isinstance(self.use_conditional_verifier, bool):
            raise TypeError("use_conditional_verifier 必须是布尔值")
        if self.verifier_aux_weight <= 0.0:
            raise ValueError("verifier_aux_weight 必须为正")
        if self.candidate_aux_weight < 0.0:
            raise ValueError("candidate_aux_weight 不能为负")
        if self.auprc_surrogate_weight < 0.0:
            raise ValueError("auprc_surrogate_weight 不能为负")
        if self.negative_spike_weight < 0.0:
            raise ValueError("negative_spike_weight 不能为负")
        if self.hard_negative_memory_size < 1:
            raise ValueError("hard_negative_memory_size 必须为正")
        if not 0.0 < self.high_recall_positive_fraction <= 1.0:
            raise ValueError("high_recall_positive_fraction 必须位于 (0, 1]")
        if not 0.0 <= self.verifier_gate_max <= 1.0:
            raise ValueError("verifier_gate_max 必须位于 [0, 1]")
        if self.candidate_anchor_weight < 0.0:
            raise ValueError("candidate_anchor_weight 不能为负")
        if not 0.0 < self.verifier_background_fraction <= 1.0:
            raise ValueError("verifier_background_fraction 必须位于 (0, 1]")
        if not 0 <= self.verifier_head_only_epochs <= self.epochs:
            raise ValueError("verifier_head_only_epochs 必须位于 [0, epochs]")
        if self.verifier_head_only_epochs and not self.use_conditional_verifier:
            raise ValueError("head-only warmup 需要启用受控转变验证器")
        if not isinstance(self.use_event_verifier, bool):
            raise TypeError("use_event_verifier 必须是布尔值")
        if self.use_event_verifier and self.use_conditional_verifier:
            raise ValueError("窗口 verifier 与事件 verifier 不能同时启用")
        if self.event_hidden_dim < 1:
            raise ValueError("event_hidden_dim 必须为正")
        if not 0.0 <= self.event_gate_max <= 1.0:
            raise ValueError("event_gate_max 必须位于 [0, 1]")
        if not 0 <= self.event_head_only_epochs <= self.epochs:
            raise ValueError("event_head_only_epochs 必须位于 [0, epochs]")
        if self.event_head_only_epochs and not self.use_event_verifier:
            raise ValueError("event head-only warmup 需要事件级验证器")
        if self.event_aux_weight <= 0.0:
            raise ValueError("event_aux_weight 必须为正")
        if not isinstance(self.event_suppression_only, bool):
            raise TypeError("event_suppression_only 必须是布尔值")
        if self.event_suppression_only and not self.use_event_verifier:
            raise ValueError("单向事件抑制需要事件级验证器")
        if self.event_penalty_cap <= 0.0:
            raise ValueError("event_penalty_cap 必须为正")
        if self.event_positive_penalty_weight < 0.0:
            raise ValueError("event_positive_penalty_weight 不能为负")
        if self.symmetry_consistency_weight < 0.0:
            raise ValueError("symmetry_consistency_weight 不能为负")
        if self.real_distortion_mode not in {None, "observation", "temporal"}:
            raise ValueError("real_distortion_mode 必须为 observation、temporal 或 null")
        if not 0.0 <= self.real_distortion_probability <= 1.0:
            raise ValueError("real_distortion_probability 必须位于 [0, 1]")
        if bool(self.real_distortion_mode) != bool(self.real_distortion_probability):
            raise ValueError("真实失真 mode 与 probability 必须同时启用或关闭")
        if self.symmetry_consistency_weight and (
            self.use_event_verifier
            or self.use_conditional_verifier
            or self.use_transition_rule_late_fusion
        ):
            raise ValueError("对称一致性当前仅支持标准 Stage-S1 分类路径")


def aggregate_topk_logits(
    logits: torch.Tensor, group_sizes: list[int], fraction: float
) -> torch.Tensor:
    """Backward-compatible Top-k mean facade for the canonical aggregator."""
    return aggregate_clip_logits(
        logits, group_sizes, mode="topk_mean", topk_fraction=fraction
    )


def smooth_max_temperature(config: MILConfig, epoch: int) -> float:
    """Return the preregistered geometric temperature schedule."""
    if not 0 <= epoch < config.epochs:
        raise ValueError("epoch 超出配置范围")
    if config.epochs == 1:
        return config.smooth_max_temperature_end
    progress = epoch / (config.epochs - 1)
    return config.smooth_max_temperature_start * (
        config.smooth_max_temperature_end
        / config.smooth_max_temperature_start
    ) ** progress


def verifier_scale(config: MILConfig, epoch: int) -> float:
    """Linearly expose the residual verifier after a head-only first epoch."""
    if not 0 <= epoch < config.epochs:
        raise ValueError("epoch 超出配置范围")
    if config.epochs == 1:
        return 0.0
    return config.verifier_gate_max * epoch / (config.epochs - 1)


def event_verifier_scale(config: MILConfig, epoch: int) -> float:
    if not 0 <= epoch < config.epochs:
        raise ValueError("epoch 超出配置范围")
    if config.epochs == 1:
        return 0.0
    return config.event_gate_max * epoch / (config.epochs - 1)


def combine_event_logits(
    candidate_logits: torch.Tensor,
    event_logits: torch.Tensor,
    scale: torch.Tensor | float,
    config: MILConfig,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Combine event evidence, optionally as a bounded downward-only penalty."""
    if candidate_logits.shape != event_logits.shape:
        raise ValueError("candidate/event logits 必须同形")
    if config.event_suppression_only:
        penalty = config.event_penalty_cap * torch.sigmoid(event_logits)
        return candidate_logits - scale * penalty, penalty
    return candidate_logits + scale * event_logits, None


def aggregate_training_logits(
    logits: torch.Tensor,
    group_sizes: list[int],
    config: MILConfig,
    *,
    epoch: int,
) -> torch.Tensor:
    """Use the configured clip aggregator shared with validation."""
    if config.clip_aggregator == "max":
        return aggregate_group_logits(logits, group_sizes, mode="max")
    if config.clip_aggregator in {"legacy_topk_train_max_eval", "topk_mean"}:
        return aggregate_group_logits(
            logits,
            group_sizes,
            mode="topk_mean",
            topk_fraction=config.mil_topk_fraction,
        )
    return aggregate_group_logits(
        logits,
        group_sizes,
        mode="smooth_max",
        temperature=smooth_max_temperature(config, epoch),
    )


def aggregate_evaluation_logits(
    logits: torch.Tensor,
    group_sizes: list[int],
    config: MILConfig,
    *,
    epoch: int,
) -> torch.Tensor:
    """Apply the configured validation aggregator to complete clip groups."""
    if config.clip_aggregator in {"legacy_topk_train_max_eval", "max"}:
        return aggregate_group_logits(logits, group_sizes, mode="max")
    if config.clip_aggregator == "topk_mean":
        return aggregate_group_logits(
            logits,
            group_sizes,
            mode="topk_mean",
            topk_fraction=config.mil_topk_fraction,
        )
    return aggregate_group_logits(
        logits,
        group_sizes,
        mode="smooth_max",
        temperature=smooth_max_temperature(config, epoch),
    )


def high_recall_auprc_surrogate(
    clip_logits: torch.Tensor,
    clip_labels: torch.Tensor,
    remembered_negatives: torch.Tensor,
    *,
    positive_fraction: float,
) -> tuple[torch.Tensor, int]:
    """Focus ranking gradients on low positives and high hard negatives.

    The detached memory supplies cross-batch controlled-transition negatives;
    current-batch negatives retain gradients. Softplus is a stable pairwise
    ranking surrogate, restricted to the positive tail that controls R90/R95.
    """
    positive = clip_logits[clip_labels > 0.5]
    negative = clip_logits[clip_labels < 0.5]
    if remembered_negatives.numel():
        negative = torch.cat((negative, remembered_negatives.to(clip_logits.device)))
    if not positive.numel() or not negative.numel():
        return clip_logits.new_zeros(()), 0
    positive_count = max(1, math.ceil(positive.numel() * positive_fraction))
    hard_positive = torch.topk(positive, positive_count, largest=False).values
    negative_count = min(negative.numel(), max(1, positive_count * 4))
    hard_negative = torch.topk(negative, negative_count).values
    pairs = hard_positive.numel() * hard_negative.numel()
    return (
        nn.functional.softplus(hard_negative[:, None] - hard_positive[None, :]).mean(),
        pairs,
    )


def hard_negative_ranking_loss(
    clip_logits: torch.Tensor,
    clip_labels: torch.Tensor,
    activities: np.ndarray,
    *,
    margin: float,
) -> torch.Tensor:
    """Require fall clips to outrank confusing posture-transition clips."""
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


STAGE_NAMES = (
    "fall_process",
    "fallen",
    "controlled_transition",
    "stationary_nonfall",
)


def build_stage_targets(semantic_codes: np.ndarray, activities: np.ndarray) -> np.ndarray:
    """Map immutable window semantics to a four-way training-only stage task."""
    semantic_codes = np.asarray(semantic_codes)
    activities = np.asarray(activities, dtype=object)
    if semantic_codes.ndim != 1 or semantic_codes.shape != activities.shape:
        raise ValueError("semantic_codes/activities 必须为同形一维数组")
    if not np.isin(semantic_codes, tuple(SEMANTIC_TO_CODE.values())).all():
        raise ValueError("semantic_codes 包含未知值")
    targets = np.full(semantic_codes.shape, 3, dtype=np.int64)
    controlled = np.isin(activities, ("lie_down", "sit_down", "stand_up"))
    targets[controlled] = 2
    # Immutable event semantics take precedence over a source-directory activity:
    # a positive incident stored below a confusing-activity directory is still a
    # fall/fallen window, not a controlled non-fall transition.
    targets[semantic_codes == SEMANTIC_TO_CODE["post_fall_state"]] = 1
    targets[semantic_codes == SEMANTIC_TO_CODE["fall_process"]] = 0
    return targets


VERIFIER_NAMES = (
    "uncontrolled_fall",
    "controlled_descent",
    "reverse_transition",
    "stationary_or_background",
)


def build_verifier_targets(
    semantic_codes: np.ndarray, activities: np.ndarray
) -> np.ndarray:
    """Build the fall-vs-controlled-transition specialist target."""
    semantic_codes = np.asarray(semantic_codes)
    activities = np.asarray(activities, dtype=object)
    if semantic_codes.ndim != 1 or semantic_codes.shape != activities.shape:
        raise ValueError("semantic_codes/activities 必须为同形一维数组")
    if not np.isin(semantic_codes, tuple(SEMANTIC_TO_CODE.values())).all():
        raise ValueError("semantic_codes 包含未知值")
    targets = np.full(semantic_codes.shape, 3, dtype=np.int64)
    targets[np.isin(activities, ("lie_down", "sit_down"))] = 1
    targets[activities == "stand_up"] = 2
    positive = np.isin(
        semantic_codes,
        (
            SEMANTIC_TO_CODE["fall_process"],
            SEMANTIC_TO_CODE["post_fall_state"],
        ),
    )
    targets[positive] = 0
    return targets


def physics_targets(features: torch.Tensor) -> torch.Tensor:
    """Build causal, bounded supervision for torso orientation and descent."""
    from models.multiscale_multistream_tcn import build_pifr_features

    pifr = build_pifr_features(features)
    height_delta = (pifr[:, -1, 1] - pifr[:, 0, 1]).clamp(-1.0, 1.0)
    bbox_aspect = (features[:, -1, 54] / features[:, -1, 55].clamp_min(1e-4)).clamp(0.0, 4.0) / 4.0
    return torch.stack((pifr[:, -1, 2], height_delta, bbox_aspect), dim=1)


def _clip_groups(
    clip_indices: np.ndarray, clips: list[dict[str, Any]]
) -> tuple[list[np.ndarray], np.ndarray]:
    if clip_indices.ndim != 1:
        raise ValueError("clip_indices 必须为一维")
    groups: list[np.ndarray] = []
    labels: list[float] = []
    for clip_index in np.unique(clip_indices):
        if clip_index < 0 or clip_index >= len(clips):
            raise ValueError("clip_indices 超出 clips metadata")
        group = np.flatnonzero(clip_indices == clip_index)
        if not group.size:
            continue
        has_fall = clips[int(clip_index)].get("has_fall")
        if not isinstance(has_fall, bool):
            raise TypeError("clips metadata 缺少 bool has_fall")
        groups.append(group)
        labels.append(float(has_fall))
    if not groups or not any(labels) or all(labels):
        raise ValueError("MIL 训练 clip 必须同时包含正负样本")
    return groups, np.asarray(labels, dtype=np.float32)


def train_epoch_mil(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    features: torch.Tensor,
    labels: torch.Tensor,
    groups: list[np.ndarray],
    clip_labels: np.ndarray,
    transition_targets: torch.Tensor | None = None,
    clip_activities: np.ndarray | None = None,
    stage_targets: torch.Tensor | None = None,
    verifier_targets: torch.Tensor | None = None,
    anchor_model: MultiStreamMultiScaleAttentionTCN | None = None,
    real_distortion_bank: RealDistortionBank | None = None,
    *,
    device: torch.device,
    config: MILConfig,
    epoch: int,
) -> tuple[float, dict[str, float | int]]:
    """Run one deterministic epoch over batches of complete clips."""
    model.train()
    verifier_head_only = bool(
        config.verifier_head_only_epochs
        and epoch < config.verifier_head_only_epochs
    )
    event_head_only = bool(
        config.event_head_only_epochs and epoch < config.event_head_only_epochs
    )
    if verifier_head_only or event_head_only:
        # Zero LR alone does not freeze BatchNorm running statistics. Keep the
        # entire pretrained candidate in eval mode and train only the new linear
        # verifier so epoch zero is prediction-identical to the checkpoint.
        model.eval()
        if not isinstance(model, MultiStreamMultiScaleAttentionTCN):
            raise TypeError("verifier head-only warmup 需要多流模型")
        if verifier_head_only:
            assert model.verifier_classifier is not None
            model.verifier_classifier.train()
        if event_head_only:
            assert model.event_projection is not None
            assert model.event_encoder is not None
            assert model.event_classifier is not None
            model.event_projection.train()
            model.event_encoder.train()
            model.event_classifier.train()
    if config.use_conditional_verifier:
        if not isinstance(model, MultiStreamMultiScaleAttentionTCN):
            raise TypeError("受控转变验证器需要 MultiStreamMultiScaleAttentionTCN")
        model.set_verifier_scale(verifier_scale(config, epoch))
    if config.use_event_verifier:
        if not isinstance(model, MultiStreamMultiScaleAttentionTCN):
            raise TypeError("事件级验证器需要 MultiStreamMultiScaleAttentionTCN")
        model.set_event_verifier_scale(event_verifier_scale(config, epoch))
    if config.candidate_anchor_weight and anchor_model is None:
        raise ValueError("候选 logit 锚定需要冻结初始化模型")
    clip_order = np.random.default_rng(config.seed + epoch).permutation(len(groups))
    window_pos_weight = float((labels.numel() - labels.sum()) / labels.sum())
    clip_pos_weight = float((clip_labels.size - clip_labels.sum()) / clip_labels.sum())
    window_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(window_pos_weight, device=device)
    )
    clip_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(clip_pos_weight, device=device)
    )
    transition_criterion = None
    if config.use_transition_rule_late_fusion:
        if transition_targets is None or transition_targets.shape != labels.shape:
            raise ValueError("转变规则辅助监督需要与窗口标签同形的 targets")
        positives = float(transition_targets.sum())
        if positives <= 0.0 or positives >= transition_targets.numel():
            raise ValueError("转变规则 targets 必须同时含正负样本")
        transition_criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(
                (transition_targets.numel() - positives) / positives, device=device
            )
        )
    if (
        config.hard_negative_ranking_weight or config.auprc_surrogate_weight
    ) and clip_activities is None:
        raise ValueError("困难负例排序损失需要每个训练 clip 的 activity")
    if config.event_suppression_only and clip_activities is None:
        raise ValueError("单向事件抑制需要每个训练 clip 的 activity")
    stage_criterion = None
    if config.use_stage_auxiliary:
        if stage_targets is None or stage_targets.shape != labels.shape:
            raise ValueError("阶段辅助监督需要与窗口标签同形的 targets")
        counts = torch.bincount(stage_targets, minlength=len(STAGE_NAMES)).float()
        if torch.any(counts == 0):
            raise ValueError("阶段辅助监督必须覆盖全部类别")
        stage_criterion = nn.CrossEntropyLoss(weight=(counts.sum() / counts).to(device))
    verifier_criterion = None
    if config.use_conditional_verifier:
        if verifier_targets is None or verifier_targets.shape != labels.shape:
            raise ValueError("受控转变验证器需要与窗口标签同形的 targets")
        counts = torch.bincount(
            verifier_targets, minlength=len(VERIFIER_NAMES)
        ).float()
        if torch.any(counts == 0):
            raise ValueError("受控转变验证器必须覆盖全部类别")
        verifier_criterion = nn.CrossEntropyLoss(
            weight=(counts.sum() / counts).to(device)
        )
    loss_sum = 0.0
    seen_clips = 0
    remembered_negatives = torch.empty(0, device=device)
    surrogate_sum = 0.0
    active_pairs = 0
    max_negative_sum = 0.0
    negative_batches = 0
    positive_penalty_sum = 0.0
    controlled_penalty_sum = 0.0
    penalty_batches = 0
    symmetry_sum = 0.0
    for start in range(0, len(groups), config.mil_clip_batch_size):
        selected = clip_order[start : start + config.mil_clip_batch_size]
        group_sizes = [int(groups[int(index)].size) for index in selected]
        window_indices = np.concatenate([groups[int(index)] for index in selected])
        batch_x = augment_training_features(
            features[window_indices].to(device),
            config,
            seed=config.seed + epoch * 1_000_003 + start,
        )
        if config.real_distortion_mode:
            if real_distortion_bank is None:
                raise ValueError("配置启用真实失真但未加载模板库")
            batch_x = replay_real_distortion(
                batch_x,
                real_distortion_bank,
                mode=config.real_distortion_mode,
                probability=config.real_distortion_probability,
                seed=config.seed + epoch * 1_000_003 + start + 97,
            )
        batch_y = labels[window_indices].to(device)
        batch_clip_y = torch.from_numpy(clip_labels[selected]).to(device)
        optimizer.zero_grad(set_to_none=True)
        physics_prediction = None
        candidate_logits = None
        verifier_logits = None
        event_clip_logits = None
        event_delta = None
        event_penalty = None
        if config.use_event_verifier:
            assert isinstance(model, MultiStreamMultiScaleAttentionTCN)
            assert model.event_verifier_scale is not None
            fused = model.encode(batch_x)
            class_logits = model.classifier(fused)
            candidate_logits = class_logits[:, 1] - class_logits[:, 0]
            candidate_clip_logits = aggregate_training_logits(
                candidate_logits, group_sizes, config, epoch=epoch
            )
            event_delta = model.event_verifier_delta(fused, group_sizes)
            event_clip_logits, event_penalty = combine_event_logits(
                candidate_clip_logits,
                event_delta,
                model.event_verifier_scale,
                config,
            )
            logits = candidate_logits
            transition_logits = None
            stage_logits = None
        elif config.use_conditional_verifier:
            if not isinstance(model, MultiStreamMultiScaleAttentionTCN):
                raise TypeError("受控转变验证器需要 MultiStreamMultiScaleAttentionTCN")
            logits, candidate_logits, verifier_logits = model.forward_with_verifier(
                batch_x
            )
            transition_logits = None
            stage_logits = None
        elif config.use_transition_rule_late_fusion:
            if not isinstance(model, MultiStreamMultiScaleAttentionTCN):
                raise TypeError(
                    "转变规则 late fusion 需要 MultiStreamMultiScaleAttentionTCN"
                )
            logits, transition_logits = model.forward_with_transition(batch_x)
        else:
            if config.use_stage_auxiliary:
                logits, stage_logits = model.forward_with_stage(batch_x)
                if config.use_physics_auxiliary:
                    _, physics_prediction = model.forward_with_physics(batch_x)
            elif config.use_physics_auxiliary:
                logits, physics_prediction = model.forward_with_physics(batch_x)
            else:
                logits = model(batch_x)
                stage_logits = None
            transition_logits = None
        clip_logits = (
            event_clip_logits
            if event_clip_logits is not None
            else aggregate_training_logits(logits, group_sizes, config, epoch=epoch)
        )
        loss = window_criterion(
            logits, batch_y
        ) + config.mil_loss_weight * clip_criterion(clip_logits, batch_clip_y)
        if event_delta is not None:
            if config.event_suppression_only:
                assert clip_activities is not None and event_penalty is not None
                controlled = torch.from_numpy(
                    np.isin(
                        clip_activities[selected],
                        ("lie_down", "lying", "stand_up", "sit_down"),
                    )
                ).to(device) & (batch_clip_y < 0.5)
                controlled_targets = controlled.to(batch_clip_y.dtype)
                loss = loss + config.event_aux_weight * nn.functional.binary_cross_entropy_with_logits(
                    event_delta, controlled_targets
                )
                positive_penalty = event_penalty[batch_clip_y > 0.5]
                if positive_penalty.numel():
                    loss = loss + config.event_positive_penalty_weight * positive_penalty.mean()
                    positive_penalty_sum += float(positive_penalty.mean().detach())
                controlled_penalty = event_penalty[controlled]
                if controlled_penalty.numel():
                    controlled_penalty_sum += float(controlled_penalty.mean().detach())
                penalty_batches += 1
            else:
                loss = loss + config.event_aux_weight * clip_criterion(
                    event_delta, batch_clip_y
                )
        if candidate_logits is not None and config.candidate_aux_weight:
            candidate_clip_logits = aggregate_training_logits(
                candidate_logits, group_sizes, config, epoch=epoch
            )
            loss = loss + config.candidate_aux_weight * (
                window_criterion(candidate_logits, batch_y)
                + config.mil_loss_weight
                * clip_criterion(candidate_clip_logits, batch_clip_y)
            )
        if config.hard_negative_ranking_weight:
            assert clip_activities is not None
            loss = loss + config.hard_negative_ranking_weight * hard_negative_ranking_loss(
                clip_logits,
                batch_clip_y,
                clip_activities[selected],
                margin=config.hard_negative_ranking_margin,
            )
        if transition_criterion is not None and transition_logits is not None:
            loss = loss + config.transition_aux_weight * transition_criterion(
                transition_logits, transition_targets[window_indices].to(device)
            )
        if stage_criterion is not None and stage_logits is not None:
            loss = loss + config.stage_aux_weight * stage_criterion(
                stage_logits, stage_targets[window_indices].to(device)
            )
        if verifier_criterion is not None and verifier_logits is not None:
            assert verifier_targets is not None
            batch_verifier_targets = verifier_targets[window_indices].to(device)
            selected_for_verifier = batch_verifier_targets != 3
            background = torch.nonzero(
                batch_verifier_targets == 3, as_tuple=False
            ).flatten()
            if background.numel():
                count = max(
                    1,
                    math.ceil(
                        background.numel() * config.verifier_background_fraction
                    ),
                )
                assert candidate_logits is not None
                hardest = background[
                    torch.topk(candidate_logits.detach()[background], count).indices
                ]
                selected_for_verifier[hardest] = True
            loss = loss + config.verifier_aux_weight * verifier_criterion(
                verifier_logits[selected_for_verifier],
                batch_verifier_targets[selected_for_verifier],
            )
        if config.candidate_anchor_weight:
            assert anchor_model is not None and candidate_logits is not None
            with torch.no_grad():
                _, anchor_candidate, _ = anchor_model.forward_with_verifier(batch_x)
            loss = loss + config.candidate_anchor_weight * nn.functional.smooth_l1_loss(
                candidate_logits, anchor_candidate
            )
        if config.negative_spike_weight:
            max_logits = aggregate_group_logits(logits, group_sizes, mode="max")
            negative_max = max_logits[batch_clip_y < 0.5]
            if negative_max.numel():
                loss = loss + config.negative_spike_weight * nn.functional.softplus(
                    negative_max
                ).mean()
                max_negative_sum += float(negative_max.max().detach())
                negative_batches += 1
        if config.auprc_surrogate_weight:
            assert clip_activities is not None
            hard_activity = torch.from_numpy(
                np.isin(
                    clip_activities[selected],
                    ("lie_down", "lying", "stand_up", "sit_down"),
                )
            ).to(device)
            rank_mask = (batch_clip_y > 0.5) | hard_activity
            surrogate, pairs = high_recall_auprc_surrogate(
                clip_logits[rank_mask],
                batch_clip_y[rank_mask],
                remembered_negatives,
                positive_fraction=config.high_recall_positive_fraction,
            )
            loss = loss + config.auprc_surrogate_weight * surrogate
            surrogate_sum += float(surrogate.detach())
            active_pairs += pairs
            new_negatives = clip_logits[
                (batch_clip_y < 0.5) & hard_activity
            ].detach()
            if new_negatives.numel():
                remembered_negatives = torch.cat(
                    (remembered_negatives, new_negatives)
                )[-config.hard_negative_memory_size :]
        if config.use_physics_auxiliary and physics_prediction is not None:
            loss = loss + config.physics_aux_weight * nn.functional.smooth_l1_loss(
                physics_prediction, physics_targets(batch_x)
            )
        if config.symmetry_consistency_weight:
            mirrored_x = horizontal_flip_features(batch_x)
            if config.use_stage_auxiliary:
                mirrored_logits, mirrored_stage = model.forward_with_stage(mirrored_x)
                assert stage_logits is not None
                symmetry_loss = nn.functional.smooth_l1_loss(
                    logits, mirrored_logits
                ) + 0.25 * nn.functional.smooth_l1_loss(
                    stage_logits, mirrored_stage
                )
            else:
                mirrored_logits = model(mirrored_x)
                symmetry_loss = nn.functional.smooth_l1_loss(logits, mirrored_logits)
            loss = loss + config.symmetry_consistency_weight * symmetry_loss
            symmetry_sum += float(symmetry_loss.detach())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        loss_sum += float(loss.detach()) * len(selected)
        seen_clips += len(selected)
    return loss_sum / seen_clips, {
        "auprc_surrogate_mean": surrogate_sum
        / max(1, math.ceil(len(groups) / config.mil_clip_batch_size)),
        "active_hard_pairs": active_pairs,
        "memory_bank_size": int(remembered_negatives.numel()),
        "max_negative_logit_mean": max_negative_sum / max(1, negative_batches),
        "positive_event_penalty_mean": positive_penalty_sum / max(1, penalty_batches),
        "controlled_event_penalty_mean": controlled_penalty_sum
        / max(1, penalty_batches),
        "symmetry_consistency_mean": symmetry_sum
        / max(1, math.ceil(len(groups) / config.mil_clip_batch_size)),
    }


@torch.inference_mode()
def predict_event_clip_scores(
    model: MultiStreamMultiScaleAttentionTCN,
    features: torch.Tensor,
    clip_indices: np.ndarray,
    clip_count: int,
    *,
    device: torch.device,
    config: MILConfig,
    epoch: int,
) -> np.ndarray:
    """Predict event-level scores while preserving complete ordered clips."""
    if not config.use_event_verifier or model.event_verifier_scale is None:
        raise ValueError("事件评分需要启用事件级验证器")
    model.eval()
    scores = np.zeros(clip_count, dtype=np.float32)
    groups = [np.flatnonzero(clip_indices == index) for index in range(clip_count)]
    present = [index for index, group in enumerate(groups) if group.size]
    for start in range(0, len(present), config.mil_clip_batch_size):
        selected = present[start : start + config.mil_clip_batch_size]
        group_sizes = [int(groups[index].size) for index in selected]
        indices = np.concatenate([groups[index] for index in selected])
        fused = model.encode(features[indices].to(device))
        class_logits = model.classifier(fused)
        candidate_logits = class_logits[:, 1] - class_logits[:, 0]
        candidate_clip_logits = aggregate_evaluation_logits(
            candidate_logits, group_sizes, config, epoch=epoch
        )
        event_delta = model.event_verifier_delta(fused, group_sizes)
        final_logits, _ = combine_event_logits(
            candidate_clip_logits,
            event_delta,
            model.event_verifier_scale,
            config,
        )
        scores[np.asarray(selected)] = torch.sigmoid(final_logits).cpu().numpy()
    return scores


def train_mil(
    *,
    config: MILConfig,
    train_cache: WindowMemmapCache,
    val_cache: WindowMemmapCache,
    train_sidecar: np.ndarray,
    val_sidecar: np.ndarray,
    output_dir: Path,
    device_name: str,
    joint_pretrain_checkpoint: Path | None = None,
    initialization_checkpoint: Path | None = None,
    real_distortion_bank_path: Path | None = None,
) -> dict[str, Any]:
    config.validate()
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("请求 CUDA 训练，但 torch.cuda.is_available() 为 False")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("输出目录非空；MIL 训练不支持 resume")
    set_deterministic(config.seed)
    if bool(config.real_distortion_mode) != bool(real_distortion_bank_path):
        raise ValueError("真实失真配置与 --real-distortion-bank 必须同时提供")
    device = torch.device(device_name)
    real_distortion_bank = (
        RealDistortionBank.load(real_distortion_bank_path, device=device)
        if real_distortion_bank_path is not None
        else None
    )
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
    clip_activities = np.asarray(
        [
            str(
                train_cache.metadata["clips"][
                    int(train_clip_indices[int(indices[0])])
                ]["clip_id"]
            ).split("/", 1)[0]
            for indices in groups
        ],
        dtype=object,
    )
    window_activities = _activity_groups(train_cache)[train_indices]
    stage_targets = (
        torch.from_numpy(
            build_stage_targets(
                np.asarray(train_cache.array("semantic_codes")[train_indices]),
                window_activities,
            )
        )
        if config.use_stage_auxiliary
        else None
    )
    verifier_targets = (
        torch.from_numpy(
            build_verifier_targets(
                np.asarray(train_cache.array("semantic_codes")[train_indices]),
                window_activities,
            )
        )
        if config.use_conditional_verifier
        else None
    )
    transition_targets = (
        torch.from_numpy(
            (
                np.asarray(train_cache.array("semantic_codes")[train_indices])
                == SEMANTIC_TO_CODE["fall_process"]
            ).astype(np.float32, copy=False)
        )
        if config.use_transition_rule_late_fusion
        else None
    )
    device = torch.device(device_name)
    model = MultiStreamMultiScaleAttentionTCN(
        stream_channels=config.stream_channels,
        stream_output_dim=config.stream_output_dim,
        dropout=config.dropout,
        use_geometry=True,
        stream_lstm_layers=config.stream_lstm_layers,
        use_rule_features=config.use_rule_features,
        use_pifr_features=config.use_pifr_features,
        use_stgcn_joint=config.use_stgcn_joint,
        use_stgcn_residual_joint=config.use_stgcn_residual_joint,
        use_transformer_encoder=config.use_transformer_encoder,
        transformer_joint_only=config.transformer_joint_only,
        use_transition_rule_late_fusion=config.use_transition_rule_late_fusion,
        use_discriminative_kinematics=config.use_discriminative_kinematics,
        kinematic_feature_dim=config.kinematic_feature_dim,
        use_gated_conv=config.use_gated_conv,
        use_motion_guided_fusion=config.use_motion_guided_fusion,
        motion_validity_mask=config.motion_validity_mask,
        per_joint_mask_correction=config.per_joint_mask_correction,
        use_stream_interaction=config.use_stream_interaction,
        use_stage_auxiliary=config.use_stage_auxiliary,
        use_physics_auxiliary=config.use_physics_auxiliary,
        use_conditional_verifier=config.use_conditional_verifier,
        use_event_verifier=config.use_event_verifier,
        event_hidden_dim=config.event_hidden_dim,
    ).to(device)
    joint_pretrain_sha256 = None
    initialization_checkpoint_sha256 = None
    initialization_loaded_keys = 0
    if initialization_checkpoint is not None:
        payload = torch.load(initialization_checkpoint, map_location="cpu", weights_only=False)
        state = payload.get("model_state") if isinstance(payload, dict) else None
        if not isinstance(state, dict):
            raise TypeError("初始化 checkpoint 缺少 model_state")
        current = model.state_dict()
        compatible = {
            key: value
            for key, value in state.items()
            if key in current and current[key].shape == value.shape
        }
        motion_key = "streams.2.input_projection.weight"
        if config.motion_validity_mask and motion_key in state and motion_key in current:
            source, target = state[motion_key], current[motion_key]
            if source.ndim == target.ndim == 3 and source.shape[0] == target.shape[0] and source.shape[2] == target.shape[2] and source.shape[1] < target.shape[1]:
                expanded = target.clone()
                expanded.zero_()
                expanded[:, : source.shape[1]] = source
                compatible[motion_key] = expanded
        if not compatible:
            raise ValueError("初始化 checkpoint 没有形状兼容的参数")
        current.update(compatible)
        model.load_state_dict(current, strict=True)
        initialization_loaded_keys = len(compatible)
        initialization_checkpoint_sha256 = _sha256_file(initialization_checkpoint)
    if joint_pretrain_checkpoint is not None:
        if config.use_stgcn_joint:
            raise ValueError("ST-GCN joint 流不能加载 StreamEncoder 预训练权重")
        joint_pretrain_checkpoint = Path(joint_pretrain_checkpoint)
        payload = torch.load(joint_pretrain_checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or not isinstance(payload.get("encoder_state"), dict):
            raise ValueError("Joint 预训练 checkpoint 缺少 encoder_state")
        load_result = model.streams[0].load_state_dict(
            payload["encoder_state"], strict=not config.use_stgcn_residual_joint
        )
        if config.use_stgcn_residual_joint and (
            load_result.unexpected_keys
            or any(
                not key.startswith("graph_residual.")
                for key in load_result.missing_keys
            )
        ):
            raise ValueError("Residual ST-GCN Joint 预训练权重不兼容")
        joint_pretrain_sha256 = _sha256_file(joint_pretrain_checkpoint)
    anchor_model = None
    if config.candidate_anchor_weight:
        if not config.use_conditional_verifier:
            raise ValueError("候选 logit 锚定需要启用受控转变验证器")
        if initialization_checkpoint is None:
            raise ValueError("候选 logit 锚定需要 initialization checkpoint")
        anchor_model = copy.deepcopy(model).eval()
        anchor_model.set_verifier_scale(0.0)
        for parameter in anchor_model.parameters():
            parameter.requires_grad_(False)
    if config.backbone_learning_rate is None:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    else:
        assert config.new_head_learning_rate is not None
        head_parameters = []
        if config.use_stage_auxiliary and model.stage_classifier is not None:
            head_parameters.extend(model.stage_classifier.parameters())
        if config.use_physics_auxiliary and model.physics_auxiliary is not None:
            head_parameters.extend(model.physics_auxiliary.parameters())
        if (
            config.use_conditional_verifier
            and model.verifier_classifier is not None
        ):
            head_parameters.extend(model.verifier_classifier.parameters())
        if config.use_event_verifier:
            assert model.event_projection is not None
            assert model.event_encoder is not None
            assert model.event_classifier is not None
            head_parameters.extend(model.event_projection.parameters())
            head_parameters.extend(model.event_encoder.parameters())
            head_parameters.extend(model.event_classifier.parameters())
        if config.use_stgcn_residual_joint:
            head_parameters.extend(model.streams[0].graph_residual.parameters())
        if not head_parameters:
            raise ValueError("分层学习率需要至少一个新辅助头")
        head_ids = {id(parameter) for parameter in head_parameters}
        backbone_parameters = [
            parameter for parameter in model.parameters() if id(parameter) not in head_ids
        ]
        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_parameters, "lr": config.backbone_learning_rate},
                {"params": head_parameters, "lr": config.new_head_learning_rate},
            ],
            weight_decay=config.weight_decay,
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, config.epochs - config.warmup_epochs)
    )
    if config.warmup_epochs:
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * 0.1
    architecture = {
        "name": "MultiStreamMultiScaleAttentionTCN",
        "streams": {
            "joint": 51,
            "bone": 32,
            "motion": 51 if config.motion_validity_mask else 34,
            "geometry_observed": 6,
            **({"rule_features": 12} if config.use_rule_features else {}),
            **({"pifr_features": 9} if config.use_pifr_features else {}),
            **(
                {"transition_rule_features": 8}
                if config.use_transition_rule_late_fusion
                else {}
            ),
            **(
                {"discriminative_kinematics": config.kinematic_feature_dim}
                if config.use_discriminative_kinematics
                else {}
            ),
        },
        "dilations": [1, 2, 4, 8],
        "kernels": [3, 5, 7],
        "attention_heads": 4,
        "stream_channels": config.stream_channels,
        "stream_output_dim": config.stream_output_dim,
        "stream_lstm_layers": config.stream_lstm_layers,
        "use_rule_features": config.use_rule_features,
        "use_pifr_features": config.use_pifr_features,
        "use_stgcn_joint": config.use_stgcn_joint,
        "use_stgcn_residual_joint": config.use_stgcn_residual_joint,
        "use_transformer_encoder": config.use_transformer_encoder,
        "transformer_joint_only": config.transformer_joint_only,
        "use_transition_rule_late_fusion": config.use_transition_rule_late_fusion,
        "use_discriminative_kinematics": config.use_discriminative_kinematics,
        "kinematic_feature_dim": config.kinematic_feature_dim,
        "use_gated_conv": config.use_gated_conv,
        "use_motion_guided_fusion": config.use_motion_guided_fusion,
        "motion_validity_mask": config.motion_validity_mask,
        "use_stream_interaction": config.use_stream_interaction,
        "use_stage_auxiliary": config.use_stage_auxiliary,
        "use_physics_auxiliary": config.use_physics_auxiliary,
        "use_conditional_verifier": config.use_conditional_verifier,
        "use_event_verifier": config.use_event_verifier,
        "training_augmentation": {
            "pose_jitter_std": config.pose_jitter_std,
            "joint_dropout_probability": config.joint_dropout_probability,
            "horizontal_flip_probability": config.horizontal_flip_probability,
            "temporal_augmentation_probability": config.temporal_augmentation_probability,
            "temporal_max_frame_fraction": config.temporal_max_frame_fraction,
            "joint_occlusion_probability": config.joint_occlusion_probability,
            "bbox_jitter_std": config.bbox_jitter_std,
            "track_break_probability": config.track_break_probability,
            "track_break_max_fraction": config.track_break_max_fraction,
            "temporal_speed_probability": config.temporal_speed_probability,
            "temporal_speed_min": config.temporal_speed_min,
            "temporal_speed_max": config.temporal_speed_max,
            "symmetry_consistency_weight": config.symmetry_consistency_weight,
            "real_distortion_mode": config.real_distortion_mode,
            "real_distortion_probability": config.real_distortion_probability,
            "real_distortion_bank_sha256": (
                _sha256_file(real_distortion_bank_path)
                if real_distortion_bank_path is not None
                else None
            ),
            "train_only": True,
        },
        "optimization": {
            "backbone_learning_rate": config.backbone_learning_rate,
            "new_head_learning_rate": config.new_head_learning_rate,
            "warmup_epochs": config.warmup_epochs,
            "scheduler": "warmup_then_cosine",
        },
        "mil": {
            "aggregate": config.clip_aggregator,
            "topk_fraction": config.mil_topk_fraction,
            "smooth_max_temperature": {
                "start": config.smooth_max_temperature_start,
                "end": config.smooth_max_temperature_end,
                "schedule": "geometric",
            },
            "loss_weight": config.mil_loss_weight,
            "clip_batch_size": config.mil_clip_batch_size,
            "transition_aux_weight": (
                config.transition_aux_weight
                if config.use_transition_rule_late_fusion
                else None
            ),
            "hard_negative_ranking": {
                "weight": config.hard_negative_ranking_weight,
                "margin": config.hard_negative_ranking_margin,
                "activities": ["lie_down", "lying", "stand_up"],
            },
            "stage_auxiliary": {
                "enabled": config.use_stage_auxiliary,
                "weight": config.stage_aux_weight if config.use_stage_auxiliary else None,
                "classes": list(STAGE_NAMES) if config.use_stage_auxiliary else None,
            },
            "conditional_verifier": {
                "enabled": config.use_conditional_verifier,
                "candidate_aux_weight": config.candidate_aux_weight,
                "verifier_aux_weight": config.verifier_aux_weight,
                "gate_max": config.verifier_gate_max,
                "gate_schedule": "linear_after_head_only_epoch",
                "candidate_anchor_weight": config.candidate_anchor_weight,
                "hard_background_fraction": config.verifier_background_fraction,
                "head_only_epochs": config.verifier_head_only_epochs,
                "classes": (
                    list(VERIFIER_NAMES)
                    if config.use_conditional_verifier
                    else None
                ),
            },
            "event_verifier": {
                "enabled": config.use_event_verifier,
                "hidden_dim": config.event_hidden_dim,
                "gate_max": config.event_gate_max,
                "head_only_epochs": config.event_head_only_epochs,
                "causal": True,
                "supervision": "clip_binary",
                "aux_weight": config.event_aux_weight,
                "suppression_only": config.event_suppression_only,
                "penalty_cap": config.event_penalty_cap,
                "positive_penalty_weight": config.event_positive_penalty_weight,
            },
            "auprc_surrogate": {
                "weight": config.auprc_surrogate_weight,
                "positive_fraction": config.high_recall_positive_fraction,
                "hard_negative_memory_size": config.hard_negative_memory_size,
            },
            "negative_spike_weight": config.negative_spike_weight,
        },
        "causal": True,
        "joint_pretrain": (
            {"protocol": "fallvision_joint_clip_mil_v1", "checkpoint_sha256": joint_pretrain_sha256}
            if joint_pretrain_sha256 is not None
            else None
        ),
        "initialization": (
            {
                "checkpoint_sha256": initialization_checkpoint_sha256,
                "compatible_state_keys": initialization_loaded_keys,
            }
            if initialization_checkpoint_sha256 is not None
            else None
        ),
    }
    root = Path(__file__).parent.parent
    signature_sha256, signature = _run_signature(
        config=config,
        train_cache=train_cache,
        val_cache=val_cache,
        device=device_name,
        max_train_samples=None,
        max_val_samples=None,
        protocol="multiscale_multistream_topk_mil_v1",
        code_paths=(
            root / "models" / "multiscale_multistream_tcn.py",
            root / "models" / "tcn_dataset.py",
            root / "tools" / "train_tcn.py",
            root / "tools" / "train_multiscale_multistream_tcn.py",
            root / "tools" / "train_multiscale_multistream_mil.py",
            *(
                (root / "models" / "fallvision_joint_mil.py",)
                if joint_pretrain_sha256 is not None
                else ()
            ),
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
        if config.warmup_epochs and epoch < config.warmup_epochs:
            warmup_factor = 0.1 + 0.9 * float(epoch) / float(config.warmup_epochs)
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * warmup_factor
        if (
            (
                config.verifier_head_only_epochs
                and epoch < config.verifier_head_only_epochs
            )
            or (
                config.event_head_only_epochs
                and epoch < config.event_head_only_epochs
            )
        ):
            if len(optimizer.param_groups) != 2:
                raise ValueError("verifier head-only warmup 需要分层学习率")
            optimizer.param_groups[0]["lr"] = 0.0
        train_loss, train_diagnostics = train_epoch_mil(
            model,
            optimizer,
            train_x,
            train_y,
            groups,
            clip_labels,
            transition_targets,
            clip_activities,
            stage_targets,
            verifier_targets,
            anchor_model,
            real_distortion_bank,
            device=device,
            config=config,
            epoch=epoch,
        )
        event_clip_scores = (
            predict_event_clip_scores(
                model,
                val_x,
                val_clip_indices,
                len(val_cache.metadata["clips"]),
                device=device,
                config=config,
                epoch=epoch,
            )
            if config.use_event_verifier
            else None
        )
        metrics = evaluate_model(
            model,
            val_x,
            val_y,
            val_clip_indices,
            val_cache.metadata["clips"],
            device=device,
            batch_size=config.batch_size,
            clip_aggregation=(
                "smooth_max"
                if config.clip_aggregator == "smooth_max"
                else "max"
                if config.clip_aggregator
                in {"legacy_topk_train_max_eval", "max"}
                else "topk_mean"
            ),
            aggregation_temperature=smooth_max_temperature(config, epoch),
            topk_fraction=config.mil_topk_fraction,
            clip_score_override=event_clip_scores,
        )
        if config.warmup_epochs and epoch + 1 == config.warmup_epochs:
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"]
        elif epoch >= config.warmup_epochs:
            scheduler.step()
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_diagnostics": train_diagnostics,
            "aggregation_temperature": smooth_max_temperature(config, epoch),
            "verifier_scale": (
                verifier_scale(config, epoch)
                if config.use_conditional_verifier
                else None
            ),
            "event_verifier_scale": (
                event_verifier_scale(config, epoch)
                if config.use_event_verifier
                else None
            ),
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
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all()
            if device.type == "cuda"
            else [],
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
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pose-cache-root", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--window-cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-sidecar", type=Path, required=True)
    parser.add_argument("--val-sidecar", type=Path, required=True)
    parser.add_argument("--dataset", default="of-syn")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--event-gate-max", type=float)
    parser.add_argument("--event-head-only-epochs", type=int)
    parser.add_argument("--joint-pretrain-checkpoint", type=Path)
    parser.add_argument("--initialization-checkpoint", type=Path)
    parser.add_argument("--real-distortion-bank", type=Path)
    args = parser.parse_args()
    config = MILConfig.from_json(args.config)
    if any(
        value is not None
        for value in (
            args.epochs,
            args.seed,
            args.event_gate_max,
            args.event_head_only_epochs,
        )
    ):
        config = replace(
            config,
            **({"epochs": args.epochs} if args.epochs is not None else {}),
            **({"seed": args.seed} if args.seed is not None else {}),
            **(
                {"event_gate_max": args.event_gate_max}
                if args.event_gate_max is not None
                else {}
            ),
            **(
                {"event_head_only_epochs": args.event_head_only_epochs}
                if args.event_head_only_epochs is not None
                else {}
            ),
        )
        config.validate()
    caches = {
        split: build_window_cache(
            args.manifest,
            args.pose_cache_root,
            args.audit_report,
            args.window_cache_root,
            dataset=args.dataset,
            split=split,
            window_size=config.window_size,
            stride=config.stride,
            min_observed_frames=config.min_observed_frames,
        )
        for split in ("train", "val")
    }
    summary = train_mil(
        config=config,
        train_cache=caches["train"],
        val_cache=caches["val"],
        train_sidecar=load_sidecar(args.train_sidecar, caches["train"]),
        val_sidecar=load_sidecar(args.val_sidecar, caches["val"]),
        output_dir=args.output_dir,
        device_name=args.device,
        joint_pretrain_checkpoint=args.joint_pretrain_checkpoint,
        initialization_checkpoint=args.initialization_checkpoint,
        real_distortion_bank_path=args.real_distortion_bank,
    )
    print(_canonical_json({"stage": "complete", **summary}), flush=True)


if __name__ == "__main__":
    main()
