"""End-to-end full-clip VideoMAE teacher for the backbone-replacement plan."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

try:
    from transformers import VideoMAEConfig, VideoMAEModel
except ImportError as exc:  # pragma: no cover - exercised only without optional dep.
    raise ImportError(
        "VideoMAE teacher 需要 transformers；请安装 requirements.txt 中的依赖"
    ) from exc


SUBTYPE_NAMES = (
    "fall",
    "lie_down",
    "lying",
    "stand_up",
    "sit_or_bend",
    "other_background",
)


@dataclass(frozen=True)
class VideoMaeTeacherOutput:
    """Clip ranking and auxiliary-activity outputs of the RGB teacher."""

    binary_logits: torch.Tensor
    subtype_logits: torch.Tensor
    token_embeddings: torch.Tensor


class VideoMaeClipTeacher(nn.Module):
    """Full-clip pretrained VideoMAE plus attention pooling and auxiliary ADL head.

    This model is intentionally a stand-alone RGB teacher: it accepts the
    complete clip as ``(B,T,C,H,W)``, contains no pose-track inputs, no hand
    engineered streams, and no max-window aggregation.
    """

    def __init__(
        self,
        backbone: VideoMAEModel,
        *,
        subtype_count: int = len(SUBTYPE_NAMES),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if subtype_count < 2 or not 0.0 <= dropout < 1.0:
            raise ValueError("VideoMAE teacher head 参数无效")
        self.backbone = backbone
        hidden_size = int(backbone.config.hidden_size)
        self.attention = nn.Linear(hidden_size, 1)
        self.head_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.binary = nn.Linear(hidden_size, 1)
        self.subtype = nn.Linear(hidden_size, subtype_count)

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        frames: int,
        image_size: int,
        cache_dir: str | None = None,
    ) -> VideoMaeClipTeacher:
        """Load official VideoMAE weights and adapt its positional grid to a clip."""
        if frames < 4 or image_size < 32:
            raise ValueError("frames/image_size 无效")
        config = VideoMAEConfig.from_pretrained(model_id, cache_dir=cache_dir)
        config.num_frames = int(frames)
        config.image_size = int(image_size)
        backbone = VideoMAEModel.from_pretrained(
            model_id,
            config=config,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=True,
        )
        return cls(backbone)

    def forward(self, pixel_values: torch.Tensor) -> VideoMaeTeacherOutput:
        if pixel_values.ndim != 5 or pixel_values.shape[2] != 3:
            raise ValueError("pixel_values 必须为 (B,T,3,H,W)")
        tokens = self.backbone(pixel_values=pixel_values).last_hidden_state
        weights = torch.softmax(self.attention(tokens).squeeze(-1), dim=1)
        pooled = torch.einsum("bt,bth->bh", weights, tokens)
        value = self.dropout(self.head_norm(pooled))
        return VideoMaeTeacherOutput(
            binary_logits=self.binary(value).squeeze(1),
            subtype_logits=self.subtype(value),
            token_embeddings=tokens,
        )

    def set_trainable_backbone_layers(self, last_layers: int | None) -> None:
        """Warm up heads and late blocks, or unfreeze the full video backbone.

        ``last_layers=None`` is the full fine-tuning phase.  A finite value
        freezes patch embedding and early encoder blocks, preserving Kinetics
        features while the new fall-ranking heads become calibrated.
        """
        layers = self.backbone.encoder.layer
        if last_layers is not None and not 0 <= last_layers <= len(layers):
            raise ValueError("last_layers 超出 VideoMAE encoder 范围")
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(last_layers is None)
        if last_layers is not None:
            for layer in layers[-last_layers:] if last_layers else ():
                for parameter in layer.parameters():
                    parameter.requires_grad_(True)
        for module in (self.attention, self.head_norm, self.binary, self.subtype):
            for parameter in module.parameters():
                parameter.requires_grad_(True)


def asymmetric_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    gamma_positive: float = 1.0,
    gamma_negative: float = 4.0,
    positive_weight: float = 1.0,
) -> torch.Tensor:
    """Binary focal loss that concentrates the teacher on difficult ADLs."""
    if logits.shape != targets.shape or logits.ndim != 1:
        raise ValueError("binary focal logits/targets 必须是同形一维张量")
    if gamma_positive < 0.0 or gamma_negative < 0.0 or positive_weight <= 0.0:
        raise ValueError("focal 参数无效")
    probabilities = torch.sigmoid(logits)
    positive = targets > 0.5
    true_probability = torch.where(positive, probabilities, 1.0 - probabilities)
    gamma = torch.where(
        positive,
        torch.full_like(true_probability, gamma_positive),
        torch.full_like(true_probability, gamma_negative),
    )
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    class_weight = torch.where(
        positive, torch.full_like(true_probability, positive_weight), torch.ones_like(true_probability)
    )
    return (class_weight * (1.0 - true_probability).pow(gamma) * bce).mean()


def teacher_loss(
    output: VideoMaeTeacherOutput,
    labels: torch.Tensor,
    subtypes: torch.Tensor,
    *,
    subtype_weight: float = 0.2,
    positive_weight: float = 1.0,
) -> torch.Tensor:
    """Report-prescribed focal objective plus activity-boundary auxiliary task."""
    if labels.shape != output.binary_logits.shape or subtypes.shape != labels.shape:
        raise ValueError("teacher labels 与 logits 不兼容")
    if subtype_weight < 0.0:
        raise ValueError("subtype_weight 必须非负")
    return asymmetric_focal_loss(
        output.binary_logits, labels, positive_weight=positive_weight
    ) + subtype_weight * (
        nn.functional.cross_entropy(output.subtype_logits, subtypes)
    )
