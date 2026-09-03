"""Controlled TCN+SA-TDGFormer and TCN+InfoGCN ablation models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from models.multiscale_multistream_tcn import MultiStreamMultiScaleAttentionTCN
from models.tri_expert_skeleton import (
    FourModalInfoGCN,
    SATDGFormerEncoder,
    build_infogcn_modalities,
)

DualVariant = Literal["sa_only", "tcn_sa", "tcn_info"]


def orthogonal_dual_loss(embeddings: torch.Tensor) -> torch.Tensor:
    """Squared cosine penalty between two expert embeddings, with no parameters."""
    if embeddings.ndim != 3 or embeddings.shape[1] != 2:
        raise ValueError("双专家正交损失输入必须是 (B,2,D)")
    normalized = F.normalize(embeddings.float(), dim=-1, eps=1e-6)
    return (normalized[:, 0] * normalized[:, 1]).sum(dim=-1).square().mean()


@dataclass(frozen=True)
class DualExpertOutput:
    logits: torch.Tensor
    embeddings: torch.Tensor
    attention_weights: torch.Tensor
    information_loss: torch.Tensor


class DualExpertSkeletonFallDetector(nn.Module):
    """A two-expert model used to isolate SA-TDGFormer and InfoGCN gains."""

    def __init__(
        self,
        *,
        variant: DualVariant,
        channels: int = 32,
        expert_dim: int = 64,
        dropout: float = 0.35,
        dynamic_layers: int = 2,
        knn_k: int = 4,
        heads: int = 4,
    ) -> None:
        super().__init__()
        if variant not in {"sa_only", "tcn_sa", "tcn_info"}:
            raise ValueError("variant 必须为 sa_only、tcn_sa 或 tcn_info")
        if channels < 4 or channels % heads or expert_dim % heads:
            raise ValueError("channels/expert_dim 必须能被 attention heads 整除")
        self.variant: DualVariant = variant
        self.tcn: nn.Module | None = None
        if variant != "sa_only":
            self.tcn = MultiStreamMultiScaleAttentionTCN(
                stream_channels=channels,
                stream_output_dim=expert_dim,
                dropout=dropout,
                use_geometry=True,
                use_rule_features=True,
                use_discriminative_kinematics=True,
                kinematic_feature_dim=12,
                use_motion_guided_fusion=True,
            )
        self.secondary: nn.Module
        if variant in {"sa_only", "tcn_sa"}:
            self.secondary = SATDGFormerEncoder(channels, expert_dim, dropout, heads)
        else:
            self.secondary = FourModalInfoGCN(
                channels,
                expert_dim,
                dropout,
                dynamic_layers=dynamic_layers,
                knn_k=knn_k,
                heads=heads,
            )
        self.expert_attention = nn.MultiheadAttention(
            expert_dim, heads, dropout=dropout, batch_first=True
        )
        self.fusion_norm = nn.LayerNorm(expert_dim)
        self.classifier = nn.Linear(expert_dim, 2)

    @property
    def expert_names(self) -> tuple[str, ...]:
        if self.variant == "sa_only":
            return ("sa_tdgformer",)
        return (
            ("tcn", "sa_tdgformer")
            if self.variant == "tcn_sa"
            else ("tcn", "infogcn")
        )

    def forward_detailed(self, x: torch.Tensor) -> DualExpertOutput:
        modalities = build_infogcn_modalities(x)
        if self.variant == "sa_only":
            secondary_embedding = self.secondary(modalities[0])
            logits = self.classifier(secondary_embedding)
            return DualExpertOutput(
                logits=logits[:, 1] - logits[:, 0],
                embeddings=secondary_embedding.unsqueeze(1),
                attention_weights=torch.ones(
                    (x.shape[0], 1), device=x.device, dtype=secondary_embedding.dtype
                ),
                information_loss=secondary_embedding.new_zeros((), dtype=torch.float32),
            )
        assert self.tcn is not None
        tcn_embedding = self.tcn.encode(x)
        if self.variant == "tcn_sa":
            secondary_embedding = self.secondary(modalities[0])
            information_loss = tcn_embedding.new_zeros((), dtype=torch.float32)
        else:
            secondary_embedding, information_loss = self.secondary(modalities)
        embeddings = torch.stack((tcn_embedding, secondary_embedding), dim=1)
        query = embeddings.mean(dim=1, keepdim=True)
        fused, weights = self.expert_attention(
            query, embeddings, embeddings, need_weights=True
        )
        class_logits = self.classifier(self.fusion_norm(query + fused).squeeze(1))
        return DualExpertOutput(
            logits=class_logits[:, 1] - class_logits[:, 0],
            embeddings=embeddings,
            attention_weights=weights.squeeze(1),
            information_loss=information_loss,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_detailed(x).logits
