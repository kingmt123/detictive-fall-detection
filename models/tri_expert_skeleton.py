"""Causal TCN + SA-TDGFormer + four-modal InfoGCN fall detector.

The skeleton encoders are compact 2-D COCO adaptations.  They preserve the
published architectural ideas while keeping the experiment suitable for a
16-frame, single-person, edge-oriented fall detector.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from models.multiscale_multistream_tcn import (
    JOINT_DIM,
    MultiStreamMultiScaleAttentionTCN,
    _coco_adjacency,
)
from models.tcn_multistream import COCO_BONE_EDGES

NUM_JOINTS = 17
NUM_MODALITIES = 4
NUM_EXPERTS = 3


def _validate_input(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 3 or x.shape[-1] < JOINT_DIM:
        raise ValueError("三专家模型输入必须是 (B,T,>=51)")
    if not torch.is_floating_point(x):
        raise TypeError("三专家模型输入必须是浮点张量")
    return x[..., :JOINT_DIM].reshape(*x.shape[:2], NUM_JOINTS, 3)


def build_infogcn_modalities(x: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Build joint, bone, joint-motion and bone-motion tensors (B,3,T,17)."""
    pose = _validate_input(x)
    xy = pose[..., :2]
    visible = pose[..., 2] > 0

    joint = pose
    bone = torch.zeros_like(pose)
    for parent, child in COCO_BONE_EDGES:
        valid = visible[..., parent] & visible[..., child]
        bone[..., child, :2] = (xy[..., child, :] - xy[..., parent, :]) * valid[
            ..., None
        ]
        bone[..., child, 2] = torch.minimum(
            pose[..., parent, 2], pose[..., child, 2]
        ) * valid

    joint_motion = torch.zeros_like(pose)
    bone_motion = torch.zeros_like(pose)
    if pose.shape[1] > 1:
        joint_valid = visible[:, 1:] & visible[:, :-1]
        joint_motion[:, 1:, :, :2] = (xy[:, 1:] - xy[:, :-1]) * joint_valid[
            ..., None
        ]
        joint_motion[:, 1:, :, 2] = joint_valid.to(pose.dtype)

        bone_visible = bone[..., 2] > 0
        bone_valid = bone_visible[:, 1:] & bone_visible[:, :-1]
        bone_motion[:, 1:, :, :2] = (
            bone[:, 1:, :, :2] - bone[:, :-1, :, :2]
        ) * bone_valid[..., None]
        bone_motion[:, 1:, :, 2] = bone_valid.to(pose.dtype)

    return tuple(
        value.permute(0, 3, 1, 2).contiguous()
        for value in (joint, bone, joint_motion, bone_motion)
    )


class CausalTemporalConv(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.padding = 2 * dilation
        self.conv = nn.Conv2d(
            channels, channels, (3, 1), dilation=(dilation, 1)
        )
        self.norm = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        temporal = self.conv(F.pad(x, (0, 0, self.padding, 0)))
        return x + self.dropout(F.gelu(self.norm(temporal)))


class AdaptiveGraphTemporalBlock(nn.Module):
    """AGCN-style learned graph residual followed by causal TDCN."""

    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.register_buffer("fixed_adjacency", _coco_adjacency())
        self.adjacency_residual = nn.Parameter(torch.zeros(NUM_JOINTS, NUM_JOINTS))
        self.spatial = nn.Conv2d(channels, channels, 1)
        self.temporal = CausalTemporalConv(channels, dilation, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        adjacency = self.fixed_adjacency + torch.tanh(self.adjacency_residual)
        graph = torch.einsum("bctv,vw->bctw", x, adjacency)
        return self.temporal(x + self.spatial(graph))


class SATDGFormerEncoder(nn.Module):
    """Compact parallel AGCN-TDCN and spatial/temporal Transformer encoder."""

    def __init__(
        self, channels: int, output_dim: int, dropout: float, heads: int = 4
    ) -> None:
        super().__init__()
        self.graph_input = nn.Conv2d(3, channels, 1)
        self.graph_blocks = nn.ModuleList(
            AdaptiveGraphTemporalBlock(channels, dilation, dropout)
            for dilation in (1, 2, 4, 8)
        )
        self.token_input = nn.Linear(3, channels)
        spatial_layer = nn.TransformerEncoderLayer(
            channels,
            heads,
            channels * 2,
            dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        temporal_layer = nn.TransformerEncoderLayer(
            channels,
            heads,
            channels * 2,
            dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.spatial_transformer = nn.TransformerEncoder(spatial_layer, 1)
        self.temporal_transformer = nn.TransformerEncoder(temporal_layer, 1)
        self.branch_attention = nn.MultiheadAttention(
            channels, heads, dropout=dropout, batch_first=True
        )
        self.branch_norm = nn.LayerNorm(channels)
        self.output = nn.Linear(channels, output_dim)

    def forward(self, joint: torch.Tensor) -> torch.Tensor:
        if joint.ndim != 4 or joint.shape[1] != 3 or joint.shape[-1] != NUM_JOINTS:
            raise ValueError("SA-TDGFormer 输入必须是 (B,3,T,17)")
        graph = self.graph_input(joint)
        for block in self.graph_blocks:
            graph = block(graph)
        graph_embedding = graph[:, :, -1].mean(dim=-1)

        tokens = self.token_input(joint.permute(0, 2, 3, 1))
        batch, length, joints, channels = tokens.shape
        spatial = self.spatial_transformer(tokens.reshape(batch * length, joints, channels))
        spatial = spatial.reshape(batch, length, joints, channels)
        temporal = spatial.permute(0, 2, 1, 3).reshape(batch * joints, length, channels)
        causal_mask = torch.ones(
            length, length, device=joint.device, dtype=torch.bool
        ).triu(1)
        temporal = self.temporal_transformer(temporal, mask=causal_mask)
        transformer_embedding = temporal[:, -1].reshape(batch, joints, channels).mean(1)

        branches = torch.stack((graph_embedding, transformer_embedding), dim=1)
        query = branches.mean(dim=1, keepdim=True)
        fused, _ = self.branch_attention(query, branches, branches, need_weights=False)
        return self.output(self.branch_norm(query + fused).squeeze(1))


class InfoGraphBlock(nn.Module):
    """Attention graph block with optional late-layer sample-specific k-NN."""

    def __init__(
        self,
        channels: int,
        dilation: int,
        dropout: float,
        *,
        dynamic_knn: bool,
        knn_k: int,
    ) -> None:
        super().__init__()
        self.dynamic_knn = dynamic_knn
        self.knn_k = knn_k
        self.register_buffer("fixed_adjacency", _coco_adjacency())
        attention_dim = max(4, channels // 4)
        self.query = (
            nn.Conv2d(channels, attention_dim, 1) if dynamic_knn else None
        )
        self.key = nn.Conv2d(channels, attention_dim, 1) if dynamic_knn else None
        self.spatial = nn.Conv2d(channels, channels, 1)
        self.temporal = CausalTemporalConv(channels, dilation, dropout)

    def _adjacency(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, length, joints = x.shape
        fixed = self.fixed_adjacency.to(x.dtype).expand(
            batch, length, joints, joints
        )
        if not self.dynamic_knn:
            return fixed
        if self.query is None or self.key is None:
            raise RuntimeError("动态 k-NN 层缺少 query/key")
        # k-NN selection is particularly prone to fp16 affinity overflow.
        # Keep topology construction in fp32 even when the rest uses AMP.
        query = self.query(x).permute(0, 2, 3, 1).float()
        key = self.key(x).permute(0, 2, 3, 1).float()
        affinity = torch.einsum("btvc,btwc->btvw", query, key)
        affinity = affinity / query.shape[-1] ** 0.5
        topk = affinity.topk(min(self.knn_k, joints), dim=-1).indices
        mask = torch.zeros_like(affinity, dtype=torch.bool).scatter(-1, topk, True)
        learned = affinity.masked_fill(~mask, -torch.inf).softmax(dim=-1)
        return fixed + learned.to(x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        adjacency = self._adjacency(x)
        graph = torch.einsum("bctv,btvw->bctw", x, adjacency)
        return self.temporal(x + self.spatial(graph))


class InfoGCNModalEncoder(nn.Module):
    """Four-layer encoder: fixed topology below, dynamic k-NN in last layers."""

    def __init__(
        self,
        channels: int,
        output_dim: int,
        dropout: float,
        *,
        dynamic_layers: int = 2,
        knn_k: int = 4,
    ) -> None:
        super().__init__()
        if dynamic_layers not in {2, 3}:
            raise ValueError("InfoGCN dynamic_layers 必须为 2 或 3")
        self.dynamic_layers = dynamic_layers
        self.input = nn.Conv2d(3, channels, 1)
        self.blocks = nn.ModuleList(
            InfoGraphBlock(
                channels,
                dilation,
                dropout,
                dynamic_knn=index >= 4 - dynamic_layers,
                knn_k=knn_k,
            )
            for index, dilation in enumerate((1, 2, 4, 8))
        )
        self.mu = nn.Linear(channels, output_dim)
        self.log_variance = nn.Linear(channels, output_dim)

    def forward(self, modality: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.input(modality)
        for block in self.blocks:
            hidden = block(hidden)
        pooled = hidden[:, :, -1].mean(dim=-1)
        mu = self.mu(pooled)
        log_variance = self.log_variance(pooled).clamp(-6.0, 3.0)
        if self.training:
            embedding = mu + torch.randn_like(mu) * torch.exp(0.5 * log_variance)
        else:
            embedding = mu
        mu_float = mu.float()
        log_variance_float = log_variance.float()
        kl = 0.5 * (
            mu_float.square()
            + log_variance_float.exp()
            - log_variance_float
            - 1.0
        ).mean()
        return embedding, kl


class FourModalInfoGCN(nn.Module):
    def __init__(
        self,
        channels: int,
        output_dim: int,
        dropout: float,
        *,
        dynamic_layers: int = 2,
        knn_k: int = 4,
        heads: int = 4,
    ) -> None:
        super().__init__()
        self.encoders = nn.ModuleList(
            InfoGCNModalEncoder(
                channels,
                output_dim,
                dropout,
                dynamic_layers=dynamic_layers,
                knn_k=knn_k,
            )
            for _ in range(NUM_MODALITIES)
        )
        self.modality_attention = nn.MultiheadAttention(
            output_dim, heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(output_dim)

    def forward(
        self, modalities: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(modalities) != NUM_MODALITIES:
            raise ValueError("InfoGCN 必须接收四种骨架模态")
        encoded = [
            encoder(modality)
            for encoder, modality in zip(self.encoders, modalities, strict=True)
        ]
        embeddings = torch.stack([item[0] for item in encoded], dim=1)
        query = embeddings.mean(dim=1, keepdim=True)
        fused, _ = self.modality_attention(query, embeddings, embeddings)
        kl = torch.stack([item[1] for item in encoded]).mean()
        return self.norm(query + fused).squeeze(1), kl


def orthogonal_expert_loss(embeddings: torch.Tensor) -> torch.Tensor:
    """Squared pairwise cosine penalty; this loss has no trainable parameters."""
    if embeddings.ndim != 3 or embeddings.shape[1] != NUM_EXPERTS:
        raise ValueError("正交损失输入必须是 (B,3,D)")
    normalized = F.normalize(embeddings.float(), dim=-1, eps=1e-6)
    similarities = torch.einsum("bed,bfd->bef", normalized, normalized)
    upper = torch.triu(
        torch.ones(NUM_EXPERTS, NUM_EXPERTS, device=embeddings.device, dtype=torch.bool),
        diagonal=1,
    )
    return similarities[:, upper].square().mean()


@dataclass(frozen=True)
class TriExpertOutput:
    logits: torch.Tensor
    embeddings: torch.Tensor
    attention_weights: torch.Tensor
    information_loss: torch.Tensor


class TriExpertSkeletonFallDetector(nn.Module):
    """Three complementary causal experts with sample-specific attention fusion."""

    def __init__(
        self,
        *,
        channels: int = 32,
        expert_dim: int = 64,
        dropout: float = 0.35,
        dynamic_layers: int = 2,
        knn_k: int = 4,
        heads: int = 4,
    ) -> None:
        super().__init__()
        if channels < 4 or channels % heads or expert_dim % heads:
            raise ValueError("channels/expert_dim 必须能被 attention heads 整除")
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
        self.sa_tdgformer = SATDGFormerEncoder(channels, expert_dim, dropout, heads)
        self.infogcn = FourModalInfoGCN(
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

    def forward_detailed(self, x: torch.Tensor) -> TriExpertOutput:
        modalities = build_infogcn_modalities(x)
        tcn_embedding = self.tcn.encode(x)
        sa_embedding = self.sa_tdgformer(modalities[0])
        info_embedding, information_loss = self.infogcn(modalities)
        embeddings = torch.stack(
            (tcn_embedding, sa_embedding, info_embedding), dim=1
        )
        query = embeddings.mean(dim=1, keepdim=True)
        fused, weights = self.expert_attention(
            query, embeddings, embeddings, need_weights=True
        )
        class_logits = self.classifier(self.fusion_norm(query + fused).squeeze(1))
        return TriExpertOutput(
            logits=class_logits[:, 1] - class_logits[:, 0],
            embeddings=embeddings,
            attention_weights=weights.squeeze(1),
            information_loss=information_loss,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_detailed(x).logits
