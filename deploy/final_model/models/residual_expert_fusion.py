"""Diversity-preserving residual fusion for causal fall classification."""

from __future__ import annotations

import torch
from torch import nn

from models.multiscale_multistream_tcn import (
    JOINT_DIM,
    MultiStreamMultiScaleAttentionTCN,
    STGCNJointEncoder,
)


class CausalTransformerExpert(nn.Module):
    """An independent global-temporal expert without a duplicated TCN path."""

    def __init__(
        self,
        channels: int,
        output_dim: int,
        dropout: float,
        *,
        layers: int = 2,
        max_frames: int = 32,
    ) -> None:
        super().__init__()
        if channels < 4 or channels % 4 or output_dim < 1:
            raise ValueError("Transformer channels/output_dim 无效")
        if layers < 1 or max_frames < 1 or not 0.0 <= dropout < 1.0:
            raise ValueError("Transformer layers/max_frames/dropout 无效")
        self.max_frames = max_frames
        self.input = nn.Linear(JOINT_DIM, channels)
        self.position = nn.Parameter(torch.zeros(1, max_frames, channels))
        layer = nn.TransformerEncoderLayer(
            channels,
            4,
            channels * 2,
            dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(channels)
        self.output = nn.Linear(channels, output_dim)
        nn.init.normal_(self.position, std=0.02)

    def forward(self, joint: torch.Tensor) -> torch.Tensor:
        if joint.ndim != 3 or joint.shape[-1] != JOINT_DIM:
            raise ValueError("Transformer expert 输入必须是 (B,T,51)")
        length = joint.shape[1]
        if length > self.max_frames:
            raise ValueError("输入帧数超过 Transformer positional budget")
        tokens = self.input(joint) + self.position[:, :length]
        causal_mask = torch.ones(
            length, length, dtype=torch.bool, device=joint.device
        ).triu(diagonal=1)
        encoded = self.encoder(tokens, mask=causal_mask)
        return self.output(self.norm(encoded[:, -1]))


class ResidualExpertFusion(nn.Module):
    """Frozen TCN base plus independent graph and Transformer delta experts."""

    def __init__(
        self,
        base: MultiStreamMultiScaleAttentionTCN,
        *,
        expert_channels: int = 32,
        expert_output_dim: int = 64,
        transformer_layers: int = 2,
        dropout: float = 0.35,
        correction_scale: float = 2.0,
        max_frames: int = 32,
        use_global_gate: bool = False,
        global_gate_initial_logit: float = -2.0,
    ) -> None:
        super().__init__()
        if correction_scale <= 0.0:
            raise ValueError("correction_scale 必须为正")
        self.base = base
        self.use_geometry = bool(base.use_geometry)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.graph_expert = STGCNJointEncoder(
            expert_channels, expert_output_dim, dropout
        )
        self.transformer_expert = CausalTransformerExpert(
            expert_channels,
            expert_output_dim,
            dropout,
            layers=transformer_layers,
            max_frames=max_frames,
        )
        self.graph_auxiliary = nn.Linear(expert_output_dim, 1)
        self.transformer_auxiliary = nn.Linear(expert_output_dim, 1)
        self.graph_delta = nn.Linear(expert_output_dim, 1)
        self.transformer_delta = nn.Linear(expert_output_dim, 1)
        self.fusion_logits = nn.Parameter(torch.zeros(2))
        self.correction_scale = float(correction_scale)
        self.global_gate_logit = (
            nn.Parameter(torch.tensor(float(global_gate_initial_logit)))
            if use_global_gate
            else None
        )
        nn.init.zeros_(self.graph_delta.weight)
        nn.init.zeros_(self.graph_delta.bias)
        nn.init.zeros_(self.transformer_delta.weight)
        nn.init.zeros_(self.transformer_delta.bias)

    def train(self, mode: bool = True) -> ResidualExpertFusion:
        super().train(mode)
        self.base.eval()
        return self

    def expert_weights(self) -> torch.Tensor:
        return torch.softmax(self.fusion_logits, dim=0)

    def global_gate(self) -> torch.Tensor:
        if self.global_gate_logit is None:
            return self.fusion_logits.new_ones(())
        return torch.sigmoid(self.global_gate_logit)

    def forward_with_experts(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 3 or x.shape[-1] < JOINT_DIM:
            raise ValueError("Residual fusion 输入必须是 (B,T,>=51)")
        with torch.no_grad():
            base_logit = self.base(x)
        joint = x[..., :JOINT_DIM]
        graph_embedding = self.graph_expert(joint)
        transformer_embedding = self.transformer_expert(joint)
        auxiliary_logits = torch.stack(
            (
                self.graph_auxiliary(graph_embedding).squeeze(-1),
                self.transformer_auxiliary(transformer_embedding).squeeze(-1),
            ),
            dim=-1,
        )
        corrections = self.correction_scale * torch.tanh(
            torch.stack(
                (
                    self.graph_delta(graph_embedding).squeeze(-1),
                    self.transformer_delta(transformer_embedding).squeeze(-1),
                ),
                dim=-1,
            )
        )
        fused = base_logit + self.global_gate() * (
            corrections * self.expert_weights()
        ).sum(dim=-1)
        return (
            fused,
            auxiliary_logits,
            corrections,
            graph_embedding,
            transformer_embedding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_experts(x)[0]


def orthogonal_expert_loss(
    graph_embedding: torch.Tensor, transformer_embedding: torch.Tensor
) -> torch.Tensor:
    """Penalize squared cosine similarity without adding inference parameters."""
    if graph_embedding.shape != transformer_embedding.shape or graph_embedding.ndim != 2:
        raise ValueError("专家 embedding 必须为同形二维张量")
    graph = nn.functional.normalize(graph_embedding, dim=-1)
    transformer = nn.functional.normalize(transformer_embedding, dim=-1)
    return (graph.mul(transformer).sum(dim=-1).square()).mean()
