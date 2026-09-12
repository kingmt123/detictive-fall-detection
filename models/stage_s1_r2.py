"""Anchor-preserving Stage-S1-R2 wrappers.

R1 intentionally owns only clip evidence pooling; context and cross-stream
residuals remain separate later-stage experiments.
"""

from __future__ import annotations

import torch
from torch import nn

from models.evidence_pooling import EvidencePoolingResidual
from models.multiscale_multistream_tcn import MultiStreamMultiScaleAttentionTCN


class StageS1R2R1(nn.Module):
    """Frozen Stage-S1 window model plus a trainable clip-level residual."""

    def __init__(self, base: MultiStreamMultiScaleAttentionTCN, pool: EvidencePoolingResidual) -> None:
        super().__init__()
        self.base = base
        self.pool = pool
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> StageS1R2R1:
        super().train(mode)
        self.base.eval()
        return self

    @torch.inference_mode()
    def window_outputs(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.base.stage_classifier is None:
            return self.base(x), None
        return self.base.forward_with_stage(x)

    def forward(
        self,
        window_logits: torch.Tensor,
        group_sizes: list[int],
        *,
        stage_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.pool(window_logits, group_sizes, stage_logits=stage_logits)


class StageS1R2Context(nn.Module):
    """M2 shared-encoder 16+48-frame causal context residual.

    ``context_x`` is a sparsely sampled 16-token representation of the last
    48 frames.  The base encoder is deliberately shared and frozen for this
    canary; its local path remains the exact Stage-S1 anchor at initialization.
    """

    def __init__(
        self,
        base: MultiStreamMultiScaleAttentionTCN,
        *,
        bottleneck_dim: int = 16,
        correction_cap: float = 0.5,
    ) -> None:
        super().__init__()
        if bottleneck_dim < 1 or correction_cap <= 0.0:
            raise ValueError("context adapter 参数无效")
        self.base = base
        self.correction_cap = float(correction_cap)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        fusion_dim = int(self.base.classifier.in_features)
        self.adapter = nn.Sequential(
            nn.LayerNorm(fusion_dim * 3),
            nn.Linear(fusion_dim * 3, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, 1),
        )
        final = self.adapter[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def train(self, mode: bool = True) -> StageS1R2Context:
        super().train(mode)
        self.base.eval()
        return self

    def forward_with_stage(
        self, local_x: torch.Tensor, context_x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if local_x.shape != context_x.shape:
            raise ValueError("local/context 输入形状必须相同")
        with torch.no_grad():
            local_embedding = self.base.encode(local_x)
            context_embedding = self.base.encode(context_x)
            class_logits = self.base.classifier(local_embedding)
            base_logit = class_logits[:, 1] - class_logits[:, 0]
            stage_logits = (
                self.base.stage_classifier(local_embedding)
                if self.base.stage_classifier is not None
                else None
            )
        features = torch.cat(
            (local_embedding, context_embedding, local_embedding - context_embedding),
            dim=-1,
        )
        delta = self.correction_cap * torch.tanh(self.adapter(features).squeeze(-1))
        return base_logit + delta, stage_logits


def build_sparse_context_windows(
    local_windows: torch.Tensor, group_sizes: list[int]
) -> torch.Tensor:
    """Derive causal 48-frame/3-stride context from contiguous local windows.

    Each local window ends at its sample time.  A clip's first local window
    establishes its first 16 frames and every following window contributes one
    new final frame.  History before that first frame is explicit all-zero
    padding, never a copied first observation.
    """
    if local_windows.ndim != 3 or not group_sizes or sum(group_sizes) != local_windows.shape[0]:
        raise ValueError("local_windows 与 group_sizes 不匹配")
    if local_windows.shape[1] != 16:
        raise ValueError("M2 需要 16 帧 local window")
    context = torch.zeros_like(local_windows)
    offset = 0
    for size in group_sizes:
        if size < 1:
            raise ValueError("每个 clip 至少需要一个窗口")
        local = local_windows[offset : offset + size]
        frames = torch.cat((local[0], local[1:, -1]), dim=0)
        for index in range(size):
            end = index + 15
            positions = end - 45 + torch.arange(0, 48, 3, device=local.device)
            valid = positions >= 0
            if valid.any():
                context[offset + index, valid] = frames[positions[valid]]
        offset += size
    return context


class StageS1R2M2(nn.Module):
    """Frozen shared-encoder local/context residual adapter for the M2 canary."""

    def __init__(
        self,
        base: MultiStreamMultiScaleAttentionTCN,
        *,
        hidden_dim: int = 32,
        correction_cap: float = 0.5,
    ) -> None:
        super().__init__()
        if hidden_dim < 1 or correction_cap <= 0.0:
            raise ValueError("M2 adapter 参数无效")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        fused_dim = int(base.classifier.in_features)
        self.correction_cap = float(correction_cap)
        self.adapter = nn.Sequential(
            nn.LayerNorm(fused_dim * 3),
            nn.Linear(fused_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        final = self.adapter[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def train(self, mode: bool = True) -> StageS1R2M2:
        super().train(mode)
        self.base.eval()
        return self

    def forward(self, local_x: torch.Tensor, context_x: torch.Tensor) -> torch.Tensor:
        if local_x.shape != context_x.shape:
            raise ValueError("local/context 输入形状必须相同")
        with torch.no_grad():
            local_embedding = self.base.encode(local_x)
            context_embedding = self.base.encode(context_x)
            base_logits = self.base.classifier(local_embedding)
            base_logit = base_logits[:, 1] - base_logits[:, 0]
        adapter_input = torch.cat(
            (local_embedding, context_embedding, local_embedding - context_embedding), dim=-1
        )
        correction = self.correction_cap * torch.tanh(self.adapter(adapter_input).squeeze(-1))
        return base_logit + correction
