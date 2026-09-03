"""Bounded transition re-ranking on top of a frozen multistream fall model."""

from __future__ import annotations

import torch
from torch import nn

from models.multiscale_multistream_tcn import MultiStreamMultiScaleAttentionTCN
from models.transition_gru import TRANSITION_FEATURE_DIM, build_transition_features


class TransitionResidualReranker(nn.Module):
    """Correct only plausible candidates while preserving the frozen backbone."""

    def __init__(
        self,
        base: MultiStreamMultiScaleAttentionTCN,
        *,
        hidden_dim: int = 64,
        dropout: float = 0.25,
        correction_scale: float = 2.0,
        gate_center: float = 0.0,
        gate_temperature: float = 1.0,
        auxiliary_classes: int = 5,
    ) -> None:
        super().__init__()
        if hidden_dim < 8 or auxiliary_classes < 2:
            raise ValueError("hidden_dim/auxiliary_classes 无效")
        if correction_scale <= 0.0 or gate_temperature <= 0.0:
            raise ValueError("correction_scale/gate_temperature 必须为正")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        fusion_dim = int(base.classifier.in_features)
        self.correction_scale = float(correction_scale)
        self.gate_center = float(gate_center)
        self.gate_temperature = float(gate_temperature)
        self.branch = nn.Sequential(
            nn.LayerNorm(fusion_dim + TRANSITION_FEATURE_DIM + 1),
            nn.Linear(fusion_dim + TRANSITION_FEATURE_DIM + 1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.correction = nn.Linear(hidden_dim, 1)
        self.auxiliary = nn.Linear(hidden_dim, auxiliary_classes)
        nn.init.zeros_(self.correction.weight)
        nn.init.zeros_(self.correction.bias)

    def train(self, mode: bool = True) -> TransitionResidualReranker:
        super().train(mode)
        self.base.eval()
        return self

    def forward_with_auxiliary(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            fused = self.base.encode(x)
            class_logits = self.base.classifier(fused)
            base_logit = class_logits[:, 1] - class_logits[:, 0]
        transition = build_transition_features(x)[:, -1]
        hidden = self.branch(torch.cat((fused, transition, base_logit[:, None]), dim=-1))
        raw_correction = self.correction(hidden).squeeze(-1)
        correction = self.correction_scale * torch.tanh(raw_correction)
        gate = torch.sigmoid(
            (base_logit.detach() - self.gate_center) / self.gate_temperature
        )
        return base_logit + gate * correction, self.auxiliary(hidden), correction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_auxiliary(x)[0]


class TemporalTransitionResidualReranker(nn.Module):
    """Frozen-backbone residual re-ranker with causal transition history."""

    def __init__(
        self,
        base: MultiStreamMultiScaleAttentionTCN,
        *,
        hidden_dim: int = 64,
        temporal_hidden_dim: int = 16,
        dropout: float = 0.25,
        correction_scale: float = 2.0,
        gate_center: float = 0.0,
        gate_temperature: float = 1.0,
        auxiliary_classes: int = 5,
    ) -> None:
        super().__init__()
        if hidden_dim < 8 or temporal_hidden_dim < 4 or auxiliary_classes < 2:
            raise ValueError("hidden_dim/temporal_hidden_dim/auxiliary_classes 无效")
        if correction_scale <= 0.0 or gate_temperature <= 0.0:
            raise ValueError("correction_scale/gate_temperature 必须为正")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        fusion_dim = int(base.classifier.in_features)
        self.correction_scale = float(correction_scale)
        self.gate_center = float(gate_center)
        self.gate_temperature = float(gate_temperature)
        self.transition_gru = nn.GRU(
            TRANSITION_FEATURE_DIM,
            temporal_hidden_dim,
            batch_first=True,
        )
        self.branch = nn.Sequential(
            nn.LayerNorm(
                fusion_dim + TRANSITION_FEATURE_DIM + temporal_hidden_dim + 1
            ),
            nn.Linear(
                fusion_dim + TRANSITION_FEATURE_DIM + temporal_hidden_dim + 1,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.correction = nn.Linear(hidden_dim, 1)
        self.auxiliary = nn.Linear(hidden_dim, auxiliary_classes)
        nn.init.zeros_(self.correction.weight)
        nn.init.zeros_(self.correction.bias)

    def train(self, mode: bool = True) -> TemporalTransitionResidualReranker:
        super().train(mode)
        self.base.eval()
        return self

    def forward_with_auxiliary(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            fused = self.base.encode(x)
            class_logits = self.base.classifier(fused)
            base_logit = class_logits[:, 1] - class_logits[:, 0]
        transition = build_transition_features(x)
        temporal, _ = self.transition_gru(transition)
        hidden = self.branch(
            torch.cat((fused, transition[:, -1], temporal[:, -1], base_logit[:, None]), dim=-1)
        )
        correction = self.correction_scale * torch.tanh(
            self.correction(hidden).squeeze(-1)
        )
        gate = torch.sigmoid(
            (base_logit.detach() - self.gate_center) / self.gate_temperature
        )
        return base_logit + gate * correction, self.auxiliary(hidden), correction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_auxiliary(x)[0]
