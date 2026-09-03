"""Low-capacity phase-motion expert for native 48-frame pose windows."""

from __future__ import annotations

import torch
from torch import nn

from models.multiscale_multistream_tcn import (
    KINEMATIC_RULE_DIM,
    build_discriminative_kinematic_features,
)

PHASE_COUNT = 3
PHASE_WIDTH = 16
PHASE_MOTION_DIM = KINEMATIC_RULE_DIM * 11


def phase_motion_features(features: torch.Tensor) -> torch.Tensor:
    """Summarize three ordered 16-frame phases from a causal 48-frame window.

    Each phase contributes mean, maximum, and minimum raw kinematic cues.  Two
    signed changes between adjacent phase means retain the event order without
    attention, learned pooling, or access to future frames.
    """
    if features.ndim != 3 or features.shape[1:] != (48, 57):
        raise ValueError("Phase-Motion 输入必须为 (B,48,57)")
    kinematics = build_discriminative_kinematic_features(features)
    phase_means: list[torch.Tensor] = []
    summaries: list[torch.Tensor] = []
    for phase in range(PHASE_COUNT):
        start = phase * PHASE_WIDTH
        values = kinematics[:, start : start + PHASE_WIDTH]
        mean = values.mean(dim=1)
        phase_means.append(mean)
        summaries.extend((mean, values.amax(dim=1), values.amin(dim=1)))
    summaries.extend(
        (
            phase_means[1] - phase_means[0],
            phase_means[2] - phase_means[1],
        )
    )
    result = torch.cat(summaries, dim=1)
    if result.shape[1] != PHASE_MOTION_DIM:
        raise AssertionError("Phase-Motion 特征维度漂移")
    return result


class PhaseMotionExpert(nn.Module):
    """Small ordered-phase head that never consumes learned e16/e48 embeddings."""

    def __init__(self, hidden_dim: int = 32, subtypes: int = 6) -> None:
        super().__init__()
        if hidden_dim < 1 or subtypes < 2:
            raise ValueError("hidden_dim/subtypes 无效")
        self.encoder = nn.Sequential(
            nn.Linear(PHASE_MOTION_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.binary = nn.Linear(hidden_dim, 1)
        self.subtype = nn.Linear(hidden_dim, subtypes)

    def forward(self, summaries: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if summaries.ndim != 2 or summaries.shape[1] != PHASE_MOTION_DIM:
            raise ValueError("Phase-Motion summary shape 无效")
        value = self.encoder(summaries)
        return self.binary(value).squeeze(1), self.subtype(value)
