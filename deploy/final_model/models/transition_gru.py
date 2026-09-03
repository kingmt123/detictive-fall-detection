"""Causal, lightweight re-ranker for fall-transition candidates."""

from __future__ import annotations

import torch
from torch import nn

from models.multiscale_multistream_tcn import (
    build_rule_features,
    build_transition_rule_features,
)

CONTROLLED_TRANSITION = 0
FALL_INCIDENT = 1
OTHER_STATE = 2
TRANSITION_CLASS_NAMES = ("controlled_transition", "fall_incident", "other_state")
TRANSITION_FEATURE_DIM = 20


def build_transition_features(x: torch.Tensor) -> torch.Tensor:
    """Return 20 continuous, causal transition cues from ``(B,T,57)`` input."""
    rules = build_rule_features(x)
    transition = build_transition_rule_features(x)
    features = torch.cat((rules, transition), dim=-1)
    if features.shape[-1] != TRANSITION_FEATURE_DIM:
        raise RuntimeError("转变特征维度契约被破坏")
    return features


class TransitionGRU(nn.Module):
    """Three-class causal GRU that vetoes controlled posture transitions.

    ``forward`` exposes a binary fall-versus-rest logit for existing evaluation
    helpers. ``forward_multiclass`` is used for its supervised three-state loss.
    """

    def __init__(
        self,
        *,
        feature_dim: int = TRANSITION_FEATURE_DIM,
        hidden_size: int = 32,
        num_layers: int = 2,
        dropout: float = 0.2,
        attention_heads: int = 4,
    ) -> None:
        super().__init__()
        if feature_dim < 1 or hidden_size < attention_heads or hidden_size % attention_heads:
            raise ValueError("feature_dim/hidden_size/attention_heads 不兼容")
        if num_layers < 1 or not 0.0 <= dropout < 1.0:
            raise ValueError("num_layers/dropout 无效")
        self.feature_norm = nn.LayerNorm(feature_dim)
        self.gru = nn.GRU(
            feature_dim,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.attention = nn.MultiheadAttention(
            hidden_size, attention_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden_size, 16), nn.GELU(), nn.Linear(16, 3)
        )

    def forward_multiclass(self, x: torch.Tensor) -> torch.Tensor:
        features = build_transition_features(x)
        sequence, _ = self.gru(self.feature_norm(features))
        query = sequence[:, -1:].contiguous()
        attended, _ = self.attention(query, sequence, sequence, need_weights=False)
        return self.classifier(self.norm((query + attended).squeeze(1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward_multiclass(x)
        rest = torch.logsumexp(
            logits[:, (CONTROLLED_TRANSITION, OTHER_STATE)], dim=-1
        )
        return logits[:, FALL_INCIDENT] - rest
