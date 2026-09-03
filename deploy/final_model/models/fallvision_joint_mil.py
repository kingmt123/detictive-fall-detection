"""Joint-stream MIL pretraining for external FallVision keypoint clips."""

from __future__ import annotations

import torch
from torch import nn

from models.multiscale_multistream_tcn import JOINT_DIM, StreamEncoder


class FallVisionJointMIL(nn.Module):
    """A joint-only encoder whose state transfers exactly into stream zero."""

    def __init__(
        self,
        *,
        channels: int = 32,
        output_dim: int = 64,
        dropout: float = 0.35,
    ) -> None:
        super().__init__()
        self.encoder = StreamEncoder(JOINT_DIM, channels, output_dim, dropout)
        self.classifier = nn.Linear(output_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != JOINT_DIM:
            raise ValueError("FallVision Joint MIL 输入必须是 (B,T,51)")
        logits = self.classifier(self.encoder(x))
        return logits[:, 1] - logits[:, 0]
