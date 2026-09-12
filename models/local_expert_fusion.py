"""Constrained local experts for frozen-anchor fall-score correction.

These modules intentionally do not own an anchor model.  A caller supplies
frozen anchor logits and fold-frozen score-band boundaries, which keeps nested
OOF construction and model architecture separate.
"""

from __future__ import annotations

import torch
from torch import nn


class PoseHeatmapBandResidual(nn.Module):
    """PoseConv-style R90 residual expert with a smooth, bounded score gate.

    The expert receives confidence-weighted joint heatmaps rather than RGB or
    coordinate vectors.  Its residual and non-negative fusion scale begin at
    zero, guaranteeing exact frozen-anchor behavior before optimization.
    """

    def __init__(
        self,
        *,
        joints: int = 17,
        hidden_dim: int = 64,
        subtype_count: int = 10,
        correction_cap: float = 1.0,
    ) -> None:
        super().__init__()
        if joints < 1 or hidden_dim < 4 or subtype_count < 2:
            raise ValueError("heatmap expert dimensions 无效")
        if correction_cap <= 0.0:
            raise ValueError("correction_cap 必须为正")
        self.joints = int(joints)
        self.hidden_dim = int(hidden_dim)
        self.spatial = nn.Sequential(
            nn.Conv3d(joints, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(hidden_dim),
            nn.GELU(),
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(hidden_dim),
            nn.GELU(),
        )
        self.temporal = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=5,
            groups=hidden_dim,
            bias=False,
        )
        self.embedding = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.residual = nn.Linear(hidden_dim, 1)
        self.subtype = nn.Linear(hidden_dim, subtype_count)
        # softplus(0) - log(2) is exactly zero.  project_alpha_ keeps the
        # underlying parameter non-negative after every optimizer update.
        self.alpha_parameter = nn.Parameter(torch.zeros(()))
        self.correction_cap = float(correction_cap)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    @staticmethod
    def score_band(
        anchor_logits: torch.Tensor,
        *,
        low: float,
        high: float,
        temperature: float,
    ) -> torch.Tensor:
        """Return a continuous membership value for a frozen score interval."""
        if anchor_logits.ndim != 1 or not low < high or temperature <= 0.0:
            raise ValueError("anchor score band 参数无效")
        return torch.sigmoid((anchor_logits - low) / temperature) - torch.sigmoid(
            (anchor_logits - high) / temperature
        )

    def alpha(self) -> torch.Tensor:
        return nn.functional.softplus(self.alpha_parameter) - self.alpha_parameter.new_tensor(
            2.0
        ).log()

    @torch.no_grad()
    def project_alpha_(self) -> None:
        """Project the scale parameter to alpha >= 0 after an optimizer step."""
        self.alpha_parameter.clamp_(min=0.0)

    def encode(self, heatmaps: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Encode padded heatmaps, excluding frames after each clip length."""
        if heatmaps.ndim != 5 or heatmaps.shape[1] != self.joints:
            raise ValueError("heatmaps 必须是 (B,J,T,H,W) 且关节数匹配")
        if lengths.shape != (heatmaps.shape[0],):
            raise ValueError("lengths 必须为 (B,)")
        steps = heatmaps.shape[2]
        if steps < 1 or torch.any(lengths < 1) or torch.any(lengths > steps):
            raise ValueError("heatmap lengths 超出有效范围")
        valid_lengths = lengths.to(device=heatmaps.device, dtype=torch.long)
        mask = (
            torch.arange(steps, device=heatmaps.device)[None, :]
            < valid_lengths[:, None]
        )
        # Mask before the 3D convolution as well as before pooling: otherwise
        # a padded frame could bleed into the final valid frame through the
        # convolution's temporal receptive field.
        spatial = self.spatial(heatmaps * mask[:, None, :, None, None]).mean(
            dim=(-1, -2)
        )
        # Left padding preserves temporal causality for clip-level deployment.
        temporal = self.temporal(nn.functional.pad(spatial, (4, 0)))[..., :steps]
        values = temporal.transpose(1, 2)
        weights = mask.unsqueeze(-1).to(values.dtype)
        mean = (values * weights).sum(dim=1) / valid_lengths[:, None].to(values.dtype)
        masked = values.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        maximum = masked.amax(dim=1)
        return self.embedding(torch.cat((mean, maximum), dim=1))

    def forward(
        self,
        heatmaps: torch.Tensor,
        lengths: torch.Tensor,
        anchor_logits: torch.Tensor,
        *,
        low: float,
        high: float,
        temperature: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if anchor_logits.shape != (heatmaps.shape[0],):
            raise ValueError("anchor logits 必须与 heatmap batch 对齐")
        embedding = self.encode(heatmaps, lengths)
        gate = self.score_band(
            anchor_logits, low=low, high=high, temperature=temperature
        )
        bounded_residual = self.correction_cap * torch.tanh(
            self.residual(embedding).squeeze(1)
        )
        final_logits = anchor_logits + self.alpha() * gate * bounded_residual
        return final_logits, gate, bounded_residual, self.subtype(embedding)
