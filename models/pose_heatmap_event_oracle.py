"""Non-causal PoseConv3D-style oracle for strong event supervision."""

from __future__ import annotations

import torch
from torch import nn


class PoseHeatmapEventOracle(nn.Module):
    """Encode pose heatmaps, then jointly classify stages, boundaries, and clips."""

    def __init__(
        self,
        *,
        joints: int = 17,
        geometry_dim: int = 6,
        hidden_dim: int = 64,
        dropout: float = 0.2,
        stage_classes: int = 4,
    ) -> None:
        super().__init__()
        if joints < 1 or geometry_dim < 1 or hidden_dim < 1 or stage_classes < 2:
            raise ValueError("PoseConv3D oracle dimensions 无效")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout 无效")
        self.joints = joints
        self.geometry_dim = geometry_dim
        self.stage_classes = stage_classes
        self.spatial_encoder = nn.Sequential(
            nn.Conv3d(joints, 32, kernel_size=(3, 5, 5), padding=(1, 2, 2), stride=(1, 2, 2)),
            nn.BatchNorm3d(32),
            nn.GELU(),
            nn.Conv3d(32, 48, kernel_size=3, padding=1, stride=(1, 2, 2)),
            nn.BatchNorm3d(48),
            nn.GELU(),
            nn.Conv3d(48, hidden_dim, kernel_size=3, padding=1, stride=(1, 2, 2)),
            nn.BatchNorm3d(hidden_dim),
            nn.GELU(),
        )
        self.geometry_projection = nn.Sequential(
            nn.LayerNorm(geometry_dim), nn.Linear(geometry_dim, hidden_dim), nn.GELU()
        )
        self.context_encoder = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=2,
            dropout=dropout,
            batch_first=True,
            bidirectional=True,
        )
        context_dim = hidden_dim * 2
        self.context_norm = nn.LayerNorm(context_dim)
        self.stage_head = nn.Linear(context_dim, stage_classes)
        self.boundary_head = nn.Linear(context_dim, 2)
        self.event_attention = nn.Linear(context_dim, 1)
        self.event_classifier = nn.Sequential(
            nn.Linear(context_dim * 2, context_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(context_dim, 1),
        )

    @staticmethod
    def _mask(lengths: torch.Tensor, steps: int) -> torch.Tensor:
        if lengths.ndim != 1 or torch.any(lengths < 1) or torch.any(lengths > steps):
            raise ValueError("lengths 必须是一维且位于有效序列范围")
        return torch.arange(steps, device=lengths.device)[None, :] < lengths[:, None]

    def forward(
        self, heatmaps: torch.Tensor, geometry: torch.Tensor, lengths: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if heatmaps.ndim != 5 or heatmaps.shape[1] != self.joints:
            raise ValueError("heatmaps 必须为 (B,J,T,H,W)")
        if geometry.ndim != 3 or geometry.shape[:2] != heatmaps.shape[:1] + heatmaps.shape[2:3]:
            raise ValueError("geometry 必须与 heatmaps 的 B/T 对齐")
        if geometry.shape[2] != self.geometry_dim:
            raise ValueError("geometry feature dim 不兼容")
        mask = self._mask(lengths, heatmaps.shape[2])
        spatial = self.spatial_encoder(heatmaps).mean(dim=(-1, -2)).transpose(1, 2)
        temporal = spatial + self.geometry_projection(geometry)
        temporal = temporal * mask.unsqueeze(-1)
        packed = nn.utils.rnn.pack_padded_sequence(
            temporal,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_context, _ = self.context_encoder(packed)
        context, _ = nn.utils.rnn.pad_packed_sequence(
            packed_context, batch_first=True, total_length=heatmaps.shape[2]
        )
        context = self.context_norm(context) * mask.unsqueeze(-1)
        stage_logits = self.stage_head(context)
        boundary_logits = self.boundary_head(context)
        attention_logits = self.event_attention(context).squeeze(-1)
        attention = torch.softmax(attention_logits.masked_fill(~mask, float("-inf")), dim=1)
        attended = torch.sum(context * attention.unsqueeze(-1), dim=1)
        maximum = context.masked_fill(~mask.unsqueeze(-1), float("-inf")).amax(dim=1)
        clip_logit = self.event_classifier(torch.cat((attended, maximum), dim=1)).squeeze(1)
        return {
            "clip_logit": clip_logit,
            "stage_logits": stage_logits,
            "boundary_logits": boundary_logits,
            "attention": attention,
            "mask": mask,
        }
