"""Non-causal long-context oracle for supervised fall-event localization."""

from __future__ import annotations

import torch
from torch import nn


class LongContextEventOracle(nn.Module):
    """Estimate event quality while supervising stages and temporal boundaries."""

    def __init__(
        self,
        *,
        input_dim: int = 57,
        hidden_dim: int = 96,
        gru_hidden_dim: int = 64,
        dropout: float = 0.2,
        stage_classes: int = 4,
    ) -> None:
        super().__init__()
        if input_dim < 1 or hidden_dim < 3 or gru_hidden_dim < 1:
            raise ValueError("oracle dimensions 必须为正且 hidden_dim 至少为 3")
        if not 0.0 <= dropout < 1.0 or stage_classes < 2:
            raise ValueError("dropout/stage_classes 无效")
        branch_dim = hidden_dim // 3
        merged_dim = branch_dim * 3
        self.input_dim = input_dim
        self.stage_classes = stage_classes
        self.input_projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.temporal_branches = nn.ModuleList(
            nn.Sequential(
                nn.Conv1d(hidden_dim, branch_dim, kernel_size=kernel, padding=kernel // 2),
                nn.GELU(),
            )
            for kernel in (3, 7, 15)
        )
        self.temporal_norm = nn.LayerNorm(merged_dim)
        self.context_encoder = nn.GRU(
            merged_dim,
            gru_hidden_dim,
            num_layers=2,
            dropout=dropout,
            batch_first=True,
            bidirectional=True,
        )
        context_dim = gru_hidden_dim * 2
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
        self, x: torch.Tensor, lengths: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if x.ndim != 3 or x.shape[-1] != self.input_dim:
            raise ValueError(f"oracle 输入必须为 (B,T,{self.input_dim})")
        mask = self._mask(lengths, x.shape[1])
        projected = self.input_projection(x) * mask.unsqueeze(-1)
        channels_first = projected.transpose(1, 2)
        temporal = torch.cat(
            [branch(channels_first) for branch in self.temporal_branches], dim=1
        ).transpose(1, 2)
        temporal = self.temporal_norm(temporal) * mask.unsqueeze(-1)
        packed = nn.utils.rnn.pack_padded_sequence(
            temporal,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_context, _ = self.context_encoder(packed)
        context, _ = nn.utils.rnn.pad_packed_sequence(
            packed_context, batch_first=True, total_length=x.shape[1]
        )
        context = self.context_norm(context) * mask.unsqueeze(-1)
        stage_logits = self.stage_head(context)
        boundary_logits = self.boundary_head(context)
        attention_logits = self.event_attention(context).squeeze(-1)
        attention_logits = attention_logits.masked_fill(~mask, float("-inf"))
        attention = torch.softmax(attention_logits, dim=1)
        attended = torch.sum(context * attention.unsqueeze(-1), dim=1)
        masked_context = context.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        maximum = masked_context.amax(dim=1)
        clip_logit = self.event_classifier(torch.cat((attended, maximum), dim=1)).squeeze(1)
        return {
            "clip_logit": clip_logit,
            "stage_logits": stage_logits,
            "boundary_logits": boundary_logits,
            "attention": attention,
            "mask": mask,
        }
