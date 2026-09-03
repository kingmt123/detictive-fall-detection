"""Bounded, zero-initialized clip evidence pooling for Stage-S1-R2."""

from __future__ import annotations

import math

import torch
from torch import nn


def clip_evidence_statistics(
    window_logits: torch.Tensor,
    group_sizes: list[int],
    *,
    stage_logits: torch.Tensor | None = None,
    peak_delta: float = 1.0,
) -> torch.Tensor:
    """Build deterministic per-clip evidence features from ordered window logits.

    The last two features are fall-process and post-fall probabilities at the
    maximum-logit window.  They are zero when no stage head is available.
    """
    if window_logits.ndim != 1 or not group_sizes or sum(group_sizes) != window_logits.numel():
        raise ValueError("window_logits 与 group_sizes 不匹配")
    if peak_delta <= 0.0:
        raise ValueError("peak_delta 必须为正")
    if stage_logits is not None and stage_logits.shape != (window_logits.numel(), 4):
        raise ValueError("stage_logits 必须为形状 (N, 4)")
    rows: list[torch.Tensor] = []
    offset = 0
    for size in group_sizes:
        if size < 1:
            raise ValueError("每个 clip 至少需要一个窗口")
        logits = window_logits[offset : offset + size]
        maximum, index_tensor = logits.max(dim=0)
        index = int(index_tensor.item())
        top3 = torch.topk(logits, min(3, size)).values.mean()
        top_count = max(1, math.ceil(size * 0.2))
        top20 = torch.topk(logits, top_count).values.mean()
        above_peak = logits >= maximum - peak_delta - torch.finfo(logits.dtype).eps
        left = index
        while left > 0 and above_peak[left - 1]:
            left -= 1
        right = index
        while right + 1 < size and above_peak[right + 1]:
            right += 1
        peak_width = logits.new_tensor((right - left + 1) / size)
        if stage_logits is None:
            stage_evidence = logits.new_zeros(2)
        else:
            stage_evidence = torch.softmax(stage_logits[offset + index], dim=-1)[:2]
        rows.append(
            torch.cat(
                (
                    maximum.reshape(1),
                    top3.reshape(1),
                    top20.reshape(1),
                    (maximum - top3).reshape(1),
                    peak_width.reshape(1),
                    stage_evidence,
                )
            )
        )
        offset += size
    return torch.stack(rows)


class EvidencePoolingResidual(nn.Module):
    """Correct max-logit clip evidence by a small bounded residual.

    The final projection is zero initialized, so construction is an exact
    max-pooling anchor irrespective of the MLP's randomly initialized layers.
    """

    feature_dim = 7

    def __init__(
        self,
        *,
        bottleneck_dim: int = 8,
        correction_cap: float = 0.5,
        peak_delta: float = 1.0,
    ) -> None:
        super().__init__()
        if bottleneck_dim < 1 or correction_cap <= 0.0 or peak_delta <= 0.0:
            raise ValueError("聚合器参数无效")
        self.correction_cap = float(correction_cap)
        self.peak_delta = float(peak_delta)
        self.mlp = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, 1),
        )
        final = self.mlp[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(
        self,
        window_logits: torch.Tensor,
        group_sizes: list[int],
        *,
        stage_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        stats = clip_evidence_statistics(
            window_logits,
            group_sizes,
            stage_logits=stage_logits,
            peak_delta=self.peak_delta,
        )
        return stats[:, 0] + self.correction_cap * torch.tanh(self.mlp(stats).squeeze(-1))
