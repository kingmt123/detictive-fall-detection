"""轻量 TCN 时序分类器：17 关键点序列 → 跌倒概率。

输入: (B, T, 17, 3) 或 (B, T, 51) 的关键点序列（x, y, conf，已归一化）
输出: (B,) 该滑窗属于跌倒过程的概率（sigmoid 前 logit）

设计要点（依据 docs/research/lightweight_options.md 第 4 节）:
- 因果膨胀卷积堆叠，感受野覆盖整个窗口，可改流式（逐帧+环形缓冲）
- 参数量目标 ≤0.5M；GPU 单窗推理 <1ms
"""
from __future__ import annotations

import torch
from torch import nn


class CausalConv1d(nn.Module):
    """左侧补零的因果卷积，保证 t 时刻输出只依赖 ≤t 的输入。"""

    def __init__(self, cin: int, cout: int, k: int, dilation: int):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.conv = nn.Conv1d(cin, cout, k, dilation=dilation)

    def forward(self, x):  # (B, C, T)
        return self.conv(nn.functional.pad(x, (self.pad, 0)))


class TCNBlock(nn.Module):
    def __init__(self, cin: int, cout: int, k: int, dilation: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(cin, cout, k, dilation),
            nn.BatchNorm1d(cout),
            nn.ReLU(),
            nn.Dropout(dropout),
            CausalConv1d(cout, cout, k, dilation),
            nn.BatchNorm1d(cout),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.res = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.res(x)


class FallTCN(nn.Module):
    """(B, T, 51) → (B,) logit。默认配置 ~0.15M 参数。"""

    def __init__(
        self,
        in_dim: int = 51,
        channels: tuple[int, ...] = (64, 64, 128),
        kernel: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        blocks = []
        cin = in_dim
        for i, c in enumerate(channels):
            blocks.append(TCNBlock(cin, c, kernel, dilation=2 ** i, dropout=dropout))
            cin = c
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Linear(cin, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, 51) 或 (B, T, 17, 3) → logit (B,)"""
        if x.dim() == 4:
            x = x.flatten(2)
        h = self.tcn(x.transpose(1, 2))      # (B, C, T)
        return self.head(h[:, :, -1]).squeeze(-1)  # 取最后时刻（因果）


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
