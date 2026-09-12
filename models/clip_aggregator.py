"""Single, differentiable contract for reducing window evidence to clip logits."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import torch

ClipAggregation = Literal["max", "smooth_max", "topk_mean", "topk_logmeanexp"]


def aggregate_clip_logits(
    logits: torch.Tensor,
    group_sizes: list[int],
    *,
    mode: ClipAggregation,
    temperature: float = 0.1,
    topk_fraction: float = 0.2,
    topk_count: int | None = None,
) -> torch.Tensor:
    """Aggregate contiguous window logits once per clip.

    ``smooth_max`` is log-mean-exp, deliberately normalized by clip length;
    ``topk_logmeanexp`` retains Challenge A's top-k log-mean-exp semantics.
    """
    if logits.ndim != 1 or not group_sizes or sum(group_sizes) != logits.numel():
        raise ValueError("logits 与 group_sizes 不匹配")
    if mode not in {"max", "smooth_max", "topk_mean", "topk_logmeanexp"}:
        raise ValueError("未知 clip 聚合器")
    if mode == "smooth_max" and temperature <= 0.0:
        raise ValueError("temperature 必须为正")
    if mode.startswith("topk_"):
        if topk_count is not None and (
            isinstance(topk_count, bool) or not isinstance(topk_count, int) or topk_count < 1
        ):
            raise ValueError("topk_count 必须为正整数")
        if topk_count is None and not 0.0 < topk_fraction <= 1.0:
            raise ValueError("topk_fraction 必须位于 (0, 1]")

    values: list[torch.Tensor] = []
    offset = 0
    for size in group_sizes:
        if size < 1:
            raise ValueError("每个 clip 至少需要一个窗口")
        group = logits[offset : offset + size]
        if mode == "max":
            value = group.max()
        elif mode == "smooth_max":
            value = temperature * (
                torch.logsumexp(group / temperature, dim=0) - math.log(size)
            )
        else:
            count = min(size, topk_count) if topk_count is not None else max(
                1, math.ceil(size * topk_fraction)
            )
            top = torch.topk(group, count).values
            value = (
                top.mean()
                if mode == "topk_mean"
                else torch.logsumexp(top, dim=0) - math.log(count)
            )
        values.append(value)
        offset += size
    return torch.stack(values)


def aggregate_indexed_clip_logits(
    logits: torch.Tensor,
    clip_indices: torch.Tensor,
    *,
    mode: ClipAggregation,
    temperature: float = 0.1,
    topk_fraction: float = 0.2,
    topk_count: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply :func:`aggregate_clip_logits` to arbitrary integer clip indices."""
    if (
        logits.ndim != 1
        or clip_indices.ndim != 1
        or logits.numel() != clip_indices.numel()
    ):
        raise ValueError("logits/clip_indices 必须是等长一维张量")
    if logits.numel() == 0:
        raise ValueError("聚合输入不能为空")
    if clip_indices.dtype not in {torch.int32, torch.int64}:
        raise TypeError("clip_indices 必须为整数张量")
    clip_indices = clip_indices.to(device=logits.device)
    unique, inverse = torch.unique(clip_indices, sorted=True, return_inverse=True)
    groups = [logits[inverse == index] for index in range(unique.numel())]
    return aggregate_clip_logits(
        torch.cat(groups),
        [group.numel() for group in groups],
        mode=mode,
        temperature=temperature,
        topk_fraction=topk_fraction,
        topk_count=topk_count,
    ), unique


def aggregate_window_scores_max(
    scores: np.ndarray,
    clip_indices: np.ndarray,
    clip_count: int,
    *,
    require_all: bool = True,
    missing_score: float | None = None,
) -> np.ndarray:
    """Maximum-pool NumPy window scores with one shared coverage policy."""
    values = np.asarray(scores, dtype=np.float32)
    indices = np.asarray(clip_indices)
    if values.ndim != 1 or indices.ndim != 1 or values.shape != indices.shape:
        raise ValueError("窗口分数与 clip_indices 必须是等长一维数组")
    if clip_count < 1 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("clip_count 必须为正且 clip_indices 必须为整数")
    if indices.size and (indices.min() < 0 or indices.max() >= clip_count):
        raise ValueError("clip_indices 超出 clip_count 范围")
    result = np.full(clip_count, -np.inf, dtype=np.float32)
    np.maximum.at(result, indices, values)
    missing = ~np.isfinite(result)
    if require_all and np.any(missing):
        raise ValueError("窗口分数未覆盖全部目标 clips")
    if missing_score is not None:
        result[missing] = float(missing_score)
    return result


def aggregate_window_logits_peak_support(
    window_logits: np.ndarray,
    clip_indices: np.ndarray,
    clip_count: int,
    *,
    support: int = 2,
    horizon: int = 3,
    require_all: bool = True,
) -> np.ndarray:
    """Return the strongest causal horizon with at least ``support`` peaks.

    For the preregistered 2-of-3 rule this is the maximum, over all causal
    three-window histories, of the second-largest logit. Histories shorter than
    ``support`` do not qualify, so an isolated one-window spike cannot define a
    clip score.
    """
    logits = np.asarray(window_logits, dtype=np.float64)
    indices = np.asarray(clip_indices, dtype=np.int64)
    if logits.ndim != 1 or indices.shape != logits.shape:
        raise ValueError("window_logits/clip_indices 必须为同形一维数组")
    if clip_count < 1 or not 1 <= support <= horizon:
        raise ValueError("clip_count/support/horizon 无效")
    if not np.isfinite(logits).all() or np.any((indices < 0) | (indices >= clip_count)):
        raise ValueError("window logits或clip indices无效")
    result = np.full(clip_count, -np.inf, dtype=np.float64)
    for clip_index in range(clip_count):
        values = logits[indices == clip_index]
        if values.size < support:
            continue
        best = -np.inf
        for end in range(support, values.size + 1):
            history = values[max(0, end - horizon) : end]
            if history.size >= support:
                candidate = np.partition(history, history.size - support)[
                    history.size - support
                ]
                best = max(best, float(candidate))
        result[clip_index] = best
    missing = ~np.isfinite(result)
    if require_all and missing.any():
        raise ValueError(f"peak-support未覆盖{int(missing.sum())}个clips")
    return result
