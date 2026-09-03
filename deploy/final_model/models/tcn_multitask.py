"""FallTCN-compatible multi-task head and window-level auxiliary targets."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch
from torch import nn

from models.tcn import FallTCN
from models.tcn_dataset import SEMANTIC_TO_CODE

AUXILIARY_CLASSES = (
    "background",
    "fall",
    "fallen",
    "lie_down",
    "lying",
    "stand_up",
    "other",
)
AUXILIARY_TO_CODE = {name: index for index, name in enumerate(AUXILIARY_CLASSES)}
_HARD_NEGATIVE_PREFIXES = frozenset({"lie_down", "lying", "stand_up"})


class MultiTaskFallTCN(FallTCN):
    """FallTCN with an auxiliary activity head.

    ``tcn`` and ``head`` deliberately retain the base model names, so a
    :class:`FallTCN` checkpoint loads with ``strict=False`` with only
    ``aux_head.weight`` and ``aux_head.bias`` missing.
    """

    def __init__(
        self,
        in_dim: int = 51,
        channels: tuple[int, ...] = (64, 64, 128),
        kernel: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__(in_dim, channels, kernel, dropout)
        self.aux_head = nn.Linear(self.head.in_features, len(AUXILIARY_CLASSES))

    def _last_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.flatten(2)
        return self.tcn(x.transpose(1, 2))[:, :, -1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the standard FallTCN binary logits with shape ``(B,)``."""
        return self.head(self._last_features(x)).squeeze(-1)

    def forward_multitask(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(binary_logits, auxiliary_logits)`` for multi-task training."""
        features = self._last_features(x)
        return self.head(features).squeeze(-1), self.aux_head(features)


def _validate_window_arrays(
    semantic_codes: np.ndarray, clip_indices: np.ndarray, clips: Sequence[Mapping[str, object]]
) -> None:
    if semantic_codes.ndim != 1 or clip_indices.ndim != 1:
        raise ValueError("semantic_codes 和 clip_indices 必须是一维数组")
    if semantic_codes.shape != clip_indices.shape:
        raise ValueError("semantic_codes 和 clip_indices 的形状必须一致")
    if not np.issubdtype(semantic_codes.dtype, np.integer):
        raise ValueError("semantic_codes 必须是整数数组")
    if not np.issubdtype(clip_indices.dtype, np.integer):
        raise ValueError("clip_indices 必须是整数数组")
    if not clips:
        raise ValueError("metadata clips 不能为空")
    valid_codes = set(SEMANTIC_TO_CODE.values())
    if not np.isin(semantic_codes, tuple(valid_codes)).all():
        raise ValueError("semantic_codes 包含未知语义编码")
    if (clip_indices < 0).any() or (clip_indices >= len(clips)).any():
        raise ValueError("clip_indices 超出 metadata clips 范围")
    for clip in clips:
        clip_id = clip.get("clip_id") if isinstance(clip, Mapping) else None
        if not isinstance(clip_id, str) or not clip_id:
            raise ValueError("metadata clips 必须包含非空 clip_id")


def build_auxiliary_targets(
    semantic_codes: np.ndarray,
    clip_indices: np.ndarray,
    clips: Sequence[Mapping[str, object]],
) -> np.ndarray:
    """Map cache window semantics to fixed seven-class auxiliary targets.

    Background and unlabeled windows become ``background``.  Hard-negative
    windows use their ``clip_id`` first path component for the three targeted
    confusing actions; every other hard-negative activity maps to ``other``.
    """
    semantic_codes = np.asarray(semantic_codes)
    clip_indices = np.asarray(clip_indices)
    _validate_window_arrays(semantic_codes, clip_indices, clips)

    targets = np.full(semantic_codes.shape, AUXILIARY_TO_CODE["background"], dtype=np.int64)
    targets[semantic_codes == SEMANTIC_TO_CODE["fall_process"]] = AUXILIARY_TO_CODE["fall"]
    targets[semantic_codes == SEMANTIC_TO_CODE["post_fall_state"]] = AUXILIARY_TO_CODE[
        "fallen"
    ]
    hard_mask = semantic_codes == SEMANTIC_TO_CODE["hard_negative"]
    for index in np.flatnonzero(hard_mask):
        clip_id = str(clips[int(clip_indices[index])]["clip_id"])
        prefix = clip_id.split("/", 1)[0]
        target_name = prefix if prefix in _HARD_NEGATIVE_PREFIXES else "other"
        targets[index] = AUXILIARY_TO_CODE[target_name]
    return targets


def auxiliary_fall_probability(auxiliary_logits: torch.Tensor) -> torch.Tensor:
    """Return the frozen deployment score ``p(fall) + p(fallen)``."""
    if auxiliary_logits.ndim != 2 or auxiliary_logits.shape[1] != len(AUXILIARY_CLASSES):
        raise ValueError("auxiliary_logits 必须为 (B, 7)")
    probabilities = torch.softmax(auxiliary_logits, dim=1)
    return probabilities[:, AUXILIARY_TO_CODE["fall"]] + probabilities[
        :, AUXILIARY_TO_CODE["fallen"]
    ]
