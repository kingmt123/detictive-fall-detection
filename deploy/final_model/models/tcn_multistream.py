"""Derived 123D TCN features and auditable 51D checkpoint expansion."""
from __future__ import annotations

from collections.abc import Mapping

import torch

from models.tcn import FallTCN

JOINT_DIM = 17 * 3
COCO_BONE_EDGES = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (0, 5),
    (0, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)
BONE_DIM = len(COCO_BONE_EDGES) * 2
MOTION_DIM = 17 * 2
BBOX_DIM = 5
OBSERVED_DIM = 1
SIDECAR_DIM = BBOX_DIM + OBSERVED_DIM
MULTISTREAM_DIM = JOINT_DIM + BONE_DIM + MOTION_DIM + SIDECAR_DIM
FEATURE_DIM = MULTISTREAM_DIM

_FIRST_CONV_KEY = "tcn.0.net.0.conv.weight"
_FIRST_RESIDUAL_KEY = "tcn.0.res.weight"


def build_multistream_features(
    joints: torch.Tensor, sidecar: torch.Tensor, *, include_extra: bool = True
) -> torch.Tensor:
    """Append bone, motion, bbox, and observed channels to 51D joints.

    ``joints`` is normalized ``(B, T, 51)`` data from the frozen base cache.
    ``sidecar`` is aligned ``(B, T, 6)`` data ordered as bbox5 then observed.
    Motion is zero for the first frame and whenever either side of a temporal
    difference is unobserved. Bone features are zero for unobserved frames.
    """
    if joints.ndim != 3 or joints.shape[-1] != JOINT_DIM:
        raise ValueError(f"joints 必须是 (B,T,{JOINT_DIM})")
    if sidecar.ndim != 3 or sidecar.shape[:2] != joints.shape[:2]:
        raise ValueError("sidecar 的 batch/time 维必须与 joints 一致")
    if sidecar.shape[-1] != SIDECAR_DIM:
        raise ValueError(f"sidecar 必须是 (B,T,{SIDECAR_DIM})")
    if joints.device != sidecar.device:
        raise ValueError("joints 和 sidecar 必须位于同一设备")
    if not joints.is_floating_point() or not sidecar.is_floating_point():
        raise ValueError("joints 和 sidecar 必须是浮点张量")

    if not include_extra:
        zeros = torch.zeros(
            *joints.shape[:2],
            MULTISTREAM_DIM - JOINT_DIM,
            dtype=joints.dtype,
            device=joints.device,
        )
        return torch.cat((joints, zeros), dim=-1)
    coordinates = joints.reshape(*joints.shape[:2], 17, 3)[..., :2]
    edges = torch.as_tensor(COCO_BONE_EDGES, device=joints.device)
    bones = coordinates[..., edges[:, 1], :] - coordinates[..., edges[:, 0], :]
    observed = sidecar[..., BBOX_DIM] > 0.5
    bones = bones * observed.unsqueeze(-1).unsqueeze(-1)

    motion = torch.zeros_like(coordinates)
    consecutive = observed[:, 1:] & observed[:, :-1]
    motion[:, 1:] = (coordinates[:, 1:] - coordinates[:, :-1]) * consecutive.unsqueeze(
        -1
    ).unsqueeze(-1)
    return torch.cat(
        (
            joints,
            bones.flatten(2),
            motion.flatten(2),
            sidecar,
        ),
        dim=-1,
    )


def expand_falltcn_51d_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    *,
    target_in_dim: int = MULTISTREAM_DIM,
) -> dict[str, torch.Tensor]:
    """Expand only FallTCN's first convolution and residual input channels."""
    if target_in_dim < JOINT_DIM:
        raise ValueError("target_in_dim 不能小于 51")
    expanded = {name: value.detach().clone() for name, value in state_dict.items()}
    for name in (_FIRST_CONV_KEY, _FIRST_RESIDUAL_KEY):
        weight = expanded.get(name)
        if weight is None or weight.ndim != 3 or weight.shape[1] != JOINT_DIM:
            raise ValueError(f"base state_dict 缺少 51D 输入权重: {name}")
        target_shape = (weight.shape[0], target_in_dim, weight.shape[2])
        target_weight = torch.zeros(target_shape, dtype=weight.dtype, device=weight.device)
        target_weight[:, :JOINT_DIM] = weight
        expanded[name] = target_weight
    return expanded


def load_expanded_checkpoint(
    model: FallTCN, state_dict: Mapping[str, torch.Tensor]
) -> None:
    """Strictly load a 51D FallTCN state into a 123D FallTCN model.

    The original 51 input channels are copied exactly; the 72 derived feature
    channels are zero-initialized, preserving initial logits for any sidecar.
    """
    first_weight = model.state_dict().get(_FIRST_CONV_KEY)
    residual_weight = model.state_dict().get(_FIRST_RESIDUAL_KEY)
    if (
        first_weight is None
        or residual_weight is None
        or first_weight.shape[1] != MULTISTREAM_DIM
        or residual_weight.shape[1] != MULTISTREAM_DIM
    ):
        raise ValueError("目标模型必须是 in_dim=123 的 FallTCN")
    expanded = expand_falltcn_51d_state_dict(state_dict)
    model.load_state_dict(expanded, strict=True)


load_falltcn_51d_into_123d = load_expanded_checkpoint
