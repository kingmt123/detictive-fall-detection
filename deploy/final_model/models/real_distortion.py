"""Replay train-only pose/track failure patterns as deterministic augmentation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class RealDistortionBank:
    joint_keep: torch.Tensor
    track_keep: torch.Tensor
    bbox_residual: torch.Tensor
    time_index: torch.Tensor

    @classmethod
    def load(cls, path: Path, *, device: torch.device) -> RealDistortionBank:
        with np.load(path, allow_pickle=False) as payload:
            required = {"joint_keep", "track_keep", "bbox_residual", "time_index"}
            if required - set(payload.files):
                raise ValueError("真实失真模板字段不完整")
            arrays = {key: np.asarray(payload[key]) for key in required}
        count, frames, joints = arrays["joint_keep"].shape
        if count < 1 or frames < 2 or joints != 17:
            raise ValueError("真实失真 joint_keep shape 无效")
        if arrays["track_keep"].shape != (count, frames):
            raise ValueError("真实失真 track_keep shape 无效")
        if arrays["bbox_residual"].shape != (count, frames, 4):
            raise ValueError("真实失真 bbox_residual shape 无效")
        if arrays["time_index"].shape != (count, frames):
            raise ValueError("真实失真 time_index shape 无效")
        if np.any(np.diff(arrays["time_index"], axis=1) < 0):
            raise ValueError("真实失真 time_index 必须单调")
        if arrays["time_index"].min() < 0 or arrays["time_index"].max() >= frames:
            raise ValueError("真实失真 time_index 越界")
        return cls(
            joint_keep=torch.as_tensor(arrays["joint_keep"], device=device),
            track_keep=torch.as_tensor(arrays["track_keep"], device=device),
            bbox_residual=torch.as_tensor(arrays["bbox_residual"], device=device),
            time_index=torch.as_tensor(arrays["time_index"], device=device, dtype=torch.long),
        )

    @property
    def count(self) -> int:
        return int(self.joint_keep.shape[0])


def replay_real_distortion(
    features: torch.Tensor,
    bank: RealDistortionBank,
    *,
    mode: str,
    probability: float,
    seed: int,
) -> torch.Tensor:
    """Apply empirical observation or temporal templates to a batch."""
    if mode not in {"observation", "temporal"}:
        raise ValueError("真实失真 mode 必须为 observation 或 temporal")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("真实失真 probability 必须位于 [0, 1]")
    if features.ndim != 3 or features.shape[1] != bank.joint_keep.shape[1]:
        raise ValueError("训练特征与真实失真模板帧数不匹配")
    generator = torch.Generator(device=features.device).manual_seed(seed)
    selected = torch.rand(features.shape[0], device=features.device, generator=generator) < probability
    if not selected.any():
        return features
    template = torch.randint(
        bank.count, (features.shape[0],), device=features.device, generator=generator
    )
    output = features.clone()
    if mode == "temporal":
        indices = bank.time_index[template]
        warped = torch.gather(output, 1, indices.unsqueeze(-1).expand_as(output))
        return torch.where(selected[:, None, None], warped, output)

    pose = output[..., :51].reshape(*output.shape[:2], 17, 3)
    keep = bank.joint_keep[template].to(pose.dtype).clamp_(0.0, 1.0)
    pose[..., 2].mul_(torch.where(selected[:, None, None], keep, torch.ones_like(keep)))
    pose[..., :2].masked_fill_((pose[..., 2] <= 0).unsqueeze(-1), 0.0)
    track_keep = bank.track_keep[template].bool()
    missing = selected[:, None] & ~track_keep
    corruption_dim = 57 if output.shape[-1] >= 57 else 51
    output[..., :corruption_dim].masked_fill_(missing.unsqueeze(-1), 0.0)
    if output.shape[-1] >= 57:
        geometry = output[..., 51:57]
        residual = bank.bbox_residual[template].to(geometry.dtype)
        active = selected[:, None] & (geometry[..., 5] > 0) & track_keep
        geometry[..., :4] = torch.where(
            active.unsqueeze(-1),
            (geometry[..., :4] + residual).clamp_min(0.0),
            geometry[..., :4],
        )
        geometry[..., :2].clamp_(0.0, 1.0)
        observed = geometry[..., 5] > 0
        geometry[..., 2:4] = torch.where(
            observed.unsqueeze(-1),
            geometry[..., 2:4].clamp_min(1e-4),
            geometry[..., 2:4],
        )
        safe_width = geometry[..., 2].clamp_min(1e-4)
        safe_height = geometry[..., 3].clamp_min(1e-4)
        aspect = torch.log(safe_width / safe_height).div(3.0).clamp(-1.0, 1.0)
        geometry[..., 4] = torch.where(active, aspect, geometry[..., 4])
    return output
