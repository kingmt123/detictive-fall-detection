"""Protocol-locked fusion for the seven-head EdgeFall ensemble.

The incumbent has three short-window 320 heads, three short-window 640 heads,
and one dense-48/320 head.  RMPTS moves mass only inside the 320 family while
keeping the proven 320/640 resolution balance fixed.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import torch
from torch import nn

RMPTS_GRID: tuple[Fraction, ...] = (
    Fraction(-1, 14),
    Fraction(0, 1),
    Fraction(1, 28),
    Fraction(1, 14),
    Fraction(1, 7),
)


@dataclass(frozen=True)
class RMPTSWeights:
    """Seven fixed weights in their protocol-defined member families."""

    short_320_each: float
    short_640_each: float
    dense_48: float

    @property
    def short_320_mass(self) -> float:
        return 3.0 * self.short_320_each

    @property
    def short_640_mass(self) -> float:
        return 3.0 * self.short_640_each

    @property
    def resolution_320_mass(self) -> float:
        return self.short_320_mass + self.dense_48

    @property
    def total_mass(self) -> float:
        return self.resolution_320_mass + self.short_640_mass


def _registered_delta(delta: float | Fraction) -> Fraction:
    if isinstance(delta, Fraction):
        candidate = delta
    else:
        value = float(delta)
        if not np.isfinite(value):
            raise ValueError("RMPTS delta 必须为有限数值")
        candidate = min(RMPTS_GRID, key=lambda item: abs(float(item) - value))
        if not np.isclose(float(candidate), value, rtol=0.0, atol=1e-12):
            raise ValueError("RMPTS delta 必须来自预注册网格")
    if candidate not in RMPTS_GRID:
        raise ValueError("RMPTS delta 必须来自预注册网格")
    return candidate


def rmpts_weights(delta: float | Fraction = Fraction(0, 1)) -> RMPTSWeights:
    """Return non-negative weights while preserving 320 and 640 family mass."""
    registered = _registered_delta(delta)
    short_320_mass = Fraction(3, 7) - registered
    dense_mass = Fraction(1, 7) + registered
    weights = RMPTSWeights(
        short_320_each=float(short_320_mass / 3),
        short_640_each=float(Fraction(1, 7)),
        dense_48=float(dense_mass),
    )
    values = np.asarray(
        [weights.short_320_each, weights.short_640_each, weights.dense_48]
    )
    if np.any(values < 0.0):
        raise AssertionError("预注册 RMPTS 网格产生了负权重")
    if not np.isclose(weights.resolution_320_mass, 4.0 / 7.0):
        raise AssertionError("RMPTS 破坏了 320 family 质量守恒")
    if not np.isclose(weights.short_640_mass, 3.0 / 7.0):
        raise AssertionError("RMPTS 破坏了 640 family 质量守恒")
    if not np.isclose(weights.total_mass, 1.0):
        raise AssertionError("RMPTS 总权重不为 1")
    return weights


def _validate_numpy_logits(
    short_320: np.ndarray,
    short_640: np.ndarray,
    dense_48: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    first = np.asarray(short_320, dtype=np.float64)
    second = np.asarray(short_640, dtype=np.float64)
    dense = np.asarray(dense_48, dtype=np.float64)
    if first.ndim < 1 or first.shape[-1] != 3:
        raise ValueError("short_320 logits 最后一维必须恰好包含三个 seed")
    if second.shape != first.shape:
        raise ValueError("short_320/short_640 logits shape 必须一致")
    if dense.shape != first.shape[:-1]:
        raise ValueError("dense_48 logits shape 必须等于短窗 logits 去掉 seed 维")
    if not all(np.isfinite(value).all() for value in (first, second, dense)):
        raise ValueError("七头 logits 必须全部为有限数值")
    return first, second, dense


def rmpts_fuse_numpy_logits(
    short_320: np.ndarray,
    short_640: np.ndarray,
    dense_48: np.ndarray,
    *,
    delta: float | Fraction = Fraction(0, 1),
) -> np.ndarray:
    """Fuse raw logits and return one logit per example."""
    first, second, dense = _validate_numpy_logits(short_320, short_640, dense_48)
    weights = rmpts_weights(delta)
    return (
        weights.short_320_each * first.sum(axis=-1)
        + weights.short_640_each * second.sum(axis=-1)
        + weights.dense_48 * dense
    )


def rmpts_fuse_numpy_probabilities(
    short_320: np.ndarray,
    short_640: np.ndarray,
    dense_48: np.ndarray,
    *,
    delta: float | Fraction = Fraction(0, 1),
) -> np.ndarray:
    """Fuse probabilities with the same protocol-locked RMPTS family mass."""
    first, second, dense = _validate_numpy_logits(short_320, short_640, dense_48)
    if any(
        np.any((value < 0.0) | (value > 1.0)) for value in (first, second, dense)
    ):
        raise ValueError("七头 probability 必须位于 [0,1]")
    weights = rmpts_weights(delta)
    fused = (
        weights.short_320_each * first.sum(axis=-1)
        + weights.short_640_each * second.sum(axis=-1)
        + weights.dense_48 * dense
    )
    if np.any((fused < 0.0) | (fused > 1.0)):
        raise AssertionError("Probability-RMPTS 输出越界")
    return fused


class SevenHeadRMPTSFusion(nn.Module):
    """Parameter-free, protocol-locked fusion of seven raw-logit heads."""

    def __init__(self, *, delta: float | Fraction = Fraction(0, 1)) -> None:
        super().__init__()
        registered = _registered_delta(delta)
        weights = rmpts_weights(registered)
        self.delta = registered
        self.register_buffer(
            "family_weights",
            torch.tensor(
                [
                    weights.short_320_each,
                    weights.short_640_each,
                    weights.dense_48,
                ],
                dtype=torch.float64,
            ),
            persistent=True,
        )

    def forward(
        self,
        short_320_logits: torch.Tensor,
        short_640_logits: torch.Tensor,
        dense_48_logits: torch.Tensor,
    ) -> torch.Tensor:
        if short_320_logits.ndim < 1 or short_320_logits.shape[-1] != 3:
            raise ValueError("short_320 logits 最后一维必须恰好包含三个 seed")
        if short_640_logits.shape != short_320_logits.shape:
            raise ValueError("short_320/short_640 logits shape 必须一致")
        if dense_48_logits.shape != short_320_logits.shape[:-1]:
            raise ValueError("dense_48 logits shape 必须等于短窗 logits 去掉 seed 维")
        tensors = (short_320_logits, short_640_logits, dense_48_logits)
        if not all(torch.isfinite(value).all().item() for value in tensors):
            raise ValueError("七头 logits 必须全部为有限数值")
        weights = self.family_weights.to(
            device=short_320_logits.device, dtype=short_320_logits.dtype
        )
        return (
            weights[0] * short_320_logits.sum(dim=-1)
            + weights[1] * short_640_logits.sum(dim=-1)
            + weights[2] * dense_48_logits
        )

    def protocol_signature(self) -> dict[str, object]:
        weights = rmpts_weights(self.delta)
        return {
            "protocol": "edgefall_seven_head_rmpts_v1",
            "delta": f"{self.delta.numerator}/{self.delta.denominator}",
            "member_order": [
                "320-short-seed1",
                "320-short-seed2",
                "320-short-seed3",
                "640-short-seed1",
                "640-short-seed2",
                "640-short-seed3",
                "320-dense48",
            ],
            "weights": {
                "short_320_each": weights.short_320_each,
                "short_640_each": weights.short_640_each,
                "dense_48": weights.dense_48,
            },
            "resolution_mass": {"320": 4.0 / 7.0, "640": 3.0 / 7.0},
        }


class SevenHeadProbabilityRMPTSFusion(nn.Module):
    """Parameter-free probability-domain RMPTS with a frozen delta grid."""

    def __init__(self, *, delta: float | Fraction = Fraction(0, 1)) -> None:
        super().__init__()
        registered = _registered_delta(delta)
        weights = rmpts_weights(registered)
        self.delta = registered
        self.register_buffer(
            "family_weights",
            torch.tensor(
                [
                    weights.short_320_each,
                    weights.short_640_each,
                    weights.dense_48,
                ],
                dtype=torch.float64,
            ),
            persistent=True,
        )

    def forward(
        self,
        short_320_probabilities: torch.Tensor,
        short_640_probabilities: torch.Tensor,
        dense_48_probabilities: torch.Tensor,
    ) -> torch.Tensor:
        if (
            short_320_probabilities.ndim < 1
            or short_320_probabilities.shape[-1] != 3
        ):
            raise ValueError("short_320 probabilities 最后一维必须恰好包含三个 seed")
        if short_640_probabilities.shape != short_320_probabilities.shape:
            raise ValueError("short_320/short_640 probabilities shape 必须一致")
        if dense_48_probabilities.shape != short_320_probabilities.shape[:-1]:
            raise ValueError("dense_48 probabilities shape 不匹配")
        tensors = (
            short_320_probabilities,
            short_640_probabilities,
            dense_48_probabilities,
        )
        if not all(torch.isfinite(value).all().item() for value in tensors):
            raise ValueError("七头 probabilities 必须全部为有限数值")
        if any(torch.any((value < 0.0) | (value > 1.0)).item() for value in tensors):
            raise ValueError("七头 probabilities 必须位于 [0,1]")
        weights = self.family_weights.to(
            device=short_320_probabilities.device,
            dtype=short_320_probabilities.dtype,
        )
        return (
            weights[0] * short_320_probabilities.sum(dim=-1)
            + weights[1] * short_640_probabilities.sum(dim=-1)
            + weights[2] * dense_48_probabilities
        )

    def protocol_signature(self) -> dict[str, object]:
        weights = rmpts_weights(self.delta)
        return {
            "protocol": "edgefall_seven_head_probability_rmpts_v1",
            "fusion_domain": "probability",
            "delta": f"{self.delta.numerator}/{self.delta.denominator}",
            "member_order": [
                "320-short-seed1",
                "320-short-seed2",
                "320-short-seed3",
                "640-short-seed1",
                "640-short-seed2",
                "640-short-seed3",
                "320-dense48",
            ],
            "weights": {
                "short_320_each": weights.short_320_each,
                "short_640_each": weights.short_640_each,
                "dense_48": weights.dense_48,
            },
            "resolution_mass": {"320": 4.0 / 7.0, "640": 3.0 / 7.0},
        }
