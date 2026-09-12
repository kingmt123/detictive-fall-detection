"""Minimal EdgeFall F1 heads for paired skeleton/person-ROI late fusion."""

from __future__ import annotations

import torch
from torch import nn


class RoiTemporalEncoder(nn.Module):
    """Encode shared-YOLO per-frame person ROI tokens with causal multi-kernel TCN."""

    def __init__(
        self,
        input_dim: int = 192,
        hidden_dim: int = 128,
        *,
        temporal_pooling: str = "mean_max",
    ) -> None:
        super().__init__()
        if input_dim < 1 or hidden_dim < 4:
            raise ValueError("ROI temporal dimensions 无效")
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.convolutions = nn.ModuleList(
            nn.Conv1d(
                hidden_dim,
                hidden_dim,
                kernel_size=kernel,
                groups=hidden_dim,
                bias=False,
            )
            for kernel in (3, 7, 15)
        )
        if temporal_pooling not in {
            "mean_max",
            "attention_max",
            "residual_attention_max",
            "residual_transition_max",
        }:
            raise ValueError("未知 ROI temporal pooling")
        self.temporal_pooling = temporal_pooling
        if temporal_pooling in {"attention_max", "residual_attention_max"}:
            self.temporal_attention = nn.Linear(hidden_dim, 1)
            if temporal_pooling == "attention_max":
                nn.init.zeros_(self.temporal_attention.weight)
                nn.init.zeros_(self.temporal_attention.bias)
        else:
            self.temporal_attention = None
        self.attention_residual_weights = (
            nn.Parameter(torch.zeros(hidden_dim))
            if temporal_pooling == "residual_attention_max"
            else None
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.transition_residual = (
            nn.Linear(hidden_dim * 2, hidden_dim)
            if temporal_pooling == "residual_transition_max"
            else None
        )
        if self.transition_residual is not None:
            nn.init.zeros_(self.transition_residual.weight)
            nn.init.zeros_(self.transition_residual.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError("person ROI tokens 必须为 (B,T,D)")
        value = self.input_projection(tokens).transpose(1, 2)
        branches = []
        for convolution in self.convolutions:
            padding = convolution.kernel_size[0] - 1
            padded = nn.functional.pad(value, (padding, 0))
            branches.append(convolution(padded)[..., : value.shape[-1]])
        context = torch.stack(branches).mean(0).transpose(1, 2)
        mean_value = context.mean(1)
        if self.temporal_attention is not None:
            attention = torch.softmax(self.temporal_attention(context), dim=1)
            attended_value = (context * attention).sum(1)
            if self.attention_residual_weights is None:
                mean_value = attended_value
            else:
                mean_value = mean_value + self.attention_residual_weights * (
                    attended_value - mean_value
                )
        pooled = torch.cat((mean_value, context.amax(1)), dim=1)
        output = self.output(pooled)
        if self.transition_residual is not None:
            endpoint_delta = context[:, -1] - context[:, 0]
            mean_step = (context[:, 1:] - context[:, :-1]).abs().mean(1)
            output = output + self.transition_residual(
                torch.cat((endpoint_delta, mean_step), dim=1)
            )
        return output


class RoiSpatialStatisticsAdapter(nn.Module):
    """Mix cached mean/std/x/y moments back to the original 192D token width.

    The additional moments start with exactly zero contribution.  This keeps a
    newly constructed spatial-statistics F1 observationally identical to the
    original mean-token F1 while adding only three scalars per output channel.
    """

    def __init__(self, p3_channels: int = 64, p4_channels: int = 128) -> None:
        super().__init__()
        if min(p3_channels, p4_channels) < 1:
            raise ValueError("P3/P4 channel 数必须为正")
        self.p3_channels = int(p3_channels)
        self.p4_channels = int(p4_channels)
        self.moment_weights = nn.Parameter(
            torch.zeros(3, self.p3_channels + self.p4_channels)
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        expected = 4 * (self.p3_channels + self.p4_channels)
        if tokens.ndim != 3 or tokens.shape[2] != expected:
            raise ValueError(f"spatial ROI tokens 必须为 (B,T,{expected})")
        p3_width = 4 * self.p3_channels
        p3 = tokens[..., :p3_width].reshape(
            *tokens.shape[:2], 4, self.p3_channels
        )
        p4 = tokens[..., p3_width:].reshape(
            *tokens.shape[:2], 4, self.p4_channels
        )
        statistics = torch.cat((p3, p4), dim=3)
        return statistics[..., 0, :] + (
            statistics[..., 1:, :] * self.moment_weights
        ).sum(dim=2)


class RoiMeanMaxAdapter(nn.Module):
    """Add a zero-initialized per-channel local-peak correction to ROI means."""

    def __init__(self, p3_channels: int = 64, p4_channels: int = 128) -> None:
        super().__init__()
        if min(p3_channels, p4_channels) < 1:
            raise ValueError("P3/P4 channel 数必须为正")
        self.p3_channels = int(p3_channels)
        self.p4_channels = int(p4_channels)
        self.peak_weights = nn.Parameter(
            torch.zeros(self.p3_channels + self.p4_channels)
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        expected = 2 * (self.p3_channels + self.p4_channels)
        if tokens.ndim != 3 or tokens.shape[2] != expected:
            raise ValueError(f"mean/max ROI tokens 必须为 (B,T,{expected})")
        p3_width = 2 * self.p3_channels
        p3 = tokens[..., :p3_width].reshape(
            *tokens.shape[:2], 2, self.p3_channels
        )
        p4 = tokens[..., p3_width:].reshape(
            *tokens.shape[:2], 2, self.p4_channels
        )
        statistics = torch.cat((p3, p4), dim=3)
        mean = statistics[..., 0, :]
        peak_delta = statistics[..., 1, :] - mean
        return mean + peak_delta * self.peak_weights


class RoiResolutionDeltaAdapter(nn.Module):
    """Learn a zero-initialized channel correction from high-resolution ROI tokens."""

    def __init__(self, channels: int = 192) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError("ROI channel 数必须为正")
        self.channels = int(channels)
        self.delta_weights = nn.Parameter(torch.zeros(self.channels))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[2] != 2 * self.channels:
            raise ValueError(
                f"dual-resolution ROI tokens 必须为 (B,T,{2 * self.channels})"
            )
        base, high_resolution = tokens.split(self.channels, dim=2)
        return base + (high_resolution - base) * self.delta_weights


class RoiResolutionStochasticAdapter(nn.Module):
    """Select one of two aligned ROI resolutions independently per sample.

    Training exposes one shared ROI head to both resolutions without blending
    their token representations.  Evaluation is deliberately fail-closed:
    callers must duplicate the resolution they want scored in both halves.
    This lets inference average the two resulting logits outside the model.
    """

    def __init__(self, channels: int = 192) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError("ROI channel 数必须为正")
        self.channels = int(channels)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[2] != 2 * self.channels:
            raise ValueError(
                f"stochastic dual-resolution ROI tokens 必须为 (B,T,{2 * self.channels})"
            )
        base, high_resolution = tokens.split(self.channels, dim=2)
        if not self.training:
            if not torch.equal(base, high_resolution):
                raise ValueError("eval 时必须在两个 half 中复制同一分辨率 token")
            return base
        use_high = torch.rand(
            (tokens.shape[0], 1, 1), device=tokens.device
        ) < 0.5
        return torch.where(use_high, high_resolution, base)


class SkeletonControlHead(nn.Module):
    def __init__(
        self, skeleton_dim: int, *, hidden_dim: int = 128, subtypes: int = 6
    ) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(skeleton_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.binary = nn.Linear(hidden_dim, 1)
        self.subtype = nn.Linear(hidden_dim, subtypes)

    def forward(self, skeleton: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        value = self.projection(skeleton)
        return self.binary(value).squeeze(1), self.subtype(value)


class EdgeFallF1Head(nn.Module):
    """Simple concat late fusion; context ROI and quality gates are intentionally absent."""

    def __init__(
        self,
        skeleton_dim: int,
        *,
        roi_dim: int = 192,
        hidden_dim: int = 128,
        subtypes: int = 6,
        roi_token_layout: str = "direct",
        roi_temporal_pooling: str = "mean_max",
    ) -> None:
        super().__init__()
        self.skeleton_projection = nn.Sequential(
            nn.Linear(skeleton_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        if roi_token_layout == "direct":
            self.roi_adapter: nn.Module = nn.Identity()
            encoder_input_dim = roi_dim
        elif roi_token_layout == "mean_max_compact":
            if roi_dim != 384:
                raise ValueError("mean_max_compact 要求 384D ROI token")
            self.roi_adapter = RoiMeanMaxAdapter()
            encoder_input_dim = 192
        elif roi_token_layout == "dual_resolution_compact":
            if roi_dim != 384:
                raise ValueError("dual_resolution_compact 要求 384D ROI token")
            self.roi_adapter = RoiResolutionDeltaAdapter()
            encoder_input_dim = 192
        elif roi_token_layout == "resolution_stochastic":
            if roi_dim != 384:
                raise ValueError("resolution_stochastic 要求 384D ROI token")
            self.roi_adapter = RoiResolutionStochasticAdapter()
            encoder_input_dim = 192
        elif roi_token_layout == "mean_std_xy_compact":
            if roi_dim != 768:
                raise ValueError("mean_std_xy_compact 要求 768D ROI token")
            self.roi_adapter = RoiSpatialStatisticsAdapter()
            encoder_input_dim = 192
        else:
            raise ValueError("未知 roi_token_layout")
        self.roi_encoder = RoiTemporalEncoder(
            encoder_input_dim,
            hidden_dim,
            temporal_pooling=roi_temporal_pooling,
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.binary = nn.Linear(hidden_dim, 1)
        self.subtype = nn.Linear(hidden_dim, subtypes)

    def forward(
        self, skeleton: torch.Tensor, person_roi: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        skeleton_value = self.skeleton_projection(skeleton)
        roi_value = self.roi_encoder(self.roi_adapter(person_roi))
        value = self.fusion(torch.cat((skeleton_value, roi_value), dim=1))
        return self.binary(value).squeeze(1), self.subtype(value)


class BandGatedRgbResidual(nn.Module):
    """A bounded RGB correction which is active only near an anchor score band.

    The RGB encoder learns a residual, never a replacement classifier.  Its last
    layer starts at zero so construction is observationally identical to the
    frozen anchor, while ``softplus`` keeps the learned correction scale
    non-negative.  The smooth sigmoid-difference gate avoids a discontinuity at
    the pre-registered score-band boundaries.
    """

    def __init__(
        self,
        rgb_dim: int,
        *,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        correction_cap: float = 1.0,
    ) -> None:
        super().__init__()
        if rgb_dim < 1 or hidden_dim < 4:
            raise ValueError("RGB residual dimensions 无效")
        if not 0.0 <= dropout < 1.0 or correction_cap <= 0.0:
            raise ValueError("RGB residual dropout/cap 无效")
        self.encoder = nn.Sequential(
            nn.Linear(rgb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.residual = nn.Linear(hidden_dim, 1)
        # ``softplus(0) - log(2)`` is exactly zero.  Training projects this
        # parameter back to the non-negative half-line after every step, so
        # alpha remains non-negative while retaining a non-zero derivative at
        # its required zero initialization.
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
        """Return the smooth score-band membership in [0, 1]."""
        if anchor_logits.ndim != 1 or not low < high or temperature <= 0.0:
            raise ValueError("anchor band 参数无效")
        return torch.sigmoid((anchor_logits - low) / temperature) - torch.sigmoid(
            (anchor_logits - high) / temperature
        )

    def alpha(self) -> torch.Tensor:
        return nn.functional.softplus(self.alpha_parameter) - self.alpha_parameter.new_tensor(2.0).log()

    @torch.no_grad()
    def project_alpha_(self) -> None:
        """Keep the reparameterized residual scale non-negative."""
        self.alpha_parameter.clamp_(min=0.0)

    def forward(
        self,
        rgb_features: torch.Tensor,
        anchor_logits: torch.Tensor,
        *,
        low: float,
        high: float,
        temperature: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if rgb_features.ndim != 2 or anchor_logits.shape != (rgb_features.shape[0],):
            raise ValueError("RGB features 与 anchor logits 不兼容")
        gate = self.score_band(
            anchor_logits, low=low, high=high, temperature=temperature
        )
        raw_residual = self.residual(self.encoder(rgb_features)).squeeze(1)
        bounded = self.correction_cap * torch.tanh(raw_residual)
        final = anchor_logits + self.alpha() * gate * bounded
        return final, gate, bounded


class EdgeFallF2ContextHead(nn.Module):
    """Freeze a passed F1 and learn only a zero-initialized context residual."""

    def __init__(self, base: EdgeFallF1Head, *, roi_dim: int = 192) -> None:
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.context_encoder = RoiTemporalEncoder(roi_dim, 128)
        self.binary_delta = nn.Linear(128, 1)
        self.subtype_delta = nn.Linear(128, 6)
        nn.init.zeros_(self.binary_delta.weight)
        nn.init.zeros_(self.binary_delta.bias)
        nn.init.zeros_(self.subtype_delta.weight)
        nn.init.zeros_(self.subtype_delta.bias)

    def train(self, mode: bool = True) -> EdgeFallF2ContextHead:
        super().train(mode)
        self.base.eval()
        return self

    def forward(
        self,
        skeleton: torch.Tensor,
        person_roi: torch.Tensor,
        context_roi: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        base_binary, base_subtype = self.base(skeleton, person_roi)
        context = self.context_encoder(context_roi)
        return (
            base_binary + self.binary_delta(context).squeeze(1),
            base_subtype + self.subtype_delta(context),
        )


class EdgeFallF3QualityCrossHead(nn.Module):
    """Freeze F1 and add a quality-gated, person-ROI cross-attention residual.

    The residual classifiers start at zero, so a newly constructed F3 is exactly
    equivalent to its frozen F1 base. Context ROI is deliberately absent: F2 did
    not clear its hard-negative gate.
    """

    def __init__(
        self,
        base: EdgeFallF1Head,
        skeleton_dim: int,
        *,
        roi_dim: int = 192,
        hidden_dim: int = 128,
        quality_dim: int = 5,
        subtypes: int = 6,
    ) -> None:
        super().__init__()
        if quality_dim < 1:
            raise ValueError("quality_dim 必须为正")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.skeleton_query = nn.Sequential(
            nn.Linear(skeleton_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.appearance_tokens = nn.Sequential(
            nn.Linear(roi_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=4, batch_first=True, dropout=0.0
        )
        self.quality = nn.Sequential(
            nn.Linear(quality_dim, 32), nn.GELU(), nn.Linear(32, 3)
        )
        self.residual = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.binary_delta = nn.Linear(hidden_dim, 1)
        self.subtype_delta = nn.Linear(hidden_dim, subtypes)
        self.quality_error = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.binary_delta.weight)
        nn.init.zeros_(self.binary_delta.bias)
        nn.init.zeros_(self.subtype_delta.weight)
        nn.init.zeros_(self.subtype_delta.bias)

    def train(self, mode: bool = True) -> EdgeFallF3QualityCrossHead:
        super().train(mode)
        self.base.eval()
        return self

    def forward(
        self,
        skeleton: torch.Tensor,
        person_roi: torch.Tensor,
        quality: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if quality.ndim != 2 or quality.shape[1] != 5:
            raise ValueError("quality 必须为 (B,5)")
        base_binary, base_subtype = self.base(skeleton, person_roi)
        query = self.skeleton_query(skeleton).unsqueeze(1)
        appearance = self.appearance_tokens(person_roi)
        crossed, _ = self.cross_attention(query, appearance, appearance, need_weights=False)
        gates = torch.sigmoid(self.quality(quality))
        # gates[:, 0] weights pose query, gates[:, 1] appearance attention, and
        # gates[:, 2] scales the new residual continuously rather than overriding F1.
        mixed = gates[:, :1] * query.squeeze(1) + gates[:, 1:2] * crossed.squeeze(1)
        value = self.residual(mixed) * gates[:, 2:3]
        return (
            base_binary + self.binary_delta(value).squeeze(1),
            base_subtype + self.subtype_delta(value),
            self.quality_error(value).squeeze(1),
        )
