"""训练四流多尺度注意力 TCN 的 train/val 预算试验。"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from models.multiscale_multistream_tcn import MultiStreamMultiScaleAttentionTCN
from models.tcn_dataset import WindowMemmapCache, build_window_cache
from tools.build_tcn_multistream_sidecar import load_sidecar
from tools.train_tcn import TrainingConfig, _canonical_json, train


def _activity_groups(cache: WindowMemmapCache) -> np.ndarray:
    """Return the OF-Syn activity prefix for every cached training window."""
    clips = cache.metadata.get("clips")
    if not isinstance(clips, list):
        raise TypeError("窗口缓存缺少 clips metadata")
    activities: list[str] = []
    for clip in clips:
        clip_id = clip.get("clip_id")
        if not isinstance(clip_id, str) or "/" not in clip_id:
            raise ValueError("困难负例配额要求 OF-Syn activity/clip_id 格式")
        activities.append(clip_id.split("/", maxsplit=1)[0])
    clip_indices = np.asarray(cache.array("clip_indices"), dtype=np.int64)
    if np.any(clip_indices < 0) or np.any(clip_indices >= len(activities)):
        raise ValueError("窗口缓存的 clip_indices 超出 metadata 范围")
    return np.asarray(activities, dtype=object)[clip_indices]


@dataclass(frozen=True)
class MultiStreamConfig(TrainingConfig):
    stream_channels: int = 60
    stream_output_dim: int = 256
    stream_lstm_layers: int = 0
    use_rule_features: bool = False
    use_pifr_features: bool = False
    use_stgcn_joint: bool = False
    use_stgcn_residual_joint: bool = False
    use_transformer_encoder: bool = False
    transformer_joint_only: bool = False
    use_transition_rule_late_fusion: bool = False
    use_discriminative_kinematics: bool = False
    kinematic_feature_dim: int = 15
    use_gated_conv: bool = False
    use_motion_guided_fusion: bool = False
    motion_validity_mask: bool = False
    per_joint_mask_correction: bool = False
    use_stream_interaction: bool = False
    use_stage_auxiliary: bool = False
    pose_jitter_std: float = 0.0
    joint_dropout_probability: float = 0.0
    horizontal_flip_probability: float = 0.0
    temporal_augmentation_probability: float = 0.0
    temporal_max_frame_fraction: float = 0.0
    joint_occlusion_probability: float = 0.0
    bbox_jitter_std: float = 0.0
    track_break_probability: float = 0.0
    track_break_max_fraction: float = 0.0
    temporal_speed_probability: float = 0.0
    temporal_speed_min: float = 1.0
    temporal_speed_max: float = 1.0
    use_physics_auxiliary: bool = False

    @classmethod
    def from_json(cls, path: Path) -> MultiStreamConfig:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("多流配置必须是 JSON 对象")
        unknown = set(payload) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"多流配置包含未知字段: {sorted(unknown)}")
        if "channels" in payload:
            payload["channels"] = tuple(payload["channels"])
        if "hard_negative_activities" in payload:
            payload["hard_negative_activities"] = tuple(
                payload["hard_negative_activities"]
            )
        config = cls(**payload)
        config.validate()
        return config

    def validate(self) -> None:
        super().validate()
        if self.window_size not in {16, 32}:
            raise ValueError("本试验仅支持 16 或 32 帧窗口")
        if self.stream_channels < 4 or self.stream_channels % 4:
            raise ValueError("stream_channels 必须是不小于 4 的 4 的倍数")
        if self.stream_output_dim < 1:
            raise ValueError("stream_output_dim 必须为正整数")
        if self.stream_lstm_layers < 0:
            raise ValueError("stream_lstm_layers 不能为非负整数")
        if not isinstance(self.use_rule_features, bool):
            raise TypeError("use_rule_features 必须是布尔值")
        if not isinstance(self.use_pifr_features, bool):
            raise TypeError("use_pifr_features 必须是布尔值")
        if not isinstance(self.use_stgcn_joint, bool):
            raise TypeError("use_stgcn_joint 必须是布尔值")
        if not isinstance(self.use_stgcn_residual_joint, bool):
            raise TypeError("use_stgcn_residual_joint 必须是布尔值")
        if self.use_stgcn_joint and self.use_stgcn_residual_joint:
            raise ValueError("ST-GCN replacement 与 residual 模式不能同时启用")
        if not isinstance(self.use_transformer_encoder, bool):
            raise TypeError("use_transformer_encoder 必须是布尔值")
        if not isinstance(self.transformer_joint_only, bool):
            raise TypeError("transformer_joint_only 必须是布尔值")
        if self.transformer_joint_only and not self.use_transformer_encoder:
            raise ValueError("transformer_joint_only 需要 use_transformer_encoder")
        if not isinstance(self.use_transition_rule_late_fusion, bool):
            raise TypeError("use_transition_rule_late_fusion 必须是布尔值")
        if not isinstance(self.use_discriminative_kinematics, bool):
            raise TypeError("use_discriminative_kinematics 必须是布尔值")
        if self.kinematic_feature_dim not in {12, 15}:
            raise ValueError("kinematic_feature_dim 必须为 12 或 15")
        if not isinstance(self.use_gated_conv, bool):
            raise TypeError("use_gated_conv 必须是布尔值")
        if not isinstance(self.use_motion_guided_fusion, bool):
            raise TypeError("use_motion_guided_fusion 必须是布尔值")
        if not isinstance(self.motion_validity_mask, bool):
            raise TypeError("motion_validity_mask 必须是布尔值")
        if not isinstance(self.per_joint_mask_correction, bool):
            raise TypeError("per_joint_mask_correction 必须是布尔值")
        if not isinstance(self.use_stream_interaction, bool):
            raise TypeError("use_stream_interaction 必须是布尔值")
        if not isinstance(self.use_stage_auxiliary, bool):
            raise TypeError("use_stage_auxiliary 必须是布尔值")
        if not isinstance(self.use_physics_auxiliary, bool):
            raise TypeError("use_physics_auxiliary 必须是布尔值")
        for name, value in (
            ("pose_jitter_std", self.pose_jitter_std),
            ("joint_dropout_probability", self.joint_dropout_probability),
            ("horizontal_flip_probability", self.horizontal_flip_probability),
            ("temporal_augmentation_probability", self.temporal_augmentation_probability),
            ("temporal_max_frame_fraction", self.temporal_max_frame_fraction),
            ("joint_occlusion_probability", self.joint_occlusion_probability),
            ("bbox_jitter_std", self.bbox_jitter_std),
            ("track_break_probability", self.track_break_probability),
            ("track_break_max_fraction", self.track_break_max_fraction),
            ("temporal_speed_probability", self.temporal_speed_probability),
        ):
            if not isinstance(value, (float, int)) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} 必须位于 [0, 1]")
        if not 0.5 <= self.temporal_speed_min <= 2.0:
            raise ValueError("temporal_speed_min 必须位于 [0.5, 2.0]")
        if not self.temporal_speed_min <= self.temporal_speed_max <= 2.0:
            raise ValueError("temporal_speed_max 必须不小于 min 且不超过 2.0")
        if self.track_break_probability and self.track_break_max_fraction <= 0.0:
            raise ValueError("track break 启用时 max fraction 必须为正")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pose-cache-root", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--window-cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", default="of-syn")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--train-sidecar", type=Path)
    parser.add_argument("--val-sidecar", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = MultiStreamConfig.from_json(args.config)
    if args.epochs is not None:
        config = replace(config, epochs=args.epochs)
        config.validate()
    caches: dict[str, WindowMemmapCache] = {}
    for split in ("train", "val"):
        print(_canonical_json({"stage": "window_cache", "split": split}), flush=True)
        caches[split] = build_window_cache(
            args.manifest,
            args.pose_cache_root,
            args.audit_report,
            args.window_cache_root,
            dataset=args.dataset,
            split=split,
            window_size=config.window_size,
            stride=config.stride,
            min_observed_frames=config.min_observed_frames,
            progress=lambda state, split=split: print(
                _canonical_json(
                    {"stage": "window_cache_progress", "split": split, **state}
                ),
                flush=True,
            ),
        )
    use_geometry = args.train_sidecar is not None or args.val_sidecar is not None
    if use_geometry and (args.train_sidecar is None or args.val_sidecar is None):
        raise ValueError("Geometry 模式必须同时传入 --train-sidecar 和 --val-sidecar")
    if (config.use_rule_features or config.use_pifr_features) and not use_geometry:
        raise ValueError("规则/PIFR 流需要同时传入 Geometry sidecar")
    sidecar_arrays = None
    if use_geometry:
        sidecar_arrays = (
            load_sidecar(args.train_sidecar, caches["train"]),
            load_sidecar(args.val_sidecar, caches["val"]),
        )
    root = Path(__file__).parent.parent
    architecture = {
        "name": "MultiStreamMultiScaleAttentionTCN",
        "streams": {
            "joint": 51,
            "bone": 32,
            "motion": 34,
            "geometry_observed": 6,
            **({"rule_features": 12} if config.use_rule_features else {}),
            **({"pifr_features": 9} if config.use_pifr_features else {}),
            **(
                {"discriminative_kinematics": config.kinematic_feature_dim}
                if config.use_discriminative_kinematics
                else {}
            ),
        }
        if use_geometry
        else {"joint": 51, "bone": 32, "motion": 34, "accel": 34},
        "dilations": [1, 2, 4, 8],
        "kernels": [3, 5, 7],
        "attention_heads": 4,
        "stream_channels": config.stream_channels,
        "stream_output_dim": config.stream_output_dim,
        "stream_lstm_layers": config.stream_lstm_layers,
        "use_rule_features": config.use_rule_features,
        "use_pifr_features": config.use_pifr_features,
        "use_stgcn_joint": config.use_stgcn_joint,
        "use_stgcn_residual_joint": config.use_stgcn_residual_joint,
        "use_transformer_encoder": config.use_transformer_encoder,
        "use_transition_rule_late_fusion": config.use_transition_rule_late_fusion,
        "use_discriminative_kinematics": config.use_discriminative_kinematics,
        "kinematic_feature_dim": config.kinematic_feature_dim,
        "use_gated_conv": config.use_gated_conv,
        "use_motion_guided_fusion": config.use_motion_guided_fusion,
        "classifier": [
            config.stream_output_dim
            if config.use_motion_guided_fusion
            else config.stream_output_dim
            * (
                4
                + int(config.use_rule_features or config.use_pifr_features)
                + int(config.use_discriminative_kinematics)
            ),
            2,
        ],
        "causal": True,
    }
    summary = train(
        config=config,
        train_cache=caches["train"],
        val_cache=caches["val"],
        output_dir=args.output_dir,
        device_name=args.device,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        model_factory=lambda: MultiStreamMultiScaleAttentionTCN(
            stream_channels=config.stream_channels,
            stream_output_dim=config.stream_output_dim,
            dropout=config.dropout,
            use_geometry=use_geometry,
            stream_lstm_layers=config.stream_lstm_layers,
            use_rule_features=config.use_rule_features,
            use_pifr_features=config.use_pifr_features,
            use_stgcn_joint=config.use_stgcn_joint,
            use_stgcn_residual_joint=config.use_stgcn_residual_joint,
            use_transformer_encoder=config.use_transformer_encoder,
            transformer_joint_only=config.transformer_joint_only,
            use_transition_rule_late_fusion=config.use_transition_rule_late_fusion,
            use_discriminative_kinematics=config.use_discriminative_kinematics,
            kinematic_feature_dim=config.kinematic_feature_dim,
            use_gated_conv=config.use_gated_conv,
            use_motion_guided_fusion=config.use_motion_guided_fusion,
            motion_validity_mask=config.motion_validity_mask,
            per_joint_mask_correction=config.per_joint_mask_correction,
            use_stream_interaction=config.use_stream_interaction,
            use_stage_auxiliary=config.use_stage_auxiliary,
        ),
        protocol="multiscale_multistream_attention_tcn_v1",
        code_paths=(
            root / "models" / "multiscale_multistream_tcn.py",
            root / "models" / "tcn_multistream.py",
            root / "models" / "tcn_window.py",
            root / "models" / "tcn_dataset.py",
            root / "tools" / "train_tcn.py",
            root / "tools" / "train_multiscale_multistream_tcn.py",
        ),
        run_metadata=architecture,
        sidecar_arrays=sidecar_arrays,
        activity_groups=(
            _activity_groups(caches["train"])
            if config.hard_negative_activities
            else None
        ),
    )
    print(_canonical_json({"stage": "complete", **summary}), flush=True)


if __name__ == "__main__":
    main()
