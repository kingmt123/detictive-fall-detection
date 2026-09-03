"""在固定、非 test 的困难负例队列上诊断已训练模型的误报。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from models.dual_expert_skeleton import DualExpertSkeletonFallDetector
from models.lstm import (
    POSE51_FEATURES,
    POSE51_VELOCITY34_FEATURES,
    FallLSTM,
    add_velocity_features,
    feature_dimension,
)
from models.multiscale_multistream_tcn import MultiStreamMultiScaleAttentionTCN
from models.residual_expert_fusion import ResidualExpertFusion
from models.tcn import FallTCN
from models.tcn_dataset import WindowMemmapCache, load_window_cache
from models.transition_residual_reranker import (
    TemporalTransitionResidualReranker,
    TransitionResidualReranker,
)
from models.tri_expert_skeleton import TriExpertSkeletonFallDetector
from tools.build_tcn_multistream_sidecar import load_sidecar
from tools.train_tcn import _atomic_json


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"不是合法 JSON: {path}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"JSON 对象预期: {path}")
    return value


def _model_from_run(
    *, model_kind: str, run: dict[str, Any], checkpoint: Path, device: torch.device
) -> tuple[torch.nn.Module, str, float]:
    config = run.get("signature", {}).get("config", {})
    if not isinstance(config, dict):
        raise TypeError("run.json 缺少 signature.config")
    input_features = config.get("input_features")
    if input_features is None:
        input_features = (
            POSE51_FEATURES if model_kind == "tcn" else POSE51_VELOCITY34_FEATURES
        )
    if model_kind == "tcn":
        model = FallTCN(
            in_dim=feature_dimension(input_features),
            channels=tuple(config["channels"]),
            kernel=int(config["kernel"]),
            dropout=float(config["dropout"]),
        )
    elif model_kind == "lstm":
        model = FallLSTM(
            input_dim=feature_dimension(input_features),
            hidden_size=int(config["hidden_size"]),
            num_layers=int(config["num_layers"]),
            dropout=float(config["dropout"]),
            input_features=input_features,
        )
    elif model_kind in {"multiscale_multistream_tcn", "mixed_fallvision_multistream_tcn"}:
        architecture = run.get("architecture")
        if not isinstance(architecture, dict):
            raise TypeError("多流 run.json 缺少 architecture")
        if model_kind == "mixed_fallvision_multistream_tcn":
            architecture = architecture.get("base_architecture")
            if not isinstance(architecture, dict):
                raise TypeError("混合训练 run.json 缺少 base_architecture")
        streams = architecture.get("streams", {})
        if not isinstance(streams, dict):
            raise TypeError("多流 run.json architecture.streams 无效")
        model = MultiStreamMultiScaleAttentionTCN(
            stream_channels=int(architecture["stream_channels"]),
            stream_output_dim=int(architecture["stream_output_dim"]),
            dropout=float(config["dropout"]),
            use_geometry=int(streams.get("geometry_observed", 0)) == 6,
            stream_lstm_layers=int(architecture.get("stream_lstm_layers", 0)),
            use_rule_features=int(streams.get("rule_features", 0)) == 12,
            use_pifr_features=int(streams.get("pifr_features", 0)) == 9,
            use_stgcn_joint=bool(architecture.get("use_stgcn_joint", False)),
            use_stgcn_residual_joint=bool(
                architecture.get("use_stgcn_residual_joint", False)
            ),
            use_transformer_encoder=bool(
                architecture.get("use_transformer_encoder", False)
            ),
            transformer_joint_only=bool(
                architecture.get("transformer_joint_only", False)
            ),
            use_transition_rule_late_fusion=bool(
                architecture.get("use_transition_rule_late_fusion", False)
            ),
            use_discriminative_kinematics=bool(
                architecture.get("use_discriminative_kinematics", False)
            ),
            kinematic_feature_dim=int(streams.get("discriminative_kinematics", 15)),
            use_gated_conv=bool(architecture.get("use_gated_conv", False)),
            use_motion_guided_fusion=bool(
                architecture.get("use_motion_guided_fusion", False)
            ),
            motion_validity_mask=bool(architecture.get("motion_validity_mask", False)),
            use_stream_interaction=bool(architecture.get("use_stream_interaction", False)),
            use_stage_auxiliary=bool(architecture.get("use_stage_auxiliary", False)),
            use_physics_auxiliary=bool(architecture.get("use_physics_auxiliary", False)),
        )
    elif model_kind == "residual_expert_fusion":
        architecture = run.get("architecture")
        if not isinstance(architecture, dict):
            raise TypeError("Residual fusion run.json 缺少 architecture")
        base_architecture = architecture.get("base_architecture")
        if not isinstance(base_architecture, dict):
            raise TypeError("Residual fusion run.json 缺少 base_architecture")
        streams = base_architecture.get("streams", {})
        if not isinstance(streams, dict):
            raise TypeError("Residual fusion base streams 无效")
        base = MultiStreamMultiScaleAttentionTCN(
            stream_channels=int(base_architecture["stream_channels"]),
            stream_output_dim=int(base_architecture["stream_output_dim"]),
            dropout=float(config["dropout"]),
            use_geometry=int(streams.get("geometry_observed", 0)) == 6,
            stream_lstm_layers=int(base_architecture.get("stream_lstm_layers", 0)),
            use_rule_features=int(streams.get("rule_features", 0)) == 12,
            use_pifr_features=int(streams.get("pifr_features", 0)) == 9,
            use_stgcn_joint=bool(base_architecture.get("use_stgcn_joint", False)),
            use_stgcn_residual_joint=bool(
                base_architecture.get("use_stgcn_residual_joint", False)
            ),
            use_transformer_encoder=bool(
                base_architecture.get("use_transformer_encoder", False)
            ),
            transformer_joint_only=bool(
                base_architecture.get("transformer_joint_only", False)
            ),
            use_transition_rule_late_fusion=bool(
                base_architecture.get("use_transition_rule_late_fusion", False)
            ),
            use_discriminative_kinematics=bool(
                base_architecture.get("use_discriminative_kinematics", False)
            ),
            kinematic_feature_dim=int(streams.get("discriminative_kinematics", 15)),
            use_gated_conv=bool(base_architecture.get("use_gated_conv", False)),
            use_motion_guided_fusion=bool(
                base_architecture.get("use_motion_guided_fusion", False)
            ),
            use_stage_auxiliary=bool(
                base_architecture.get("use_stage_auxiliary", False)
            ),
        )
        model = ResidualExpertFusion(
            base,
            expert_channels=int(architecture["expert_channels"]),
            expert_output_dim=int(architecture["expert_output_dim"]),
            transformer_layers=int(architecture["transformer_layers"]),
            dropout=float(config["dropout"]),
            correction_scale=float(architecture["correction_scale"]),
            max_frames=max(32, int(config["window_size"])),
            use_global_gate=bool(architecture.get("use_global_gate", False)),
            global_gate_initial_logit=float(
                architecture.get("global_gate_initial_logit", -2.0)
            ),
        )
    elif model_kind in {
        "transition_residual_reranker",
        "temporal_transition_residual_reranker",
    }:
        architecture = run.get("architecture")
        if not isinstance(architecture, dict):
            raise TypeError("重排 run.json 缺少 architecture")
        base_architecture = architecture.get("base_architecture")
        if not isinstance(base_architecture, dict):
            raise TypeError("重排 run.json 缺少 base_architecture")
        streams = base_architecture.get("streams", {})
        if not isinstance(streams, dict):
            raise TypeError("重排 base streams 无效")
        base = MultiStreamMultiScaleAttentionTCN(
            stream_channels=int(base_architecture["stream_channels"]),
            stream_output_dim=int(base_architecture["stream_output_dim"]),
            dropout=float(config["dropout"]),
            use_geometry=int(streams.get("geometry_observed", 0)) == 6,
            stream_lstm_layers=int(base_architecture.get("stream_lstm_layers", 0)),
            use_rule_features=int(streams.get("rule_features", 0)) == 12,
            use_pifr_features=int(streams.get("pifr_features", 0)) == 9,
            use_stgcn_joint=bool(base_architecture.get("use_stgcn_joint", False)),
            use_stgcn_residual_joint=bool(
                base_architecture.get("use_stgcn_residual_joint", False)
            ),
            use_transformer_encoder=bool(
                base_architecture.get("use_transformer_encoder", False)
            ),
            transformer_joint_only=bool(
                base_architecture.get("transformer_joint_only", False)
            ),
            use_transition_rule_late_fusion=bool(
                base_architecture.get("use_transition_rule_late_fusion", False)
            ),
            use_discriminative_kinematics=bool(
                base_architecture.get("use_discriminative_kinematics", False)
            ),
            kinematic_feature_dim=int(streams.get("discriminative_kinematics", 15)),
            use_gated_conv=bool(base_architecture.get("use_gated_conv", False)),
            use_motion_guided_fusion=bool(
                base_architecture.get("use_motion_guided_fusion", False)
            ),
            use_stage_auxiliary=bool(
                base_architecture.get("use_stage_auxiliary", False)
            ),
        )
        reranker_kwargs: dict[str, Any] = {
            "hidden_dim": int(architecture["hidden_dim"]),
            "dropout": float(architecture["dropout"]),
            "correction_scale": float(architecture["correction_scale"]),
            "gate_center": float(architecture["gate_center"]),
            "gate_temperature": float(architecture["gate_temperature"]),
            "auxiliary_classes": len(architecture["auxiliary_classes"]),
        }
        if model_kind == "temporal_transition_residual_reranker":
            model = TemporalTransitionResidualReranker(
                base,
                temporal_hidden_dim=int(architecture["temporal_transition_hidden_dim"]),
                **reranker_kwargs,
            )
        else:
            model = TransitionResidualReranker(base, **reranker_kwargs)
    elif model_kind == "tri_expert_skeleton":
        architecture = run.get("architecture")
        if not isinstance(architecture, dict):
            raise TypeError("三专家 run.json 缺少 architecture")
        topology = architecture.get("infogcn_topology")
        if not isinstance(topology, dict):
            raise TypeError("三专家 run.json 缺少 InfoGCN topology")
        model = TriExpertSkeletonFallDetector(
            channels=int(architecture["channels"]),
            expert_dim=int(architecture["expert_dim"]),
            dropout=float(config["dropout"]),
            dynamic_layers=int(topology["dynamic_knn_last_layers"]),
            knn_k=int(topology["knn_k"]),
            heads=int(architecture["attention_heads"]),
        )
    elif model_kind == "dual_expert_skeleton":
        architecture = run.get("architecture")
        if not isinstance(architecture, dict):
            raise TypeError("双专家 run.json 缺少 architecture")
        topology = architecture.get("infogcn_topology") or {}
        if not isinstance(topology, dict):
            raise TypeError("双专家 run.json InfoGCN topology 无效")
        model = DualExpertSkeletonFallDetector(
            variant=str(architecture["variant"]),
            channels=int(architecture["channels"]),
            expert_dim=int(architecture["expert_dim"]),
            dropout=float(config["dropout"]),
            dynamic_layers=int(topology.get("dynamic_knn_last_layers", 2)),
            knn_k=int(topology.get("knn_k", 4)),
            heads=int(architecture["attention_heads"]),
        )
    else:
        raise ValueError(
            "model_kind 必须为 tcn、lstm、multiscale_multistream_tcn、mixed_fallvision_multistream_tcn、transition_residual_reranker、temporal_transition_residual_reranker "
            "、residual_expert_fusion、tri_expert_skeleton 或 dual_expert_skeleton"
        )
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"])
    point = payload.get("metrics", {}).get("val", {}).get("clip_r90_point")
    if not isinstance(point, dict) or "threshold" not in point:
        raise ValueError("best checkpoint 缺少验证集 R90 阈值")
    return model.to(device).eval(), input_features, float(point["threshold"])


def _cohort_indices(cache: WindowMemmapCache, clip_ids: set[str]) -> np.ndarray:
    clips = cache.metadata.get("clips")
    if not isinstance(clips, list):
        raise TypeError("窗口缓存缺少 clips metadata")
    matching = [
        index for index, clip in enumerate(clips) if clip["clip_id"] in clip_ids
    ]
    if len(matching) != len(clip_ids):
        found = {clips[index]["clip_id"] for index in matching}
        missing = sorted(clip_ids - found)
        raise ValueError(f"困难负例不在窗口缓存中: {missing[:3]}")
    return np.flatnonzero(np.isin(cache.array("clip_indices"), matching))


@torch.inference_mode()
def score_cohort(
    *,
    model: torch.nn.Module,
    model_kind: str,
    input_features: str,
    cache: WindowMemmapCache,
    cohort: dict[str, Any],
    threshold: float,
    device: torch.device,
    batch_size: int,
    sidecar: np.ndarray | None = None,
) -> dict[str, Any]:
    """在指定阈值下汇总 cohort 的 clip max 分数和分组误报率。"""
    by_activity = cohort.get("clips_by_activity")
    if not isinstance(by_activity, dict) or not by_activity:
        raise ValueError("cohort 缺少 clips_by_activity")
    all_ids = {clip_id for values in by_activity.values() for clip_id in values}
    if len(all_ids) != int(cohort.get("clip_count", -1)):
        raise ValueError("cohort clip_count 与 clips_by_activity 不一致")
    indices = _cohort_indices(cache, all_ids)
    features = cache.array("features")
    if getattr(model, "use_geometry", False) or model_kind in {
        "tri_expert_skeleton",
        "dual_expert_skeleton",
    }:
        if sidecar is None:
            raise ValueError("几何多流模型需要通过 --sidecar 提供经审计的 6D sidecar")
        if sidecar.shape[:2] != features.shape[:2] or sidecar.shape[2] != 6:
            raise ValueError("sidecar 与窗口缓存的 shape 不兼容")
    clip_indices = np.asarray(cache.array("clip_indices")[indices], dtype=np.int64)
    probabilities: list[np.ndarray] = []
    for start in range(0, indices.size, batch_size):
        selected_indices = indices[start : start + batch_size]
        batch = torch.from_numpy(
            np.asarray(features[selected_indices], dtype=np.float32)
        )
        if sidecar is not None:
            geometry = np.array(sidecar[selected_indices], dtype=np.float32, copy=True)
            batch = torch.cat((batch.flatten(2), torch.from_numpy(geometry)), dim=-1)
        if model_kind == "tcn" and input_features == POSE51_VELOCITY34_FEATURES:
            batch = add_velocity_features(batch)
        probabilities.append(torch.sigmoid(model(batch.to(device))).cpu().numpy())
    scores = np.full(len(cache.metadata["clips"]), -np.inf, dtype=np.float32)
    np.maximum.at(scores, clip_indices, np.concatenate(probabilities))
    clip_to_score = {
        clip["clip_id"]: float(scores[index])
        for index, clip in enumerate(cache.metadata["clips"])
        if clip["clip_id"] in all_ids
    }
    groups: dict[str, Any] = {}
    for activity, clip_ids in by_activity.items():
        values = np.asarray(
            [clip_to_score[clip_id] for clip_id in clip_ids], dtype=np.float32
        )
        false_positives = int(np.count_nonzero(values >= threshold))
        groups[activity] = {
            "clip_count": int(values.size),
            "false_positive_count": false_positives,
            "false_positive_rate": false_positives / values.size,
            "score_mean": float(values.mean()),
            "score_p95": float(np.quantile(values, 0.95)),
        }
    all_values = np.asarray(list(clip_to_score.values()), dtype=np.float32)
    total_fp = int(np.count_nonzero(all_values >= threshold))
    return {
        "threshold": threshold,
        "clip_count": int(all_values.size),
        "false_positive_count": total_fp,
        "false_positive_rate": total_fp / all_values.size,
        "score_mean": float(all_values.mean()),
        "groups": groups,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--model-kind",
        choices=(
            "tcn",
            "lstm",
            "multiscale_multistream_tcn",
            "mixed_fallvision_multistream_tcn",
            "transition_residual_reranker",
            "temporal_transition_residual_reranker",
            "residual_expert_fusion",
            "tri_expert_skeleton",
            "dual_expert_skeleton",
        ),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--sidecar",
        type=Path,
        help="几何多流模型所需的经审计 bbox/observed sidecar 目录",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"输出已存在，拒绝覆盖: {args.output}")
    cohort = _load_json(args.cohort)
    if cohort.get("split") == "test":
        raise ValueError("困难负例诊断禁止使用 test split")
    cache = load_window_cache(args.cache)
    run = _load_json(args.run)
    device = torch.device(args.device)
    model, input_features, threshold = _model_from_run(
        model_kind=args.model_kind,
        run=run,
        checkpoint=args.checkpoint,
        device=device,
    )
    sidecar = load_sidecar(args.sidecar, cache) if args.sidecar else None
    report = score_cohort(
        model=model,
        model_kind=args.model_kind,
        input_features=input_features,
        cache=cache,
        cohort=cohort,
        threshold=threshold,
        device=device,
        batch_size=args.batch_size,
        sidecar=sidecar,
    )
    report.update(
        {
            "protocol": "hard_negative_cohort_score_v1",
            "model_kind": args.model_kind,
            "input_features": input_features,
            "cohort_path": str(args.cohort.resolve()),
            "cohort_sha256": _sha256_file(args.cohort),
            "checkpoint_sha256": _sha256_file(args.checkpoint),
            "run_signature_sha256": run.get("signature_sha256"),
            "cache_signature_sha256": cache.metadata.get("signature_sha256"),
        }
    )
    _atomic_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
