"""Distill the frozen train-OOF three-route ensemble into one deployable TCN.

Teacher targets are computed at clip level with the exact frozen max-window and
standardized-logit fusion protocol.  Only the OF-Syn train split is used for
optimization; validation remains an epoch-selection split.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from models.clip_aggregator import aggregate_window_scores_max
from models.multiscale_multistream_tcn import (
    MultiStreamMultiScaleAttentionTCN,
    count_params,
)
from models.tcn_dataset import WindowMemmapCache, build_window_cache
from tools.build_tcn_multistream_sidecar import load_sidecar
from tools.evaluate_hard_negative_cohort import _model_from_run
from tools.evaluate_oof_multistream_fusion import _logit
from tools.train_multiscale_multistream_mil import (
    MILConfig,
    _clip_groups,
    aggregate_training_logits,
    augment_training_features,
)
from tools.train_multiscale_multistream_tcn import _activity_groups
from tools.train_tcn import (
    _append_sidecar,
    _atomic_checkpoint,
    _atomic_json,
    _canonical_json,
    _materialize,
    _run_signature,
    _sha256_file,
    evaluate_model,
    select_training_indices,
    set_deterministic,
)


@dataclass(frozen=True)
class DistillationConfig(MILConfig):
    distillation_alpha: float = 0.3
    distillation_temperature: float = 2.0

    @classmethod
    def from_json(cls, path: Path) -> DistillationConfig:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("蒸馏配置必须是 JSON 对象")
        unknown = set(payload) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"蒸馏配置包含未知字段: {sorted(unknown)}")
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
        if not 0.0 < self.distillation_alpha < 1.0:
            raise ValueError("distillation_alpha 必须位于 (0, 1)")
        if self.distillation_temperature <= 0.0:
            raise ValueError("distillation_temperature 必须为正")


def fused_teacher_logits(
    route_probabilities: np.ndarray,
    *,
    mean: np.ndarray,
    scale: np.ndarray,
    coefficients: np.ndarray,
) -> np.ndarray:
    """Apply the immutable OOF-fitted standardized-logit fusion."""
    probabilities = np.asarray(route_probabilities, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] not in {2, 3}:
        raise ValueError("教师概率必须为 (clips, 2|3)")
    if mean.shape != (probabilities.shape[1],) or scale.shape != mean.shape:
        raise ValueError("融合 mean/scale 与教师路数不匹配")
    if coefficients.shape != (probabilities.shape[1] + 1,):
        raise ValueError("融合 coefficients 与教师路数不匹配")
    if np.any(scale <= 0.0) or not np.all(np.isfinite(probabilities)):
        raise ValueError("教师融合输入无效")
    features = np.column_stack([_logit(probabilities[:, i]) for i in range(probabilities.shape[1])])
    return (((features - mean) / scale) @ coefficients[:-1] + coefficients[-1]).astype(
        np.float32
    )


def fixed_weight_teacher_logits(
    route_probabilities: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Blend frozen route logits using preregistered non-negative weights."""
    probabilities = np.asarray(route_probabilities, dtype=np.float64)
    route_weights = np.asarray(weights, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] < 1:
        raise ValueError("教师概率必须为 (clips, routes)")
    if route_weights.shape != (probabilities.shape[1],):
        raise ValueError("教师权重与路数不匹配")
    if (
        not np.all(np.isfinite(probabilities))
        or not np.all(np.isfinite(route_weights))
        or np.any(route_weights < 0.0)
        or not np.isclose(route_weights.sum(), 1.0, rtol=0.0, atol=1e-8)
    ):
        raise ValueError("固定教师融合输入无效")
    logits = np.column_stack(
        [_logit(probabilities[:, index]) for index in range(probabilities.shape[1])]
    )
    return (logits @ route_weights).astype(np.float32)


@torch.inference_mode()
def score_teacher_route(
    model: nn.Module,
    cache: WindowMemmapCache,
    sidecar: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Return exact max-window probability for every clip without materializing RAM."""
    if sidecar.shape[:2] != (
        cache.sample_count,
        int(cache.metadata["window_config"]["window_size"]),
    ):
        raise ValueError("教师 sidecar 与窗口 cache 不匹配")
    features = cache.array("features")
    clip_indices = cache.array("clip_indices")
    window_scores: list[np.ndarray] = []
    model.eval()
    for start in range(0, cache.sample_count, batch_size):
        end = min(start + batch_size, cache.sample_count)
        pose = torch.from_numpy(
            np.array(features[start:end], copy=True).reshape(
                end - start, features.shape[1], 51
            )
        )
        extra = torch.from_numpy(np.array(sidecar[start:end], copy=True))
        batch = torch.cat((pose, extra), dim=-1).to(device)
        window_scores.append(
            torch.sigmoid(model(batch)).cpu().numpy().astype(np.float32)
        )
    return aggregate_window_scores_max(
        np.concatenate(window_scores),
        np.asarray(clip_indices),
        len(cache.metadata["clips"]),
        require_all=True,
    )


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    if student_logits.shape != teacher_logits.shape or student_logits.ndim != 1:
        raise ValueError("学生与教师 clip logits 必须为同形一维张量")
    teacher_soft = torch.sigmoid(teacher_logits / temperature)
    return temperature**2 * nn.functional.binary_cross_entropy_with_logits(
        student_logits / temperature, teacher_soft
    )


def _student(config: DistillationConfig) -> MultiStreamMultiScaleAttentionTCN:
    return MultiStreamMultiScaleAttentionTCN(
        stream_channels=config.stream_channels,
        stream_output_dim=config.stream_output_dim,
        dropout=config.dropout,
        use_geometry=True,
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
        use_stage_auxiliary=config.use_stage_auxiliary,
    )


def train_epoch_distilled(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    features: torch.Tensor,
    labels: torch.Tensor,
    groups: list[np.ndarray],
    clip_labels: np.ndarray,
    teacher_clip_logits: torch.Tensor,
    *,
    device: torch.device,
    config: DistillationConfig,
    epoch: int,
) -> float:
    model.train()
    clip_order = np.random.default_rng(config.seed + epoch).permutation(len(groups))
    window_pos_weight = float((labels.numel() - labels.sum()) / labels.sum())
    clip_pos_weight = float((clip_labels.size - clip_labels.sum()) / clip_labels.sum())
    window_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(window_pos_weight, device=device)
    )
    clip_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(clip_pos_weight, device=device)
    )
    loss_sum = 0.0
    seen_clips = 0
    for start in range(0, len(groups), config.mil_clip_batch_size):
        selected = clip_order[start : start + config.mil_clip_batch_size]
        group_sizes = [int(groups[int(index)].size) for index in selected]
        window_indices = np.concatenate([groups[int(index)] for index in selected])
        batch_x = augment_training_features(
            features[window_indices].to(device),
            config,
            seed=config.seed + epoch * 1_000_003 + start,
        )
        batch_y = labels[window_indices].to(device)
        batch_clip_y = torch.from_numpy(clip_labels[selected]).to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_x)
        clip_logits = aggregate_training_logits(
            logits, group_sizes, config, epoch=epoch
        )
        hard_loss = window_criterion(logits, batch_y) + config.mil_loss_weight * clip_criterion(
            clip_logits, batch_clip_y
        )
        soft_loss = distillation_loss(
            clip_logits,
            teacher_clip_logits[selected].to(device),
            temperature=config.distillation_temperature,
        )
        loss = (1.0 - config.distillation_alpha) * hard_loss + config.distillation_alpha * soft_loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        loss_sum += float(loss.detach()) * len(selected)
        seen_clips += len(selected)
    return loss_sum / seen_clips


def train_distilled(
    *,
    config: DistillationConfig,
    train_cache: WindowMemmapCache,
    val_cache: WindowMemmapCache,
    train_sidecar: np.ndarray,
    val_sidecar: np.ndarray,
    fusion_report_path: Path,
    oof_teacher_targets_path: Path | None,
    teacher_runs: list[Path],
    teacher_checkpoints: list[Path],
    joint_pretrain_checkpoint: Path | None,
    student_init_checkpoint: Path | None = None,
    output_dir: Path,
    device_name: str,
) -> dict[str, Any]:
    config.validate()
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("请求 CUDA 训练，但 torch.cuda.is_available() 为 False")
    if len(teacher_runs) != len(teacher_checkpoints) or len(teacher_runs) not in {2, 3, 4}:
        raise ValueError("教师必须提供相同数量的 2、3 或 4 个 run/checkpoint")
    if (joint_pretrain_checkpoint is None) == (student_init_checkpoint is None):
        raise ValueError("必须且只能选择 Joint 预训练或完整学生初始化 checkpoint")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("输出目录非空；蒸馏训练不支持 resume")
    set_deterministic(config.seed)
    device = torch.device(device_name)
    report = json.loads(Path(fusion_report_path).read_text(encoding="utf-8"))
    fixed_report = report.get("protocol") == "fixed_pre_registered_logit_blend_v1"
    oof_report = report.get("protocol") == "nonnegative_l2_train_oof_stacking_v1"
    if fixed_report:
        if val_cache.metadata.get("signature_sha256") != report.get(
            "cache_signature_sha256"
        ):
            raise ValueError("固定教师融合报告与 val cache 签名不匹配")
    elif oof_report:
        if train_cache.metadata.get("signature_sha256") != report.get(
            "cache_signatures", {}
        ).get("train"):
            raise ValueError("OOF 教师报告与 train cache 签名不匹配")
        if oof_teacher_targets_path is None:
            raise ValueError("OOF 教师报告必须提供 --oof-teacher-targets")
    else:
        expected_signature = report.get("inputs", {}).get("train_cache_signature")
        if train_cache.metadata.get("signature_sha256") != expected_signature:
            raise ValueError("教师融合报告与 train cache 签名不匹配")
    fusion = report.get("fusion", {})
    route_scores: list[np.ndarray] = []
    teacher_hashes: list[dict[str, str]] = []
    for index, (run_path, checkpoint_path) in enumerate(zip(teacher_runs, teacher_checkpoints, strict=True)):
        if not oof_report:
            run = json.loads(Path(run_path).read_text(encoding="utf-8"))
            model, _, _ = _model_from_run(
                model_kind="multiscale_multistream_tcn",
                run=run,
                checkpoint=Path(checkpoint_path),
                device=device,
            )
            route_scores.append(
                score_teacher_route(
                    model,
                    train_cache,
                    train_sidecar,
                    device=device,
                    batch_size=config.batch_size,
                )
            )
        teacher_hashes.append(
            {
                "run_sha256": _sha256_file(Path(run_path)),
                "checkpoint_sha256": _sha256_file(Path(checkpoint_path)),
            }
        )
        if not oof_report:
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        print(_canonical_json({"stage": "teacher_route", "index": index, "complete": True}), flush=True)
    if oof_report:
        entries = report.get("members", [])
        if len(entries) != len(teacher_hashes):
            raise ValueError("OOF 教师报告与教师数量不匹配")
        for entry, actual in zip(entries, teacher_hashes, strict=True):
            if entry.get("run_sha256") != actual["run_sha256"] or entry.get(
                "checkpoint_sha256"
            ) != actual["checkpoint_sha256"]:
                raise ValueError("OOF 教师报告与教师哈希不匹配")
        target_entry = report.get("oof_teacher_targets", {})
        if target_entry.get("sha256") != _sha256_file(oof_teacher_targets_path):
            raise ValueError("OOF teacher target 哈希不匹配")
        with np.load(oof_teacher_targets_path, allow_pickle=False) as payload:
            required = {"clip_id", "label", "fold", "score"}
            if required - set(payload.files):
                raise ValueError("OOF teacher target 字段不完整")
            target_clip_ids = np.asarray(payload["clip_id"]).astype(str)
            target_labels = np.asarray(payload["label"])
            target_scores = np.asarray(payload["score"], dtype=np.float64)
            target_folds = np.asarray(payload["fold"])
        clips = train_cache.metadata["clips"]
        expected_clip_ids = np.asarray([str(clip["clip_id"]) for clip in clips])
        expected_labels = np.asarray([bool(clip["has_fall"]) for clip in clips])
        if not np.array_equal(target_clip_ids, expected_clip_ids):
            raise ValueError("OOF teacher target clip 顺序不匹配")
        if not np.array_equal(target_labels.astype(bool), expected_labels):
            raise ValueError("OOF teacher target label 不匹配")
        if target_scores.shape != (len(clips),) or not np.all(
            np.isfinite(target_scores) & (target_scores > 0.0) & (target_scores < 1.0)
        ):
            raise ValueError("OOF teacher target score 无效")
        if target_folds.shape != (len(clips),) or np.any(target_folds < 0):
            raise ValueError("OOF teacher target fold 无效")
        teacher_all = _logit(target_scores).astype(np.float32)
    elif fixed_report:
        entries = [report["primary"], report["expert"], *report.get("additional_experts", [])]
        if len(entries) != len(teacher_hashes):
            raise ValueError("固定融合报告与教师数量不匹配")
        for entry, actual in zip(entries, teacher_hashes, strict=True):
            if entry.get("run_sha256") != actual["run_sha256"] or entry.get(
                "checkpoint_sha256"
            ) != actual["checkpoint_sha256"]:
                raise ValueError("固定融合报告与教师哈希不匹配")
        teacher_all = fixed_weight_teacher_logits(
            np.column_stack(route_scores),
            np.asarray([entry["weight"] for entry in entries]),
        )
    else:
        teacher_all = fused_teacher_logits(
            np.column_stack(route_scores),
            mean=np.asarray(fusion["mean"]),
            scale=np.asarray(fusion["scale"]),
            coefficients=np.asarray(fusion["standardized_logit_coefficients"]),
        )

    train_indices = select_training_indices(
        train_cache.array("labels"),
        negative_ratio=config.negative_ratio,
        seed=config.seed,
        activity_groups=_activity_groups(train_cache),
        hard_negative_activities=config.hard_negative_activities,
        hard_negative_fraction=config.hard_negative_fraction,
    )
    train_x, train_y, train_clip_indices = _materialize(train_cache, train_indices)
    val_x, val_y, val_clip_indices = _materialize(val_cache, None)
    train_x = _append_sidecar(train_x, train_sidecar, train_indices)
    val_x = _append_sidecar(val_x, val_sidecar, None)
    groups, clip_labels = _clip_groups(train_clip_indices, train_cache.metadata["clips"])
    selected_clip_ids = np.unique(train_clip_indices)
    teacher_clip_logits = torch.from_numpy(teacher_all[selected_clip_ids])

    model = _student(config).to(device)
    if student_init_checkpoint is not None:
        init_payload = torch.load(
            student_init_checkpoint, map_location="cpu", weights_only=False
        )
        if not isinstance(init_payload, dict) or not isinstance(
            init_payload.get("model_state"), dict
        ):
            raise TypeError("学生初始化 checkpoint 缺少 model_state")
        model.load_state_dict(init_payload["model_state"], strict=True)
        initialization = {
            "kind": "full_student",
            "checkpoint_sha256": _sha256_file(student_init_checkpoint),
        }
    else:
        assert joint_pretrain_checkpoint is not None
        pretrain_payload = torch.load(
            joint_pretrain_checkpoint, map_location="cpu", weights_only=False
        )
        if not isinstance(pretrain_payload, dict) or not isinstance(
            pretrain_payload.get("encoder_state"), dict
        ):
            raise TypeError("Joint 预训练 checkpoint 缺少 encoder_state")
        model.streams[0].load_state_dict(pretrain_payload["encoder_state"], strict=True)
        initialization = {
            "kind": "joint_pretrain",
            "checkpoint_sha256": _sha256_file(joint_pretrain_checkpoint),
        }
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    architecture = {
        "name": "MultiStreamMultiScaleAttentionTCN",
        "streams": {"joint": 51, "bone": 32, "motion": 34, "geometry_observed": 6, "rule_features": 12, "discriminative_kinematics": config.kinematic_feature_dim},
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
        "transformer_joint_only": config.transformer_joint_only,
        "use_transition_rule_late_fusion": config.use_transition_rule_late_fusion,
        "use_discriminative_kinematics": config.use_discriminative_kinematics,
        "kinematic_feature_dim": config.kinematic_feature_dim,
        "use_gated_conv": config.use_gated_conv,
        "use_motion_guided_fusion": config.use_motion_guided_fusion,
        "causal": True,
        "initialization": initialization,
        "distillation": {
            "protocol": (
                "oof_clip_teacher_shared_aggregator_student_kd_v1"
                if oof_report
                else "fixed_clip_teacher_shared_aggregator_student_kd_v2"
            ),
            "alpha": config.distillation_alpha,
            "temperature": config.distillation_temperature,
            "fusion_report_sha256": _sha256_file(fusion_report_path),
            "teachers": teacher_hashes,
            "fit_split": "of-syn_train_only",
        },
    }
    root = Path(__file__).parent.parent
    signature_sha256, signature = _run_signature(
        config=config,
        train_cache=train_cache,
        val_cache=val_cache,
        device=device_name,
        max_train_samples=None,
        max_val_samples=None,
        protocol="multistream_clip_kd_v1",
        code_paths=(
            root / "models" / "multiscale_multistream_tcn.py",
            root / "models" / "clip_aggregator.py",
            root / "models" / "tcn_dataset.py",
            root / "tools" / "train_tcn.py",
            root / "tools" / "train_multiscale_multistream_mil.py",
            root / "tools" / "train_distilled_multistream_mil.py",
        ),
        extra_signature=architecture,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        output_dir / "run.json",
        {
            "signature_sha256": signature_sha256,
            "signature": signature,
            "architecture": architecture,
            "parameter_count": count_params(model),
            "train_samples": int(train_y.numel()),
            "train_positive": int(train_y.sum()),
            "train_clips": len(groups),
            "val_samples": int(val_y.numel()),
            "pilot": False,
        },
    )
    np.save(output_dir / "teacher_clip_logits.npy", teacher_all, allow_pickle=False)
    best_map = -math.inf
    history: list[dict[str, Any]] = []
    for epoch in range(config.epochs):
        train_loss = train_epoch_distilled(
            model,
            optimizer,
            train_x,
            train_y,
            groups,
            clip_labels,
            teacher_clip_logits,
            device=device,
            config=config,
            epoch=epoch,
        )
        metrics = evaluate_model(
            model,
            val_x,
            val_y,
            val_clip_indices,
            val_cache.metadata["clips"],
            device=device,
            batch_size=config.batch_size,
            clip_aggregation=(
                "max"
                if config.clip_aggregator == "legacy_topk_train_max_eval"
                else config.clip_aggregator
            ),
            aggregation_temperature=config.smooth_max_temperature_end,
            topk_fraction=config.mil_topk_fraction,
        )
        scheduler.step()
        record = {"epoch": epoch, "train_loss": train_loss, "learning_rate": optimizer.param_groups[0]["lr"], "val": metrics}
        with (output_dir / "history.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(_canonical_json(record) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        history.append(record)
        improved = metrics["clip_map"] > best_map
        if improved:
            best_map = metrics["clip_map"]
        checkpoint = {
            "epoch": epoch,
            "run_signature_sha256": signature_sha256,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_map": best_map,
            "metrics": record,
        }
        _atomic_checkpoint(output_dir / "last.pt", checkpoint)
        if improved:
            _atomic_checkpoint(output_dir / "best.pt", checkpoint)
        print(_canonical_json({"stage": "epoch", **record}), flush=True)
    summary = {"run_signature_sha256": signature_sha256, "epochs_completed": config.epochs, "best_clip_map": best_map, "last": history[-1], "output_dir": str(output_dir.resolve())}
    _atomic_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pose-cache-root", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--window-cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-sidecar", type=Path, required=True)
    parser.add_argument("--val-sidecar", type=Path, required=True)
    parser.add_argument("--fusion-report", type=Path, required=True)
    parser.add_argument("--oof-teacher-targets", type=Path)
    parser.add_argument("--teacher-run", type=Path, action="append", required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, action="append", required=True)
    initialization = parser.add_mutually_exclusive_group(required=True)
    initialization.add_argument("--joint-pretrain-checkpoint", type=Path)
    initialization.add_argument("--student-init-checkpoint", type=Path)
    parser.add_argument("--dataset", default="of-syn")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()
    config = DistillationConfig.from_json(args.config)
    if args.epochs is not None:
        config = replace(config, epochs=args.epochs)
        config.validate()
    caches = {
        split: build_window_cache(
            args.manifest,
            args.pose_cache_root,
            args.audit_report,
            args.window_cache_root,
            dataset=args.dataset,
            split=split,
            window_size=config.window_size,
            stride=config.stride,
            min_observed_frames=config.min_observed_frames,
        )
        for split in ("train", "val")
    }
    summary = train_distilled(
        config=config,
        train_cache=caches["train"],
        val_cache=caches["val"],
        train_sidecar=load_sidecar(args.train_sidecar, caches["train"]),
        val_sidecar=load_sidecar(args.val_sidecar, caches["val"]),
        fusion_report_path=args.fusion_report,
        oof_teacher_targets_path=args.oof_teacher_targets,
        teacher_runs=args.teacher_run,
        teacher_checkpoints=args.teacher_checkpoint,
        joint_pretrain_checkpoint=args.joint_pretrain_checkpoint,
        student_init_checkpoint=args.student_init_checkpoint,
        output_dir=args.output_dir,
        device_name=args.device,
    )
    print(_canonical_json({"stage": "complete", **summary}), flush=True)


if __name__ == "__main__":
    main()
