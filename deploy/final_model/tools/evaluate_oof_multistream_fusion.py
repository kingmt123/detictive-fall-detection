"""Train-only cross-fitted logit fusion for two or three multistream designs.

The fusion coefficients are fitted only from out-of-fold (OOF) clip scores.
Validation is scored once after the coefficients are frozen.  Test splits are
explicitly rejected by the cache loader and by the checks below.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from eval.metrics import competition_map
from models.clip_aggregator import aggregate_window_scores_max
from models.multiscale_multistream_tcn import MultiStreamMultiScaleAttentionTCN
from models.tcn_dataset import WindowMemmapCache, load_window_cache
from tools.build_tcn_multistream_sidecar import load_sidecar
from tools.evaluate_hard_negative_cohort import _model_from_run
from tools.train_long_context_event_oracle import group_fold, template_group
from tools.train_multiscale_multistream_mil import (
    MILConfig,
    _clip_groups,
    train_epoch_mil,
)
from tools.train_multiscale_multistream_tcn import _activity_groups
from tools.train_tcn import (
    _append_sidecar,
    _atomic_json,
    _materialize,
    predict_probabilities,
    select_training_indices,
    set_deterministic,
)

_EPS = 1e-5
_L2 = 1e-3


def _stratified_clip_folds(clips: list[dict[str, Any]], folds: int) -> np.ndarray:
    """Assign every clip to a deterministic, label-stratified fold."""
    if folds < 2:
        raise ValueError("OOF 至少需要两个 folds")
    assignment = np.full(len(clips), -1, dtype=np.int16)
    for label in (False, True):
        indices = [
            index
            for index, clip in enumerate(clips)
            if bool(clip.get("has_fall")) is label
        ]
        if len(indices) < folds:
            raise ValueError("每个标签至少需要与 folds 数相同的 clips")
        ordered = sorted(
            indices,
            key=lambda index: hashlib.sha256(
                str(clips[index].get("clip_id", "")).encode()
            ).digest(),
        )
        for offset, index in enumerate(ordered):
            assignment[index] = offset % folds
    if np.any(assignment < 0):
        raise ValueError("clip 缺少 bool has_fall")
    return assignment


def _template_grouped_folds(
    clips: list[dict[str, Any]], folds: int, *, seed: int
) -> np.ndarray:
    """Assign every OF-Syn template group to one fold without group leakage."""
    if folds < 2:
        raise ValueError("OOF 至少需要两个 folds")
    groups = [template_group(str(clip.get("clip_id", ""))) for clip in clips]
    assignment = np.asarray(
        [group_fold(group, folds=folds, seed=seed) for group in groups], dtype=np.int16
    )
    if set(assignment.tolist()) != set(range(folds)):
        raise ValueError("模板分组未覆盖全部 folds")
    for fold in range(folds):
        labels = [bool(clip.get("has_fall")) for index, clip in enumerate(clips) if assignment[index] == fold]
        if not labels or all(labels) or not any(labels):
            raise ValueError(f"fold {fold} 未同时覆盖正负 clips")
    for group in set(groups):
        group_folds = {
            assignment[index] for index, value in enumerate(groups) if value == group
        }
        if len(group_folds) != 1:
            raise AssertionError("template group 跨 OOF fold 泄漏")
    return assignment


def _logit(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    return np.log(
        np.clip(probabilities, _EPS, 1.0 - _EPS)
        / np.clip(1.0 - probabilities, _EPS, 1.0)
    )


def _fit_logistic(
    features: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a deterministic L2-regularized, class-balanced logistic fusion."""
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if features.ndim != 2 or features.shape[1] not in {2, 3}:
        raise ValueError("融合需要两个或三个 OOF logit 特征")
    if labels.shape != (features.shape[0],) or not np.all(
        (labels == 0) | (labels == 1)
    ):
        raise ValueError("融合标签必须为二分类一维数组")
    positives = float(labels.sum())
    if positives == 0.0 or positives == labels.size:
        raise ValueError("融合 OOF 必须同时含正负 clips")
    mean = features.mean(axis=0)
    scale = features.std(axis=0).clip(min=1e-6)
    design = np.column_stack(((features - mean) / scale, np.ones(features.shape[0])))
    sample_weight = np.where(labels == 1, (labels.size - positives) / positives, 1.0)
    feature_count = features.shape[1]
    coefficients = np.zeros(feature_count + 1, dtype=np.float64)
    penalty = np.diag([_L2] * feature_count + [0.0])
    for _ in range(40):
        logits = np.clip(design @ coefficients, -40.0, 40.0)
        probability = 1.0 / (1.0 + np.exp(-logits))
        gradient = (
            design.T @ (sample_weight * (probability - labels)) + penalty @ coefficients
        )
        curvature = sample_weight * probability * (1.0 - probability)
        hessian = design.T @ (design * curvature[:, None]) + penalty
        step = np.linalg.solve(hessian, gradient)
        coefficients -= step
        if np.linalg.vector_norm(step) < 1e-8:
            break
    return (
        coefficients.astype(np.float32),
        mean.astype(np.float32),
        scale.astype(np.float32),
    )


def _model(config: MILConfig) -> MultiStreamMultiScaleAttentionTCN:
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
        motion_validity_mask=config.motion_validity_mask,
        per_joint_mask_correction=config.per_joint_mask_correction,
        use_stream_interaction=config.use_stream_interaction,
        use_stage_auxiliary=config.use_stage_auxiliary,
    )


def _clip_score_array(
    probabilities: np.ndarray,
    clip_indices: np.ndarray,
    clip_count: int,
    *,
    require_all: bool = True,
) -> np.ndarray:
    if probabilities.shape != clip_indices.shape:
        raise ValueError("窗口概率与 clip_indices 不匹配")
    try:
        return aggregate_window_scores_max(
            probabilities, clip_indices, clip_count, require_all=require_all
        )
    except ValueError as error:
        if require_all and "未覆盖" in str(error):
            raise ValueError("OOF 预测未覆盖全部目标 clips") from error
        raise


def _metrics(clips: list[dict[str, Any]], scores: np.ndarray) -> dict[str, float]:
    if scores.shape != (len(clips),):
        raise ValueError("clip 分数与 metadata 不匹配")
    labels = {str(clip["clip_id"]): bool(clip["has_fall"]) for clip in clips}
    predictions = {
        str(clip["clip_id"]): float(scores[index]) for index, clip in enumerate(clips)
    }
    result = competition_map(labels, predictions, mode="clip")
    return {
        "clip_map": float(result["map"]),
        "clip_map_percent": float(result["map_percent"]),
        "clip_p_at_r90": float(result["p_at_r90"]),
        "clip_p_at_r95": float(result["p_at_r95"]),
    }


def _fold_scores(
    *,
    config: MILConfig,
    cache: WindowMemmapCache,
    sidecar: np.ndarray,
    clip_folds: np.ndarray,
    fold: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit one fold and return only its held-out clip ids and scores."""
    window_clip_indices = np.asarray(cache.array("clip_indices"), dtype=np.int64)
    held_window = np.flatnonzero(clip_folds[window_clip_indices] == fold)
    train_pool = np.flatnonzero(clip_folds[window_clip_indices] != fold)
    if not held_window.size or not train_pool.size:
        raise ValueError("OOF fold 为空")
    labels = np.asarray(cache.array("labels"), dtype=np.uint8)
    groups = _activity_groups(cache)
    selected_local = select_training_indices(
        labels[train_pool],
        negative_ratio=config.negative_ratio,
        seed=config.seed + fold,
        activity_groups=groups[train_pool],
        hard_negative_activities=config.hard_negative_activities,
        hard_negative_fraction=config.hard_negative_fraction,
    )
    selected = train_pool[selected_local]
    train_x, train_y, train_clip_indices = _materialize(cache, selected)
    train_x = _append_sidecar(train_x, sidecar, selected)
    clip_groups, clip_labels = _clip_groups(train_clip_indices, cache.metadata["clips"])
    fold_config = replace(config, seed=config.seed + fold)
    set_deterministic(fold_config.seed)
    model = _model(fold_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=fold_config.learning_rate,
        weight_decay=fold_config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=fold_config.epochs
    )
    for epoch in range(fold_config.epochs):
        train_epoch_mil(
            model,
            optimizer,
            train_x,
            train_y,
            clip_groups,
            clip_labels,
            device=device,
            config=fold_config,
            epoch=epoch,
        )
        scheduler.step()
    held_x, _, held_clip_indices = _materialize(cache, held_window)
    held_x = _append_sidecar(held_x, sidecar, held_window)
    probabilities = predict_probabilities(
        model, held_x, device=device, batch_size=fold_config.batch_size
    )
    held_ids = np.unique(held_clip_indices)
    held_scores = _clip_score_array(
        probabilities,
        held_clip_indices,
        len(cache.metadata["clips"]),
        require_all=False,
    )[held_ids]
    return held_ids, held_scores


def _cross_fitted_scores(
    *,
    config: MILConfig,
    cache: WindowMemmapCache,
    sidecar: np.ndarray,
    folds: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    clips = cache.metadata.get("clips")
    if not isinstance(clips, list):
        raise TypeError("窗口缓存缺少 clips metadata")
    assignment = _template_grouped_folds(clips, folds, seed=config.seed)
    scores = np.full(len(clips), np.nan, dtype=np.float32)
    for fold in range(folds):
        held_ids, held_scores = _fold_scores(
            config=config,
            cache=cache,
            sidecar=sidecar,
            clip_folds=assignment,
            fold=fold,
            device=device,
        )
        if np.any(np.isfinite(scores[held_ids])):
            raise ValueError("OOF clip 被重复预测")
        scores[held_ids] = held_scores
        print(json.dumps({"stage": "oof_fold_complete", "fold": fold}), flush=True)
    if np.any(~np.isfinite(scores)):
        raise ValueError("OOF 未覆盖全部训练 clips")
    return scores, assignment


def _checkpoint_scores(
    *,
    cache: WindowMemmapCache,
    sidecar: np.ndarray,
    run: Path,
    checkpoint: Path,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    payload = json.loads(run.read_text(encoding="utf-8"))
    model, _, _ = _model_from_run(
        model_kind="multiscale_multistream_tcn",
        run=payload,
        checkpoint=checkpoint,
        device=device,
    )
    features, _, clip_indices = _materialize(cache, None)
    features = _append_sidecar(features, sidecar, None)
    probabilities = predict_probabilities(
        model, features, device=device, batch_size=batch_size
    )
    return _clip_score_array(probabilities, clip_indices, len(cache.metadata["clips"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--train-sidecar", type=Path, required=True)
    parser.add_argument("--val-sidecar", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--kinematics-config", type=Path, required=True)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--kinematics-run", type=Path, required=True)
    parser.add_argument("--kinematics-checkpoint", type=Path, required=True)
    parser.add_argument("--third-config", type=Path)
    parser.add_argument("--third-run", type=Path)
    parser.add_argument("--third-checkpoint", type=Path)
    parser.add_argument("--folds", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("融合输出已存在，拒绝覆盖")
    train = load_window_cache(args.train_cache)
    val = load_window_cache(args.val_cache)
    if train.metadata.get("split") == "test" or val.metadata.get("split") == "test":
        raise ValueError("OOF 融合禁止读取 test split")
    device = torch.device(args.device)
    base_config = MILConfig.from_json(args.base_config)
    second_config = MILConfig.from_json(args.kinematics_config)
    third_values = (args.third_config, args.third_run, args.third_checkpoint)
    if any(value is not None for value in third_values) and not all(
        value is not None for value in third_values
    ):
        raise ValueError("第三路模型必须同时提供 config、run 和 checkpoint")
    third_config = MILConfig.from_json(args.third_config) if args.third_config else None
    train_sidecar, val_sidecar = (
        load_sidecar(args.train_sidecar, train),
        load_sidecar(args.val_sidecar, val),
    )
    base_oof, folds = _cross_fitted_scores(
        config=base_config,
        cache=train,
        sidecar=train_sidecar,
        folds=args.folds,
        device=device,
    )
    second_oof, repeated_folds = _cross_fitted_scores(
        config=second_config,
        cache=train,
        sidecar=train_sidecar,
        folds=args.folds,
        device=device,
    )
    if not np.array_equal(folds, repeated_folds):
        raise RuntimeError("两路模型 OOF 折分不一致")
    oof_scores = [base_oof, second_oof]
    if third_config is not None:
        third_oof, third_folds = _cross_fitted_scores(
            config=third_config,
            cache=train,
            sidecar=train_sidecar,
            folds=args.folds,
            device=device,
        )
        if not np.array_equal(folds, third_folds):
            raise RuntimeError("第三路模型 OOF 折分不一致")
        oof_scores.append(third_oof)
    clips = train.metadata["clips"]
    labels = np.asarray(
        [float(bool(clip["has_fall"])) for clip in clips], dtype=np.float32
    )
    oof_features = np.column_stack([_logit(scores) for scores in oof_scores])
    coefficients, mean, scale = _fit_logistic(oof_features, labels)
    base_val = _checkpoint_scores(
        cache=val,
        sidecar=val_sidecar,
        run=args.base_run,
        checkpoint=args.base_checkpoint,
        device=device,
        batch_size=base_config.batch_size,
    )
    second_val = _checkpoint_scores(
        cache=val,
        sidecar=val_sidecar,
        run=args.kinematics_run,
        checkpoint=args.kinematics_checkpoint,
        device=device,
        batch_size=second_config.batch_size,
    )
    val_scores = [base_val, second_val]
    if third_config is not None:
        third_val = _checkpoint_scores(
            cache=val,
            sidecar=val_sidecar,
            run=args.third_run,
            checkpoint=args.third_checkpoint,
            device=device,
            batch_size=third_config.batch_size,
        )
        val_scores.append(third_val)
    val_features = np.column_stack([_logit(scores) for scores in val_scores])
    fused_logit = ((val_features - mean) / scale) @ coefficients[:-1] + coefficients[-1]
    fused_val = (1.0 / (1.0 + np.exp(-np.clip(fused_logit, -40.0, 40.0)))).astype(
        np.float32
    )
    report = {
        "protocol": "multistream_oof_logit_fusion_v3",
        "training": {
            "fit_split": "train_oof_only",
            "folds": args.folds,
            "fold_assignment": folds.tolist(),
            "fold_assignment_sha256": hashlib.sha256(folds.tobytes()).hexdigest(),
            "models_oof": [_metrics(clips, scores) for scores in oof_scores],
            "fusion_oof": _metrics(
                clips,
                (
                    1.0
                    / (
                        1.0
                        + np.exp(
                            -np.clip(
                                ((oof_features - mean) / scale) @ coefficients[:-1]
                                + coefficients[-1],
                                -40.0,
                                40.0,
                            )
                        )
                    )
                ).astype(np.float32),
            ),
        },
        "fusion": {
            "standardized_logit_coefficients": coefficients.tolist(),
            "mean": mean.tolist(),
            "scale": scale.tolist(),
            "l2": _L2,
        },
        "validation_once": {
            "models": [
                _metrics(val.metadata["clips"], scores) for scores in val_scores
            ],
            "fused": _metrics(val.metadata["clips"], fused_val),
        },
        "inputs": {
            "runs": [
                str(path.resolve())
                for path in [
                    args.base_run,
                    args.kinematics_run,
                    *([args.third_run] if args.third_run else []),
                ]
            ],
            "checkpoints": [
                str(path.resolve())
                for path in [
                    args.base_checkpoint,
                    args.kinematics_checkpoint,
                    *([args.third_checkpoint] if args.third_checkpoint else []),
                ]
            ],
            "train_cache_signature": train.metadata.get("signature_sha256"),
            "val_cache_signature": val.metadata.get("signature_sha256"),
        },
    }
    _atomic_json(args.output, report)
    print(json.dumps(report["validation_once"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
