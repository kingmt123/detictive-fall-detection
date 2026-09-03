"""Strict train-OOF fusion of the main TCN and causal transition GRU.

The sealed test split is never accepted.  Fusion parameters are fit only from
held-out train clips; the validation split is scored once after they are fixed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from models.tcn_dataset import WindowMemmapCache, load_window_cache
from tools.build_tcn_multistream_sidecar import load_sidecar
from tools.evaluate_oof_multistream_fusion import (
    _checkpoint_scores,
    _clip_score_array,
    _cross_fitted_scores,
    _fit_logistic,
    _logit,
    _metrics,
    _stratified_clip_folds,
)
from tools.train_multiscale_multistream_mil import MILConfig
from tools.train_tcn import (
    _append_sidecar,
    _atomic_json,
    _materialize,
    predict_probabilities,
    set_deterministic,
)
from tools.train_transition_gru import (
    TransitionGRUConfig,
    _balanced_indices,
    _model,
    _targets,
    _train_epoch,
)


def _transition_fold_scores(
    *,
    config: TransitionGRUConfig,
    cache: WindowMemmapCache,
    sidecar: np.ndarray,
    clip_folds: np.ndarray,
    fold: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    window_clips = np.asarray(cache.array("clip_indices"), dtype=np.int64)
    held = np.flatnonzero(clip_folds[window_clips] == fold)
    train_pool = np.flatnonzero(clip_folds[window_clips] != fold)
    if not held.size or not train_pool.size:
        raise ValueError("Transition OOF fold 为空")
    all_targets = _targets(cache.array("semantic_codes"))
    selected = train_pool[_balanced_indices(all_targets[train_pool], config.seed + fold)]
    train_x, _, _ = _materialize(cache, selected)
    train_x = _append_sidecar(train_x, sidecar, selected)
    train_targets = torch.from_numpy(all_targets[selected])
    fold_config = TransitionGRUConfig(**{**config.__dict__, "seed": config.seed + fold})
    set_deterministic(fold_config.seed)
    model = _model(fold_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=fold_config.learning_rate, weight_decay=fold_config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=fold_config.epochs)
    for epoch in range(fold_config.epochs):
        _train_epoch(model, optimizer, train_x, train_targets, config=fold_config, device=device, epoch=epoch)
        scheduler.step()
    held_x, _, held_window_clips = _materialize(cache, held)
    probabilities = predict_probabilities(model, _append_sidecar(held_x, sidecar, held), device=device, batch_size=fold_config.batch_size)
    held_ids = np.unique(held_window_clips)
    scores = _clip_score_array(probabilities, held_window_clips, len(cache.metadata["clips"]), require_all=False)[held_ids]
    return held_ids, scores


def _transition_oof_scores(
    *,
    config: TransitionGRUConfig,
    cache: WindowMemmapCache,
    sidecar: np.ndarray,
    folds: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    clips = cache.metadata.get("clips")
    if not isinstance(clips, list):
        raise TypeError("窗口缓存缺少 clips metadata")
    assignment = _stratified_clip_folds(clips, folds)
    scores = np.full(len(clips), np.nan, dtype=np.float32)
    for fold in range(folds):
        ids, held_scores = _transition_fold_scores(config=config, cache=cache, sidecar=sidecar, clip_folds=assignment, fold=fold, device=device)
        if np.any(np.isfinite(scores[ids])):
            raise RuntimeError("Transition OOF clip 重复预测")
        scores[ids] = held_scores
        print(json.dumps({"stage": "transition_oof_fold_complete", "fold": fold}), flush=True)
    if np.any(~np.isfinite(scores)):
        raise RuntimeError("Transition OOF 未覆盖全部训练 clips")
    return scores, assignment


def _transition_checkpoint_scores(
    *, cache: WindowMemmapCache, sidecar: np.ndarray, checkpoint: Path, config: TransitionGRUConfig, device: torch.device
) -> np.ndarray:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = _model(config).to(device)
    model.load_state_dict(payload["model_state"])
    features, _, clip_indices = _materialize(cache, None)
    probabilities = predict_probabilities(model, _append_sidecar(features, sidecar, None), device=device, batch_size=config.batch_size)
    return _clip_score_array(probabilities, clip_indices, len(cache.metadata["clips"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("train-cache", "val-cache", "train-sidecar", "val-sidecar", "base-config", "base-run", "base-checkpoint", "transition-config", "transition-checkpoint", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("融合输出已存在，拒绝覆盖")
    train, val = load_window_cache(args.train_cache), load_window_cache(args.val_cache)
    if train.metadata.get("split") != "train" or val.metadata.get("split") != "val":
        raise ValueError("融合仅允许 train/val，禁止 test")
    device = torch.device(args.device)
    base_config = MILConfig.from_json(args.base_config)
    transition_config = TransitionGRUConfig.from_json(args.transition_config)
    train_sidecar, val_sidecar = load_sidecar(args.train_sidecar, train), load_sidecar(args.val_sidecar, val)
    base_oof, assignment = _cross_fitted_scores(config=base_config, cache=train, sidecar=train_sidecar, folds=args.folds, device=device)
    transition_oof, repeated = _transition_oof_scores(config=transition_config, cache=train, sidecar=train_sidecar, folds=args.folds, device=device)
    if not np.array_equal(assignment, repeated):
        raise RuntimeError("两路 OOF 折分不一致")
    clips = train.metadata["clips"]
    labels = np.asarray([float(bool(clip["has_fall"])) for clip in clips], dtype=np.float32)
    features = np.column_stack((_logit(base_oof), _logit(transition_oof)))
    coefficients, mean, scale = _fit_logistic(features, labels)
    base_val = _checkpoint_scores(cache=val, sidecar=val_sidecar, run=args.base_run, checkpoint=args.base_checkpoint, device=device, batch_size=base_config.batch_size)
    transition_val = _transition_checkpoint_scores(cache=val, sidecar=val_sidecar, checkpoint=args.transition_checkpoint, config=transition_config, device=device)
    val_features = np.column_stack((_logit(base_val), _logit(transition_val)))
    fused = 1.0 / (1.0 + np.exp(-np.clip(((val_features - mean) / scale) @ coefficients[:-1] + coefficients[-1], -40.0, 40.0)))
    oof_fused = 1.0 / (
        1.0
        + np.exp(
            -np.clip(
                ((features - mean) / scale) @ coefficients[:-1] + coefficients[-1],
                -40.0,
                40.0,
            )
        )
    )
    report = {
        "protocol": "tcn_transition_gru_oof_logit_fusion_v1",
        "training": {
            "fit_split": "train_oof_only",
            "folds": args.folds,
            "fold_assignment_sha256": hashlib.sha256(assignment.tobytes()).hexdigest(),
            "models_oof": [_metrics(clips, base_oof), _metrics(clips, transition_oof)],
            "fusion_oof": _metrics(clips, oof_fused),
        },
        "fusion": {
            "standardized_logit_coefficients": coefficients.tolist(),
            "mean": mean.tolist(),
            "scale": scale.tolist(),
            "l2": 1e-3,
        },
        "validation_once": {
            "models": [
                _metrics(val.metadata["clips"], base_val),
                _metrics(val.metadata["clips"], transition_val),
            ],
            "fused": _metrics(val.metadata["clips"], fused),
        },
        "inputs": {
            "base_run": str(args.base_run.resolve()),
            "base_checkpoint": str(args.base_checkpoint.resolve()),
            "transition_checkpoint": str(args.transition_checkpoint.resolve()),
            "train_cache_signature": train.metadata.get("signature_sha256"),
            "val_cache_signature": val.metadata.get("signature_sha256"),
        },
    }
    _atomic_json(args.output, report)
    print(json.dumps(report["validation_once"], ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
