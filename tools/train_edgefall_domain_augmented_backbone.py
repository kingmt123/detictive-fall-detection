"""Retrain one F1 backbone with fixed CAUCAFall MIL domain augmentation."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from models.edgefall_f1 import EdgeFallF1Head, SkeletonControlHead
from models.tcn_dataset import load_window_cache
from tools.audit_paired_clip_predictions import metrics_from_arrays
from tools.build_tcn_multistream_sidecar import load_sidecar
from tools.train_edgefall_domain_augmented_head import _load_roi, _person_roi
from tools.train_edgefall_f1 import (
    F1Config,
    _head_scores,
    _train_heads,
    build_skeleton_model,
    clip_embeddings,
    explicit_fold_ids,
    load_joint_pretrain,
    nested_fold_masks,
    subtype_targets,
)
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
    _sha256_file,
    select_training_indices,
    set_deterministic,
)

PROTOCOL = "edgefall_caucafall_domain_augmented_backbone_fold_canary_v1"
EXTERNAL_GROUP_REPEATS = 19


def repeated_shifted_groups(
    groups: list[np.ndarray],
    *,
    offset: int,
    repeats: int = EXTERNAL_GROUP_REPEATS,
) -> list[np.ndarray]:
    if offset < 0 or repeats < 1 or not groups:
        raise ValueError("external group repeat 配置无效")
    shifted = [np.asarray(group, dtype=np.int64) + offset for group in groups]
    return [group.copy() for _ in range(repeats) for group in shifted]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--joint-pretrain", type=Path, required=True)
    parser.add_argument("--base-window-cache", type=Path, required=True)
    parser.add_argument("--base-sidecar", type=Path, required=True)
    parser.add_argument("--base-roi-cache", type=Path, required=True)
    parser.add_argument("--fold-map", type=Path, required=True)
    parser.add_argument("--external-window-cache", type=Path, required=True)
    parser.add_argument("--external-sidecar", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--embedding-batch-size", type=int, default=1024)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("domain augmented backbone 输出目录必须为空")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference = torch.load(
        args.reference_checkpoint, map_location="cpu", weights_only=False
    )
    config = F1Config(**reference["config"])
    if config.fold != args.fold or not 0 <= args.fold < config.folds:
        raise ValueError("reference checkpoint/config 与请求 fold 不匹配")
    base_cache = load_window_cache(args.base_window_cache, verify_hashes=True)
    base_sidecar = load_sidecar(args.base_sidecar, base_cache)
    clips = list(base_cache.metadata["clips"])
    fold_ids, fold_assignment = explicit_fold_ids(
        args.fold_map,
        clips,
        folds=config.folds,
        expected_cache_signature=str(base_cache.metadata["signature_sha256"]),
    )
    if reference.get("fold_assignment") != fold_assignment:
        raise ValueError("reference checkpoint fold assignment 不匹配")
    train_mask, heldout_mask, _ = nested_fold_masks(
        fold_ids, heldout_fold=args.fold, excluded_folds=()
    )
    all_labels = np.asarray(base_cache.array("labels"))
    all_clip_indices = np.asarray(base_cache.array("clip_indices"), dtype=np.int64)
    candidate_windows = np.flatnonzero(train_mask[all_clip_indices])
    activities = _activity_groups(base_cache)
    local = select_training_indices(
        all_labels[candidate_windows],
        negative_ratio=config.negative_ratio,
        seed=config.seed,
        activity_groups=activities[candidate_windows],
        hard_negative_activities=("lie_down", "lying", "stand_up"),
        hard_negative_fraction=0.4,
    )
    selected_windows = candidate_windows[local]
    train_x, train_y, train_window_clips = _materialize(
        base_cache, selected_windows
    )
    train_x = _append_sidecar(train_x, base_sidecar, selected_windows)
    base_groups, base_group_labels = _clip_groups(train_window_clips, clips)
    base_group_indices = np.unique(train_window_clips)
    base_group_activities = np.asarray(
        [str(clips[int(index)]["clip_id"]).split("/", 1)[0] for index in base_group_indices],
        dtype=object,
    )

    external_cache = load_window_cache(args.external_window_cache, verify_hashes=True)
    external_sidecar = load_sidecar(args.external_sidecar, external_cache)
    external_windows = np.arange(external_cache.sample_count, dtype=np.int64)
    external_x, external_y, external_window_clips = _materialize(
        external_cache, external_windows
    )
    external_x = _append_sidecar(
        external_x, external_sidecar, external_windows
    )
    external_clips = list(external_cache.metadata["clips"])
    external_groups, external_group_labels = _clip_groups(
        external_window_clips, external_clips
    )
    if len(external_groups) != 100 or int(external_group_labels.sum()) != 50:
        raise ValueError("CAUCAFall development identity 无效")
    external_group_activities = np.asarray(
        ["fall" if label == 1 else "other" for label in external_group_labels],
        dtype=object,
    )
    features = torch.cat((train_x, external_x))
    labels = torch.cat((train_y, external_y))
    repeated_groups = repeated_shifted_groups(
        external_groups, offset=len(train_x)
    )
    groups = base_groups + repeated_groups
    group_labels = np.concatenate(
        (base_group_labels, np.tile(external_group_labels, EXTERNAL_GROUP_REPEATS))
    )
    group_activities = np.concatenate(
        (
            base_group_activities,
            np.tile(external_group_activities, EXTERNAL_GROUP_REPEATS),
        )
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("请求 CUDA，但当前不可用")
    set_deterministic(config.seed)
    model = build_skeleton_model(config).to(device)
    pretrain = torch.load(args.joint_pretrain, map_location="cpu", weights_only=False)
    if not isinstance(pretrain, dict) or not isinstance(
        pretrain.get("encoder_state"), dict
    ):
        raise TypeError("joint pretrain 缺少 encoder_state")
    missing_keys = load_joint_pretrain(model, pretrain["encoder_state"], config)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.backbone_learning_rate, weight_decay=1e-3
    )
    mil_config = MILConfig(
        epochs=config.backbone_epochs,
        seed=config.seed,
        use_rule_features=True,
        use_discriminative_kinematics=True,
        kinematic_feature_dim=15,
        horizontal_flip_probability=0.5,
        mil_clip_batch_size=config.clip_batch_size,
        hard_negative_activities=("lie_down", "lying", "stand_up"),
        hard_negative_fraction=0.4,
    )
    for epoch in range(config.backbone_epochs):
        started = time.perf_counter()
        loss, diagnostics = train_epoch_mil(
            model,
            optimizer,
            features,
            labels,
            groups,
            group_labels,
            clip_activities=group_activities,
            device=device,
            config=mil_config,
            epoch=epoch,
        )
        print(
            json.dumps(
                {
                    "stage": "domain_augmented_backbone",
                    "epoch": epoch,
                    "epoch_seconds": time.perf_counter() - started,
                    "loss": loss,
                    **diagnostics,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    del features, labels, external_x, external_y
    heldout_windows = np.flatnonzero(heldout_mask[all_clip_indices])
    heldout_x, _, heldout_window_clips = _materialize(
        base_cache, heldout_windows
    )
    heldout_x = _append_sidecar(heldout_x, base_sidecar, heldout_windows)
    train_embedding_clips, train_embeddings = clip_embeddings(
        model,
        train_x,
        train_window_clips,
        device=device,
        batch_size=args.embedding_batch_size,
    )
    heldout_embedding_clips, heldout_embeddings = clip_embeddings(
        model,
        heldout_x,
        heldout_window_clips,
        device=device,
        batch_size=args.embedding_batch_size,
    )
    selected_indices = np.concatenate(
        (train_embedding_clips, heldout_embedding_clips)
    )
    selected_ids = [str(clips[int(index)]["clip_id"]) for index in selected_indices]
    roi_ids, roi_tokens, roi_labels = _load_roi(args.base_roi_cache)
    person_roi = _person_roi(selected_ids, roi_ids, roi_tokens)
    head_labels = torch.tensor(
        [float(clips[int(index)]["has_fall"]) for index in selected_indices]
    )
    if not np.array_equal(
        head_labels.numpy(),
        np.asarray([roi_labels[roi_ids.index(clip_id)] for clip_id in selected_ids]),
    ):
        raise ValueError("OF-Syn ROI label 不匹配")
    subtypes = torch.from_numpy(subtype_targets(clips, selected_indices))
    skeleton = torch.from_numpy(
        np.concatenate((train_embeddings, heldout_embeddings))
    )
    head_train_count = len(train_embedding_clips)
    head_train = np.arange(head_train_count, dtype=np.int64)
    head_heldout = np.arange(head_train_count, len(selected_indices), dtype=np.int64)
    set_deterministic(20260829)
    control = SkeletonControlHead(skeleton.shape[1]).to(device)
    fusion = EdgeFallF1Head(
        skeleton.shape[1],
        roi_dim=person_roi.shape[2],
        roi_token_layout="direct",
        roi_temporal_pooling="mean_max",
    ).to(device)
    _train_heads(
        control,
        fusion,
        skeleton,
        person_roi,
        head_labels,
        subtypes,
        head_train,
        config=config,
        device=device,
    )
    scores = _head_scores(
        fusion,
        skeleton,
        person_roi,
        head_heldout,
        device=device,
        batch_size=config.head_batch_size,
    )
    heldout_ids = selected_ids[head_train_count:]
    heldout_labels = head_labels[head_heldout].numpy().astype(np.uint8)
    predictions = args.output_dir / "oof_predictions.npz"
    np.savez_compressed(
        predictions,
        clip_id=np.asarray(heldout_ids),
        label=heldout_labels,
        score=scores.astype(np.float32),
        fold=np.full(len(heldout_ids), args.fold, dtype=np.int64),
    )
    checkpoint_path = args.output_dir / "last.pt"
    torch.save(
        {
            "protocol": PROTOCOL,
            "config": asdict(config),
            "fold_assignment": fold_assignment,
            "skeleton_model_state": model.state_dict(),
            "fusion_state": fusion.state_dict(),
        },
        checkpoint_path,
    )
    summary = {
        "protocol": PROTOCOL,
        "fold": args.fold,
        "base_train_groups": len(base_groups),
        "external_unique_groups": len(external_groups),
        "external_group_repeats": EXTERNAL_GROUP_REPEATS,
        "external_group_fraction": len(repeated_groups) / len(groups),
        "heldout_clips": len(heldout_ids),
        "joint_pretrain_missing_keys": missing_keys,
        "member_metrics": metrics_from_arrays(heldout_labels, scores),
        "input_sha256": {
            "reference_checkpoint": _sha256_file(args.reference_checkpoint),
            "joint_pretrain": _sha256_file(args.joint_pretrain),
            "base_roi_cache": _sha256_file(args.base_roi_cache),
            "fold_map": _sha256_file(args.fold_map),
        },
        "output_sha256": {
            "checkpoint": _sha256_file(checkpoint_path),
            "predictions": _sha256_file(predictions),
        },
        "test_accessed": False,
        "upfall_accessed": False,
    }
    _atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
