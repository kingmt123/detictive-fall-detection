"""Train the frozen EdgeFall F1 recipe on OF-Syn train and evaluate val once."""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from models.edgefall_f1 import EdgeFallF1Head, SkeletonControlHead
from models.tcn_dataset import WindowMemmapCache, load_window_cache
from tools.build_tcn_multistream_sidecar import load_sidecar
from tools.train_edgefall_f1 import (
    F1Config,
    _head_scores,
    _train_heads,
    build_skeleton_model,
    clip_embeddings,
    load_joint_pretrain,
    metric_result,
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
    _atomic_checkpoint,
    _atomic_json,
    _materialize,
    _sha256_file,
    select_training_indices,
    set_deterministic,
)


def validate_cache_split(cache: WindowMemmapCache, split: str) -> None:
    """Fail closed before materialising a wrong or reserved split."""
    if split not in {"train", "val"}:
        raise ValueError("F1 full-validation 只允许 train/val")
    if cache.metadata.get("dataset") != "of-syn":
        raise ValueError("F1 full-validation 只接受 OF-Syn cache")
    if cache.metadata.get("split") != split:
        raise ValueError(f"F1 cache split 必须为 {split}")


def ensure_clip_coverage(
    selected: np.ndarray, clip_indices: np.ndarray, clip_count: int
) -> np.ndarray:
    """Add the first valid window only for clips absent after fixed sampling."""
    selected = np.asarray(selected, dtype=np.int64)
    clip_indices = np.asarray(clip_indices, dtype=np.int64)
    if selected.ndim != 1 or clip_indices.ndim != 1 or clip_count < 1:
        raise ValueError("clip coverage 输入 shape 无效")
    if selected.size == 0 or np.any(selected < 0) or np.any(selected >= clip_indices.size):
        raise ValueError("clip coverage selected index 无效")
    missing = np.setdiff1d(
        np.arange(clip_count, dtype=np.int64),
        np.unique(clip_indices[selected]),
        assume_unique=False,
    )
    additions = []
    for clip_index in missing:
        candidates = np.flatnonzero(clip_indices == clip_index)
        if candidates.size == 0:
            raise ValueError(f"clip {clip_index} 没有合法窗口")
        additions.append(int(candidates[0]))
    if not additions:
        return np.sort(np.unique(selected))
    return np.sort(
        np.unique(np.concatenate((selected, np.asarray(additions, dtype=np.int64))))
    )


def load_person_roi(path: Path, clips: list[dict[str, object]]) -> torch.Tensor:
    """Load cache-ordered person ROI tokens with exact clip coverage."""
    with np.load(path, allow_pickle=False) as payload:
        clip_ids = [str(value) for value in payload["clip_ids"]]
        tokens = np.asarray(payload["tokens"], dtype=np.float32)
    expected = [str(clip["clip_id"]) for clip in clips]
    if clip_ids != expected:
        raise ValueError("ROI cache clip 顺序与 window cache 不一致")
    if tokens.ndim != 4 or tokens.shape[:3] != (len(clips), 16, 2):
        raise ValueError("ROI cache token 形状无效")
    return torch.from_numpy(tokens[:, :, 0].copy())


def _all_clip_embeddings(
    model: torch.nn.Module,
    cache: WindowMemmapCache,
    sidecar: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, torch.Tensor]:
    indices = np.arange(int(cache.metadata["sample_count"]), dtype=np.int64)
    features, _, clip_indices = _materialize(cache, indices)
    features = _append_sidecar(features, sidecar, indices)
    embedded_clips, embeddings = clip_embeddings(
        model,
        features,
        clip_indices,
        device=device,
        batch_size=batch_size,
    )
    return embedded_clips, torch.from_numpy(embeddings)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--train-sidecar", type=Path, required=True)
    parser.add_argument("--train-roi-cache", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--val-sidecar", type=Path, required=True)
    parser.add_argument("--val-roi-cache", type=Path, required=True)
    parser.add_argument("--joint-pretrain", type=Path, required=True)
    parser.add_argument(
        "--reuse-backbone-checkpoint",
        type=Path,
        help="严格匹配 full-train 配置后复用 skeleton backbone",
    )
    parser.add_argument("--head-seed", type=int)
    parser.add_argument(
        "--ensure-all-clips-for-head",
        action="store_true",
        help="为固定采样遗漏的 clip 确定性补入第一个合法窗口",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backbone-epochs", type=int, default=12)
    parser.add_argument("--head-epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args()

    config = F1Config(
        fold=0,
        backbone_epochs=args.backbone_epochs,
        head_epochs=args.head_epochs,
        seed=args.seed,
        use_transformer_encoder=True,
        transformer_joint_only=True,
    )
    config.validate()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("F1 full-validation 输出目录必须为空")

    train_cache = load_window_cache(args.train_cache, verify_hashes=True)
    val_cache = load_window_cache(args.val_cache, verify_hashes=True)
    validate_cache_split(train_cache, "train")
    validate_cache_split(val_cache, "val")
    train_sidecar = load_sidecar(args.train_sidecar, train_cache)
    val_sidecar = load_sidecar(args.val_sidecar, val_cache)
    train_clips = train_cache.metadata["clips"]
    val_clips = val_cache.metadata["clips"]
    train_roi = load_person_roi(args.train_roi_cache, train_clips)
    val_roi = load_person_roi(args.val_roi_cache, val_clips)

    all_labels = np.asarray(train_cache.array("labels"))
    activities = _activity_groups(train_cache)
    selected_windows = select_training_indices(
        all_labels,
        negative_ratio=config.negative_ratio,
        seed=config.seed,
        activity_groups=activities,
        hard_negative_activities=("lie_down", "lying", "stand_up"),
        hard_negative_fraction=0.4,
    )
    if args.ensure_all_clips_for_head:
        selected_windows = ensure_clip_coverage(
            selected_windows,
            np.asarray(train_cache.array("clip_indices"), dtype=np.int64),
            len(train_cache.metadata["clips"]),
        )
    train_x, train_y, train_window_clips = _materialize(train_cache, selected_windows)
    train_x = _append_sidecar(train_x, train_sidecar, selected_windows)
    groups, group_labels = _clip_groups(train_window_clips, train_clips)
    group_clip_indices = np.unique(train_window_clips)
    group_activities = np.asarray(
        [str(train_clips[int(index)]["clip_id"]).split("/", 1)[0] for index in group_clip_indices],
        dtype=object,
    )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("请求 CUDA，但当前不可用")
    set_deterministic(config.seed)
    model = build_skeleton_model(config).to(device)
    pretrain = torch.load(args.joint_pretrain, map_location="cpu", weights_only=False)
    if not isinstance(pretrain, dict) or not isinstance(pretrain.get("encoder_state"), dict):
        raise TypeError("FallVision joint pretrain 缺少 encoder_state")
    missing_keys = load_joint_pretrain(model, pretrain["encoder_state"], config)
    reused_backbone_sha256: str | None = None
    if args.reuse_backbone_checkpoint is not None:
        reused = torch.load(
            args.reuse_backbone_checkpoint, map_location="cpu", weights_only=False
        )
        if not isinstance(reused, dict) or not isinstance(
            reused.get("skeleton_model_state"), dict
        ):
            raise TypeError("复用 checkpoint 缺少 skeleton_model_state")
        if reused.get("config") != asdict(config):
            raise ValueError("复用 full-train backbone 的配置不匹配")
        model.load_state_dict(reused["skeleton_model_state"], strict=True)
        reused_backbone_sha256 = _sha256_file(args.reuse_backbone_checkpoint)
    else:
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
            loss, diagnostics = train_epoch_mil(
                model,
                optimizer,
                train_x,
                train_y,
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
                        "stage": "skeleton_backbone",
                        "epoch": epoch,
                        "loss": loss,
                        **diagnostics,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    train_embedding_clips, train_embeddings_np = clip_embeddings(
        model,
        train_x,
        train_window_clips,
        device=device,
        batch_size=config.batch_size,
    )
    if not np.array_equal(train_embedding_clips, np.arange(len(train_clips))):
        raise ValueError("采样后的 train windows 未覆盖全部 train clips")
    _, val_embeddings = _all_clip_embeddings(
        model,
        val_cache,
        val_sidecar,
        device=device,
        batch_size=config.batch_size,
    )
    train_embeddings = torch.from_numpy(train_embeddings_np)
    train_labels = torch.tensor([float(clip["has_fall"]) for clip in train_clips])
    train_subtypes = torch.from_numpy(
        subtype_targets(train_clips, np.arange(len(train_clips), dtype=np.int64))
    )
    train_indices = np.arange(len(train_clips), dtype=np.int64)
    skeleton_dim = int(train_embeddings.shape[1])
    roi_token_dim = int(train_roi.shape[2])
    if int(val_roi.shape[2]) != roi_token_dim:
        raise ValueError("train/val ROI token 维度不一致")
    if args.head_seed is not None:
        set_deterministic(args.head_seed)
    control = SkeletonControlHead(skeleton_dim).to(device)
    fusion = EdgeFallF1Head(skeleton_dim, roi_dim=roi_token_dim).to(device)
    initial_control = copy.deepcopy(control.state_dict())
    _train_heads(
        control,
        fusion,
        train_embeddings,
        train_roi,
        train_labels,
        train_subtypes,
        train_indices,
        config=config,
        device=device,
    )

    val_indices = np.arange(len(val_clips), dtype=np.int64)
    control_scores = _head_scores(
        control,
        val_embeddings,
        None,
        val_indices,
        device=device,
        batch_size=config.head_batch_size,
    )
    fusion_scores = _head_scores(
        fusion,
        val_embeddings,
        val_roi,
        val_indices,
        device=device,
        batch_size=config.head_batch_size,
    )
    val_ids = [str(clip["clip_id"]) for clip in val_clips]
    val_labels = np.asarray([float(clip["has_fall"]) for clip in val_clips])
    control_metrics = metric_result(val_ids, val_labels, control_scores)
    fusion_metrics = metric_result(val_ids, val_labels, fusion_scores)
    delta = {
        key: fusion_metrics[key] - control_metrics[key]
        for key in ("clip_map", "clip_map_percent", "clip_p_at_r90", "clip_p_at_r95")
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "config": asdict(config),
        "roi_token_dim": roi_token_dim,
        "skeleton_model_state": model.state_dict(),
        "control_state": control.state_dict(),
        "fusion_state": fusion.state_dict(),
        "initial_control_state": initial_control,
        "control_metrics": control_metrics,
        "fusion_metrics": fusion_metrics,
        "head_seed": args.head_seed,
        "ensure_all_clips_for_head": args.ensure_all_clips_for_head,
        "reused_backbone_checkpoint_sha256": reused_backbone_sha256,
    }
    _atomic_checkpoint(args.output_dir / "last.pt", checkpoint)
    np.savez_compressed(
        args.output_dir / "val_predictions.npz",
        clip_id=np.asarray(val_ids),
        label=val_labels,
        control_score=control_scores,
        score=fusion_scores,
    )
    summary = {
        "protocol": "edgefall_f1_frozen_recipe_full_train_single_random_validation_v1",
        "config": asdict(config),
        "train_clips": len(train_clips),
        "validation_clips": len(val_clips),
        "control": control_metrics,
        "fusion": fusion_metrics,
        "delta": delta,
        "joint_pretrain_sha256": _sha256_file(args.joint_pretrain),
        "joint_pretrain_missing_keys": list(missing_keys),
        "head_seed": args.head_seed,
        "reused_backbone_checkpoint_sha256": reused_backbone_sha256,
        "train_cache_signature_sha256": train_cache.metadata["signature_sha256"],
        "val_cache_signature_sha256": val_cache.metadata["signature_sha256"],
        "train_roi_cache_sha256": _sha256_file(args.train_roi_cache),
        "val_roi_cache_sha256": _sha256_file(args.val_roi_cache),
        "roi_token_dim": roi_token_dim,
        "checkpoint_sha256": _sha256_file(args.output_dir / "last.pt"),
        "validation_uses": 1,
        "validation_used_for_selection": False,
        "test_accessed": False,
    }
    _atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
