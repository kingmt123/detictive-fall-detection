"""Train one leak-free, template-grouped Stage-S1 train-only OOF fold."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from models.real_distortion import RealDistortionBank
from models.tcn_dataset import load_window_cache
from tools.build_tcn_multistream_sidecar import load_sidecar
from tools.evaluate_oof_multistream_fusion import _clip_score_array, _model
from tools.train_long_context_event_oracle import template_group
from tools.train_multiscale_multistream_mil import (
    MILConfig,
    _clip_groups,
    build_stage_targets,
    train_epoch_mil,
)
from tools.train_multiscale_multistream_tcn import _activity_groups
from tools.train_tcn import (
    _append_sidecar,
    _atomic_checkpoint,
    _atomic_json,
    _materialize,
    _sha256_file,
    predict_probabilities,
    select_training_indices,
    set_deterministic,
)


def _fold_assignment(path: Path, clips: list[dict[str, object]], *, fold: int) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or len(rows) != len(clips):
        raise ValueError("OOF fold map 与 cache clips 不匹配")
    assignments: list[int] = []
    for expected, row in zip(clips, rows, strict=True):
        if not isinstance(row, dict) or row.get("clip_id") != expected.get("clip_id"):
            raise ValueError("OOF fold map clip 顺序不匹配")
        if row.get("group_id") != template_group(str(expected["clip_id"])):
            raise ValueError("OOF fold map template group 不匹配")
        assignments.append(int(row["fold"]))
    result = np.asarray(assignments, dtype=np.int16)
    if fold not in result:
        raise ValueError("请求的 OOF fold 为空")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--joint-pretrain-checkpoint", type=Path, required=True)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--train-sidecar", type=Path, required=True)
    parser.add_argument("--fold-map", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--real-distortion-bank", type=Path)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("输出目录必须为空")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("请求 CUDA 但不可用")
    config = MILConfig.from_json(args.config)
    if args.seed is not None:
        config = replace(config, seed=args.seed)
        config.validate()
    cache = load_window_cache(args.train_cache, verify_hashes=True)
    if cache.metadata.get("split") != "train":
        raise ValueError("OOF trainer 只允许 train cache")
    clips = cache.metadata["clips"]
    assignment = _fold_assignment(args.fold_map, clips, fold=args.fold)
    all_clip_indices = np.asarray(cache.array("clip_indices"), dtype=np.int64)
    held_out = np.flatnonzero(assignment[all_clip_indices] == args.fold)
    train_pool = np.flatnonzero(assignment[all_clip_indices] != args.fold)
    if not held_out.size or not train_pool.size:
        raise ValueError("OOF train/held-out windows 不能为空")
    held_groups = {template_group(str(clips[index]["clip_id"])) for index in np.unique(all_clip_indices[held_out])}
    train_groups = {template_group(str(clips[index]["clip_id"])) for index in np.unique(all_clip_indices[train_pool])}
    if held_groups & train_groups:
        raise AssertionError("template group 跨 OOF fold 泄漏")
    args.output_dir.mkdir(parents=True)
    _atomic_json(args.output_dir / "status.json", {"stage": "materializing_train"})
    set_deterministic(config.seed + args.fold)
    labels = np.asarray(cache.array("labels"), dtype=np.uint8)
    activities = _activity_groups(cache)
    selected = train_pool[
        select_training_indices(
            labels[train_pool],
            negative_ratio=config.negative_ratio,
            seed=config.seed + args.fold,
            activity_groups=activities[train_pool],
            hard_negative_activities=config.hard_negative_activities,
            hard_negative_fraction=config.hard_negative_fraction,
        )
    ]
    sidecar = load_sidecar(args.train_sidecar, cache)
    train_x, train_y, train_clip_indices = _materialize(cache, selected)
    train_x = _append_sidecar(train_x, sidecar, selected)
    groups, clip_labels = _clip_groups(train_clip_indices, clips)
    stage_targets = torch.from_numpy(
        build_stage_targets(np.asarray(cache.array("semantic_codes")[selected]), activities[selected])
    )
    device = torch.device(args.device)
    if bool(config.real_distortion_mode) != bool(args.real_distortion_bank):
        raise ValueError("real distortion config 与 bank 参数必须同时启用")
    distortion_evidence: dict[str, object] | None = None
    real_distortion_bank = None
    if args.real_distortion_bank is not None:
        evidence_path = args.real_distortion_bank.with_suffix(".json")
        if not evidence_path.exists():
            raise FileNotFoundError("fold-local distortion bank 缺少 JSON 证据")
        distortion_evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        if distortion_evidence.get("source_cache_signature") != cache.metadata.get(
            "signature_sha256"
        ):
            raise ValueError("distortion bank 与训练 cache 不匹配")
        fold_evidence = distortion_evidence.get("fold_assignment")
        if not isinstance(fold_evidence, dict) or fold_evidence.get(
            "excluded_fold"
        ) != args.fold:
            raise ValueError("distortion bank 未排除当前 held-out fold")
        if distortion_evidence.get("sha256") != _sha256_file(
            args.real_distortion_bank
        ):
            raise ValueError("distortion bank SHA256 不匹配")
        real_distortion_bank = RealDistortionBank.load(
            args.real_distortion_bank, device=device
        )
    model = _model(config).to(device)
    pretrain = torch.load(args.joint_pretrain_checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(pretrain, dict) or not isinstance(pretrain.get("encoder_state"), dict):
        raise TypeError("Joint 外部预训练 checkpoint 缺少 encoder_state")
    model.streams[0].load_state_dict(pretrain["encoder_state"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    history: list[dict[str, float | int]] = []
    for epoch in range(config.epochs):
        _atomic_json(args.output_dir / "status.json", {"stage": "training", "epoch": epoch})
        loss, diagnostics = train_epoch_mil(
            model, optimizer, train_x, train_y, groups, clip_labels,
            stage_targets=stage_targets,
            real_distortion_bank=real_distortion_bank,
            device=device,
            config=config,
            epoch=epoch,
        )
        scheduler.step()
        record = {"epoch": epoch, "train_loss": loss, **diagnostics}
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
    _atomic_json(args.output_dir / "status.json", {"stage": "scoring_held_out"})
    held_x, _, held_clip_indices = _materialize(cache, held_out)
    held_x = _append_sidecar(held_x, sidecar, held_out)
    probabilities = predict_probabilities(model, held_x, device=device, batch_size=config.batch_size)
    scores = _clip_score_array(probabilities, held_clip_indices, len(clips), require_all=False)
    held_clip_ids = np.unique(held_clip_indices)
    np.savez_compressed(
        args.output_dir / "oof_predictions.npz",
        clip_id=np.asarray([clips[index]["clip_id"] for index in held_clip_ids]),
        group_id=np.asarray([template_group(str(clips[index]["clip_id"])) for index in held_clip_ids]),
        label=np.asarray([bool(clips[index]["has_fall"]) for index in held_clip_ids]),
        score=scores[held_clip_ids], fold=np.full(held_clip_ids.size, args.fold, dtype=np.int16),
    )
    checkpoint = {"model_state": model.state_dict(), "epoch": config.epochs - 1, "config": config.__dict__}
    _atomic_checkpoint(args.output_dir / "final.pt", checkpoint)
    _atomic_json(args.output_dir / "summary.json", {
        "protocol": "stage_s1_r2_template_grouped_train_only_oof_v1",
        "fold": args.fold, "seed": config.seed, "train_clips": len(groups),
        "held_out_clips": int(held_clip_ids.size), "history": history,
        "real_distortion_bank": distortion_evidence,
    })
    _atomic_json(args.output_dir / "status.json", {"stage": "complete"})


if __name__ == "__main__":
    main()
