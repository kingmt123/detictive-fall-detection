"""Train one grouped, train-only, full-clip VideoMAE RGB-teacher canary.

The canary is the first radical-backbone gate.  It does not read OF-Syn val,
OF-Syn test, or URFD.  It never uses pose semantics to crop a video: every
input is a uniformly sampled full clip, and the final epoch is pre-fixed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from models.tcn_dataset import load_window_cache
from models.videomae_clip_teacher import (
    SUBTYPE_NAMES,
    VideoMaeClipTeacher,
    teacher_loss,
)
from pipeline.video_source import VideoSourceResolver
from tools.train_edgefall_f1 import metric_result
from tools.train_event_aligned_rgb_oracle import load_train_rows
from tools.train_long_context_event_oracle import group_fold, template_group
from tools.train_tcn import (
    _atomic_checkpoint,
    _atomic_json,
    _sha256_file,
    set_deterministic,
)


@dataclass(frozen=True)
class VideoMaeCanaryConfig:
    frames: int = 64
    image_size: int = 112
    batch_size: int = 1
    decode_workers: int = 1
    epochs: int = 1
    learning_rate: float = 2e-5
    weight_decay: float = 0.05
    subtype_weight: float = 0.2
    folds: int = 5
    fold: int = 0
    seed: int = 20260825
    max_train_clips: int | None = None
    max_heldout_clips: int | None = None
    warmup_epochs: int = 1
    warmup_last_layers: int = 4

    def validate(self) -> None:
        if self.frames < 4 or self.image_size < 32 or self.batch_size < 1:
            raise ValueError("frames/image_size/batch_size 无效")
        if not 1 <= self.decode_workers <= self.batch_size:
            raise ValueError("decode_workers 必须位于 [1,batch_size]")
        if self.epochs < 1 or self.folds < 2 or not 0 <= self.fold < self.folds:
            raise ValueError("epochs/folds/fold 无效")
        if not 0 <= self.warmup_epochs <= self.epochs or self.warmup_last_layers < 0:
            raise ValueError("warmup 参数无效")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("optimizer 参数无效")
        for value in (self.max_train_clips, self.max_heldout_clips):
            if value is not None and value < 2:
                raise ValueError("clip cap 必须至少为 2")


def subtype_target(row: dict[str, str]) -> int:
    """Map the manifest activity to a stable auxiliary ADL target."""
    activity = row["clip_id"].split("/", 1)[0]
    if row["has_fall"] == "1":
        activity = "fall"
    aliases = {"sit_down": "sit_or_bend", "sitting": "sit_or_bend"}
    canonical = aliases.get(activity, activity)
    if canonical not in SUBTYPE_NAMES:
        canonical = "other_background"
    return SUBTYPE_NAMES.index(canonical)


def deterministic_activity_stratified_subset(
    indices: np.ndarray,
    rows: list[dict[str, str]],
    *,
    limit: int | None,
    seed: int,
) -> np.ndarray:
    """Cap a canary while retaining ADL hard-negative strata and positives."""
    indices = np.asarray(indices, dtype=np.int64)
    if limit is None or indices.size <= limit:
        return indices
    buckets: dict[int, list[int]] = {index: [] for index in range(len(SUBTYPE_NAMES))}
    for index in indices:
        buckets[subtype_target(rows[int(index)])].append(int(index))
    nonempty = {label: values for label, values in buckets.items() if values}
    if len(nonempty) < 2 or limit < len(nonempty):
        raise ValueError("canary cap 无法保持活动分层")
    quotas = {label: max(1, int(limit * len(values) // indices.size)) for label, values in nonempty.items()}
    while sum(quotas.values()) < limit:
        label = max(
            nonempty,
            key=lambda key: (limit * len(nonempty[key]) / indices.size - quotas[key], -key),
        )
        quotas[label] += 1
    while sum(quotas.values()) > limit:
        label = max(quotas, key=lambda key: (quotas[key], -key))
        if quotas[label] == 1:
            raise ValueError("canary cap 无法保留每个活动")
        quotas[label] -= 1
    selected: list[int] = []
    for label, values in nonempty.items():
        ordered = sorted(
            values,
            key=lambda index: hashlib.sha256(
                f"{seed}:{label}:{rows[index]['clip_id']}".encode()
            ).digest(),
        )
        selected.extend(ordered[: quotas[label]])
    return np.asarray(sorted(selected), dtype=np.int64)


def decode_uniform_full_clip(path: Path, *, frames: int, image_size: int) -> torch.Tensor:
    """Decode a complete video timeline; no event crop or track is consulted."""
    capture = cv2.VideoCapture(str(path))
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count < 2:
            raise ValueError(f"视频帧数无效: {path}")
        positions = np.linspace(0, frame_count - 1, frames).round().astype(np.int64)
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(positions[0]))
        current = int(positions[0]) - 1
        decoded: list[np.ndarray] = []
        for position in positions:
            while current < int(position):
                ok, frame = capture.read()
                current += 1
                if not ok or frame is None:
                    raise ValueError(f"视频解码失败: {path}@{position}")
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width = rgb.shape[:2]
            if min(height, width) < 2:
                raise ValueError(f"视频尺寸无效: {path}")
            scale = image_size / min(height, width)
            resized = cv2.resize(
                rgb,
                (round(width * scale), round(height * scale)),
                interpolation=cv2.INTER_AREA,
            )
            top = (resized.shape[0] - image_size) // 2
            left = (resized.shape[1] - image_size) // 2
            decoded.append(resized[top : top + image_size, left : left + image_size])
    finally:
        capture.release()
    values = torch.from_numpy(np.asarray(decoded)).float().permute(0, 3, 1, 2) / 255.0
    # Official VideoMAE preprocessing uses Kinetics-style 0.45 normalization.
    return (values - 0.45) / 0.225


def _decode_batch(
    rows: list[dict[str, str]],
    *,
    resolver: VideoSourceResolver,
    pool: ThreadPoolExecutor,
    config: VideoMaeCanaryConfig,
) -> torch.Tensor:
    with ExitStack() as stack:
        paths = [
            stack.enter_context(resolver.materialize(row["video_path"])).local_path
            for row in rows
        ]
        clips = list(
            pool.map(
                lambda path: decode_uniform_full_clip(
                    path, frames=config.frames, image_size=config.image_size
                ),
                paths,
            )
        )
    return torch.stack(clips)


@torch.inference_mode()
def _scores(
    model: VideoMaeClipTeacher,
    rows: list[dict[str, str]],
    *,
    resolver: VideoSourceResolver,
    pool: ThreadPoolExecutor,
    config: VideoMaeCanaryConfig,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    values: list[np.ndarray] = []
    for start in range(0, len(rows), config.batch_size):
        batch = _decode_batch(rows[start : start + config.batch_size], resolver=resolver, pool=pool, config=config)
        output = model(batch.to(device, non_blocking=True))
        values.append(torch.sigmoid(output.binary_logits).cpu().numpy())
    return np.concatenate(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--temp-root", type=Path, required=True)
    parser.add_argument("--hf-cache", type=Path, required=True)
    parser.add_argument("--model-id", default="MCG-NJU/videomae-base-finetuned-kinetics")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--decode-workers", type=int, default=1)
    parser.add_argument("--max-train-clips", type=int)
    parser.add_argument("--max-heldout-clips", type=int)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--warmup-last-layers", type=int, default=4)
    args = parser.parse_args()
    config = VideoMaeCanaryConfig(
        fold=args.fold, epochs=args.epochs, frames=args.frames, image_size=args.image_size,
        batch_size=args.batch_size, decode_workers=args.decode_workers, seed=args.seed,
        max_train_clips=args.max_train_clips, max_heldout_clips=args.max_heldout_clips,
        warmup_epochs=args.warmup_epochs, warmup_last_layers=args.warmup_last_layers,
    )
    config.validate()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("VideoMAE canary 输出目录非空，拒绝覆盖")
    cache = load_window_cache(args.cache, verify_hashes=True)
    rows = load_train_rows(args.manifest, cache)
    groups = np.asarray([template_group(row["clip_id"]) for row in rows], dtype=object)
    folds = np.asarray([group_fold(group, folds=config.folds, seed=20260824) for group in groups])
    train_indices = deterministic_activity_stratified_subset(
        np.flatnonzero(folds != config.fold), rows, limit=config.max_train_clips, seed=config.seed
    )
    held_indices = deterministic_activity_stratified_subset(
        np.flatnonzero(folds == config.fold), rows, limit=config.max_heldout_clips, seed=config.seed + 1
    )
    if set(groups[train_indices]) & set(groups[held_indices]):
        raise AssertionError("template group 跨 VideoMAE fold 泄漏")
    train_rows = [rows[index] for index in train_indices]
    held_rows = [rows[index] for index in held_indices]
    train_labels = torch.tensor([float(row["has_fall"]) for row in train_rows])
    held_labels = np.asarray([float(row["has_fall"]) for row in held_rows], dtype=np.float32)
    if not train_labels.any() or train_labels.all() or not held_labels.any() or held_labels.all():
        raise ValueError("VideoMAE train/held-out 均需包含正负 clips")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("请求 CUDA，但当前不可用")
    set_deterministic(config.seed)
    model = VideoMaeClipTeacher.from_pretrained(
        args.model_id, frames=config.frames, image_size=config.image_size, cache_dir=str(args.hf_cache)
    ).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    with VideoSourceResolver(args.temp_root) as resolver, ThreadPoolExecutor(max_workers=config.decode_workers) as pool:
        for epoch in range(config.epochs):
            warmup = epoch < config.warmup_epochs
            model.set_trainable_backbone_layers(
                config.warmup_last_layers if warmup else None
            )
            optimizer = torch.optim.AdamW(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                lr=config.learning_rate * (1.0 if warmup else 0.5),
                weight_decay=config.weight_decay,
            )
            model.train()
            order = np.random.default_rng(config.seed + epoch).permutation(len(train_rows))
            total_loss = 0.0
            seen = 0
            for start in range(0, order.size, config.batch_size):
                offsets = order[start : start + config.batch_size]
                batch_rows = [train_rows[int(offset)] for offset in offsets]
                pixels = _decode_batch(batch_rows, resolver=resolver, pool=pool, config=config).to(device, non_blocking=True)
                labels = train_labels[offsets].to(device)
                subtypes = torch.tensor([subtype_target(row) for row in batch_rows], device=device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    output = model(pixels)
                    positive_weight = float(
                        (len(train_rows) - train_labels.sum()) / train_labels.sum()
                    )
                    loss = teacher_loss(
                        output,
                        labels,
                        subtypes,
                        subtype_weight=config.subtype_weight,
                        positive_weight=positive_weight,
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                total_loss += float(loss.detach()) * len(batch_rows)
                seen += len(batch_rows)
            scores = _scores(model, held_rows, resolver=resolver, pool=pool, config=config, device=device)
            record = {
                "epoch": epoch,
                "phase": "warmup" if warmup else "full_finetune",
                "train_loss": total_loss / seen,
                "oof": metric_result(
                    [row["clip_id"] for row in held_rows], held_labels, scores
                ),
            }
            history.append(record)
            print(json.dumps({"stage": "videomae_rgb_teacher", **record}, ensure_ascii=False, sort_keys=True), flush=True)
        final_scores = _scores(
            model,
            held_rows,
            resolver=resolver,
            pool=pool,
            config=config,
            device=device,
        )
    _atomic_checkpoint(args.output_dir / "last.pt", {"model_state": model.state_dict(), "config": asdict(config), "history": history})
    np.savez_compressed(args.output_dir / "oof_predictions.npz", clip_ids=np.asarray([row["clip_id"] for row in held_rows]), labels=held_labels, scores=final_scores, fold=np.full(len(held_rows), config.fold, dtype=np.int64))
    summary = {
        "protocol": "template_grouped_train_only_full_clip_videomae_rgb_teacher_canary_v1",
        "qualification": "outer_fold_canary_only; full_five_fold_oof_required_for_teacher_gate",
        "model_id": args.model_id, "manifest_sha256": _sha256_file(args.manifest),
        "cache_signature_sha256": cache.metadata["signature_sha256"], "fold": config.fold,
        "train_clips": len(train_rows), "oof_clips": len(held_rows), "config": asdict(config),
        "final_metrics": history[-1]["oof"], "checkpoint_sha256": _sha256_file(args.output_dir / "last.pt"),
        "test_accessed": False,
    }
    _atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
