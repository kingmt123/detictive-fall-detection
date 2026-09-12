"""Train one frozen-backbone F1 head with fixed CAUCAFall domain augmentation."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from models.edgefall_f1 import EdgeFallF1Head, SkeletonControlHead
from models.tcn_dataset import load_window_cache
from tools.audit_paired_clip_predictions import metrics_from_arrays
from tools.evaluate_gmdcsa24_seven_head import _features
from tools.train_edgefall_f1 import (
    SUBTYPES,
    F1Config,
    _head_scores,
    _state_dict_sha256,
    _train_heads,
    build_skeleton_model,
    clip_embeddings,
    load_embedding_cache,
    subtype_targets,
)
from tools.train_tcn import _atomic_json, _sha256_file, set_deterministic

PROTOCOL = "edgefall_caucafall_domain_augmented_head_fold_canary_v1"
EXTERNAL_DOMAIN_FRACTION = 0.20


def balanced_external_repeat_indices(
    train_count: int,
    external_count: int,
    *,
    target_fraction: float = EXTERNAL_DOMAIN_FRACTION,
) -> np.ndarray:
    """Repeat every external clip equally without exceeding the target fraction."""
    if train_count < 1 or external_count < 1 or not 0.0 < target_fraction < 0.5:
        raise ValueError("domain augmentation count/fraction 无效")
    target = train_count * target_fraction / (1.0 - target_fraction)
    repeats = max(1, math.floor(target / external_count))
    return np.tile(np.arange(external_count, dtype=np.int64), repeats)


def _load_roi(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        required = {"clip_ids", "tokens", "labels"}
        if not required.issubset(payload.files):
            raise ValueError(f"ROI cache 字段不完整: {path}")
        ids = np.asarray(payload["clip_ids"]).astype(str).tolist()
        tokens = np.asarray(payload["tokens"], dtype=np.float32)
        labels = np.asarray(payload["labels"], dtype=np.float32)
    if len(ids) != len(set(ids)) or tokens.shape[:2] != (len(ids), 16):
        raise ValueError(f"ROI cache 身份/shape 无效: {path}")
    return ids, tokens, labels


def _person_roi(ids: list[str], roi_ids: list[str], tokens: np.ndarray) -> torch.Tensor:
    index = {clip_id: offset for offset, clip_id in enumerate(roi_ids)}
    if any(clip_id not in index for clip_id in ids):
        raise ValueError("ROI cache 缺少请求 clip")
    return torch.from_numpy(
        np.stack([tokens[index[clip_id], :, 0] for clip_id in ids])
    )


def _external_subtypes(ids: list[str], labels: np.ndarray) -> torch.Tensor:
    values = []
    for clip_id, label in zip(ids, labels, strict=True):
        if label == 1:
            subtype = "fall"
        elif "sitdown" in clip_id.lower():
            subtype = "sit_down"
        else:
            subtype = "other"
        values.append(SUBTYPES.index(subtype))
    return torch.tensor(values, dtype=torch.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument("--base-window-cache", type=Path, required=True)
    parser.add_argument("--base-roi-cache", type=Path, required=True)
    parser.add_argument("--external-window-cache", type=Path, required=True)
    parser.add_argument("--external-sidecar", type=Path, required=True)
    parser.add_argument("--external-roi-cache", type=Path, required=True)
    parser.add_argument("--external-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--embedding-batch-size", type=int, default=1024)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("domain augmented head 输出目录必须为空")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(
        checkpoint.get("skeleton_model_state"), dict
    ):
        raise TypeError("base checkpoint 缺少 skeleton_model_state")
    config = F1Config(**checkpoint["config"])
    if config.fold != 0:
        raise ValueError("首轮 domain augmentation 只允许 fold0 canary")
    base_cache = load_window_cache(args.base_window_cache, verify_hashes=True)
    metadata = json.loads(str(np.load(args.embedding_cache, allow_pickle=False)["metadata"]))
    if (
        metadata.get("config") != asdict(config)
        or metadata.get("cache_signature_sha256")
        != base_cache.metadata.get("signature_sha256")
        or metadata.get("skeleton_state_sha256")
        != _state_dict_sha256(checkpoint["skeleton_model_state"])
    ):
        raise ValueError("embedding cache 与 base checkpoint/cache 不匹配")
    train_clips, train_embeddings, heldout_clips, heldout_embeddings = (
        load_embedding_cache(args.embedding_cache, expected_metadata=metadata)
    )
    clips = list(base_cache.metadata["clips"])
    base_indices = np.concatenate((train_clips, heldout_clips))
    base_ids = [str(clips[int(index)]["clip_id"]) for index in base_indices]
    base_roi_ids, base_roi_tokens, base_roi_labels = _load_roi(args.base_roi_cache)
    base_person_roi = _person_roi(base_ids, base_roi_ids, base_roi_tokens)
    base_labels = torch.tensor(
        [float(clips[int(index)]["has_fall"]) for index in base_indices]
    )
    if not np.array_equal(
        base_labels.numpy(),
        np.asarray(
            [base_roi_labels[base_roi_ids.index(clip_id)] for clip_id in base_ids]
        ),
    ):
        raise ValueError("OF-Syn ROI label 与 window cache 不匹配")
    base_subtypes = torch.from_numpy(subtype_targets(clips, base_indices))

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("请求 CUDA，但当前不可用")
    model = build_skeleton_model(config).to(device)
    model.load_state_dict(checkpoint["skeleton_model_state"], strict=True)
    external_cache, external_features, external_window_clips = _features(
        args.external_window_cache, args.external_sidecar
    )
    external_indices, external_embeddings = clip_embeddings(
        model,
        external_features,
        external_window_clips,
        device=device,
        batch_size=args.embedding_batch_size,
    )
    if not np.array_equal(
        external_indices, np.arange(len(external_cache.metadata["clips"]))
    ):
        raise ValueError("external embedding coverage 无效")
    external_ids = [
        str(row["clip_id"]) for row in external_cache.metadata["clips"]
    ]
    with args.external_manifest.open(encoding="utf-8", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    manifest = {row["clip_id"]: row for row in manifest_rows}
    if set(external_ids) != set(manifest):
        raise ValueError("external manifest/cache clip 集合不匹配")
    external_labels_np = np.asarray(
        [int(manifest[clip_id]["has_fall"]) for clip_id in external_ids],
        dtype=np.float32,
    )
    external_roi_ids, external_roi_tokens, external_roi_labels = _load_roi(
        args.external_roi_cache
    )
    if not np.array_equal(
        external_labels_np,
        np.asarray(
            [external_roi_labels[external_roi_ids.index(clip_id)] for clip_id in external_ids]
        ),
    ):
        raise ValueError("external ROI label 与 manifest 不匹配")
    external_person_roi = _person_roi(
        external_ids, external_roi_ids, external_roi_tokens
    )
    external_subtypes = _external_subtypes(external_ids, external_labels_np)

    repeats = balanced_external_repeat_indices(
        len(train_clips), len(external_ids)
    )
    train_count = len(train_clips)
    heldout_count = len(heldout_clips)
    skeleton = torch.from_numpy(
        np.concatenate(
            (
                train_embeddings,
                external_embeddings[repeats],
                heldout_embeddings,
            )
        )
    )
    person_roi = torch.cat(
        (
            base_person_roi[:train_count],
            external_person_roi[repeats],
            base_person_roi[train_count:],
        )
    )
    labels = torch.cat(
        (
            base_labels[:train_count],
            torch.from_numpy(external_labels_np[repeats]),
            base_labels[train_count:],
        )
    )
    subtypes = torch.cat(
        (
            base_subtypes[:train_count],
            external_subtypes[repeats],
            base_subtypes[train_count:],
        )
    )
    augmented_train_count = train_count + repeats.size
    train_indices = np.arange(augmented_train_count, dtype=np.int64)
    heldout_indices = np.arange(
        augmented_train_count,
        augmented_train_count + heldout_count,
        dtype=np.int64,
    )
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
        labels,
        subtypes,
        train_indices,
        config=config,
        device=device,
    )
    scores = _head_scores(
        fusion,
        skeleton,
        person_roi,
        heldout_indices,
        device=device,
        batch_size=config.head_batch_size,
    )
    heldout_ids = base_ids[train_count:]
    heldout_labels = base_labels[train_count:].numpy().astype(np.uint8)
    predictions = args.output_dir / "oof_predictions.npz"
    np.savez_compressed(
        predictions,
        clip_id=np.asarray(heldout_ids),
        label=heldout_labels,
        score=scores.astype(np.float32),
        fold=np.zeros(heldout_count, dtype=np.int64),
    )
    output_checkpoint = args.output_dir / "last.pt"
    torch.save(
        {
            "protocol": PROTOCOL,
            "config": asdict(config),
            "skeleton_model_state": checkpoint["skeleton_model_state"],
            "fusion_state": fusion.state_dict(),
            "external_domain_fraction": float(
                repeats.size / (train_count + repeats.size)
            ),
        },
        output_checkpoint,
    )
    summary = {
        "protocol": PROTOCOL,
        "fold": 0,
        "base_train_clips": train_count,
        "external_unique_clips": len(external_ids),
        "external_repeated_instances": int(repeats.size),
        "external_domain_fraction": float(repeats.size / augmented_train_count),
        "heldout_clips": heldout_count,
        "member_metrics": metrics_from_arrays(heldout_labels, scores),
        "input_sha256": {
            "base_checkpoint": _sha256_file(args.base_checkpoint),
            "embedding_cache": _sha256_file(args.embedding_cache),
            "base_roi_cache": _sha256_file(args.base_roi_cache),
            "external_manifest": _sha256_file(args.external_manifest),
            "external_roi_cache": _sha256_file(args.external_roi_cache),
        },
        "output_sha256": {
            "checkpoint": _sha256_file(output_checkpoint),
            "predictions": _sha256_file(predictions),
        },
        "selection_data": "OF-Syn train fold0 plus CAUCAFall development-only",
        "test_accessed": False,
        "upfall_accessed": False,
    }
    _atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
