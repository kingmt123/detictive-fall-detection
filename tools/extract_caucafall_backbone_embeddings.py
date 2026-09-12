"""Extract fold-averaged short/dense backbone embeddings for CAUCAFall development."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tools.benchmark_edgefall_seven_head_tail import _load_checkpoint
from tools.evaluate_caucafall_external_confirmation import (
    _validate_protocol as validate_external_protocol,
)
from tools.evaluate_caucafall_external_confirmation import (
    manifest_identity,
)
from tools.evaluate_gmdcsa24_seven_head import (
    _atomic_npz,
    _features,
    _short_checkpoint_grid,
)
from tools.train_edgefall_f1 import F1Config, build_skeleton_model, clip_embeddings
from tools.train_tcn import _atomic_json, _sha256_file


def average_fold_embeddings(values: list[np.ndarray]) -> np.ndarray:
    """Validate and average embeddings from the five frozen fold backbones."""
    if len(values) != 5:
        raise ValueError("CAUCAFall embedding 必须恰好包含五个 fold")
    shapes = {np.asarray(value).shape for value in values}
    if len(shapes) != 1:
        raise ValueError("CAUCAFall embedding fold shape 不一致")
    stacked = np.stack(values).astype(np.float64, copy=False)
    if stacked.ndim != 3 or stacked.shape[1] != 100 or not np.isfinite(stacked).all():
        raise ValueError("CAUCAFall embedding fold 内容无效")
    return stacked.mean(axis=0).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--member-report", type=Path, required=True)
    parser.add_argument("--dense-checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--short-cache", type=Path, required=True)
    parser.add_argument("--short-sidecar", type=Path, required=True)
    parser.add_argument("--dense-cache", type=Path, required=True)
    parser.add_argument("--dense-sidecar", type=Path, required=True)
    parser.add_argument("--existing-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise FileExistsError("CAUCAFall embedding 输出已存在，拒绝覆盖")
    if len(args.dense_checkpoint) != 5:
        raise ValueError("CAUCAFall embedding 必须提供五个 dense checkpoints")
    lock = json.loads(args.protocol_lock.read_text(encoding="utf-8"))
    if (
        lock.get("protocol") != "caucafall_backbone_embedding_development_v1"
        or lock.get("status") != "locked_before_embedding_extraction"
    ):
        raise ValueError("CAUCAFall embedding protocol lock 无效")
    for path_text, expected in lock.get("artifact_sha256", {}).items():
        path = Path(path_text)
        if not path.is_file() or _sha256_file(path) != expected:
            raise ValueError(f"CAUCAFall embedding artifact hash 不匹配: {path}")
    external_lock = validate_external_protocol(Path(lock["base_external_protocol_lock"]))
    if external_lock["dataset"].get("role") != "validation_only":
        raise ValueError("CAUCAFall base external protocol role 无效")
    clip_ids, labels, groups = manifest_identity(args.manifest)
    with np.load(args.existing_predictions, allow_pickle=False) as payload:
        if (
            np.asarray(payload["clip_id"]).astype(str).tolist() != clip_ids
            or not np.array_equal(np.asarray(payload["label"], dtype=np.uint8), labels)
        ):
            raise ValueError("CAUCAFall embedding existing prediction identity 不匹配")
        member_logits = np.asarray(payload["member_logits"], dtype=np.float32)
    short_cache, short_features, short_window_clips = _features(
        args.short_cache, args.short_sidecar
    )
    dense_cache, dense_features, dense_window_clips = _features(
        args.dense_cache, args.dense_sidecar
    )
    for cache in (short_cache, dense_cache):
        ids = [str(row["clip_id"]) for row in cache.metadata["clips"]]
        if ids != clip_ids:
            raise ValueError("CAUCAFall embedding cache identity 不匹配")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CAUCAFall embedding 请求 CUDA，但当前不可用")
    short_grid = _short_checkpoint_grid(args.member_report)
    short_folds: list[np.ndarray] = []
    dense_folds: list[np.ndarray] = []
    evidence = []
    for fold in range(5):
        short_path = short_grid[0][fold]
        dense_path = args.dense_checkpoint[fold]
        fold_evidence = {"fold": fold, "short": str(short_path), "dense": str(dense_path)}
        for kind, path, features, window_clips in (
            ("short", short_path, short_features, short_window_clips),
            ("dense", dense_path, dense_features, dense_window_clips),
        ):
            payload = _load_checkpoint(path)
            config = F1Config(**payload["config"])
            if config.fold != fold:
                raise ValueError(f"CAUCAFall embedding checkpoint fold 不匹配: {path}")
            model = build_skeleton_model(config).to(device)
            model.load_state_dict(payload["skeleton_model_state"], strict=True)
            indices, embeddings = clip_embeddings(
                model,
                features,
                window_clips,
                device=device,
                batch_size=args.batch_size,
            )
            if not np.array_equal(indices, np.arange(len(clip_ids))):
                raise ValueError("CAUCAFall embedding clip coverage 无效")
            if kind == "short":
                short_folds.append(embeddings)
            else:
                dense_folds.append(embeddings)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        evidence.append(fold_evidence)
    short_average = average_fold_embeddings(short_folds)
    dense_average = average_fold_embeddings(dense_folds)
    if short_average.shape[0] != 100 or dense_average.shape[0] != 100:
        raise ValueError("CAUCAFall embedding output shape 无效")
    _atomic_npz(
        args.output,
        clip_id=np.asarray(clip_ids),
        label=labels,
        group_id=np.asarray(groups),
        member_logits=member_logits,
        short_embedding=short_average,
        dense_embedding=dense_average,
    )
    report = {
        "protocol": "caucafall_backbone_embedding_development_v1",
        "aggregation": "mean_embedding_across_five_fold_backbones",
        "clips": len(clip_ids),
        "short_shape": list(short_average.shape),
        "dense_shape": list(dense_average.shape),
        "checkpoints": evidence,
        "inputs": {
            "protocol_lock_sha256": _sha256_file(args.protocol_lock),
            "manifest_sha256": _sha256_file(args.manifest),
            "existing_predictions_sha256": _sha256_file(args.existing_predictions),
        },
        "output_sha256": _sha256_file(args.output),
        "external_metrics_inspected": False,
        "test_accessed": False,
    }
    _atomic_json(args.report, report)
    print(json.dumps({"output": str(args.output), "shapes": report["short_shape"]}))


if __name__ == "__main__":
    main()
