"""One-shot external validation of two frozen seven-member fusion rules."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from models.tcn_dataset import load_window_cache
from tools.audit_paired_clip_predictions import (
    metrics_from_arrays,
    paired_group_bootstrap,
)
from tools.benchmark_edgefall_seven_head_tail import _head, _load_checkpoint
from tools.build_tcn_multistream_sidecar import load_sidecar
from tools.evaluate_probability_rmpts_calibration_crossfit import (
    calibrated_probability_rmpts,
)
from tools.train_edgefall_f1 import (
    F1Config,
    _head_scores,
    _state_dict_sha256,
    build_skeleton_model,
    clip_embeddings,
)
from tools.train_tcn import (
    _append_sidecar,
    _atomic_json,
    _materialize,
    _sha256_file,
)


def _logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    return np.log(clipped / (1.0 - clipped))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))


def _manifest_identity(path: Path) -> tuple[list[str], np.ndarray, list[str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(row.get("dataset") != "gmdcsa24" for row in rows):
        raise ValueError("GMDCSA24 manifest 身份无效")
    clip_ids = [row["clip_id"] for row in rows]
    labels = np.asarray([int(row["has_fall"]) for row in rows], dtype=np.uint8)
    groups = [row["group_id"] for row in rows]
    if len(clip_ids) != 160 or labels.sum() != 79 or len(set(clip_ids)) != 160:
        raise ValueError("GMDCSA24 manifest 数量或标签无效")
    if len(set(groups)) != 4:
        raise ValueError("GMDCSA24 manifest 必须恰好包含四个 subject groups")
    return clip_ids, labels, groups


def _load_roi(path: Path, clip_ids: list[str], labels: np.ndarray) -> torch.Tensor:
    with np.load(path, allow_pickle=False) as payload:
        ids = [str(value) for value in payload["clip_ids"]]
        tokens = np.asarray(payload["tokens"], dtype=np.float32)
        roi_labels = np.asarray(payload["labels"], dtype=np.uint8)
    if ids != clip_ids or not np.array_equal(roi_labels, labels):
        raise ValueError(f"GMDCSA24 ROI identity 不匹配: {path}")
    if tokens.shape != (len(clip_ids), 16, 2, 192):
        raise ValueError(f"GMDCSA24 ROI token shape 不匹配: {tokens.shape}")
    return torch.from_numpy(tokens[:, :, 0])


def _short_checkpoint_grid(member_report: Path) -> list[list[Path]]:
    report = json.loads(member_report.read_text(encoding="utf-8"))
    members = report.get("members")
    if not isinstance(members, list) or len(members) != 6:
        raise ValueError("GMDCSA24 七头评估要求六个短窗成员")
    grid: list[list[Path]] = []
    for member in members:
        rows = member.get("oof") if isinstance(member, dict) else None
        if not isinstance(rows, list) or len(rows) != 5:
            raise ValueError("GMDCSA24 短窗成员必须有五个 fold checkpoint")
        paths = [Path(row["path"]).parent / "last.pt" for row in rows]
        if any(not path.is_file() for path in paths):
            raise FileNotFoundError("GMDCSA24 短窗 checkpoint 缺失")
        grid.append(paths)
    return grid


def _features(cache_path: Path, sidecar_path: Path) -> tuple[Any, torch.Tensor, np.ndarray]:
    cache = load_window_cache(cache_path, verify_hashes=True)
    sidecar = load_sidecar(sidecar_path, cache)
    indices = np.arange(int(cache.metadata["sample_count"]), dtype=np.int64)
    features, _, clip_indices = _materialize(cache, indices)
    return cache, _append_sidecar(features, sidecar, indices), clip_indices


def _validate_lock(path: Path) -> dict[str, Any]:
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock.get("protocol") != "gmdcsa24_one_shot_external_validation_v1":
        raise ValueError("GMDCSA24 validation lock protocol 无效")
    if lock.get("status") != "locked_before_model_inference":
        raise ValueError("GMDCSA24 validation lock status 无效")
    artifacts = lock.get("artifact_sha256")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("GMDCSA24 validation lock 缺少 artifact hashes")
    for path_text, expected in artifacts.items():
        artifact = Path(path_text)
        if not artifact.is_file() or _sha256_file(artifact) != expected:
            raise ValueError(f"GMDCSA24 validation artifact hash 不匹配: {artifact}")
    return lock


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


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
    parser.add_argument("--roi-320", type=Path, required=True)
    parser.add_argument("--roi-640", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--embedding-batch-size", type=int, default=1024)
    parser.add_argument("--head-batch-size", type=int, default=256)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260831)
    args = parser.parse_args()
    if args.output.exists() or args.predictions.exists():
        raise FileExistsError("GMDCSA24 validation 输出已存在，拒绝覆盖")
    if len(args.dense_checkpoint) != 5:
        raise ValueError("GMDCSA24 validation 必须提供五个 dense checkpoints")
    if args.bootstrap_replicates < 100:
        raise ValueError("GMDCSA24 bootstrap 至少需要 100 次")
    lock = _validate_lock(args.protocol_lock)
    temperature = float(lock["challenger"]["shared_temperature"])
    if lock["challenger"].get("delta") != "1/7" or not np.isclose(
        temperature, 0.541200679214602, rtol=0.0, atol=1e-15
    ):
        raise ValueError("GMDCSA24 challenger 不符合冻结签名")
    if lock.get("fold_checkpoint_aggregation") != "mean_logit_within_each_member":
        raise ValueError("GMDCSA24 fold aggregation 不符合冻结签名")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("GMDCSA24 validation 请求 CUDA，但当前不可用")
    clip_ids, labels, groups = _manifest_identity(args.manifest)
    short_cache, short_features, short_window_clips = _features(
        args.short_cache, args.short_sidecar
    )
    dense_cache, dense_features, dense_window_clips = _features(
        args.dense_cache, args.dense_sidecar
    )
    for cache in (short_cache, dense_cache):
        cache_ids = [str(row["clip_id"]) for row in cache.metadata["clips"]]
        if cache_ids != clip_ids or any(int(row["window_count"]) <= 0 for row in cache.metadata["clips"]):
            raise ValueError("GMDCSA24 window cache 未完整覆盖 manifest")
    roi_320 = _load_roi(args.roi_320, clip_ids, labels)
    roi_640 = _load_roi(args.roi_640, clip_ids, labels)
    short_grid = _short_checkpoint_grid(args.member_report)
    fold_logits = np.empty((5, len(clip_ids), 7), dtype=np.float64)
    checkpoint_evidence: list[dict[str, Any]] = []

    for fold in range(5):
        short_paths = [short_grid[member][fold] for member in range(6)]
        short_payloads = [_load_checkpoint(path) for path in short_paths]
        configs = [F1Config(**payload["config"]) for payload in short_payloads]
        if any(config.fold != fold for config in configs) or any(
            config != configs[0] for config in configs[1:]
        ):
            raise ValueError(f"GMDCSA24 short checkpoint fold/config 不匹配: {fold}")
        backbone_hashes = {
            _state_dict_sha256(payload["skeleton_model_state"])
            for payload in short_payloads
        }
        if len(backbone_hashes) != 1:
            raise ValueError(f"GMDCSA24 short heads 未共享同一 fold backbone: {fold}")
        short_model = build_skeleton_model(configs[0]).to(device)
        short_model.load_state_dict(
            short_payloads[0]["skeleton_model_state"], strict=True
        )
        short_clip_indices, short_embeddings = clip_embeddings(
            short_model,
            short_features,
            short_window_clips,
            device=device,
            batch_size=args.embedding_batch_size,
        )
        if not np.array_equal(short_clip_indices, np.arange(len(clip_ids))):
            raise ValueError("GMDCSA24 short embeddings clip coverage 无效")
        short_tensor = torch.from_numpy(short_embeddings)
        all_indices = np.arange(len(clip_ids), dtype=np.int64)
        for member, payload in enumerate(short_payloads):
            head = _head(payload).to(device)
            roi = roi_320 if member % 2 == 0 else roi_640
            probabilities = _head_scores(
                head,
                short_tensor,
                roi,
                all_indices,
                device=device,
                batch_size=args.head_batch_size,
            )
            fold_logits[fold, :, member] = _logit(probabilities)

        dense_path = args.dense_checkpoint[fold]
        dense_payload = _load_checkpoint(dense_path)
        dense_config = F1Config(**dense_payload["config"])
        if dense_config.fold != fold:
            raise ValueError(f"GMDCSA24 dense checkpoint fold 不匹配: {fold}")
        dense_model = build_skeleton_model(dense_config).to(device)
        dense_model.load_state_dict(dense_payload["skeleton_model_state"], strict=True)
        dense_clip_indices, dense_embeddings = clip_embeddings(
            dense_model,
            dense_features,
            dense_window_clips,
            device=device,
            batch_size=args.embedding_batch_size,
        )
        if not np.array_equal(dense_clip_indices, np.arange(len(clip_ids))):
            raise ValueError("GMDCSA24 dense embeddings clip coverage 无效")
        dense_probabilities = _head_scores(
            _head(dense_payload).to(device),
            torch.from_numpy(dense_embeddings),
            roi_320,
            all_indices,
            device=device,
            batch_size=args.head_batch_size,
        )
        fold_logits[fold, :, 6] = _logit(dense_probabilities)
        checkpoint_evidence.append(
            {
                "fold": fold,
                "short": [
                    {"path": str(path), "sha256": _sha256_file(path)}
                    for path in short_paths
                ],
                "dense": {
                    "path": str(dense_path),
                    "sha256": _sha256_file(dense_path),
                },
            }
        )
        del short_model, dense_model, short_tensor
        if device.type == "cuda":
            torch.cuda.empty_cache()

    member_logits = fold_logits.mean(axis=0)
    incumbent_scores = _sigmoid(member_logits.mean(axis=1))
    challenger_scores = calibrated_probability_rmpts(
        member_logits[:, :6],
        member_logits[:, 6],
        np.full(7, temperature, dtype=np.float64),
    )
    incumbent_metrics = metrics_from_arrays(labels, incumbent_scores)
    challenger_metrics = metrics_from_arrays(labels, challenger_scores)
    delta = {
        key: challenger_metrics[key] - incumbent_metrics[key]
        for key in incumbent_metrics
    }
    bootstrap = paired_group_bootstrap(
        np.asarray(clip_ids),
        labels,
        incumbent_scores,
        challenger_scores,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    subject_rows = []
    for group in sorted(set(groups)):
        held = np.asarray([value == group for value in groups])
        baseline = metrics_from_arrays(labels[held], incumbent_scores[held])
        challenger = metrics_from_arrays(labels[held], challenger_scores[held])
        subject_rows.append(
            {
                "group_id": group,
                "clips": int(held.sum()),
                "incumbent_metrics": baseline,
                "challenger_metrics": challenger,
                "delta": {key: challenger[key] - baseline[key] for key in baseline},
            }
        )
    subject_map_deltas = [row["delta"]["clip_map_percent"] for row in subject_rows]
    gates = {
        "global_map_improves": delta["clip_map_percent"] > 0.0,
        "global_p90_nonnegative": delta["clip_p_at_r90"] >= 0.0,
        "global_p95_nonnegative": delta["clip_p_at_r95"] >= 0.0,
        "map_bootstrap_lower_positive": bootstrap["clip_map_percent"][0] > 0.0,
        "at_least_three_subject_map_nonnegative": sum(
            value >= 0.0 for value in subject_map_deltas
        )
        >= 3,
        "no_subject_map_regression_over_two_points": min(subject_map_deltas) >= -2.0,
    }
    _atomic_npz(
        args.predictions,
        clip_id=np.asarray(clip_ids),
        label=labels,
        group_id=np.asarray(groups),
        fold_member_logits=fold_logits.astype(np.float32),
        member_logits=member_logits.astype(np.float32),
        incumbent_score=incumbent_scores.astype(np.float32),
        challenger_score=challenger_scores.astype(np.float32),
    )
    report = {
        "protocol": "gmdcsa24_one_shot_external_validation_v1",
        "qualification": "new_cross_domain_one_shot_validation",
        "fold_checkpoint_aggregation": "mean_logit_within_each_member",
        "incumbent": "seven_member_equal_logit",
        "challenger": {
            "fusion": "probability_rmpts",
            "delta": "1/7",
            "shared_temperature": temperature,
        },
        "incumbent_metrics": incumbent_metrics,
        "challenger_metrics": challenger_metrics,
        "delta": delta,
        "paired_subject_bootstrap_delta_95ci": bootstrap,
        "subjects": subject_rows,
        "promotion_gates": gates,
        "passes_all_promotion_gates": all(gates.values()),
        "inputs": {
            "protocol_lock_sha256": _sha256_file(args.protocol_lock),
            "manifest_sha256": _sha256_file(args.manifest),
            "member_report_sha256": _sha256_file(args.member_report),
            "short_cache_metadata_sha256": _sha256_file(args.short_cache / "metadata.json"),
            "dense_cache_metadata_sha256": _sha256_file(args.dense_cache / "metadata.json"),
            "short_sidecar_metadata_sha256": _sha256_file(args.short_sidecar / "metadata.json"),
            "dense_sidecar_metadata_sha256": _sha256_file(args.dense_sidecar / "metadata.json"),
            "roi_320_sha256": _sha256_file(args.roi_320),
            "roi_640_sha256": _sha256_file(args.roi_640),
            "predictions_sha256": _sha256_file(args.predictions),
            "checkpoints": checkpoint_evidence,
        },
        "test_accessed": False,
        "limitations": [
            "GMDCSA24 has only four subjects and about 21 minutes of staged video",
            "the external inference model averages five fold checkpoints per member rather than using unavailable all-data retrained checkpoints",
            "this validates cross-domain ranking, not clinical performance or real-world prevalence",
            "GMDCSA24 was not used for training, calibration, threshold selection, or candidate selection",
        ],
    }
    _atomic_json(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "passes": report["passes_all_promotion_gates"],
            }
        )
    )


if __name__ == "__main__":
    main()
