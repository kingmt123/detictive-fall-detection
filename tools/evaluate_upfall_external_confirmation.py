"""One-shot UP-Fall confirmation of a frozen temporal-contrast adapter."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tools.audit_paired_clip_predictions import metrics_from_arrays
from tools.benchmark_edgefall_seven_head_tail import _head, _load_checkpoint
from tools.evaluate_gmdcsa24_seven_head import (
    _atomic_npz,
    _features,
    _load_roi,
    _logit,
    _short_checkpoint_grid,
    _sigmoid,
)
from tools.evaluate_gmdcsa24_seven_head import (
    _validate_lock as validate_base_model_lock,
)
from tools.freeze_caucafall_temporal_contrast_adapter import (
    FIXED_CANDIDATE,
    score_frozen_adapter,
)
from tools.train_edgefall_f1 import (
    F1Config,
    _head_scores,
    _state_dict_sha256,
    build_skeleton_model,
    clip_embeddings,
)
from tools.train_tcn import _atomic_json, _sha256_file

METRICS = ("clip_map_percent", "clip_p_at_r90", "clip_p_at_r95")
HARD_NEGATIVE_SEMANTICS = ("fallen", "sitting", "lying", "standing", "other")


def _manifest_identity(
    path: Path,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(row.get("dataset") != "upfall_external" for row in rows):
        raise ValueError("UP-Fall manifest identity 无效")
    clip_ids = [row["clip_id"] for row in rows]
    labels = np.asarray([int(row["has_fall"]) for row in rows], dtype=np.uint8)
    groups = np.asarray([row["group_id"] for row in rows])
    cameras = np.asarray([row["camera"] for row in rows])
    semantics = np.asarray([row["activity_semantics"] for row in rows])
    if (
        len(clip_ids) != 2378
        or len(set(clip_ids)) != 2378
        or labels.sum() != 502
        or len(set(groups.tolist())) != 17
        or set(cameras.tolist()) != {"cam1", "cam2"}
        or any((cameras == camera).sum() != 1189 for camera in ("cam1", "cam2"))
    ):
        raise ValueError("UP-Fall manifest counts 无效")
    return clip_ids, labels, groups, cameras, semantics


def _group_bootstrap(
    labels: np.ndarray,
    baseline: np.ndarray,
    challenger: np.ndarray,
    groups: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, list[float]]:
    group_rows = [
        np.flatnonzero(groups == group) for group in sorted(set(groups.tolist()))
    ]
    rng = np.random.default_rng(seed)
    deltas = np.empty((replicates, len(METRICS)), dtype=np.float64)
    for replicate in range(replicates):
        indices = np.concatenate(
            [group_rows[index] for index in rng.integers(0, len(group_rows), len(group_rows))]
        )
        base_metrics = metrics_from_arrays(labels[indices], baseline[indices])
        candidate_metrics = metrics_from_arrays(labels[indices], challenger[indices])
        deltas[replicate] = [
            candidate_metrics[name] - base_metrics[name] for name in METRICS
        ]
    return {
        name: [
            float(np.quantile(deltas[:, index], 0.025)),
            float(np.quantile(deltas[:, index], 0.975)),
        ]
        for index, name in enumerate(METRICS)
    }


def _delta(
    baseline: dict[str, float], challenger: dict[str, float]
) -> dict[str, float]:
    return {name: challenger[name] - baseline[name] for name in METRICS}


def _slice_metrics(
    labels: np.ndarray,
    baseline: np.ndarray,
    challenger: np.ndarray,
    values: np.ndarray,
    selected_values: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows = []
    for value in selected_values:
        selected = values == value
        base_metrics = metrics_from_arrays(labels[selected], baseline[selected])
        candidate_metrics = metrics_from_arrays(labels[selected], challenger[selected])
        rows.append(
            {
                "slice": value,
                "clips": int(selected.sum()),
                "incumbent_metrics": base_metrics,
                "challenger_metrics": candidate_metrics,
                "delta": _delta(base_metrics, candidate_metrics),
            }
        )
    return rows


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
    parser.add_argument("--frozen-adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--embedding-batch-size", type=int, default=1024)
    parser.add_argument("--head-batch-size", type=int, default=256)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260831)
    args = parser.parse_args()
    if args.output.exists() or args.predictions.exists():
        raise FileExistsError("UP-Fall confirmation 输出已存在，拒绝覆盖")
    if len(args.dense_checkpoint) != 5 or args.bootstrap_replicates < 100:
        raise ValueError("UP-Fall checkpoint/bootstrap 配置无效")
    lock = json.loads(args.protocol_lock.read_text(encoding="utf-8"))
    if (
        lock.get("protocol") != "upfall_temporal_contrast_external_confirmation_v4"
        or lock.get("status")
        != "locked_after_preinference_sidecar_float32_validation_fix"
        or lock.get("challenger", {}).get("fixed_candidate") != FIXED_CANDIDATE
        or lock.get("fold_checkpoint_aggregation") != "mean_logit_within_each_member"
    ):
        raise ValueError("UP-Fall confirmation protocol lock 无效")
    for path_text, expected in lock.get("artifact_sha256", {}).items():
        artifact = Path(path_text)
        if not artifact.is_file() or _sha256_file(artifact) != expected:
            raise ValueError(f"UP-Fall confirmation artifact hash 不匹配: {artifact}")
    validate_base_model_lock(Path(lock["base_model_protocol_lock"]))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("UP-Fall confirmation 请求 CUDA，但当前不可用")
    clip_ids, labels, groups, cameras, semantics = _manifest_identity(args.manifest)
    short_cache, short_features, short_window_clips = _features(
        args.short_cache, args.short_sidecar
    )
    dense_cache, dense_features, dense_window_clips = _features(
        args.dense_cache, args.dense_sidecar
    )
    for cache in (short_cache, dense_cache):
        cache_ids = [str(row["clip_id"]) for row in cache.metadata["clips"]]
        if cache_ids != clip_ids or any(
            int(row["window_count"]) <= 0 for row in cache.metadata["clips"]
        ):
            raise ValueError("UP-Fall window cache 未完整覆盖 manifest")
    roi_320 = _load_roi(args.roi_320, clip_ids, labels)
    roi_640 = _load_roi(args.roi_640, clip_ids, labels)
    short_grid = _short_checkpoint_grid(args.member_report)
    fold_logits = np.empty((5, len(clip_ids), 7), dtype=np.float64)
    short_folds = []
    dense_folds = []
    checkpoint_evidence = []
    all_indices = np.arange(len(clip_ids), dtype=np.int64)
    for fold in range(5):
        short_paths = [short_grid[member][fold] for member in range(6)]
        short_payloads = [_load_checkpoint(path) for path in short_paths]
        configs = [F1Config(**payload["config"]) for payload in short_payloads]
        if any(config.fold != fold for config in configs) or any(
            config != configs[0] for config in configs[1:]
        ):
            raise ValueError(f"UP-Fall short checkpoint config 不匹配: {fold}")
        if len(
            {
                _state_dict_sha256(payload["skeleton_model_state"])
                for payload in short_payloads
            }
        ) != 1:
            raise ValueError(f"UP-Fall short heads 未共享 fold backbone: {fold}")
        short_model = build_skeleton_model(configs[0]).to(device)
        short_model.load_state_dict(short_payloads[0]["skeleton_model_state"], strict=True)
        short_indices, short_embeddings = clip_embeddings(
            short_model,
            short_features,
            short_window_clips,
            device=device,
            batch_size=args.embedding_batch_size,
        )
        if not np.array_equal(short_indices, all_indices):
            raise ValueError("UP-Fall short embedding coverage 无效")
        short_folds.append(short_embeddings)
        short_tensor = torch.from_numpy(short_embeddings)
        for member, payload in enumerate(short_payloads):
            roi = roi_320 if member % 2 == 0 else roi_640
            probabilities = _head_scores(
                _head(payload).to(device),
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
            raise ValueError(f"UP-Fall dense checkpoint fold 不匹配: {fold}")
        dense_model = build_skeleton_model(dense_config).to(device)
        dense_model.load_state_dict(dense_payload["skeleton_model_state"], strict=True)
        dense_indices, dense_embeddings = clip_embeddings(
            dense_model,
            dense_features,
            dense_window_clips,
            device=device,
            batch_size=args.embedding_batch_size,
        )
        if not np.array_equal(dense_indices, all_indices):
            raise ValueError("UP-Fall dense embedding coverage 无效")
        dense_folds.append(dense_embeddings)
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
                "short": [str(path) for path in short_paths],
                "dense": str(dense_path),
            }
        )
        del short_model, dense_model, short_tensor
        if device.type == "cuda":
            torch.cuda.empty_cache()
    member_logits = fold_logits.mean(axis=0)
    short_average = np.stack(short_folds).mean(axis=0)
    dense_average = np.stack(dense_folds).mean(axis=0)
    with np.load(args.frozen_adapter, allow_pickle=False) as adapter:
        if str(adapter["candidate"]) != FIXED_CANDIDATE:
            raise ValueError("UP-Fall frozen adapter candidate 无效")
        challenger_scores = score_frozen_adapter(
            short_average,
            dense_average,
            member_logits,
            mean=adapter["mean"],
            scale=adapter["scale"],
            basis=adapter["basis"],
            projected_scale=adapter["projected_scale"],
            beta=adapter["beta"],
        )
    incumbent_scores = _sigmoid(member_logits.mean(axis=1))
    incumbent_metrics = metrics_from_arrays(labels, incumbent_scores)
    challenger_metrics = metrics_from_arrays(labels, challenger_scores)
    delta = _delta(incumbent_metrics, challenger_metrics)
    bootstrap = _group_bootstrap(
        labels,
        incumbent_scores,
        challenger_scores,
        groups,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    subject_rows = _slice_metrics(
        labels,
        incumbent_scores,
        challenger_scores,
        groups,
        tuple(sorted(set(groups.tolist()))),
    )
    camera_rows = _slice_metrics(
        labels,
        incumbent_scores,
        challenger_scores,
        cameras,
        ("cam1", "cam2"),
    )
    hard_negative_rows = []
    for semantic in HARD_NEGATIVE_SEMANTICS:
        selected = (labels == 1) | (semantics == semantic)
        base_metrics = metrics_from_arrays(labels[selected], incumbent_scores[selected])
        candidate_metrics = metrics_from_arrays(
            labels[selected], challenger_scores[selected]
        )
        hard_negative_rows.append(
            {
                "negative_semantic": semantic,
                "clips": int(selected.sum()),
                "delta": _delta(base_metrics, candidate_metrics),
            }
        )
    subject_map = [row["delta"]["clip_map_percent"] for row in subject_rows]
    camera_deltas = [value for row in camera_rows for value in row["delta"].values()]
    hard_negative_map = [
        row["delta"]["clip_map_percent"] for row in hard_negative_rows
    ]
    gates = {
        "global_map_improves": delta["clip_map_percent"] > 0.0,
        "global_p90_nonnegative": delta["clip_p_at_r90"] >= 0.0,
        "global_p95_nonnegative": delta["clip_p_at_r95"] >= 0.0,
        "subject_bootstrap_map_lower_positive": bootstrap["clip_map_percent"][0]
        > 0.0,
        "at_least_thirteen_subject_map_nonnegative": sum(
            value >= 0.0 for value in subject_map
        )
        >= 13,
        "no_subject_map_regression_over_two_points": min(subject_map) >= -2.0,
        "both_cameras_all_metrics_nonnegative": all(
            value >= 0.0 for value in camera_deltas
        ),
        "at_least_four_hard_negative_map_nonnegative": sum(
            value >= 0.0 for value in hard_negative_map
        )
        >= 4,
        "no_hard_negative_map_regression_over_two_points": min(hard_negative_map)
        >= -2.0,
    }
    _atomic_npz(
        args.predictions,
        clip_id=np.asarray(clip_ids),
        label=labels,
        group_id=groups,
        camera=cameras,
        semantics=semantics,
        member_logits=member_logits.astype(np.float32),
        short_embedding=short_average.astype(np.float32),
        dense_embedding=dense_average.astype(np.float32),
        incumbent_score=incumbent_scores.astype(np.float32),
        challenger_score=challenger_scores.astype(np.float32),
    )
    report = {
        "protocol": "upfall_temporal_contrast_external_confirmation_v4",
        "qualification": "third_untouched_cross_domain_one_shot_confirmation",
        "incumbent": "seven_member_equal_logit",
        "challenger": FIXED_CANDIDATE,
        "fold_checkpoint_aggregation": "mean_logit_within_each_member",
        "incumbent_metrics": incumbent_metrics,
        "challenger_metrics": challenger_metrics,
        "delta": delta,
        "paired_subject_bootstrap_delta_95ci": bootstrap,
        "subjects": subject_rows,
        "cameras": camera_rows,
        "hard_negative_semantics": hard_negative_rows,
        "promotion_gates": gates,
        "passes_all_promotion_gates": all(gates.values()),
        "decision": (
            "external_confirmation_pass"
            if all(gates.values())
            else "retain_equal_logit_incumbent"
        ),
        "inputs": {
            "protocol_lock_sha256": _sha256_file(args.protocol_lock),
            "manifest_sha256": _sha256_file(args.manifest),
            "frozen_adapter_sha256": _sha256_file(args.frozen_adapter),
            "predictions_sha256": _sha256_file(args.predictions),
            "checkpoints": checkpoint_evidence,
        },
        "test_accessed": False,
        "limitations": [
            "UP-Fall consists of staged actions by young participants",
            "two synchronized views are correlated and retained within subject bootstrap blocks",
            "the archive contains action segments rather than untrimmed source recordings",
            "five fold checkpoints are averaged because an all-data retrained deployment checkpoint is unavailable",
        ],
    }
    _atomic_json(args.output, report)
    print(json.dumps({"output": str(args.output), "decision": report["decision"]}))


if __name__ == "__main__":
    main()
