"""One-shot CAUCAFall confirmation and two-dataset cumulative audit."""

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
    _validate_lock as validate_base_lock,
)
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
from tools.train_tcn import _atomic_json, _sha256_file

METRIC_NAMES = ("clip_map_percent", "clip_p_at_r90", "clip_p_at_r95")


def manifest_identity(path: Path) -> tuple[list[str], np.ndarray, list[str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(row.get("dataset") != "caucafall_external" for row in rows):
        raise ValueError("CAUCAFall external manifest 身份无效")
    clip_ids = [row["clip_id"] for row in rows]
    labels = np.asarray([int(row["has_fall"]) for row in rows], dtype=np.uint8)
    groups = [row["group_id"] for row in rows]
    if (
        len(clip_ids) != 100
        or labels.sum() != 50
        or len(set(clip_ids)) != 100
        or len(set(groups)) != 10
    ):
        raise ValueError("CAUCAFall external manifest 数量或分组无效")
    return clip_ids, labels, groups


def explicit_group_bootstrap(
    labels: np.ndarray,
    baseline: np.ndarray,
    challenger: np.ndarray,
    groups: np.ndarray,
    datasets: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, list[float]]:
    """Bootstrap subjects inside each fixed dataset and preserve both domains."""
    dataset_names = sorted(set(datasets.tolist()))
    rows_by_dataset = []
    for dataset in dataset_names:
        names = sorted(set(groups[datasets == dataset].tolist()))
        rows_by_dataset.append(
            [np.flatnonzero((datasets == dataset) & (groups == group)) for group in names]
        )
    rng = np.random.default_rng(seed)
    deltas = np.empty((replicates, len(METRIC_NAMES)), dtype=np.float64)
    for replicate in range(replicates):
        indices = np.concatenate(
            [
                rows[index]
                for rows in rows_by_dataset
                for index in rng.integers(0, len(rows), len(rows))
            ]
        )
        base_metrics = metrics_from_arrays(labels[indices], baseline[indices])
        candidate_metrics = metrics_from_arrays(labels[indices], challenger[indices])
        deltas[replicate] = [
            candidate_metrics[name] - base_metrics[name] for name in METRIC_NAMES
        ]
    return {
        name: [
            float(np.quantile(deltas[:, index], 0.025)),
            float(np.quantile(deltas[:, index], 0.975)),
        ]
        for index, name in enumerate(METRIC_NAMES)
    }


def _metrics_delta(
    labels: np.ndarray, baseline: np.ndarray, challenger: np.ndarray
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    base_metrics = metrics_from_arrays(labels, baseline)
    candidate_metrics = metrics_from_arrays(labels, challenger)
    delta = {
        name: candidate_metrics[name] - base_metrics[name] for name in METRIC_NAMES
    }
    return base_metrics, candidate_metrics, delta


def _validate_protocol(path: Path) -> dict[str, Any]:
    lock = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        lock.get("protocol") != "caucafall_external_confirmation_v1"
        or lock.get("status") != "locked_before_model_inference"
    ):
        raise ValueError("CAUCAFall confirmation protocol lock 无效")
    for path_text, expected in lock.get("artifact_sha256", {}).items():
        artifact = Path(path_text)
        if not artifact.is_file() or _sha256_file(artifact) != expected:
            raise ValueError(f"CAUCAFall confirmation artifact hash 不匹配: {artifact}")
    validate_base_lock(Path(lock["base_protocol_lock"]))
    return lock


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
    parser.add_argument("--gmdcsa-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--embedding-batch-size", type=int, default=1024)
    parser.add_argument("--head-batch-size", type=int, default=256)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260831)
    args = parser.parse_args()
    if args.output.exists() or args.predictions.exists():
        raise FileExistsError("CAUCAFall confirmation 输出已存在，拒绝覆盖")
    if len(args.dense_checkpoint) != 5 or args.bootstrap_replicates < 100:
        raise ValueError("CAUCAFall confirmation checkpoint/bootstrap 配置无效")
    lock = _validate_protocol(args.protocol_lock)
    temperature = float(lock["challenger"]["shared_temperature"])
    if (
        lock["challenger"].get("delta") != "1/7"
        or not np.isclose(temperature, 0.541200679214602, rtol=0.0, atol=1e-15)
        or lock.get("fold_checkpoint_aggregation")
        != "mean_logit_within_each_member"
    ):
        raise ValueError("CAUCAFall confirmation candidate 签名无效")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CAUCAFall confirmation 请求 CUDA，但当前不可用")
    clip_ids, labels, groups = manifest_identity(args.manifest)
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
            raise ValueError("CAUCAFall window cache 未完整覆盖 manifest")
    roi_320 = _load_roi(args.roi_320, clip_ids, labels)
    roi_640 = _load_roi(args.roi_640, clip_ids, labels)
    short_grid = _short_checkpoint_grid(args.member_report)
    fold_logits = np.empty((5, len(clip_ids), 7), dtype=np.float64)

    for fold in range(5):
        short_paths = [short_grid[member][fold] for member in range(6)]
        short_payloads = [_load_checkpoint(path) for path in short_paths]
        configs = [F1Config(**payload["config"]) for payload in short_payloads]
        if any(config.fold != fold for config in configs) or any(
            config != configs[0] for config in configs[1:]
        ):
            raise ValueError(f"CAUCAFall short checkpoint fold/config 不匹配: {fold}")
        backbone_hashes = {
            _state_dict_sha256(payload["skeleton_model_state"])
            for payload in short_payloads
        }
        if len(backbone_hashes) != 1:
            raise ValueError(f"CAUCAFall short backbone 不一致: {fold}")
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
            raise ValueError("CAUCAFall short embeddings coverage 无效")
        short_tensor = torch.from_numpy(short_embeddings)
        all_indices = np.arange(len(clip_ids), dtype=np.int64)
        for member, payload in enumerate(short_payloads):
            probabilities = _head_scores(
                _head(payload).to(device),
                short_tensor,
                roi_320 if member % 2 == 0 else roi_640,
                all_indices,
                device=device,
                batch_size=args.head_batch_size,
            )
            fold_logits[fold, :, member] = _logit(probabilities)

        dense_payload = _load_checkpoint(args.dense_checkpoint[fold])
        dense_config = F1Config(**dense_payload["config"])
        if dense_config.fold != fold:
            raise ValueError(f"CAUCAFall dense checkpoint fold 不匹配: {fold}")
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
            raise ValueError("CAUCAFall dense embeddings coverage 无效")
        dense_probabilities = _head_scores(
            _head(dense_payload).to(device),
            torch.from_numpy(dense_embeddings),
            roi_320,
            all_indices,
            device=device,
            batch_size=args.head_batch_size,
        )
        fold_logits[fold, :, 6] = _logit(dense_probabilities)
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
    base_metrics, candidate_metrics, delta = _metrics_delta(
        labels, incumbent_scores, challenger_scores
    )
    group_array = np.asarray(groups)
    dataset_array = np.full(len(labels), "caucafall", dtype="U16")
    bootstrap = explicit_group_bootstrap(
        labels,
        incumbent_scores,
        challenger_scores,
        group_array,
        dataset_array,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    subject_rows = []
    for group in sorted(set(groups)):
        held = group_array == group
        b, c, d = _metrics_delta(
            labels[held], incumbent_scores[held], challenger_scores[held]
        )
        subject_rows.append(
            {"group_id": group, "incumbent": b, "challenger": c, "delta": d}
        )
    subject_map = [row["delta"]["clip_map_percent"] for row in subject_rows]
    caucafall_gates = {
        "global_map_improves": delta["clip_map_percent"] > 0.0,
        "global_p90_nonnegative": delta["clip_p_at_r90"] >= 0.0,
        "global_p95_nonnegative": delta["clip_p_at_r95"] >= 0.0,
        "map_bootstrap_lower_positive": bootstrap["clip_map_percent"][0] > 0.0,
        "at_least_eight_subject_map_nonnegative": sum(x >= 0 for x in subject_map) >= 8,
        "no_subject_map_regression_over_two_points": min(subject_map) >= -2.0,
    }

    with np.load(args.gmdcsa_predictions, allow_pickle=False) as payload:
        g_labels = np.asarray(payload["label"], dtype=np.uint8)
        g_groups = np.asarray(payload["group_id"]).astype(str)
        g_base = np.asarray(payload["incumbent_score"], dtype=np.float64)
        g_candidate = np.asarray(payload["challenger_score"], dtype=np.float64)
    cumulative_labels = np.concatenate([g_labels, labels])
    cumulative_groups = np.concatenate([g_groups, group_array])
    cumulative_datasets = np.concatenate(
        [np.full(len(g_labels), "gmdcsa24", dtype="U16"), dataset_array]
    )
    cumulative_base = np.concatenate([g_base, incumbent_scores])
    cumulative_candidate = np.concatenate([g_candidate, challenger_scores])
    cum_base, cum_candidate, cum_delta = _metrics_delta(
        cumulative_labels, cumulative_base, cumulative_candidate
    )
    cumulative_bootstrap = explicit_group_bootstrap(
        cumulative_labels,
        cumulative_base,
        cumulative_candidate,
        cumulative_groups,
        cumulative_datasets,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    dataset_deltas = {}
    for dataset in sorted(set(cumulative_datasets.tolist())):
        held = cumulative_datasets == dataset
        _, _, dataset_deltas[dataset] = _metrics_delta(
            cumulative_labels[held], cumulative_base[held], cumulative_candidate[held]
        )
    cumulative_gates = {
        "global_map_improves": cum_delta["clip_map_percent"] > 0.0,
        "global_p90_nonnegative": cum_delta["clip_p_at_r90"] >= 0.0,
        "global_p95_nonnegative": cum_delta["clip_p_at_r95"] >= 0.0,
        "map_bootstrap_lower_positive": cumulative_bootstrap["clip_map_percent"][0]
        > 0.0,
        "both_dataset_map_nonnegative": all(
            row["clip_map_percent"] >= 0.0 for row in dataset_deltas.values()
        ),
    }
    _atomic_npz(
        args.predictions,
        clip_id=np.asarray(clip_ids),
        label=labels,
        group_id=group_array,
        fold_member_logits=fold_logits.astype(np.float32),
        member_logits=member_logits.astype(np.float32),
        incumbent_score=incumbent_scores.astype(np.float32),
        challenger_score=challenger_scores.astype(np.float32),
    )
    report = {
        "protocol": "caucafall_external_confirmation_v1",
        "incumbent": "seven_member_equal_logit",
        "challenger": lock["challenger"],
        "caucafall": {
            "incumbent_metrics": base_metrics,
            "challenger_metrics": candidate_metrics,
            "delta": delta,
            "subject_bootstrap_delta_95ci": bootstrap,
            "subjects": subject_rows,
            "promotion_gates": caucafall_gates,
            "passes": all(caucafall_gates.values()),
        },
        "cumulative_gmdcsa24_plus_caucafall": {
            "subjects": 14,
            "clips": 260,
            "incumbent_metrics": cum_base,
            "challenger_metrics": cum_candidate,
            "delta": cum_delta,
            "dataset_stratified_subject_bootstrap_delta_95ci": cumulative_bootstrap,
            "dataset_deltas": dataset_deltas,
            "support_gates": cumulative_gates,
            "passes": all(cumulative_gates.values()),
        },
        "qualifies_for_promotion": all(caucafall_gates.values())
        and all(cumulative_gates.values()),
        "inputs": {
            "protocol_lock_sha256": _sha256_file(args.protocol_lock),
            "manifest_sha256": _sha256_file(args.manifest),
            "gmdcsa_predictions_sha256": _sha256_file(args.gmdcsa_predictions),
            "predictions_sha256": _sha256_file(args.predictions),
        },
        "test_accessed": False,
        "limitations": [
            "CAUCAFall source sequences were reconstructed from the downloadable action segments",
            "both external datasets contain staged falls by healthy participants",
            "the inference ensemble averages five fold checkpoints per member because all-data retrained checkpoints are unavailable",
            "no external data was used for training, calibration, thresholds, or candidate selection",
        ],
    }
    _atomic_json(args.output, report)
    print(json.dumps({"output": str(args.output), "qualifies": report["qualifies_for_promotion"]}))


if __name__ == "__main__":
    main()
