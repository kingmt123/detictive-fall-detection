"""Evaluate the frozen CAUCA temporal-contrast adapter on strict OF-Syn OOF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from models.tcn_dataset import load_window_cache
from tools.audit_paired_clip_predictions import (
    metrics_from_arrays,
    paired_group_bootstrap,
)
from tools.benchmark_edgefall_seven_head_tail import _load_checkpoint
from tools.build_tcn_multistream_sidecar import load_sidecar
from tools.evaluate_gmdcsa24_seven_head import _atomic_npz, _short_checkpoint_grid
from tools.evaluate_rmpts_diagnostic import _logit, load_member_matrix
from tools.freeze_caucafall_temporal_contrast_adapter import (
    FIXED_CANDIDATE,
    score_frozen_adapter,
)
from tools.train_edgefall_f1 import (
    F1Config,
    build_skeleton_model,
    clip_embeddings,
)
from tools.train_tcn import _append_sidecar, _atomic_json, _materialize, _sha256_file

PROTOCOL = "edgefall_ofsyn_frozen_temporal_contrast_adapter_oof_v1"
METRICS = ("clip_map_percent", "clip_p_at_r90", "clip_p_at_r95")


def heldout_window_indices(
    window_clip_indices: np.ndarray,
    clip_folds: np.ndarray,
    heldout_fold: int,
) -> np.ndarray:
    """Return windows belonging only to the requested OOF fold."""
    windows = np.asarray(window_clip_indices, dtype=np.int64)
    folds = np.asarray(clip_folds, dtype=np.int64)
    if windows.ndim != 1 or folds.ndim != 1 or windows.size == 0:
        raise ValueError("OOF embedding window/fold shape invalid")
    if windows.min() < 0 or windows.max() >= folds.size:
        raise ValueError("OOF embedding window clip index invalid")
    selected = np.flatnonzero(folds[windows] == int(heldout_fold))
    if selected.size == 0:
        raise ValueError("OOF embedding heldout fold is empty")
    return selected


def gate_decision(
    delta: dict[str, float],
    bootstrap: dict[str, list[float]],
    fold_map_deltas: list[float],
) -> dict[str, bool]:
    """Apply the pre-registered deployment qualification gates."""
    gates = {
        "map_at_least_plus_one_point": delta["clip_map_percent"] >= 1.0,
        "p90_nonnegative": delta["clip_p_at_r90"] >= 0.0,
        "p95_nonnegative": delta["clip_p_at_r95"] >= 0.0,
        "map_bootstrap_lower_positive": bootstrap["clip_map_percent"][0] > 0.0,
        "at_least_four_of_five_fold_map_nonnegative": sum(
            value >= 0.0 for value in fold_map_deltas
        )
        >= 4,
    }
    gates["passes_all"] = all(gates.values())
    return gates


def _cache_identity(cache: Any) -> list[str]:
    ids = [str(row["clip_id"]) for row in cache.metadata.get("clips", [])]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("OOF embedding cache clip identity invalid")
    return ids


def _extract_fold_embeddings(
    *,
    checkpoint: Path,
    cache: Any,
    sidecar: Any,
    cache_fold_ids: np.ndarray,
    fold: int,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    all_window_clips = np.asarray(cache.array("clip_indices"), dtype=np.int64)
    selected_windows = heldout_window_indices(all_window_clips, cache_fold_ids, fold)
    features, _, original_clips = _materialize(cache, selected_windows)
    features = _append_sidecar(features, sidecar, selected_windows)
    payload = _load_checkpoint(checkpoint)
    config = F1Config(**payload["config"])
    if config.fold != fold:
        raise ValueError(f"OOF embedding checkpoint fold mismatch: {checkpoint}")
    model = build_skeleton_model(config).to(device)
    model.load_state_dict(payload["skeleton_model_state"], strict=True)
    clip_indices, embeddings = clip_embeddings(
        model,
        features,
        original_clips,
        device=device,
        batch_size=batch_size,
    )
    expected = np.flatnonzero(cache_fold_ids == fold)
    if not np.array_equal(clip_indices, expected):
        raise ValueError("OOF embedding heldout clip coverage mismatch")
    del model, features
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return clip_indices, embeddings


def _comparison(
    clip_ids: np.ndarray,
    labels: np.ndarray,
    baseline: np.ndarray,
    challenger: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, float], dict[str, list[float]]]:
    baseline_metrics = metrics_from_arrays(labels, baseline)
    challenger_metrics = metrics_from_arrays(labels, challenger)
    delta = {
        name: challenger_metrics[name] - baseline_metrics[name] for name in METRICS
    }
    bootstrap = paired_group_bootstrap(
        clip_ids,
        labels,
        baseline,
        challenger,
        replicates=replicates,
        seed=seed,
    )
    return (
        {
            "baseline": baseline_metrics,
            "challenger": challenger_metrics,
            "delta": delta,
            "paired_template_group_bootstrap_delta_95ci": bootstrap,
        },
        delta,
        bootstrap,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--member-report", type=Path, required=True)
    parser.add_argument("--dense-oof", type=Path, required=True)
    parser.add_argument("--dense-checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--short-cache", type=Path, required=True)
    parser.add_argument("--short-sidecar", type=Path, required=True)
    parser.add_argument("--dense-cache", type=Path, required=True)
    parser.add_argument("--dense-sidecar", type=Path, required=True)
    parser.add_argument("--frozen-adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--embedding-batch-size", type=int, default=1024)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260901)
    args = parser.parse_args()
    if args.output.exists() or args.predictions.exists():
        raise FileExistsError("OF-Syn temporal-contrast output already exists")
    if len(args.dense_checkpoint) != 5 or args.bootstrap_replicates < 100:
        raise ValueError("OF-Syn temporal-contrast fold/bootstrap configuration invalid")
    lock = json.loads(args.protocol_lock.read_text(encoding="utf-8"))
    if (
        lock.get("protocol") != PROTOCOL
        or lock.get("status") != "locked_before_ofsyn_embedding_extraction"
        or lock.get("fixed_candidate") != FIXED_CANDIDATE
        or lock.get("success_gate", {}).get("map_delta_points") != 1.0
    ):
        raise ValueError("OF-Syn temporal-contrast protocol lock invalid")
    artifacts = lock.get("artifact_sha256")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("OF-Syn temporal-contrast lock has no artifact hashes")
    for path_text, expected in artifacts.items():
        artifact = Path(path_text)
        if not artifact.is_file() or _sha256_file(artifact) != expected:
            raise ValueError(f"OF-Syn temporal-contrast hash mismatch: {artifact}")

    with np.load(args.dense_oof, allow_pickle=False) as payload:
        required = {"clip_id", "label", "fold", "dense48_score"}
        if not required.issubset(payload.files):
            raise ValueError("OF-Syn dense OOF identity fields missing")
        clip_ids = np.asarray(payload["clip_id"]).astype(str)
        labels = np.asarray(payload["label"], dtype=np.uint8)
        folds = np.asarray(payload["fold"], dtype=np.int64)
        dense_score = np.asarray(payload["dense48_score"], dtype=np.float64)
    if (
        clip_ids.shape != (9600,)
        or len(set(clip_ids.tolist())) != 9600
        or sorted(set(folds.tolist())) != list(range(5))
    ):
        raise ValueError("OF-Syn dense OOF identity invalid")
    short_logits, _ = load_member_matrix(args.member_report, clip_ids, labels)
    member_logits = np.column_stack((short_logits, _logit(dense_score)))
    baseline = 1.0 / (1.0 + np.exp(-np.clip(member_logits.mean(axis=1), -30.0, 30.0)))

    short_cache = load_window_cache(args.short_cache, verify_hashes=True)
    dense_cache = load_window_cache(args.dense_cache, verify_hashes=True)
    for cache in (short_cache, dense_cache):
        if cache.metadata.get("dataset") != "of-syn" or cache.metadata.get("split") != "train":
            raise ValueError("OF-Syn temporal-contrast accepts train OOF cache only")
    short_sidecar = load_sidecar(args.short_sidecar, short_cache)
    dense_sidecar = load_sidecar(args.dense_sidecar, dense_cache)
    reference_fold = dict(zip(clip_ids.tolist(), folds.tolist(), strict=True))

    def cache_folds(cache: Any) -> tuple[list[str], np.ndarray]:
        ids = _cache_identity(cache)
        if set(ids) != set(reference_fold):
            raise ValueError("OF-Syn temporal-contrast cache universe mismatch")
        return ids, np.asarray([reference_fold[item] for item in ids], dtype=np.int64)

    short_ids, short_fold_ids = cache_folds(short_cache)
    dense_ids, dense_fold_ids = cache_folds(dense_cache)
    short_grid = _short_checkpoint_grid(args.member_report)
    challenger = np.full(labels.shape, np.nan, dtype=np.float64)
    checkpoint_evidence = []
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("OF-Syn temporal-contrast requested unavailable CUDA")
    with np.load(args.frozen_adapter, allow_pickle=False) as adapter:
        if str(adapter["candidate"]) != FIXED_CANDIDATE:
            raise ValueError("OF-Syn temporal-contrast frozen candidate mismatch")
        adapter_values = {name: np.asarray(adapter[name]).copy() for name in adapter.files}
    for fold in range(5):
        short_path = short_grid[0][fold]
        dense_path = args.dense_checkpoint[fold]
        short_indices, short_embedding = _extract_fold_embeddings(
            checkpoint=short_path,
            cache=short_cache,
            sidecar=short_sidecar,
            cache_fold_ids=short_fold_ids,
            fold=fold,
            device=device,
            batch_size=args.embedding_batch_size,
        )
        dense_indices, dense_embedding = _extract_fold_embeddings(
            checkpoint=dense_path,
            cache=dense_cache,
            sidecar=dense_sidecar,
            cache_fold_ids=dense_fold_ids,
            fold=fold,
            device=device,
            batch_size=args.embedding_batch_size,
        )
        short_fold_clip_ids = np.asarray([short_ids[index] for index in short_indices])
        dense_fold_clip_ids = np.asarray([dense_ids[index] for index in dense_indices])
        if not np.array_equal(short_fold_clip_ids, dense_fold_clip_ids):
            raise ValueError("OF-Syn short/dense heldout identity mismatch")
        output_index = {item: index for index, item in enumerate(clip_ids.tolist())}
        order = np.asarray([output_index[item] for item in short_fold_clip_ids], dtype=np.int64)
        if not np.all(folds[order] == fold):
            raise ValueError("OF-Syn extracted fold identity mismatch")
        challenger[order] = score_frozen_adapter(
            short_embedding,
            dense_embedding,
            member_logits[order],
            mean=adapter_values["mean"],
            scale=adapter_values["scale"],
            basis=adapter_values["basis"],
            projected_scale=adapter_values["projected_scale"],
            beta=adapter_values["beta"],
        )
        checkpoint_evidence.append(
            {
                "fold": fold,
                "clips": int(order.size),
                "short_checkpoint": str(short_path),
                "dense_checkpoint": str(dense_path),
            }
        )
        print(json.dumps({"stage": "fold_embeddings", **checkpoint_evidence[-1]}), flush=True)
    if not np.isfinite(challenger).all():
        raise ValueError("OF-Syn temporal-contrast OOF coverage incomplete")

    comparison, delta, bootstrap = _comparison(
        clip_ids,
        labels,
        baseline,
        challenger,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    fold_rows = []
    for fold in range(5):
        selected = folds == fold
        base_metrics = metrics_from_arrays(labels[selected], baseline[selected])
        candidate_metrics = metrics_from_arrays(labels[selected], challenger[selected])
        fold_rows.append(
            {
                "fold": fold,
                "clips": int(selected.sum()),
                "baseline": base_metrics,
                "challenger": candidate_metrics,
                "delta": {
                    name: candidate_metrics[name] - base_metrics[name] for name in METRICS
                },
            }
        )
    gates = gate_decision(
        delta,
        bootstrap,
        [row["delta"]["clip_map_percent"] for row in fold_rows],
    )
    _atomic_npz(
        args.predictions,
        clip_id=clip_ids,
        label=labels,
        fold=folds,
        incumbent_score=baseline.astype(np.float32),
        challenger_score=challenger.astype(np.float32),
    )
    report = {
        "protocol": PROTOCOL,
        "qualification": "strict_template_grouped_oof_frozen_external_adapter",
        "fixed_candidate": FIXED_CANDIDATE,
        "comparison": comparison,
        "folds": fold_rows,
        "gates": gates,
        "decision": "promote" if gates["passes_all"] else "stop_and_switch_direction",
        "checkpoints": checkpoint_evidence,
        "input_sha256": {
            "protocol_lock": _sha256_file(args.protocol_lock),
            "member_report": _sha256_file(args.member_report),
            "dense_oof": _sha256_file(args.dense_oof),
            "frozen_adapter": _sha256_file(args.frozen_adapter),
        },
        "predictions_sha256": _sha256_file(args.predictions),
        "test_accessed": False,
        "upfall_accessed": False,
        "limitations": [
            "the adapter candidate was selected post hoc on CAUCAFall development",
            "OF-Syn OOF is a development confirmation, not a sealed-test estimate",
            "the challenger adds an embedding-residual adapter to the seven-head deployment",
        ],
    }
    _atomic_json(args.output, report)
    print(json.dumps({"delta": delta, "gates": gates}, sort_keys=True))


if __name__ == "__main__":
    main()
