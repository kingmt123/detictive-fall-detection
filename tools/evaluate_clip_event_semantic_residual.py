"""Evaluate frozen CLIP event-frame semantics as a seven-head OOF residual."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

from models.tcn_dataset import load_window_cache
from pipeline.video_source import VideoSourceResolver
from tools.audit_paired_clip_predictions import (
    metrics_from_arrays,
    paired_group_bootstrap,
)
from tools.evaluate_gmdcsa24_seven_head import _atomic_npz
from tools.evaluate_ofsyn_frozen_temporal_contrast_adapter import gate_decision
from tools.evaluate_rmpts_diagnostic import _logit, load_member_matrix
from tools.evaluate_support_contact_residual_crossfit import (
    RESIDUAL_L2,
    apply_residual,
    fit_residual,
)
from tools.train_event_aligned_rgb_oracle import crop_intervals, load_train_rows
from tools.train_tcn import _atomic_json, _sha256_file

PROTOCOL = "edgefall_clip_event_semantic_residual_crossfit_v1"
PROMPTS = (
    "a person accidentally falling to the ground",
    "a person losing balance and falling",
    "a person lying on the floor after a fall",
    "a person intentionally lying down",
    "a person sitting down safely",
    "a person resting or exercising on the floor",
    "a person standing up from the floor",
    "a person being helped by another person",
)
FRAME_NAMES = ("start", "maximum_descent", "end")


def frame_times(
    interval: tuple[float, float] | None,
    event_time: float,
    *,
    duration: float,
) -> np.ndarray:
    """Return fixed start/event/end times clipped to the valid video duration."""
    if not np.isfinite(duration) or duration <= 0.0 or not np.isfinite(event_time):
        raise ValueError("CLIP event video timing invalid")
    if interval is None:
        left, right = 0.0, duration
    else:
        left, right = map(float, interval)
        left = float(np.clip(left, 0.0, duration))
        right = float(np.clip(right, left, duration))
    if right <= left:
        right = min(duration, left + max(duration / 100.0, 1e-3))
    event = float(np.clip(event_time, left, right))
    return np.asarray((left, event, right), dtype=np.float64)


def semantic_features(
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
) -> np.ndarray:
    """Return the locked 3-frame by 8-prompt cosine feature matrix."""
    images = np.asarray(image_embeddings, dtype=np.float64)
    texts = np.asarray(text_embeddings, dtype=np.float64)
    if images.ndim != 3 or images.shape[1:] != (3, 512):
        raise ValueError("CLIP image embedding shape invalid")
    if texts.shape != (len(PROMPTS), 512):
        raise ValueError("CLIP text embedding shape invalid")
    images /= np.linalg.norm(images, axis=2, keepdims=True).clip(min=1e-12)
    texts /= np.linalg.norm(texts, axis=1, keepdims=True).clip(min=1e-12)
    result = np.einsum("bfd,pd->bfp", images, texts).reshape(images.shape[0], -1)
    if result.shape[1] != len(FRAME_NAMES) * len(PROMPTS) or not np.isfinite(result).all():
        raise ValueError("CLIP semantic feature values invalid")
    return result.astype(np.float32)


def decode_event_frames(
    path: Path,
    interval: tuple[float, float] | None,
    event_time: float,
) -> list[np.ndarray]:
    """Decode exactly three RGB frames from one materialized video."""
    capture = cv2.VideoCapture(str(path))
    try:
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if count < 2 or not np.isfinite(fps) or fps <= 0.0:
            raise ValueError(f"CLIP event video metadata invalid: {path}")
        duration = (count - 1) / fps
        times = frame_times(interval, event_time, duration=duration)
        positions = np.clip(np.rint(times * fps).astype(np.int64), 0, count - 1)
        frames = []
        for position in positions:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(position))
            ok, frame = capture.read()
            if not ok or frame is None:
                raise ValueError(f"CLIP event frame decode failed: {path}@{position}")
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return frames
    finally:
        capture.release()


@torch.inference_mode()
def extract_features(
    rows: list[dict[str, str]],
    intervals: list[tuple[float, float] | None],
    event_times: np.ndarray,
    *,
    model_root: Path,
    protocol_lock_sha256: str,
    work_dir: Path,
    temp_root: Path,
    device: torch.device,
    batch_size: int,
    decode_workers: int,
) -> np.ndarray:
    total = len(rows)
    if len(intervals) != total or np.asarray(event_times).shape != (total,):
        raise ValueError("CLIP event extraction identity length mismatch")
    work_dir.mkdir(parents=True, exist_ok=True)
    feature_path = work_dir / "features.npy"
    progress_path = work_dir / "progress.json"
    feature_dim = len(FRAME_NAMES) * len(PROMPTS)
    if progress_path.exists() or feature_path.exists():
        if not progress_path.is_file() or not feature_path.is_file():
            raise ValueError("CLIP event resume state incomplete")
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if (
            progress.get("protocol") != PROTOCOL
            or progress.get("protocol_lock_sha256") != protocol_lock_sha256
            or progress.get("total") != total
        ):
            raise ValueError("CLIP event resume signature mismatch")
        complete = int(progress.get("complete", -1))
        features = np.lib.format.open_memmap(feature_path, mode="r+")
        if features.shape != (total, feature_dim) or not 0 <= complete <= total:
            raise ValueError("CLIP event resume feature shape invalid")
    else:
        complete = 0
        features = np.lib.format.open_memmap(
            feature_path, mode="w+", dtype=np.float32, shape=(total, feature_dim)
        )
        _atomic_json(
            progress_path,
            {
                "protocol": PROTOCOL,
                "protocol_lock_sha256": protocol_lock_sha256,
                "complete": 0,
                "total": total,
            },
        )
    if complete == total:
        return np.asarray(features).copy()
    processor = CLIPProcessor.from_pretrained(
        model_root, local_files_only=True, use_fast=False
    )
    model = CLIPModel.from_pretrained(
        model_root,
        local_files_only=True,
        use_safetensors=False,
    ).eval().to(device)
    text_inputs = processor(text=list(PROMPTS), return_tensors="pt", padding=True)
    text_inputs = {name: value.to(device) for name, value in text_inputs.items()}
    text_embeddings = model.get_text_features(**text_inputs).float().cpu().numpy()
    with VideoSourceResolver(temp_root) as resolver, ThreadPoolExecutor(
        max_workers=decode_workers
    ) as pool:
        for start in range(complete, total, batch_size):
            stop = min(start + batch_size, total)
            batch_rows = rows[start:stop]
            with ExitStack() as stack:
                local_paths = [
                    stack.enter_context(resolver.materialize(row["video_path"])).local_path
                    for row in batch_rows
                ]
                decoded = list(
                    pool.map(
                        lambda item: decode_event_frames(*item),
                        zip(
                            local_paths,
                            intervals[start:stop],
                            event_times[start:stop].tolist(),
                            strict=True,
                        ),
                    )
                )
            flat_images = [frame for triplet in decoded for frame in triplet]
            image_inputs = processor(images=flat_images, return_tensors="pt")
            pixel_values = image_inputs["pixel_values"].to(device, non_blocking=True)
            image_embeddings = (
                model.get_image_features(pixel_values=pixel_values)
                .float()
                .cpu()
                .numpy()
                .reshape(stop - start, 3, 512)
            )
            features[start:stop] = semantic_features(image_embeddings, text_embeddings)
            features.flush()
            _atomic_json(
                progress_path,
                {
                    "protocol": PROTOCOL,
                    "protocol_lock_sha256": protocol_lock_sha256,
                    "complete": stop,
                    "total": total,
                },
            )
            print(json.dumps({"stage": "clip_features", "complete": stop, "total": total}), flush=True)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.asarray(features).copy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--event-triplet", type=Path, required=True)
    parser.add_argument("--member-report", type=Path, required=True)
    parser.add_argument("--dense-oof", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--temp-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--decode-workers", type=int, default=8)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260901)
    args = parser.parse_args()
    if args.output.exists() or args.predictions.exists():
        raise FileExistsError("CLIP event semantic final output already exists")
    if (
        args.batch_size < 1
        or not 1 <= args.decode_workers <= args.batch_size
        or args.bootstrap_replicates < 100
    ):
        raise ValueError("CLIP event runtime configuration invalid")
    lock = json.loads(args.protocol_lock.read_text(encoding="utf-8"))
    if (
        lock.get("protocol") != PROTOCOL
        or lock.get("status") != "locked_before_clip_inference"
        or tuple(lock.get("prompts", [])) != PROMPTS
        or lock.get("residual_l2") != RESIDUAL_L2
    ):
        raise ValueError("CLIP event semantic protocol lock invalid")
    for path_text, expected in lock.get("artifact_sha256", {}).items():
        path = Path(path_text)
        if not path.is_file() or _sha256_file(path) != expected:
            raise ValueError(f"CLIP event artifact hash mismatch: {path}")
    cache = load_window_cache(args.cache, verify_hashes=True)
    rows = load_train_rows(args.manifest, cache)
    cache_ids = [str(row["clip_id"]) for row in cache.metadata["clips"]]
    intervals = crop_intervals(cache, context_seconds=1.0)
    with np.load(args.event_triplet, allow_pickle=False) as payload:
        if np.asarray(payload["clip_ids"]).astype(str).tolist() != cache_ids:
            raise ValueError("CLIP event triplet identity mismatch")
        event_times = np.asarray(payload["event_times"], dtype=np.float64)
    with np.load(args.dense_oof, allow_pickle=False) as payload:
        clip_ids = np.asarray(payload["clip_id"]).astype(str)
        labels = np.asarray(payload["label"], dtype=np.uint8)
        folds = np.asarray(payload["fold"], dtype=np.int64)
        dense_score = np.asarray(payload["dense48_score"], dtype=np.float64)
    if clip_ids.shape != (9600,) or sorted(set(folds.tolist())) != list(range(5)):
        raise ValueError("CLIP event OOF identity invalid")
    if set(cache_ids) != set(clip_ids.tolist()):
        raise ValueError("CLIP event cache/OOF universe mismatch")
    protocol_hash = _sha256_file(args.protocol_lock)
    cache_features = extract_features(
        rows,
        intervals,
        event_times,
        model_root=args.model_root,
        protocol_lock_sha256=protocol_hash,
        work_dir=args.work_dir,
        temp_root=args.temp_root,
        device=torch.device(args.device),
        batch_size=args.batch_size,
        decode_workers=args.decode_workers,
    )
    cache_index = {clip_id: index for index, clip_id in enumerate(cache_ids)}
    features = cache_features[[cache_index[clip_id] for clip_id in clip_ids]]
    short_logits, _ = load_member_matrix(args.member_report, clip_ids, labels)
    baseline_logits = np.column_stack((short_logits, _logit(dense_score))).mean(axis=1)
    baseline = 1.0 / (1.0 + np.exp(-np.clip(baseline_logits, -30.0, 30.0)))
    challenger = np.empty(labels.shape, dtype=np.float64)
    fold_models: list[dict[str, Any]] = []
    for fold in range(5):
        train = folds != fold
        held = ~train
        fitted = fit_residual(features[train], labels[train], baseline_logits[train])
        challenger[held] = apply_residual(features[held], baseline_logits[held], fitted)
        fold_models.append(
            {
                "fold": fold,
                "mean": fitted["mean"].tolist(),
                "scale": fitted["scale"].tolist(),
                "coefficients": fitted["coefficients"].tolist(),
            }
        )
    names = ("clip_map_percent", "clip_p_at_r90", "clip_p_at_r95")
    baseline_metrics = metrics_from_arrays(labels, baseline)
    challenger_metrics = metrics_from_arrays(labels, challenger)
    delta = {name: challenger_metrics[name] - baseline_metrics[name] for name in names}
    bootstrap = paired_group_bootstrap(
        clip_ids,
        labels,
        baseline,
        challenger,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    fold_rows = []
    for fold in range(5):
        held = folds == fold
        first = metrics_from_arrays(labels[held], baseline[held])
        second = metrics_from_arrays(labels[held], challenger[held])
        fold_rows.append(
            {"fold": fold, "delta": {name: second[name] - first[name] for name in names}}
        )
    gates = gate_decision(
        delta,
        bootstrap,
        [row["delta"]["clip_map_percent"] for row in fold_rows],
    )
    feature_names = np.asarray(
        [f"{frame}:{prompt}" for frame in FRAME_NAMES for prompt in PROMPTS]
    )
    _atomic_npz(
        args.predictions,
        clip_id=clip_ids,
        label=labels,
        fold=folds,
        feature_names=feature_names,
        features=features,
        incumbent_score=baseline.astype(np.float32),
        challenger_score=challenger.astype(np.float32),
    )
    report = {
        "protocol": PROTOCOL,
        "qualification": "strict_template_grouped_crossfit_frozen_clip_semantics",
        "model_root": str(args.model_root),
        "prompts": list(PROMPTS),
        "frame_names": list(FRAME_NAMES),
        "residual_l2": RESIDUAL_L2,
        "comparison": {
            "baseline": baseline_metrics,
            "challenger": challenger_metrics,
            "delta": delta,
            "paired_template_group_bootstrap_delta_95ci": bootstrap,
        },
        "folds": fold_rows,
        "fold_models": fold_models,
        "gates": gates,
        "decision": "promote" if gates["passes_all"] else "stop_and_switch_direction",
        "input_sha256": {
            "protocol_lock": protocol_hash,
            "manifest": _sha256_file(args.manifest),
            "event_triplet": _sha256_file(args.event_triplet),
            "member_report": _sha256_file(args.member_report),
            "dense_oof": _sha256_file(args.dense_oof),
        },
        "predictions_sha256": _sha256_file(args.predictions),
        "test_accessed": False,
        "upfall_accessed": False,
        "limitations": [
            "CLIP prompt semantics are English zero-shot priors, not controlledness labels",
            "the residual is cross-fit on OF-Syn development OOF",
            "the route adds three CLIP image forwards per clip unless embeddings are cached",
        ],
    }
    _atomic_json(args.output, report)
    print(json.dumps({"delta": delta, "gates": gates}, sort_keys=True))


if __name__ == "__main__":
    main()
