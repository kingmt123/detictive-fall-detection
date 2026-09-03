"""Evaluate train-grouped SVM and compact clip heads on frozen Stage-S1 evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from models.tcn_dataset import WindowMemmapCache, load_window_cache
from tools.build_tcn_multistream_sidecar import load_sidecar
from tools.train_long_context_event_oracle import group_fold, template_group
from tools.train_multiscale_multistream_mil import MILConfig
from tools.train_stage_s1_r2 import _base_model, _metrics
from tools.train_tcn import (
    _append_sidecar,
    _atomic_json,
    _sha256_file,
    set_deterministic,
)

FEATURE_NAMES = (
    "max_logit",
    "mean_logit",
    "std_logit",
    "min_logit",
    "q50_logit",
    "q75_logit",
    "q90_logit",
    "q95_logit",
    "top05_mean",
    "top10_mean",
    "top20_mean",
    "top1_top2_gap",
    "positive_fraction",
    "strong_fraction",
    "max_position",
    "log_window_count",
    "argmax_stage_fall_process",
    "argmax_stage_fallen",
    "argmax_stage_controlled",
    "argmax_stage_stationary",
    "max_stage_fall_process",
    "max_stage_fallen",
    "max_stage_controlled",
    "max_stage_stationary",
    "mean_stage_fall_process",
    "mean_stage_fallen",
    "mean_stage_controlled",
    "mean_stage_stationary",
    "argmax_pose_confidence",
    "argmax_joint_visibility",
    "argmax_track_observed",
    "argmax_bbox_step",
    "mean_pose_confidence",
    "mean_joint_visibility",
    "mean_track_observed",
    "mean_bbox_step",
)


@dataclass(frozen=True)
class HeadSpec:
    name: str
    kind: str
    l2: float
    gamma: float = 0.0


def _window_outputs(
    cache: WindowMemmapCache,
    sidecar: np.ndarray,
    model: nn.Module,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    logits = np.empty(cache.sample_count, dtype=np.float32)
    stages = np.empty((cache.sample_count, 4), dtype=np.float32)
    quality = np.empty((cache.sample_count, 4), dtype=np.float32)
    features = cache.array("features")
    model.eval()
    with torch.inference_mode():
        for start in range(0, cache.sample_count, batch_size):
            end = min(start + batch_size, cache.sample_count)
            pose = np.asarray(features[start:end], dtype=np.float32)
            geometry = np.asarray(sidecar[start:end], dtype=np.float32)
            x = _append_sidecar(torch.from_numpy(pose.copy()), geometry, None).to(
                device
            )
            fused = model.encode(x)
            class_logits = model.classifier(fused)
            assert model.stage_classifier is not None
            logits[start:end] = (class_logits[:, 1] - class_logits[:, 0]).cpu().numpy()
            stages[start:end] = model.stage_classifier(fused).softmax(-1).cpu().numpy()
            confidence = pose[..., 2]
            quality[start:end, 0] = confidence.mean((1, 2))
            quality[start:end, 1] = (confidence > 0).mean((1, 2))
            quality[start:end, 2] = geometry[..., 5].mean(1)
            bbox_step = np.linalg.norm(np.diff(geometry[..., :2], axis=1), axis=-1)
            quality[start:end, 3] = bbox_step.mean(1)
    return logits, stages, quality


def _top_mean(values: np.ndarray, fraction: float) -> float:
    count = max(1, math.ceil(values.size * fraction))
    return float(np.partition(values, values.size - count)[-count:].mean())


def build_clip_evidence(
    cache: WindowMemmapCache,
    logits: np.ndarray,
    stages: np.ndarray,
    quality: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    clips = cache.metadata["clips"]
    result = np.empty((len(clips), len(FEATURE_NAMES)), dtype=np.float32)
    labels = np.empty(len(clips), dtype=np.float32)
    base = np.empty(len(clips), dtype=np.float32)
    for index, clip in enumerate(clips):
        start, count = int(clip["window_start"]), int(clip["window_count"])
        if count < 1:
            raise ValueError("clip没有窗口")
        values = logits[start : start + count]
        stage = stages[start : start + count]
        q = quality[start : start + count]
        maximum = int(np.argmax(values))
        ordered = np.sort(values)
        gap = float(ordered[-1] - ordered[-2]) if count > 1 else 0.0
        row = (
            float(values.max()),
            float(values.mean()),
            float(values.std()),
            float(values.min()),
            *[float(value) for value in np.quantile(values, (0.5, 0.75, 0.9, 0.95))],
            _top_mean(values, 0.05),
            _top_mean(values, 0.10),
            _top_mean(values, 0.20),
            gap,
            float((values > 0).mean()),
            float((values > 1).mean()),
            maximum / max(1, count - 1),
            math.log1p(count),
            *stage[maximum].tolist(),
            *stage.max(0).tolist(),
            *stage.mean(0).tolist(),
            *q[maximum].tolist(),
            *q.mean(0).tolist(),
        )
        result[index] = np.asarray(row, dtype=np.float32)
        labels[index] = float(bool(clip["has_fall"]))
        base[index] = float(values[maximum])
    return result, labels, base


class ClipHead(nn.Module):
    def __init__(self, dimension: int, kind: str, *, seed: int, gamma: float) -> None:
        super().__init__()
        self.kind = kind
        if kind == "rff_svm":
            generator = torch.Generator().manual_seed(seed)
            self.register_buffer(
                "rff_weight",
                torch.randn(dimension, 256, generator=generator) * math.sqrt(2 * gamma),
            )
            self.register_buffer(
                "rff_bias", torch.rand(256, generator=generator) * (2 * math.pi)
            )
            output_dimension = 256
        else:
            self.rff_weight = None
            self.rff_bias = None
            output_dimension = dimension
        self.predictor = (
            nn.Sequential(nn.Linear(dimension, 32), nn.GELU(), nn.Linear(32, 1))
            if kind == "mlp"
            else nn.Linear(output_dimension, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "rff_svm":
            assert self.rff_weight is not None and self.rff_bias is not None
            x = math.sqrt(2 / 256) * torch.cos(x @ self.rff_weight + self.rff_bias)
        return self.predictor(x).squeeze(-1)


def _standardize(
    train: np.ndarray, other: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(0)
    scale = train.std(0)
    scale[scale < 1e-6] = 1.0
    return (train - mean) / scale, (other - mean) / scale, np.stack((mean, scale))


def fit_head(
    train_x: np.ndarray,
    train_y: np.ndarray,
    score_x: np.ndarray,
    spec: HeadSpec,
    *,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, object]]:
    normalized_train, normalized_score, scaler = _standardize(train_x, score_x)
    x = torch.from_numpy(normalized_train.astype(np.float32)).to(device)
    y = torch.from_numpy(train_y.astype(np.float32)).to(device)
    model = ClipHead(x.shape[1], spec.kind, seed=seed, gamma=spec.gamma).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03, weight_decay=0.0)
    positive_weight = float((train_y.size - train_y.sum()) / train_y.sum())
    for _ in range(300):
        scores = model(x)
        if spec.kind in {"linear_svm", "rff_svm"}:
            signs = y * 2 - 1
            weights = torch.where(y > 0, positive_weight, 1.0)
            data_loss = (weights * torch.relu(1 - signs * scores)).mean()
        else:
            data_loss = nn.functional.binary_cross_entropy_with_logits(
                scores, y, pos_weight=torch.tensor(positive_weight, device=device)
            )
        penalty = sum(parameter.square().sum() for parameter in model.parameters())
        loss = data_loss + spec.l2 * penalty
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.inference_mode():
        predictions = (
            model(torch.from_numpy(normalized_score.astype(np.float32)).to(device))
            .cpu()
            .numpy()
        )
    payload = {
        "state": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "scaler": scaler,
        "spec": spec.__dict__,
    }
    return predictions, payload


def _score_metrics(
    clips: list[dict[str, object]], scores: np.ndarray
) -> dict[str, float]:
    return _metrics(1 / (1 + np.exp(-np.clip(scores, -30, 30))), clips)


def _objective(metrics: dict[str, float]) -> float:
    return (
        0.5 * metrics["clip_map"]
        + 0.2 * metrics["clip_p_at_r90"]
        + 0.3 * metrics["clip_p_at_r95"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--train-sidecar", type=Path, required=True)
    parser.add_argument("--val-sidecar", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("输出目录必须为空")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("请求CUDA但不可用")
    set_deterministic(20260826)
    config = MILConfig.from_json(args.config)
    train_cache = load_window_cache(args.train_cache, verify_hashes=True)
    val_cache = load_window_cache(args.val_cache, verify_hashes=True)
    if (
        train_cache.metadata.get("split") != "train"
        or val_cache.metadata.get("split") != "val"
    ):
        raise ValueError("只允许train/val cache")
    model = _base_model(config, args.checkpoint, device)
    args.output_dir.mkdir(parents=True)
    _atomic_json(args.output_dir / "status.json", {"stage": "extract_train"})
    train_outputs = _window_outputs(
        train_cache,
        load_sidecar(args.train_sidecar, train_cache),
        model,
        device=device,
        batch_size=args.batch_size,
    )
    train_x, train_y, train_base = build_clip_evidence(train_cache, *train_outputs)
    del train_outputs
    _atomic_json(args.output_dir / "status.json", {"stage": "extract_val"})
    val_outputs = _window_outputs(
        val_cache,
        load_sidecar(args.val_sidecar, val_cache),
        model,
        device=device,
        batch_size=args.batch_size,
    )
    val_x, val_y, val_base = build_clip_evidence(val_cache, *val_outputs)
    del val_outputs
    folds = np.asarray(
        [
            group_fold(template_group(str(clip["clip_id"])), folds=5, seed=20260826)
            for clip in train_cache.metadata["clips"]
        ],
        dtype=np.int16,
    )
    specs = (
        HeadSpec("linear_svm_l2_1e-3", "linear_svm", 1e-3),
        HeadSpec("linear_svm_l2_1e-2", "linear_svm", 1e-2),
        HeadSpec("rff_rbf_svm_g0.1", "rff_svm", 1e-3, 0.1),
        HeadSpec("rff_rbf_svm_g0.5", "rff_svm", 1e-3, 0.5),
        HeadSpec("logistic_l2_1e-3", "logistic", 1e-3),
        HeadSpec("logistic_l2_1e-2", "logistic", 1e-2),
        HeadSpec("mlp32_l2_1e-3", "mlp", 1e-3),
    )
    alphas = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
    train_clips, val_clips = train_cache.metadata["clips"], val_cache.metadata["clips"]
    results: list[dict[str, object]] = []
    best_key: tuple[float, str, float] | None = None
    best: tuple[HeadSpec, float] | None = None
    _atomic_json(args.output_dir / "status.json", {"stage": "train_grouped_oof_heads"})
    for spec_index, spec in enumerate(specs):
        oof = np.empty(train_y.size, dtype=np.float32)
        for fold in range(5):
            held = folds == fold
            oof[held], _ = fit_head(
                train_x[~held],
                train_y[~held],
                train_x[held],
                spec,
                seed=20260826 + spec_index * 10 + fold,
                device=device,
            )
        head_metrics = _score_metrics(train_clips, oof)
        oof_head_standard = (oof - oof.mean()) / max(float(oof.std()), 1e-6)
        base_standard = (train_base - train_base.mean()) / max(
            float(train_base.std()), 1e-6
        )
        blend_rows = []
        for alpha in alphas:
            blended = base_standard + alpha * oof_head_standard
            metrics = _score_metrics(train_clips, blended)
            blend_rows.append(
                {"alpha": alpha, "metrics": metrics, "objective": _objective(metrics)}
            )
            key = (_objective(metrics), spec.name, alpha)
            if best_key is None or key > best_key:
                best_key, best = key, (spec, alpha)
        results.append(
            {"spec": spec.__dict__, "oof_head": head_metrics, "oof_blends": blend_rows}
        )
        print(json.dumps({"spec": spec.name, "oof_head": head_metrics}), flush=True)
    assert best is not None
    _atomic_json(
        args.output_dir / "status.json", {"stage": "fit_full_and_validate_once"}
    )
    val_rows = []
    saved_models: dict[str, object] = {}
    for spec_index, spec in enumerate(specs):
        val_head, payload = fit_head(
            train_x,
            train_y,
            val_x,
            spec,
            seed=20260926 + spec_index,
            device=device,
        )
        train_head, _ = fit_head(
            train_x,
            train_y,
            train_x,
            spec,
            seed=20260926 + spec_index,
            device=device,
        )
        selected_alpha = best[1] if spec.name == best[0].name else 0.0
        head_standard = (val_head - train_head.mean()) / max(
            float(train_head.std()), 1e-6
        )
        base_standard = (val_base - train_base.mean()) / max(
            float(train_base.std()), 1e-6
        )
        blended = base_standard + selected_alpha * head_standard
        val_rows.append(
            {
                "spec": spec.__dict__,
                "selected": spec.name == best[0].name,
                "alpha": selected_alpha,
                "head_metrics": _score_metrics(val_clips, val_head),
                "blend_metrics": _score_metrics(val_clips, blended),
            }
        )
        saved_models[spec.name] = payload
    np.savez_compressed(
        args.output_dir / "clip_evidence.npz",
        feature_names=np.asarray(FEATURE_NAMES),
        train_x=train_x,
        train_y=train_y,
        train_base=train_base,
        folds=folds,
        val_x=val_x,
        val_y=val_y,
        val_base=val_base,
    )
    torch.save(saved_models, args.output_dir / "heads.pt")
    report = {
        "protocol": "stage_s1_frozen_clip_evidence_grouped_head_oof_v1",
        "integrity": {
            "head_selection": "train template-grouped OOF only",
            "validation": "single final pass after selection",
            "base_representation_caveat": "Stage-S1 was trained on all train clips; only the clip head is OOF",
            "test_accessed": False,
        },
        "checkpoint_sha256": _sha256_file(args.checkpoint),
        "feature_names": list(FEATURE_NAMES),
        "fold_assignment_sha256": hashlib.sha256(folds.tobytes()).hexdigest(),
        "base_validation": _score_metrics(val_clips, val_base),
        "oof_results": results,
        "selected": {
            "spec": best[0].__dict__,
            "alpha": best[1],
            "objective": best_key[0],
        },
        "validation_results": val_rows,
    }
    _atomic_json(args.output_dir / "report.json", report)
    _atomic_json(args.output_dir / "status.json", {"stage": "complete"})
    print(
        json.dumps(
            {
                "stage": "complete",
                "selected": report["selected"],
                "validation": val_rows,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
