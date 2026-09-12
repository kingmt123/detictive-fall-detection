"""从冻结的 pose cache 构建窗口并训练可复现的 FallTCN。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

# PyTorch 的严格 CUDA 确定性在线性层首次创建 cuBLAS handle 前需要此配置。
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn

from eval.metrics import competition_map
from models.clip_aggregator import aggregate_clip_logits
from models.lstm import (
    POSE51_FEATURES,
    POSE51_VELOCITY34_FEATURES,
    SUPPORTED_INPUT_FEATURES,
    add_velocity_features,
    feature_dimension,
)
from models.tcn import FallTCN, count_params
from models.tcn_dataset import WindowMemmapCache, build_window_cache


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "missing"


@dataclass(frozen=True)
class TrainingConfig:
    window_size: int = 16
    stride: int = 1
    min_observed_frames: int = 8
    epochs: int = 50
    batch_size: int = 1024
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    negative_ratio: float = 2.0
    hard_negative_activities: tuple[str, ...] = ()
    hard_negative_fraction: float = 0.0
    seed: int = 20260817
    channels: tuple[int, ...] = (64, 64, 128)
    kernel: int = 3
    dropout: float = 0.2
    input_features: str = POSE51_FEATURES
    grad_clip: float = 5.0

    @classmethod
    def from_json(cls, path: Path) -> TrainingConfig:
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("TCN config 不是合法 JSON") from exc
        if not isinstance(payload, dict):
            raise TypeError("TCN config 必须是 JSON 对象")
        unknown = set(payload) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"TCN config 包含未知字段: {sorted(unknown)}")
        if "channels" in payload:
            payload["channels"] = tuple(payload["channels"])
        if "hard_negative_activities" in payload:
            payload["hard_negative_activities"] = tuple(
                payload["hard_negative_activities"]
            )
        config = cls(**payload)
        config.validate()
        return config

    def validate(self) -> None:
        if self.window_size < 1 or self.stride < 1:
            raise ValueError("window_size 和 stride 必须为正整数")
        if not 1 <= self.min_observed_frames <= self.window_size:
            raise ValueError("min_observed_frames 必须位于 [1, window_size]")
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs 和 batch_size 必须为正整数")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate 必须为正，weight_decay 不能为负")
        if self.negative_ratio <= 0 or self.grad_clip <= 0:
            raise ValueError("negative_ratio 和 grad_clip 必须为正")
        if not 0.0 <= self.hard_negative_fraction <= 1.0:
            raise ValueError("hard_negative_fraction 必须位于 [0, 1]")
        if self.hard_negative_activities:
            if self.hard_negative_fraction <= 0.0:
                raise ValueError("指定 hard_negative_activities 时 fraction 必须为正")
            if any(not isinstance(name, str) or not name for name in self.hard_negative_activities):
                raise ValueError("hard_negative_activities 必须为非空字符串")
        elif self.hard_negative_fraction != 0.0:
            raise ValueError("hard_negative_fraction 需要 hard_negative_activities")
        if not self.channels or any(channel < 1 for channel in self.channels):
            raise ValueError("channels 必须包含正整数")
        if self.kernel < 1 or not 0 <= self.dropout < 1:
            raise ValueError("kernel 必须为正，dropout 必须位于 [0,1)")
        if self.input_features not in SUPPORTED_INPUT_FEATURES:
            raise ValueError(f"input_features 必须为 {SUPPORTED_INPUT_FEATURES}")


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    torch.use_deterministic_algorithms(True)


def select_training_indices(
    labels: np.ndarray,
    *,
    negative_ratio: float,
    seed: int,
    max_samples: int | None = None,
    activity_groups: np.ndarray | None = None,
    hard_negative_activities: tuple[str, ...] = (),
    hard_negative_fraction: float = 0.0,
) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.ndim != 1 or not np.all((labels == 0) | (labels == 1)):
        raise ValueError("labels 必须是一维二分类数组")
    if negative_ratio <= 0:
        raise ValueError("negative_ratio 必须为正")
    if max_samples is not None and max_samples < 2:
        raise ValueError("max_samples 必须至少为 2")
    if not 0.0 <= hard_negative_fraction <= 1.0:
        raise ValueError("hard_negative_fraction 必须位于 [0, 1]")
    if hard_negative_activities and hard_negative_fraction <= 0.0:
        raise ValueError("hard_negative_activities 需要正的 hard_negative_fraction")
    if not hard_negative_activities and hard_negative_fraction != 0.0:
        raise ValueError("hard_negative_fraction 需要 hard_negative_activities")
    positives = np.flatnonzero(labels == 1)
    negatives = np.flatnonzero(labels == 0)
    if not positives.size or not negatives.size:
        raise ValueError("训练集必须同时包含正负窗口")
    positive_target = positives.size
    if max_samples is not None:
        positive_target = min(
            positive_target,
            max(1, int(max_samples / (1.0 + negative_ratio))),
        )
    negative_target = min(negatives.size, round(positive_target * negative_ratio))
    if max_samples is not None:
        negative_target = min(negative_target, max_samples - positive_target)
    rng = np.random.default_rng(seed)
    selected_positive = rng.choice(positives, size=positive_target, replace=False)
    if not hard_negative_activities:
        selected_negative = rng.choice(negatives, size=negative_target, replace=False)
    else:
        if activity_groups is None:
            raise ValueError("困难负例配额需要 activity_groups")
        activity_groups = np.asarray(activity_groups)
        if activity_groups.shape != labels.shape:
            raise ValueError("activity_groups 必须与 labels 同形")
        is_hard = np.isin(activity_groups, hard_negative_activities)
        hard_candidates = negatives[is_hard[negatives]]
        other_candidates = negatives[~is_hard[negatives]]
        hard_target = round(negative_target * hard_negative_fraction)
        other_target = negative_target - hard_target
        if hard_candidates.size < hard_target or other_candidates.size < other_target:
            raise ValueError("困难负例配额超过可用的无重复负窗口数量")
        selected_hard = rng.choice(hard_candidates, size=hard_target, replace=False)
        selected_other = rng.choice(other_candidates, size=other_target, replace=False)
        selected_negative = np.concatenate((selected_hard, selected_other))
    return np.sort(np.concatenate((selected_positive, selected_negative))).astype(
        np.int64, copy=False
    )


def _materialize(
    cache: WindowMemmapCache,
    indices: np.ndarray | None,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    features = cache.array("features")
    labels = cache.array("labels")
    clip_indices = cache.array("clip_indices")
    if indices is None:
        x = np.array(features, dtype=np.float32, copy=True)
        y = np.array(labels, dtype=np.float32, copy=True)
        clips = np.array(clip_indices, dtype=np.int64, copy=True)
    else:
        x = np.asarray(features[indices], dtype=np.float32)
        y = np.asarray(labels[indices], dtype=np.float32)
        clips = np.asarray(clip_indices[indices], dtype=np.int64)
    return torch.from_numpy(x), torch.from_numpy(y), clips


def _append_sidecar(
    features: torch.Tensor,
    sidecar: np.ndarray,
    indices: np.ndarray | None,
) -> torch.Tensor:
    """Append a prevalidated per-window sidecar without mutating the base cache."""
    if features.ndim != 4 or features.shape[-2:] != (17, 3):
        raise ValueError("sidecar 仅支持原始 (B,T,17,3) 窗口")
    if sidecar.ndim != 3 or sidecar.shape[1] != features.shape[1]:
        raise ValueError("sidecar 的窗口维度必须与 features 一致")
    selected = (
        np.array(sidecar, dtype=np.float32, copy=True)
        if indices is None
        else np.array(sidecar[indices], dtype=np.float32, copy=True)
    )
    if selected.shape[0] != features.shape[0]:
        raise ValueError("sidecar 样本数与 features 不一致")
    return torch.cat((features.flatten(2), torch.from_numpy(selected)), dim=-1)


def train_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
    pos_weight: float,
    grad_clip: float,
    seed: int,
) -> float:
    model.train()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    order = torch.randperm(labels.numel(), generator=generator)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, device=device)
    )
    loss_sum = 0.0
    seen = 0
    for start in range(0, order.numel(), batch_size):
        batch_indices = order[start : start + batch_size]
        batch_x = features[batch_indices].to(device)
        batch_y = labels[batch_indices].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        count = batch_y.numel()
        loss_sum += float(loss.detach()) * count
        seen += count
    return loss_sum / seen


@torch.inference_mode()
def predict_logits(
    model: nn.Module,
    features: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    logits: list[np.ndarray] = []
    for start in range(0, features.shape[0], batch_size):
        batch_logits = model(features[start : start + batch_size].to(device))
        logits.append(batch_logits.cpu().numpy())
    return np.concatenate(logits).astype(np.float32, copy=False)


@torch.inference_mode()
def predict_probabilities(
    model: nn.Module,
    features: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    logits = predict_logits(model, features, device=device, batch_size=batch_size)
    return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32, copy=False)


def aggregate_group_logits(
    logits: torch.Tensor,
    group_sizes: list[int],
    *,
    mode: str,
    temperature: float = 0.1,
    topk_fraction: float = 0.2,
) -> torch.Tensor:
    """Backward-compatible entry point for the canonical clip aggregator."""
    return aggregate_clip_logits(
        logits, group_sizes, mode=mode, temperature=temperature,
        topk_fraction=topk_fraction,
    )


def evaluate_model(
    model: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    clip_indices: np.ndarray,
    clips: list[dict[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
    clip_aggregation: str = "max",
    aggregation_temperature: float = 0.1,
    topk_fraction: float = 0.2,
    clip_score_override: np.ndarray | None = None,
) -> dict[str, Any]:
    window_logits = predict_logits(
        model, features, device=device, batch_size=batch_size
    )
    probabilities = 1.0 / (1.0 + np.exp(-window_logits))
    labels_np = labels.numpy()
    eps = 1e-7
    loss = -np.mean(
        labels_np * np.log(np.clip(probabilities, eps, 1.0))
        + (1.0 - labels_np)
        * np.log(np.clip(1.0 - probabilities, eps, 1.0))
    )
    predictions = probabilities >= 0.5
    positives = labels_np == 1.0
    tp = int(np.count_nonzero(predictions & positives))
    fp = int(np.count_nonzero(predictions & ~positives))
    fn = int(np.count_nonzero(~predictions & positives))
    group_sizes: list[int] = []
    grouped_logits: list[np.ndarray] = []
    for clip_index in range(len(clips)):
        group = window_logits[clip_indices == clip_index]
        if not group.size:
            group = np.asarray([-np.inf], dtype=np.float32)
        grouped_logits.append(group)
        group_sizes.append(int(group.size))
    contiguous_logits = torch.from_numpy(np.concatenate(grouped_logits))
    clip_logits = aggregate_group_logits(
        contiguous_logits,
        group_sizes,
        mode=clip_aggregation,
        temperature=aggregation_temperature,
        topk_fraction=topk_fraction,
    ).numpy()
    clip_scores = 1.0 / (1.0 + np.exp(-clip_logits))
    clip_scores[~np.isfinite(clip_scores)] = 0.0
    if clip_score_override is not None:
        override = np.asarray(clip_score_override, dtype=np.float32)
        if override.shape != (len(clips),) or not np.isfinite(override).all():
            raise ValueError("clip_score_override 必须是有限的一维 clip 分数")
        clip_scores = override
    gt_by_clip = {clip["clip_id"]: bool(clip["has_fall"]) for clip in clips}
    pred_by_clip = {
        clip["clip_id"]: float(clip_scores[index])
        for index, clip in enumerate(clips)
    }
    result = competition_map(gt_by_clip, pred_by_clip, mode="clip")
    curve = result["curve"]

    def best_point(target: float) -> dict[str, Any] | None:
        feasible = [point for point in curve if point.recall >= target]
        if not feasible:
            return None
        point = max(feasible, key=lambda item: (item.precision, item.threshold))
        return {
            "threshold": point.threshold,
            "precision": point.precision,
            "recall": point.recall,
            "tp": point.tp,
            "fp": point.fp,
            "fn": point.fn,
        }

    return {
        "loss": float(loss),
        "window_accuracy": float(np.mean(predictions == positives)),
        "window_precision": tp / (tp + fp) if tp + fp else 0.0,
        "window_recall": tp / (tp + fn) if tp + fn else 0.0,
        "clip_p_at_r90": float(result["p_at_r90"]),
        "clip_p_at_r95": float(result["p_at_r95"]),
        "clip_map": float(result["map"]),
        "clip_map_percent": float(result["map_percent"]),
        "clip_r90_point": best_point(0.90),
        "clip_r95_point": best_point(0.95),
    }


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_checkpoint(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    try:
        torch.save(value, temp_name)
        with Path(temp_name).open("rb+") as handle:
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def _run_signature(
    *,
    config: TrainingConfig,
    train_cache: WindowMemmapCache,
    val_cache: WindowMemmapCache,
    device: str,
    max_train_samples: int | None,
    max_val_samples: int | None,
    protocol: str = "fall_tcn_training_v1",
    code_paths: tuple[Path, ...] | None = None,
    extra_signature: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    project_root = Path(__file__).parent.parent
    if code_paths is None:
        code_paths = (
            project_root / "models" / "tcn.py",
            project_root / "models" / "tcn_window.py",
            project_root / "models" / "tcn_dataset.py",
            project_root / "tools" / "train_tcn.py",
        )
    payload = {
        "protocol": protocol,
        "feature_protocol": config.input_features,
        "config": asdict(config),
        "train_window_signature": train_cache.metadata["signature_sha256"],
        "val_window_signature": val_cache.metadata["signature_sha256"],
        "device": device,
        "max_train_samples": max_train_samples,
        "max_val_samples": max_val_samples,
        "code_sha256": {
            str(path.relative_to(project_root)): _sha256_file(path)
            for path in code_paths
        },
        "dependencies": {
            name: _package_version(name) for name in ("numpy", "torch")
        },
    }
    if extra_signature is not None:
        payload["extra"] = extra_signature
    digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    return digest, payload


def train(
    *,
    config: TrainingConfig,
    train_cache: WindowMemmapCache,
    val_cache: WindowMemmapCache,
    output_dir: Path,
    device_name: str,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    resume: bool = False,
    model_factory: Callable[[], nn.Module] | None = None,
    protocol: str = "fall_tcn_training_v1",
    code_paths: tuple[Path, ...] | None = None,
    run_metadata: dict[str, Any] | None = None,
    sidecar_arrays: tuple[np.ndarray, np.ndarray] | None = None,
    activity_groups: np.ndarray | None = None,
) -> dict[str, Any]:
    config.validate()
    set_deterministic(config.seed)
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("请求 CUDA 训练，但 torch.cuda.is_available() 为 False")
    device = torch.device(device_name)
    train_labels = train_cache.array("labels")
    train_indices = select_training_indices(
        train_labels,
        negative_ratio=config.negative_ratio,
        seed=config.seed,
        max_samples=max_train_samples,
        activity_groups=activity_groups,
        hard_negative_activities=config.hard_negative_activities,
        hard_negative_fraction=config.hard_negative_fraction,
    )
    val_indices: np.ndarray | None = None
    if max_val_samples is not None and max_val_samples < val_cache.sample_count:
        val_indices = np.linspace(
            0, val_cache.sample_count - 1, max_val_samples, dtype=np.int64
        )
    train_x, train_y, _ = _materialize(train_cache, train_indices)
    val_x, val_y, val_clip_indices = _materialize(val_cache, val_indices)
    if sidecar_arrays is not None:
        train_x = _append_sidecar(train_x, sidecar_arrays[0], train_indices)
        val_x = _append_sidecar(val_x, sidecar_arrays[1], val_indices)
    elif config.input_features == POSE51_VELOCITY34_FEATURES:
        train_x = add_velocity_features(train_x)
        val_x = add_velocity_features(val_x)
    model = (
        model_factory()
        if model_factory is not None
        else FallTCN(
            in_dim=feature_dimension(config.input_features),
            channels=config.channels,
            kernel=config.kernel,
            dropout=config.dropout,
        )
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )
    signature_sha256, signature = _run_signature(
        config=config,
        train_cache=train_cache,
        val_cache=val_cache,
        device=device_name,
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
        protocol=protocol,
        code_paths=code_paths,
        extra_signature=run_metadata,
    )
    output_dir = Path(output_dir)
    run_path = output_dir / "run.json"
    start_epoch = 0
    best_map = -math.inf
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise ValueError("输出目录非空；如需续跑请显式传入 --resume")
    output_dir.mkdir(parents=True, exist_ok=True)
    if resume:
        try:
            existing = json.loads(run_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("resume 缺少合法 run.json") from exc
        if existing.get("signature_sha256") != signature_sha256:
            raise ValueError("resume run signature 不匹配")
        checkpoint = torch.load(
            output_dir / "last.pt", map_location=device, weights_only=False
        )
        if checkpoint.get("run_signature_sha256") != signature_sha256:
            raise ValueError("resume checkpoint signature 不匹配")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_map = float(checkpoint["best_map"])
    else:
        run_payload = {
                "signature_sha256": signature_sha256,
                "signature": signature,
                "parameter_count": count_params(model),
                "train_samples": int(train_y.numel()),
                "train_positive": int(train_y.sum()),
                "val_samples": int(val_y.numel()),
                "pilot": max_train_samples is not None or max_val_samples is not None,
            }
        if run_metadata is not None:
            run_payload["architecture"] = run_metadata
        _atomic_json(run_path, run_payload)
    positives = float(train_y.sum())
    negatives = float(train_y.numel() - positives)
    pos_weight = negatives / positives
    history_path = output_dir / "history.jsonl"
    history: list[dict[str, Any]] = []
    for epoch in range(start_epoch, config.epochs):
        train_loss = train_epoch(
            model,
            optimizer,
            train_x,
            train_y,
            device=device,
            batch_size=config.batch_size,
            pos_weight=pos_weight,
            grad_clip=config.grad_clip,
            seed=config.seed + epoch,
        )
        metrics = evaluate_model(
            model,
            val_x,
            val_y,
            val_clip_indices,
            val_cache.metadata["clips"],
            device=device,
            batch_size=config.batch_size,
        )
        scheduler.step()
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "val": metrics,
        }
        with history_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(_canonical_json(record) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        history.append(record)
        improved = metrics["clip_map"] > best_map
        if improved:
            best_map = metrics["clip_map"]
        checkpoint = {
            "epoch": epoch,
            "run_signature_sha256": signature_sha256,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": (
                torch.cuda.get_rng_state_all() if device.type == "cuda" else []
            ),
            "best_map": best_map,
            "metrics": record,
        }
        _atomic_checkpoint(output_dir / "last.pt", checkpoint)
        if improved:
            _atomic_checkpoint(output_dir / "best.pt", checkpoint)
        print(_canonical_json({"stage": "epoch", **record}), flush=True)
    summary = {
        "run_signature_sha256": signature_sha256,
        "epochs_completed": config.epochs,
        "best_clip_map": best_map,
        "last": history[-1] if history else None,
        "output_dir": str(output_dir.resolve()),
    }
    _atomic_json(output_dir / "summary.json", summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pose-cache-root", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--window-cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", default="of-syn")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = TrainingConfig.from_json(args.config)
    if args.epochs is not None:
        config = replace(config, epochs=args.epochs)
        config.validate()
    caches: dict[str, WindowMemmapCache] = {}
    for split in ("train", "val"):
        print(_canonical_json({"stage": "window_cache", "split": split}), flush=True)
        caches[split] = build_window_cache(
            args.manifest,
            args.pose_cache_root,
            args.audit_report,
            args.window_cache_root,
            dataset=args.dataset,
            split=split,
            window_size=config.window_size,
            stride=config.stride,
            min_observed_frames=config.min_observed_frames,
            progress=lambda state, split=split: print(
                _canonical_json(
                    {"stage": "window_cache_progress", "split": split, **state}
                ),
                flush=True,
            ),
        )
        print(
            _canonical_json(
                {
                    "stage": "window_cache_ready",
                    "split": split,
                    "samples": caches[split].sample_count,
                    "path": str(caches[split].root.resolve()),
                }
            ),
            flush=True,
        )
    summary = train(
        config=config,
        train_cache=caches["train"],
        val_cache=caches["val"],
        output_dir=args.output_dir,
        device_name=args.device,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        resume=args.resume,
    )
    print(_canonical_json({"stage": "complete", **summary}), flush=True)


if __name__ == "__main__":
    main()
