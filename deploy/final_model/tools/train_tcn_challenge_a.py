"""Precision Challenge A：语义/clip 均衡、难负挖掘与 clip-level MIL。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from models.clip_aggregator import aggregate_indexed_clip_logits
from models.tcn import FallTCN, count_params
from models.tcn_dataset import SEMANTICS, WindowMemmapCache, load_window_cache
from tools.train_tcn import evaluate_model, set_deterministic

PROTOCOL = "fall_tcn_precision_challenge_a_v1"
CHECKPOINT_SELECTION = "max_min_joint_gain_then_sum"
MODEL_CHANNELS = (64, 64, 128)
MODEL_KERNEL = 3
MODEL_DROPOUT = 0.2


@dataclass(frozen=True)
class ChallengeAConfig:
    epochs: int = 30
    clips_per_batch: int = 64
    inference_batch_size: int = 2048
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    semantic_per_clip: int = 4
    hard_negative_top_k: int = 8
    mil_top_k: int = 4
    clip_loss_weight: float = 1.0

    @classmethod
    def from_json(cls, path: Path) -> ChallengeAConfig:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("Challenge A config 必须是 JSON 对象")
        unknown = set(payload) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Challenge A config 包含未知字段: {sorted(unknown)}")
        config = cls(**payload)
        config.validate()
        return config

    def validate(self) -> None:
        integers = (
            self.epochs,
            self.clips_per_batch,
            self.inference_batch_size,
            self.semantic_per_clip,
            self.hard_negative_top_k,
            self.mil_top_k,
        )
        if any(value < 1 for value in integers):
            raise ValueError("Challenge A 的整数配置必须为正")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate/weight_decay 无效")
        if self.grad_clip <= 0 or self.clip_loss_weight < 0:
            raise ValueError("grad_clip/clip_loss_weight 无效")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def build_run_record(
    *,
    config: ChallengeAConfig,
    seeds: list[int],
    baseline_checkpoint: Path,
    baseline_ablation: Path,
    train_cache: WindowMemmapCache,
    val_cache: WindowMemmapCache,
    pilot: bool,
) -> dict[str, Any]:
    project_root = Path(__file__).parent.parent
    signature = {
        "protocol": PROTOCOL,
        "checkpoint_selection": CHECKPOINT_SELECTION,
        "config": {
            **asdict(config),
            "channels": list(MODEL_CHANNELS),
            "kernel": MODEL_KERNEL,
            "dropout": MODEL_DROPOUT,
        },
        "seeds": seeds,
        "train_window_signature": train_cache.metadata["signature_sha256"],
        "val_window_signature": val_cache.metadata["signature_sha256"],
        "source_sha256": {
            "baseline_checkpoint": _sha256_file(baseline_checkpoint),
            "baseline_ablation": _sha256_file(baseline_ablation),
        },
        "code_sha256": {
            relative: _sha256_file(project_root / relative)
            for relative in (
                "models/tcn.py",
                "models/tcn_dataset.py",
                "tools/train_tcn_challenge_a.py",
            )
        },
    }
    signature_sha256 = hashlib.sha256(
        _canonical_json(signature).encode("utf-8")
    ).hexdigest()
    return {
        "protocol": PROTOCOL,
        "pilot": pilot,
        "parameter_count": count_params(FallTCN()),
        "signature": signature,
        "signature_sha256": signature_sha256,
        "test_accessed": False,
    }


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def _atomic_checkpoint(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def select_clip_balanced_indices(
    labels: np.ndarray,
    clip_indices: np.ndarray,
    semantic_codes: np.ndarray,
    baseline_scores: np.ndarray,
    clip_targets: np.ndarray,
    *,
    semantic_per_clip: int,
    hard_negative_top_k: int,
    seed: int,
) -> np.ndarray:
    """每 clip/语义等额采样，并为负 clip 强制加入最高分窗口。"""
    arrays = tuple(
        np.asarray(value)
        for value in (labels, clip_indices, semantic_codes, baseline_scores)
    )
    if any(value.ndim != 1 for value in arrays) or len({len(v) for v in arrays}) != 1:
        raise ValueError("窗口数组必须是等长一维数组")
    labels, clip_indices, semantic_codes, baseline_scores = arrays
    clip_targets = np.asarray(clip_targets, dtype=np.bool_)
    if semantic_per_clip < 1 or hard_negative_top_k < 1:
        raise ValueError("采样数量必须为正")
    if np.any(np.diff(clip_indices) < 0):
        raise ValueError("clip_indices 必须按 clip 连续递增")
    if clip_indices.size == 0 or int(clip_indices[-1]) >= len(clip_targets):
        raise ValueError("clip_indices 与 clip_targets 不一致")
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for clip_index in range(len(clip_targets)):
        start = int(np.searchsorted(clip_indices, clip_index, side="left"))
        end = int(np.searchsorted(clip_indices, clip_index, side="right"))
        if start == end:
            raise ValueError(f"clip {clip_index} 没有窗口")
        local = np.arange(start, end, dtype=np.int64)
        chosen: set[int] = set()
        for code in np.unique(semantic_codes[local]):
            candidates = local[semantic_codes[local] == code]
            count = min(semantic_per_clip, len(candidates))
            chosen.update(
                map(int, rng.choice(candidates, size=count, replace=False))
            )
        if not clip_targets[clip_index]:
            order = np.lexsort((local, -baseline_scores[local]))
            chosen.update(map(int, local[order[:hard_negative_top_k]]))
        selected.append(np.asarray(sorted(chosen), dtype=np.int64))
    result = np.concatenate(selected)
    if not np.any(labels[result] == 1) or not np.any(labels[result] == 0):
        raise ValueError("采样结果必须同时包含正负窗口")
    return result


def aggregate_clip_logits(
    logits: torch.Tensor, clip_indices: torch.Tensor, *, top_k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward-compatible Challenge A top-k log-mean-exp facade."""
    if top_k < 1:
        raise ValueError("top_k 必须为正")
    if logits.ndim != 1 or clip_indices.ndim != 1 or logits.numel() != clip_indices.numel():
        raise ValueError("logits/clip_indices 必须是等长一维张量")
    if logits.numel() == 0:
        raise ValueError("聚合输入不能为空")
    return aggregate_indexed_clip_logits(
        logits,
        clip_indices,
        mode="topk_logmeanexp",
        topk_count=top_k,
    )


def checkpoint_rank(
    metrics: dict[str, Any], baseline_metrics: dict[str, Any]
) -> tuple[float, float]:
    """按 MAP/P@R95 相对 fallback 的最小增益选择唯一 checkpoint。"""
    map_gain = float(metrics["clip_map"]) - float(baseline_metrics["map"])
    p95_gain = float(metrics["clip_p_at_r95"]) - float(
        baseline_metrics["p_at_r95"]
    )
    return min(map_gain, p95_gain), map_gain + p95_gain


@torch.inference_mode()
def predict_cache(
    model: nn.Module,
    cache: WindowMemmapCache,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    features = cache.array("features")
    output = np.empty(cache.sample_count, dtype=np.float32)
    for start in range(0, cache.sample_count, batch_size):
        end = min(start + batch_size, cache.sample_count)
        batch = torch.from_numpy(np.array(features[start:end], dtype=np.float32, copy=True))
        output[start:end] = torch.sigmoid(model(batch.to(device))).cpu().numpy()
    return output


def _load_model(checkpoint: Path, device: torch.device) -> FallTCN:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = FallTCN().to(device)
    model.load_state_dict(payload["model_state"])
    return model


def _train_seed(
    *,
    seed: int,
    config: ChallengeAConfig,
    baseline_checkpoint: Path,
    baseline_metrics: dict[str, Any],
    run_signature_sha256: str,
    train_cache: WindowMemmapCache,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    val_clip_indices: np.ndarray,
    val_clips: list[dict[str, Any]],
    selected_indices: np.ndarray,
    clip_targets: np.ndarray,
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    set_deterministic(seed)
    model = _load_model(baseline_checkpoint, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )
    train_features = torch.from_numpy(
        np.asarray(train_cache.array("features")[selected_indices], dtype=np.float32)
    )
    train_labels = torch.from_numpy(
        np.asarray(train_cache.array("labels")[selected_indices], dtype=np.float32)
    )
    selected_clips = np.asarray(
        train_cache.array("clip_indices")[selected_indices], dtype=np.int64
    )
    window_positive = float(train_labels.sum())
    window_pos_weight = (train_labels.numel() - window_positive) / window_positive
    clip_positive = float(np.count_nonzero(clip_targets))
    clip_pos_weight = (len(clip_targets) - clip_positive) / clip_positive
    window_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(window_pos_weight, device=device)
    )
    clip_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(clip_pos_weight, device=device)
    )
    history: list[dict[str, Any]] = []
    best_rank = (-math.inf, -math.inf)
    best_record: dict[str, Any] | None = None
    rng = np.random.default_rng(seed)
    for epoch in range(config.epochs):
        model.train()
        clip_order = rng.permutation(len(clip_targets))
        loss_sum = 0.0
        batches = 0
        for offset in range(0, len(clip_order), config.clips_per_batch):
            batch_clips = clip_order[offset : offset + config.clips_per_batch]
            positions = []
            for clip_index in batch_clips:
                left = int(np.searchsorted(selected_clips, clip_index, side="left"))
                right = int(np.searchsorted(selected_clips, clip_index, side="right"))
                positions.append(np.arange(left, right, dtype=np.int64))
            batch_positions = np.concatenate(positions)
            batch_x = train_features[batch_positions].to(device)
            batch_y = train_labels[batch_positions].to(device)
            batch_clip_ids = torch.from_numpy(
                selected_clips[batch_positions]
            ).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            window_loss = window_criterion(logits, batch_y)
            clip_logits, unique_clips = aggregate_clip_logits(
                logits, batch_clip_ids, top_k=config.mil_top_k
            )
            targets = torch.from_numpy(
                clip_targets[unique_clips.cpu().numpy()].astype(np.float32)
            ).to(device)
            clip_loss = clip_criterion(clip_logits, targets)
            loss = window_loss + config.clip_loss_weight * clip_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            loss_sum += float(loss.detach())
            batches += 1
        metrics = evaluate_model(
            model,
            val_features,
            val_labels,
            val_clip_indices,
            val_clips,
            device=device,
            batch_size=config.inference_batch_size,
        )
        scheduler.step()
        record = {
            "epoch": epoch,
            "train_loss": loss_sum / batches,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "val": metrics,
        }
        history.append(record)
        rank = checkpoint_rank(metrics, baseline_metrics)
        if rank > best_rank:
            best_rank = rank
            best_record = record
            _atomic_checkpoint(
                output_dir / f"seed-{seed}" / "best.pt",
                {
                    "protocol": PROTOCOL,
                    "run_signature_sha256": run_signature_sha256,
                    "seed": seed,
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "metrics": metrics,
                },
            )
        print(
            json.dumps({"stage": "epoch", "seed": seed, **record}, sort_keys=True),
            flush=True,
        )
    assert best_record is not None
    _atomic_json(output_dir / f"seed-{seed}" / "history.json", history)
    return {"seed": seed, "best": best_record}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-ablation", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument(
        "--epochs",
        type=int,
        help="仅用于受控冒烟运行；省略时使用配置文件中的正式 epoch 数",
    )
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("seeds 不能重复")
    config = ChallengeAConfig.from_json(args.config)
    if args.epochs is not None:
        config = replace(config, epochs=args.epochs)
        config.validate()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"输出目录必须为空: {args.output_dir}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("请求 CUDA，但当前不可用")
    train_cache = load_window_cache(args.train_cache, verify_hashes=True)
    val_cache = load_window_cache(args.val_cache, verify_hashes=True)
    baseline = json.loads(args.baseline_ablation.read_text(encoding="utf-8"))
    if baseline.get("split") != "val" or baseline.get("test_accessed") is True:
        raise ValueError("baseline 必须来自未访问 test 的 validation")
    baseline_model = _load_model(args.baseline_checkpoint, device)
    baseline_scores = predict_cache(
        baseline_model,
        train_cache,
        device=device,
        batch_size=config.inference_batch_size,
    )
    del baseline_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    clip_targets = np.asarray(
        [bool(item["has_fall"]) for item in train_cache.metadata["clips"]],
        dtype=np.bool_,
    )
    val_features = torch.from_numpy(
        np.array(val_cache.array("features"), dtype=np.float32, copy=True)
    )
    val_labels = torch.from_numpy(
        np.array(val_cache.array("labels"), dtype=np.float32, copy=True)
    )
    val_clip_indices = np.array(
        val_cache.array("clip_indices"), dtype=np.int64, copy=True
    )
    results = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run = build_run_record(
        config=config,
        seeds=args.seeds,
        baseline_checkpoint=args.baseline_checkpoint,
        baseline_ablation=args.baseline_ablation,
        train_cache=train_cache,
        val_cache=val_cache,
        pilot=args.epochs is not None,
    )
    _atomic_json(args.output_dir / "run.json", run)
    for seed in args.seeds:
        selected = select_clip_balanced_indices(
            train_cache.array("labels"),
            train_cache.array("clip_indices"),
            train_cache.array("semantic_codes"),
            baseline_scores,
            clip_targets,
            semantic_per_clip=config.semantic_per_clip,
            hard_negative_top_k=config.hard_negative_top_k,
            seed=seed,
        )
        semantic_codes = np.asarray(
            train_cache.array("semantic_codes")[selected], dtype=np.int64
        )
        selected_labels = np.asarray(train_cache.array("labels")[selected])
        results.append(
            _train_seed(
                seed=seed,
                config=config,
                baseline_checkpoint=args.baseline_checkpoint,
                baseline_metrics=baseline["tcn_only"],
                run_signature_sha256=run["signature_sha256"],
                train_cache=train_cache,
                val_features=val_features,
                val_labels=val_labels,
                val_clip_indices=val_clip_indices,
                val_clips=val_cache.metadata["clips"],
                selected_indices=selected,
                clip_targets=clip_targets,
                output_dir=args.output_dir,
                device=device,
            )
        )
        results[-1]["selected_windows"] = len(selected)
        results[-1]["selected_positive_windows"] = int(
            np.count_nonzero(selected_labels)
        )
        results[-1]["selected_semantic_counts"] = {
            name: int(np.count_nonzero(semantic_codes == code))
            for code, name in enumerate(SEMANTICS)
        }
    baseline_metrics = baseline["tcn_only"]
    map_deltas = np.asarray(
        [100.0 * (item["best"]["val"]["clip_map"] - baseline_metrics["map"]) for item in results]
    )
    p95_deltas = np.asarray(
        [100.0 * (item["best"]["val"]["clip_p_at_r95"] - baseline_metrics["p_at_r95"]) for item in results]
    )
    passed_seeds = int(np.count_nonzero((map_deltas >= 3.0) & (p95_deltas >= 3.0)))
    summary = {
        "protocol": PROTOCOL,
        "run_signature_sha256": run["signature_sha256"],
        "checkpoint_selection": CHECKPOINT_SELECTION,
        "config": asdict(config),
        "seeds": args.seeds,
        "results": results,
        "baseline": {
            "map": baseline_metrics["map"],
            "p_at_r95": baseline_metrics["p_at_r95"],
        },
        "delta_map_points": map_deltas.tolist(),
        "delta_p_at_r95_points": p95_deltas.tolist(),
        "median_delta_map_points": float(np.median(map_deltas)),
        "median_delta_p_at_r95_points": float(np.median(p95_deltas)),
        "passed_seeds": passed_seeds,
        "challenge_a_pass": bool(
            len(results) >= 3
            and passed_seeds >= math.ceil(len(results) / 2)
            and np.median(map_deltas) >= 3.0
            and np.median(p95_deltas) >= 3.0
        ),
        "parameter_count": count_params(FallTCN()),
        "artifact_sha256": {
            "baseline_checkpoint": _sha256_file(args.baseline_checkpoint),
            "baseline_ablation": _sha256_file(args.baseline_ablation),
            "train_cache_metadata": _sha256_file(args.train_cache / "metadata.json"),
            "val_cache_metadata": _sha256_file(args.val_cache / "metadata.json"),
            "training_code": _sha256_file(Path(__file__)),
        },
        "test_accessed": False,
    }
    _atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps({"stage": "complete", **summary}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
