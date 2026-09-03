"""在冻结的 TCN 窗口缓存上训练公平的 LSTM+速度特征基线。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from models.lstm import (
    POSE51_VELOCITY34_FEATURES,
    SUPPORTED_INPUT_FEATURES,
    FallLSTM,
    count_params,
    feature_dimension,
)
from models.tcn_dataset import WindowMemmapCache, build_window_cache
from tools.train_tcn import (
    _atomic_checkpoint,
    _atomic_json,
    _canonical_json,
    _package_version,
    _sha256_file,
    evaluate_model,
    select_training_indices,
    set_deterministic,
    train_epoch,
)


@dataclass(frozen=True)
class LSTMTrainingConfig:
    window_size: int = 16
    stride: int = 1
    min_observed_frames: int = 8
    epochs: int = 50
    batch_size: int = 1024
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    negative_ratio: float = 2.0
    seed: int = 20260817
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.3
    input_features: str = POSE51_VELOCITY34_FEATURES
    grad_clip: float = 5.0

    @classmethod
    def from_json(cls, path: Path) -> LSTMTrainingConfig:
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("LSTM config 不是合法 JSON") from exc
        if not isinstance(payload, dict):
            raise TypeError("LSTM config 必须是 JSON 对象")
        unknown = set(payload) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"LSTM config 包含未知字段: {sorted(unknown)}")
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
        if self.hidden_size < 1 or self.num_layers < 1:
            raise ValueError("hidden_size 和 num_layers 必须为正整数")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout 必须位于 [0, 1)")
        if self.input_features not in SUPPORTED_INPUT_FEATURES:
            raise ValueError(f"input_features 必须为 {SUPPORTED_INPUT_FEATURES}")


def _materialize(
    cache: WindowMemmapCache, indices: np.ndarray | None
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


def _run_signature(
    *,
    config: LSTMTrainingConfig,
    train_cache: WindowMemmapCache,
    val_cache: WindowMemmapCache,
    device: str,
    max_train_samples: int | None,
    max_val_samples: int | None,
) -> tuple[str, dict[str, Any]]:
    project_root = Path(__file__).parent.parent
    payload = {
        "protocol": "fall_lstm_velocity85_training_v1",
        "feature_protocol": config.input_features,
        "config": asdict(config),
        "train_window_signature": train_cache.metadata["signature_sha256"],
        "val_window_signature": val_cache.metadata["signature_sha256"],
        "device": device,
        "max_train_samples": max_train_samples,
        "max_val_samples": max_val_samples,
        "reference": {
            "repository": "https://github.com/jiminnote/pose-based-realtime-fall-detection",
            "borrowed_idea": "85D pose plus frame-difference velocity and 2-layer LSTM",
            "not_reproduced": "AI Hub data, 40-frame windows, and scene-level labels",
        },
        "code_sha256": {
            str(path.relative_to(project_root)): _sha256_file(path)
            for path in (
                project_root / "models" / "lstm.py",
                project_root / "models" / "tcn_window.py",
                project_root / "models" / "tcn_dataset.py",
                project_root / "tools" / "train_tcn.py",
                project_root / "tools" / "train_lstm.py",
            )
        },
        "dependencies": {name: _package_version(name) for name in ("numpy", "torch")},
    }
    digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    return digest, payload


def train(
    *,
    config: LSTMTrainingConfig,
    train_cache: WindowMemmapCache,
    val_cache: WindowMemmapCache,
    output_dir: Path,
    device_name: str,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    config.validate()
    set_deterministic(config.seed)
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("请求 CUDA 训练，但 torch.cuda.is_available() 为 False")
    device = torch.device(device_name)
    train_indices = select_training_indices(
        train_cache.array("labels"),
        negative_ratio=config.negative_ratio,
        seed=config.seed,
        max_samples=max_train_samples,
    )
    val_indices = None
    if max_val_samples is not None and max_val_samples < val_cache.sample_count:
        val_indices = np.linspace(0, val_cache.sample_count - 1, max_val_samples, dtype=np.int64)
    train_x, train_y, _ = _materialize(train_cache, train_indices)
    val_x, val_y, val_clip_indices = _materialize(val_cache, val_indices)
    model = FallLSTM(
        input_dim=feature_dimension(config.input_features),
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
        input_features=config.input_features,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    signature_sha256, signature = _run_signature(
        config=config,
        train_cache=train_cache,
        val_cache=val_cache,
        device=device_name,
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
    )
    output_dir = Path(output_dir)
    run_path = output_dir / "run.json"
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise ValueError("输出目录非空；如需续跑请显式传入 --resume")
    output_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = 0
    best_map = -math.inf
    if resume:
        try:
            existing = json.loads(run_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("resume 缺少合法 run.json") from exc
        if existing.get("signature_sha256") != signature_sha256:
            raise ValueError("resume run signature 不匹配")
        checkpoint = torch.load(output_dir / "last.pt", map_location=device, weights_only=False)
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
        _atomic_json(
            run_path,
            {
                "signature_sha256": signature_sha256,
                "signature": signature,
                "parameter_count": count_params(model),
                "train_samples": int(train_y.numel()),
                "train_positive": int(train_y.sum()),
                "val_samples": int(val_y.numel()),
                "pilot": max_train_samples is not None or max_val_samples is not None,
            },
        )
    positives = float(train_y.sum())
    pos_weight = float(train_y.numel() - positives) / positives
    history: list[dict[str, Any]] = []
    history_path = output_dir / "history.jsonl"
    for epoch in range(start_epoch, config.epochs):
        train_loss = train_epoch(
            model, optimizer, train_x, train_y, device=device, batch_size=config.batch_size,
            pos_weight=pos_weight, grad_clip=config.grad_clip, seed=config.seed + epoch,
        )
        metrics = evaluate_model(
            model, val_x, val_y, val_clip_indices, val_cache.metadata["clips"],
            device=device, batch_size=config.batch_size,
        )
        scheduler.step()
        record = {"epoch": epoch, "train_loss": train_loss,
                  "learning_rate": optimizer.param_groups[0]["lr"], "val": metrics}
        with history_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(_canonical_json(record) + "\n")
            handle.flush()
        history.append(record)
        improved = metrics["clip_map"] > best_map
        if improved:
            best_map = metrics["clip_map"]
        checkpoint = {
            "epoch": epoch, "run_signature_sha256": signature_sha256,
            "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(), "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            "best_map": best_map, "metrics": record,
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
    config = LSTMTrainingConfig.from_json(args.config)
    if args.epochs is not None:
        config = replace(config, epochs=args.epochs)
        config.validate()
    caches: dict[str, WindowMemmapCache] = {}
    for split in ("train", "val"):
        print(_canonical_json({"stage": "window_cache", "split": split}), flush=True)
        caches[split] = build_window_cache(
            args.manifest, args.pose_cache_root, args.audit_report, args.window_cache_root,
            dataset=args.dataset, split=split, window_size=config.window_size,
            stride=config.stride, min_observed_frames=config.min_observed_frames,
            progress=lambda state, split=split: print(
                _canonical_json({"stage": "window_cache_progress", "split": split, **state}),
                flush=True,
            ),
        )
        print(_canonical_json({"stage": "window_cache_ready", "split": split,
                               "samples": caches[split].sample_count,
                               "path": str(caches[split].root.resolve())}), flush=True)
    summary = train(
        config=config, train_cache=caches["train"], val_cache=caches["val"],
        output_dir=args.output_dir, device_name=args.device,
        max_train_samples=args.max_train_samples, max_val_samples=args.max_val_samples,
        resume=args.resume,
    )
    print(_canonical_json({"stage": "complete", **summary}), flush=True)


if __name__ == "__main__":
    main()
