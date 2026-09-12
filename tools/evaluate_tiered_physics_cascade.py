"""验证冻结 MIL 加 train-only 物理规则复核的层级推理。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from eval.metrics import competition_map
from models.multiscale_multistream_tcn import build_transition_rule_features
from models.tcn_dataset import WindowMemmapCache, load_window_cache
from tools.build_tcn_multistream_sidecar import load_sidecar
from tools.evaluate_hard_negative_cohort import _model_from_run
from tools.train_multiscale_multistream_tcn import _activity_groups
from tools.train_tcn import _atomic_json, select_training_indices

RULE_DIM = 8
AMBIGUITY_HALF_WIDTH = 0.35
RULE_LOGIT_WEIGHT = 0.35


def _fit_logistic(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a deterministic weighted linear physical-rule classifier on train only."""
    if features.ndim != 2 or features.shape[1] != RULE_DIM:
        raise ValueError("物理规则特征维度无效")
    labels = np.asarray(labels, dtype=np.float64)
    if labels.shape != (features.shape[0],) or not np.all((labels == 0) | (labels == 1)):
        raise ValueError("labels 必须是二分类一维数组")
    positive = float(labels.sum())
    if positive == 0.0 or positive == labels.size:
        raise ValueError("物理规则拟合需要正负样本")
    mean = features.mean(axis=0, dtype=np.float64)
    scale = features.std(axis=0, dtype=np.float64).clip(min=1e-6)
    x = (features.astype(np.float64) - mean) / scale
    design = np.column_stack((x, np.ones(x.shape[0], dtype=np.float64)))
    weights = np.where(labels == 1, (labels.size - positive) / positive, 1.0)
    coefficients = np.zeros(RULE_DIM + 1, dtype=np.float64)
    penalty = np.diag([1e-3] * RULE_DIM + [0.0])
    for _ in range(30):
        logits = np.clip(design @ coefficients, -40.0, 40.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        gradient = design.T @ (weights * (probabilities - labels)) + penalty @ coefficients
        curvature = weights * probabilities * (1.0 - probabilities)
        hessian = design.T @ (design * curvature[:, None]) + penalty
        step = np.linalg.solve(hessian, gradient)
        coefficients -= step
        if np.linalg.vector_norm(step) < 1e-8:
            break
    return coefficients.astype(np.float32), mean.astype(np.float32), scale.astype(np.float32)


@torch.inference_mode()
def _transition_features(features: np.ndarray, sidecar: np.ndarray, batch_size: int) -> np.ndarray:
    values: list[np.ndarray] = []
    for start in range(0, features.shape[0], batch_size):
        pose = torch.from_numpy(np.array(features[start : start + batch_size], dtype=np.float32, copy=True))
        geometry = torch.from_numpy(np.array(sidecar[start : start + batch_size], dtype=np.float32, copy=True))
        values.append(build_transition_rule_features(torch.cat((pose.flatten(2), geometry), dim=-1)).numpy()[:, -1])
    return np.concatenate(values).astype(np.float32, copy=False)


@torch.inference_mode()
def _mil_logits(
    model: torch.nn.Module, features: np.ndarray, sidecar: np.ndarray, device: torch.device, batch_size: int
) -> np.ndarray:
    values: list[np.ndarray] = []
    for start in range(0, features.shape[0], batch_size):
        pose = torch.from_numpy(np.array(features[start : start + batch_size], dtype=np.float32, copy=True))
        geometry = torch.from_numpy(np.array(sidecar[start : start + batch_size], dtype=np.float32, copy=True))
        values.append(model(torch.cat((pose.flatten(2), geometry), dim=-1).to(device)).cpu().numpy())
    return np.concatenate(values).astype(np.float32, copy=False)


def _clip_metrics(cache: WindowMemmapCache, logits: np.ndarray) -> dict[str, float]:
    if logits.shape != (cache.sample_count,):
        raise ValueError("logits 与窗口缓存不一致")
    scores = np.full(len(cache.metadata["clips"]), -np.inf, dtype=np.float32)
    np.maximum.at(scores, cache.array("clip_indices"), 1.0 / (1.0 + np.exp(-logits)))
    prediction = {clip["clip_id"]: float(scores[index]) for index, clip in enumerate(cache.metadata["clips"])}
    ground_truth = {clip["clip_id"]: bool(clip["has_fall"]) for clip in cache.metadata["clips"]}
    result = competition_map(ground_truth, prediction, mode="clip")
    return {
        "clip_map": float(result["map"]),
        "clip_map_percent": float(result["map_percent"]),
        "clip_p_at_r90": float(result["p_at_r90"]),
        "clip_p_at_r95": float(result["p_at_r95"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--train-sidecar", type=Path, required=True)
    parser.add_argument("--val-sidecar", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("输出已存在，拒绝覆盖")
    train_cache, val_cache = load_window_cache(args.train_cache), load_window_cache(args.val_cache)
    train_sidecar, val_sidecar = load_sidecar(args.train_sidecar, train_cache), load_sidecar(args.val_sidecar, val_cache)
    run = json.loads(args.run.read_text(encoding="utf-8"))
    model, _, _ = _model_from_run(
        model_kind="multiscale_multistream_tcn", run=run, checkpoint=args.checkpoint, device=torch.device(args.device)
    )
    config = run["signature"]["config"]
    selected = select_training_indices(
        train_cache.array("labels"), negative_ratio=float(config["negative_ratio"]), seed=int(config["seed"]),
        activity_groups=_activity_groups(train_cache),
        hard_negative_activities=tuple(config["hard_negative_activities"]),
        hard_negative_fraction=float(config["hard_negative_fraction"]),
    )
    train_rules = _transition_features(train_cache.array("features")[selected], train_sidecar[selected], args.batch_size)
    coefficients, mean, scale = _fit_logistic(train_rules, train_cache.array("labels")[selected])
    val_rules = _transition_features(val_cache.array("features"), val_sidecar, args.batch_size)
    rule_logits = (val_rules - mean) @ (coefficients[:-1] / scale) + coefficients[-1]
    base_logits = _mil_logits(model, val_cache.array("features"), val_sidecar, torch.device(args.device), args.batch_size)
    base_probability = 1.0 / (1.0 + np.exp(-base_logits))
    ambiguity = np.clip(1.0 - np.abs(base_probability - 0.5) / AMBIGUITY_HALF_WIDTH, 0.0, 1.0)
    cascade_logits = base_logits + RULE_LOGIT_WEIGHT * ambiguity * rule_logits
    report: dict[str, Any] = {
        "protocol": "tiered_physics_cascade_v1",
        "frozen": {"ambiguity_half_width": AMBIGUITY_HALF_WIDTH, "rule_logit_weight": RULE_LOGIT_WEIGHT},
        "base_mil": _clip_metrics(val_cache, base_logits),
        "physical_rule_only": _clip_metrics(val_cache, rule_logits.astype(np.float32)),
        "tiered_cascade": _clip_metrics(val_cache, cascade_logits.astype(np.float32)),
        "training": {"rule_fit_split": "train", "selected_windows": int(selected.size)},
    }
    _atomic_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
