"""Train one template-grouped train-only OOF fold of the long-context oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from eval.metrics import competition_map
from models.long_context_event_oracle import LongContextEventOracle
from models.multiscale_multistream_tcn import (
    build_discriminative_kinematic_features,
    build_rule_features,
    build_transition_rule_features,
)
from models.tcn_dataset import SEMANTIC_TO_CODE, WindowMemmapCache, load_window_cache
from tools.build_tcn_multistream_sidecar import load_sidecar
from tools.train_tcn import (
    _atomic_checkpoint,
    _atomic_json,
    _sha256_file,
    set_deterministic,
)

STAGE_NAMES = ("background", "controlled_transition", "fall", "post_fall")
INPUT_DIMS = {"last_frame": 57, "rich_window": 219}


@dataclass(frozen=True)
class OracleConfig:
    max_steps: int = 128
    epochs: int = 8
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-3
    hidden_dim: int = 96
    gru_hidden_dim: int = 64
    dropout: float = 0.2
    stage_weight: float = 1.0
    boundary_weight: float = 0.5
    smooth_weight: float = 0.1
    boundary_radius: int = 1
    folds: int = 5
    seed: int = 20260824
    feature_protocol: str = "rich_window"

    def validate(self) -> None:
        if self.max_steps < 16 or self.epochs < 1 or self.batch_size < 1:
            raise ValueError("max_steps/epochs/batch_size 无效")
        if self.folds < 2 or self.boundary_radius < 0:
            raise ValueError("folds/boundary_radius 无效")
        if min(self.stage_weight, self.boundary_weight, self.smooth_weight) < 0.0:
            raise ValueError("loss weights 不能为负")
        if self.feature_protocol not in INPUT_DIMS:
            raise ValueError("feature_protocol 无效")


def template_group(clip_id: str) -> str:
    """Group cross-activity synthetic clips sharing actor/view template IDs."""
    parts = clip_id.rsplit("_", 2)
    if len(parts) != 3 or not parts[-1].isdigit() or not parts[-2]:
        raise ValueError(f"无法解析 OF-Syn template group: {clip_id}")
    return f"{parts[-2]}_{parts[-1]}"


def group_fold(group: str, *, folds: int, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{group}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % folds


def stage_targets(semantic_codes: np.ndarray) -> np.ndarray:
    codes = np.asarray(semantic_codes)
    if codes.ndim != 1:
        raise ValueError("semantic_codes 必须是一维")
    if not np.isin(codes, tuple(SEMANTIC_TO_CODE.values())).all():
        raise ValueError("semantic_codes 含未知语义")
    targets = np.zeros(codes.shape, dtype=np.int64)
    targets[codes == SEMANTIC_TO_CODE["hard_negative"]] = 1
    targets[codes == SEMANTIC_TO_CODE["fall_process"]] = 2
    targets[codes == SEMANTIC_TO_CODE["post_fall_state"]] = 3
    return targets


def boundary_targets(stages: np.ndarray, *, radius: int) -> np.ndarray:
    """Mark fall onset/offset, including clips that begin or end during a fall."""
    stages = np.asarray(stages)
    if stages.ndim != 1 or radius < 0:
        raise ValueError("stages/radius 无效")
    result = np.zeros((stages.size, 2), dtype=np.float32)
    fall = stages == 2
    if not fall.any():
        return result
    onset = np.flatnonzero(fall & ~np.r_[False, fall[:-1]])
    offset = np.flatnonzero(fall & ~np.r_[fall[1:], False])
    for column, indices in enumerate((onset, offset)):
        for index in indices:
            left = max(0, int(index) - radius)
            right = min(stages.size, int(index) + radius + 1)
            result[left:right, column] = 1.0
    return result


class OracleDataset(Dataset[tuple[torch.Tensor, ...]]):
    def __init__(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor,
        stages: torch.Tensor,
        boundaries: torch.Tensor,
        labels: torch.Tensor,
        indices: np.ndarray,
    ) -> None:
        self.features = features
        self.lengths = lengths
        self.stages = stages
        self.boundaries = boundaries
        self.labels = labels
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        selected = int(self.indices[index])
        return (
            self.features[selected],
            self.lengths[selected],
            self.stages[selected],
            self.boundaries[selected],
            self.labels[selected],
            torch.tensor(selected, dtype=torch.long),
        )


def rich_window_descriptors(x: torch.Tensor) -> torch.Tensor:
    """Compress every full 16-frame window without discarding local dynamics."""
    if x.ndim != 3 or x.shape[1:] != (16, 57):
        raise ValueError("rich window 输入必须为 (N,16,57)")

    def summarize(values: torch.Tensor) -> torch.Tensor:
        return torch.cat((values[:, -1], values.mean(1), values.amax(1)), dim=1)

    raw = torch.cat((x[:, -1], x[:, -1] - x[:, 0]), dim=1)
    rules = summarize(build_rule_features(x))
    transitions = summarize(build_transition_rule_features(x))
    kinematics = summarize(build_discriminative_kinematic_features(x))
    result = torch.cat((raw, rules, transitions, kinematics), dim=1)
    if result.shape[1] != INPUT_DIMS["rich_window"]:
        raise RuntimeError("rich window descriptor 维度契约被破坏")
    return result


def _all_window_descriptors(
    cache: WindowMemmapCache,
    sidecar: np.ndarray,
    *,
    feature_protocol: str,
) -> np.ndarray:
    pose = cache.array("features")
    if feature_protocol == "last_frame":
        return np.concatenate(
            (
                np.asarray(pose[:, -1]).reshape(cache.sample_count, 51),
                np.asarray(sidecar[:, -1]),
            ),
            axis=1,
        ).astype(np.float32, copy=False)
    descriptors = np.empty(
        (cache.sample_count, INPUT_DIMS["rich_window"]), dtype=np.float16
    )
    for start in range(0, cache.sample_count, 4096):
        end = min(cache.sample_count, start + 4096)
        windows = torch.from_numpy(
            np.concatenate(
                (
                    np.asarray(pose[start:end]).reshape(end - start, 16, 51),
                    np.asarray(sidecar[start:end]),
                ),
                axis=2,
            )
        )
        descriptors[start:end] = rich_window_descriptors(windows).numpy().astype(
            np.float16
        )
    return descriptors


def materialize_long_context(
    cache: WindowMemmapCache,
    sidecar: np.ndarray,
    *,
    max_steps: int,
    boundary_radius: int,
    feature_protocol: str = "rich_window",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if cache.metadata.get("split") != "train":
        raise ValueError("long-context OOF 只接受 train cache")
    if sidecar.shape != (cache.sample_count, 16, 6):
        raise ValueError("bbox sidecar 与 train cache 不兼容")
    clips = cache.metadata["clips"]
    if feature_protocol not in INPUT_DIMS:
        raise ValueError("feature_protocol 无效")
    descriptor_dim = INPUT_DIMS[feature_protocol]
    descriptor_dtype = np.float16 if feature_protocol == "rich_window" else np.float32
    compact = np.zeros(
        (len(clips), max_steps, descriptor_dim), dtype=descriptor_dtype
    )
    stages = np.full((len(clips), max_steps), -100, dtype=np.int64)
    boundaries = np.zeros((len(clips), max_steps, 2), dtype=np.float32)
    lengths = np.zeros(len(clips), dtype=np.int64)
    labels = np.asarray([bool(clip["has_fall"]) for clip in clips], dtype=np.float32)
    descriptors = _all_window_descriptors(
        cache, sidecar, feature_protocol=feature_protocol
    )
    semantic_codes = cache.array("semantic_codes")
    end_times = cache.array("end_times")
    for clip_index, clip in enumerate(clips):
        start = int(clip["window_start"])
        count = int(clip["window_count"])
        order = np.argsort(np.asarray(end_times[start : start + count]), kind="stable")
        if count > max_steps:
            sampled = np.linspace(0, count - 1, max_steps).round().astype(np.int64)
            order = order[sampled]
        selected = start + order
        length = int(selected.size)
        compact[clip_index, :length] = descriptors[selected]
        clip_stages = stage_targets(np.asarray(semantic_codes[selected]))
        stages[clip_index, :length] = clip_stages
        boundaries[clip_index, :length] = boundary_targets(
            clip_stages, radius=boundary_radius
        )
        lengths[clip_index] = length
    return tuple(
        torch.from_numpy(array)
        for array in (compact, lengths, stages, boundaries, labels)
    )  # type: ignore[return-value]


def _smooth_loss(stage_logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    log_probabilities = torch.log_softmax(stage_logits, dim=-1)
    adjacent = mask[:, 1:] & mask[:, :-1]
    if not adjacent.any():
        return stage_logits.sum() * 0.0
    differences = (log_probabilities[:, 1:] - log_probabilities[:, :-1]).square()
    return torch.clamp(differences, max=16.0)[adjacent].mean()


def _metric_result(ids: list[str], labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    result = competition_map(
        dict(zip(ids, (bool(label) for label in labels), strict=True)),
        dict(zip(ids, scores.astype(float), strict=True)),
        mode="clip",
    )

    def best_point(target: float) -> dict[str, float | int] | None:
        feasible = [point for point in result["curve"] if point.recall >= target]
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
        "clip_map": float(result["map"]),
        "clip_map_percent": float(result["map_percent"]),
        "clip_p_at_r90": float(result["p_at_r90"]),
        "clip_p_at_r95": float(result["p_at_r95"]),
        "clip_r90_point": best_point(0.90),
        "clip_r95_point": best_point(0.95),
    }


def _class_weights(stages: torch.Tensor, lengths: torch.Tensor, indices: np.ndarray) -> torch.Tensor:
    counts = torch.zeros(len(STAGE_NAMES), dtype=torch.float64)
    for index in indices:
        values = stages[int(index), : int(lengths[int(index)])]
        counts += torch.bincount(values, minlength=len(STAGE_NAMES))
    if torch.any(counts == 0):
        raise ValueError("每个训练 fold 必须覆盖全部阶段")
    return (counts.sum() / (len(STAGE_NAMES) * counts)).float()


def _boundary_pos_weight(
    boundaries: torch.Tensor, lengths: torch.Tensor, indices: np.ndarray
) -> torch.Tensor:
    positives = torch.zeros(2, dtype=torch.float64)
    total = 0
    for index in indices:
        length = int(lengths[int(index)])
        positives += boundaries[int(index), :length].sum(0)
        total += length
    if torch.any(positives == 0):
        raise ValueError("训练 fold 缺少 onset/offset 边界")
    return ((total - positives) / positives).float()


def train_epoch(
    model: LongContextEventOracle,
    loader: DataLoader[tuple[torch.Tensor, ...]],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    stage_criterion: nn.Module,
    boundary_criterion: nn.Module,
    clip_criterion: nn.Module,
    config: OracleConfig,
) -> dict[str, float]:
    model.train()
    totals = {name: 0.0 for name in ("loss", "clip", "stage", "boundary", "smooth")}
    seen = 0
    for features, lengths, stages, boundaries, labels, _ in loader:
        features = features.to(device, non_blocking=True).float()
        lengths = lengths.to(device, non_blocking=True)
        stages = stages.to(device, non_blocking=True)
        boundaries = boundaries.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(features, lengths)
        mask = outputs["mask"]
        clip_loss = clip_criterion(outputs["clip_logit"], labels)
        stage_loss = stage_criterion(outputs["stage_logits"][mask], stages[mask])
        boundary_loss = boundary_criterion(
            outputs["boundary_logits"][mask], boundaries[mask]
        )
        smooth_loss = _smooth_loss(outputs["stage_logits"], mask)
        loss = (
            clip_loss
            + config.stage_weight * stage_loss
            + config.boundary_weight * boundary_loss
            + config.smooth_weight * smooth_loss
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        batch = int(labels.numel())
        seen += batch
        for name, value in (
            ("loss", loss),
            ("clip", clip_loss),
            ("stage", stage_loss),
            ("boundary", boundary_loss),
            ("smooth", smooth_loss),
        ):
            totals[name] += float(value.detach()) * batch
    return {name: value / seen for name, value in totals.items()}


@torch.inference_mode()
def evaluate(
    model: LongContextEventOracle,
    loader: DataLoader[tuple[torch.Tensor, ...]],
    *,
    device: torch.device,
    clip_ids: list[str],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    model.eval()
    scores = np.zeros(len(clip_ids), dtype=np.float32)
    labels_by_index = np.zeros(len(clip_ids), dtype=np.float32)
    stage_correct = 0
    stage_total = 0
    boundary_tp = boundary_fp = boundary_fn = 0
    candidate_positive = candidate_recovered = 0
    evaluated: list[int] = []
    for features, lengths, stages, boundaries, labels, indices in loader:
        features = features.to(device, non_blocking=True).float()
        lengths_device = lengths.to(device, non_blocking=True)
        outputs = model(features, lengths_device)
        mask = outputs["mask"]
        probabilities = torch.sigmoid(outputs["clip_logit"]).cpu().numpy()
        selected = indices.numpy()
        scores[selected] = probabilities
        labels_by_index[selected] = labels.numpy()
        evaluated.extend(selected.tolist())
        stages_device = stages.to(device)
        stage_predictions = outputs["stage_logits"].argmax(-1)
        stage_correct += int((stage_predictions[mask] == stages_device[mask]).sum())
        stage_total += int(mask.sum())
        boundary_predictions = torch.sigmoid(outputs["boundary_logits"]) >= 0.5
        boundary_truth = boundaries.to(device) >= 0.5
        boundary_tp += int((boundary_predictions & boundary_truth & mask.unsqueeze(-1)).sum())
        boundary_fp += int((boundary_predictions & ~boundary_truth & mask.unsqueeze(-1)).sum())
        boundary_fn += int((~boundary_predictions & boundary_truth & mask.unsqueeze(-1)).sum())
        fall_probability = torch.softmax(outputs["stage_logits"], -1)[..., 2]
        for row, label in enumerate(labels):
            if label > 0.5:
                candidate_positive += 1
                length = int(lengths[row])
                candidate_recovered += int(float(fall_probability[row, :length].max()) >= 0.5)
    unique = np.asarray(sorted(set(evaluated)), dtype=np.int64)
    if unique.size != len(evaluated):
        raise RuntimeError("OOF evaluation clip 重复")
    metrics = _metric_result(
        [clip_ids[index] for index in unique],
        labels_by_index[unique],
        scores[unique],
    )
    precision = boundary_tp / max(1, boundary_tp + boundary_fp)
    recall = boundary_tp / max(1, boundary_tp + boundary_fn)
    metrics.update(
        {
            "stage_accuracy": stage_correct / max(1, stage_total),
            "boundary_precision": precision,
            "boundary_recall": recall,
            "boundary_f1": 2 * precision * recall / max(1e-12, precision + recall),
            "candidate_recall_at_stage_0_5": candidate_recovered
            / max(1, candidate_positive),
            "clips": int(unique.size),
        }
    )
    return metrics, unique, scores[unique]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument(
        "--feature-protocol", choices=tuple(INPUT_DIMS), default="rich_window"
    )
    args = parser.parse_args()
    config = OracleConfig(
        epochs=args.epochs, seed=args.seed, feature_protocol=args.feature_protocol
    )
    config.validate()
    if not 0 <= args.fold < config.folds:
        raise ValueError("fold 必须位于 [0, folds)")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("oracle 输出目录必须为空")
    cache = load_window_cache(args.cache, verify_hashes=True)
    if cache.metadata.get("split") != "train":
        raise ValueError("oracle 禁止读取非 train cache")
    sidecar = load_sidecar(args.sidecar, cache)
    set_deterministic(config.seed + args.fold)
    clip_ids = [str(clip["clip_id"]) for clip in cache.metadata["clips"]]
    groups = [template_group(clip_id) for clip_id in clip_ids]
    folds = np.asarray(
        [group_fold(group, folds=config.folds, seed=config.seed) for group in groups]
    )
    held_out = np.flatnonzero(folds == args.fold)
    training = np.flatnonzero(folds != args.fold)
    if set(np.asarray(groups)[held_out]) & set(np.asarray(groups)[training]):
        raise AssertionError("template group 跨 OOF fold 泄漏")
    features, lengths, stages, boundaries, labels = materialize_long_context(
        cache,
        sidecar,
        max_steps=config.max_steps,
        boundary_radius=config.boundary_radius,
        feature_protocol=config.feature_protocol,
    )
    generator = torch.Generator().manual_seed(config.seed + args.fold)
    train_loader = DataLoader(
        OracleDataset(features, lengths, stages, boundaries, labels, training),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=True,
    )
    evaluation_loader = DataLoader(
        OracleDataset(features, lengths, stages, boundaries, labels, held_out),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("请求 CUDA，但当前不可用")
    model = LongContextEventOracle(
        input_dim=INPUT_DIMS[config.feature_protocol],
        hidden_dim=config.hidden_dim,
        gru_hidden_dim=config.gru_hidden_dim,
        dropout=config.dropout,
    ).to(device)
    stage_criterion = nn.CrossEntropyLoss(
        weight=_class_weights(stages, lengths, training).to(device)
    )
    boundary_criterion = nn.BCEWithLogitsLoss(
        pos_weight=_boundary_pos_weight(boundaries, lengths, training).to(device)
    )
    positive = float(labels[training].sum())
    clip_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor((training.size - positive) / positive, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    final_indices = final_scores = None
    for epoch in range(config.epochs):
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            stage_criterion=stage_criterion,
            boundary_criterion=boundary_criterion,
            clip_criterion=clip_criterion,
            config=config,
        )
        evaluation, final_indices, final_scores = evaluate(
            model, evaluation_loader, device=device, clip_ids=clip_ids
        )
        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "oof": evaluation,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False, sort_keys=True), flush=True)
        scheduler.step()
    assert final_indices is not None and final_scores is not None
    checkpoint = {
        "epoch": config.epochs - 1,
        "model_state": model.state_dict(),
        "config": asdict(config),
        "fold": args.fold,
        "metrics": history[-1]["oof"],
    }
    _atomic_checkpoint(args.output_dir / "last.pt", checkpoint)
    np.savez_compressed(
        args.output_dir / "oof_predictions.npz",
        clip_indices=final_indices,
        clip_ids=np.asarray([clip_ids[index] for index in final_indices]),
        labels=labels[final_indices].numpy(),
        scores=final_scores,
        fold=np.full(final_indices.size, args.fold, dtype=np.int64),
    )
    signature = {
        "protocol": "template_grouped_train_only_long_context_event_oracle_v1",
        "cache_signature_sha256": cache.metadata["signature_sha256"],
        "sidecar_sha256": _sha256_file(args.sidecar / "sidecar.bin"),
        "fold": args.fold,
        "template_groups_total": len(set(groups)),
        "train_clips": int(training.size),
        "oof_clips": int(held_out.size),
        "train_positive": int(labels[training].sum()),
        "oof_positive": int(labels[held_out].sum()),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "config": asdict(config),
        "final_metrics": history[-1]["oof"],
        "checkpoint_sha256": _sha256_file(args.output_dir / "last.pt"),
    }
    _atomic_json(args.output_dir / "summary.json", signature)
    (args.output_dir / "history.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in history),
        encoding="utf-8",
    )
    print(json.dumps(signature, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
