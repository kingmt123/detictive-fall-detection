"""Build train-only empirical pose/track distortion replay templates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from models.tcn_dataset import load_window_cache
from tools.build_tcn_multistream_sidecar import load_sidecar
from tools.train_long_context_event_oracle import template_group


def eligible_window_indices(
    clip_indices: np.ndarray,
    clips: list[dict[str, object]],
    fold_map_path: Path | None,
    exclude_fold: int | None,
) -> tuple[np.ndarray, dict[str, object] | None]:
    """Return train-window rows allowed to contribute distortion templates."""
    if (fold_map_path is None) != (exclude_fold is None):
        raise ValueError("fold-map 与 exclude-fold 必须同时提供")
    all_rows = np.arange(clip_indices.size, dtype=np.int64)
    if fold_map_path is None:
        return all_rows, None
    payload = json.loads(fold_map_path.read_text(encoding="utf-8"))
    rows = payload.get("rows") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or len(rows) != len(clips):
        raise ValueError("fold map 与 cache clips 不匹配")
    assignments = np.empty(len(clips), dtype=np.int16)
    for index, (clip, row) in enumerate(zip(clips, rows, strict=True)):
        if not isinstance(row, dict) or row.get("clip_id") != clip.get("clip_id"):
            raise ValueError("fold map clip 顺序不匹配")
        if row.get("group_id") != template_group(str(clip["clip_id"])):
            raise ValueError("fold map template group 不匹配")
        assignments[index] = int(row["fold"])
    if exclude_fold not in assignments:
        raise ValueError("exclude-fold 在 fold map 中不存在")
    eligible = all_rows[assignments[clip_indices] != exclude_fold]
    if not eligible.size:
        raise ValueError("fold-local distortion bank 没有可用训练窗口")
    return eligible, {
        "path": str(fold_map_path),
        "sha256": hashlib.sha256(fold_map_path.read_bytes()).hexdigest(),
        "excluded_fold": int(exclude_fold),
    }


def _smooth_bbox(box: np.ndarray) -> np.ndarray:
    padded = np.pad(box, ((0, 0), (1, 1), (0, 0)), mode="edge")
    return (padded[:, :-2] + 2.0 * padded[:, 1:-1] + padded[:, 2:]) / 4.0


def _time_maps(pose: np.ndarray, observed: np.ndarray) -> np.ndarray:
    xy = pose[..., :2]
    conf = pose[..., 2] > 0
    valid = conf[:, 1:] & conf[:, :-1]
    displacement = np.linalg.norm(xy[:, 1:] - xy[:, :-1], axis=-1)
    speed = np.divide(
        (displacement * valid).sum(-1), valid.sum(-1),
        out=np.zeros(displacement.shape[:2], dtype=np.float32), where=valid.sum(-1) > 0,
    )
    positive = speed[speed > 0]
    low, high = (np.quantile(positive, [0.2, 0.8]) if positive.size else (0.0, 1.0))
    increments = np.ones_like(speed, dtype=np.float32)
    increments[speed <= low] = 0.0
    increments[speed >= high] = 2.0
    increments[~observed[:, 1:]] = 0.0
    cumulative = np.concatenate((np.zeros((pose.shape[0], 1), np.float32), np.cumsum(increments, axis=1)), axis=1)
    endpoint = pose.shape[1] - 1
    scale = np.maximum(cumulative[:, -1:], 1.0)
    return np.rint(cumulative / scale * endpoint).astype(np.int16).clip(0, endpoint)


def select_template_rows(
    quality_loss: np.ndarray,
    *,
    count: int,
    selection: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Select empirical failures without turning replay into an extreme-tail prior."""
    quality_loss = np.asarray(quality_loss, dtype=np.float64)
    if quality_loss.ndim != 1 or count < 1 or count > quality_loss.size:
        raise ValueError("distortion template selection shape/count 无效")
    order = np.argsort(quality_loss, kind="stable")
    if selection == "worst":
        return order[-count:]
    if selection != "severity_stratified":
        raise ValueError("未知 distortion template selection")
    # Replay is already applied to only 20% of training samples.  Within that
    # subset retain broad real failures: 50% moderate, 30% high, 20% extreme.
    boundaries = (
        (0.40, 0.80, count // 2),
        (0.80, 0.95, 3 * count // 10),
        (0.95, 1.00, count - count // 2 - 3 * count // 10),
    )
    selected: list[np.ndarray] = []
    for low, high, take in boundaries:
        start = int(np.floor(low * order.size))
        stop = max(start + 1, int(np.floor(high * order.size)))
        pool = order[start:stop]
        if take > pool.size:
            raise ValueError("severity stratum 太小，无法无放回抽样")
        selected.append(rng.choice(pool, take, replace=False))
    return np.concatenate(selected)


def build_bank(
    cache_path: Path,
    sidecar_path: Path,
    output: Path,
    *,
    count: int,
    seed: int,
    fold_map_path: Path | None = None,
    exclude_fold: int | None = None,
    selection: str = "worst",
) -> dict[str, object]:
    cache = load_window_cache(cache_path, verify_hashes=True)
    if cache.metadata.get("split") != "train":
        raise ValueError("真实失真模板只允许从 train cache 构建")
    sidecar = load_sidecar(sidecar_path, cache, verify_hashes=True)
    rng = np.random.default_rng(seed)
    clip_indices = np.asarray(cache.array("clip_indices"), dtype=np.int64)
    eligible, fold_evidence = eligible_window_indices(
        clip_indices,
        cache.metadata["clips"],
        fold_map_path,
        exclude_fold,
    )
    candidate_count = min(eligible.size, max(count * 32, count))
    candidates = rng.choice(eligible, candidate_count, replace=False)
    pose = np.asarray(cache.array("features")[candidates], dtype=np.float32)
    geometry = np.asarray(sidecar[candidates], dtype=np.float32)
    observed = geometry[..., 5] > 0
    confidence = pose[..., 2]
    quality_loss = (1.0 - confidence).mean((1, 2)) + (~observed).mean(1)
    bbox = geometry[..., :4]
    residual = bbox - _smooth_bbox(bbox)
    quality_loss += np.linalg.norm(residual[..., :2], axis=-1).mean(1)
    chosen_local = select_template_rows(
        quality_loss, count=count, selection=selection, rng=rng
    )
    pose, geometry, observed = pose[chosen_local], geometry[chosen_local], observed[chosen_local]
    confidence = pose[..., 2]
    per_window_high = np.quantile(confidence, 0.9, axis=(1, 2), keepdims=True).clip(1e-3)
    joint_keep = np.clip(confidence / per_window_high, 0.0, 1.0).astype(np.float16)
    bbox_residual = (geometry[..., :4] - _smooth_bbox(geometry[..., :4])).astype(np.float16)
    time_index = _time_maps(pose, observed)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output, joint_keep=joint_keep, track_keep=observed,
        bbox_residual=bbox_residual, time_index=time_index,
        source_window_index=candidates[chosen_local].astype(np.int32),
    )
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    report = {
        "protocol": "train_only_empirical_pose_track_distortion_bank_v1",
        "source_cache_signature": cache.metadata["signature_sha256"],
        "source_eligible_windows": int(eligible.size),
        "fold_assignment": fold_evidence,
        "templates": int(count), "seed": seed, "sha256": digest,
        "selection": selection,
        "statistics": {
            "mean_joint_keep": float(joint_keep.mean()),
            "mean_track_keep": float(observed.mean()),
            "mean_repeated_steps": float((np.diff(time_index, axis=1) == 0).sum(1).mean()),
            "mean_skipped_steps": float((np.diff(time_index, axis=1) > 1).sum(1).mean()),
            "bbox_residual_rms": float(np.sqrt(np.mean(bbox_residual.astype(np.float32) ** 2))),
        },
    }
    output.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--train-sidecar", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--fold-map", type=Path)
    parser.add_argument("--exclude-fold", type=int)
    parser.add_argument(
        "--selection",
        choices=("worst", "severity_stratified"),
        default="worst",
    )
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError("真实失真模板输出已存在，拒绝覆盖")
    if args.count < 1:
        raise ValueError("count 必须为正")
    print(json.dumps(build_bank(
        args.train_cache,
        args.train_sidecar,
        args.output,
        count=args.count,
        seed=args.seed,
        fold_map_path=args.fold_map,
        exclude_fold=args.exclude_fold,
        selection=args.selection,
    ), ensure_ascii=False))


if __name__ == "__main__":
    main()
