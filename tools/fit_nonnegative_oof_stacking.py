"""Fit nonnegative L2 static weights from complete train-only OOF predictions."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from models.oof_stacking import apply_nonnegative_l2, fit_nonnegative_l2
from models.tcn_dataset import load_window_cache
from tools.build_tcn_multistream_sidecar import load_sidecar
from tools.evaluate_oof_multistream_fusion import (
    _checkpoint_scores,
    _logit,
    _metrics,
)
from tools.train_long_context_event_oracle import template_group
from tools.train_tcn import _atomic_json


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_member_oof(
    pattern: str, clips: list[dict[str, Any]]
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    paths = sorted(Path(path) for path in glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"OOF pattern 未匹配文件: {pattern}")
    expected = {str(clip["clip_id"]): index for index, clip in enumerate(clips)}
    scores = np.full(len(clips), np.nan, dtype=np.float32)
    folds = np.full(len(clips), -1, dtype=np.int16)
    sources = []
    for path in paths:
        with np.load(path, allow_pickle=False) as payload:
            required = {"clip_id", "group_id", "label", "score", "fold"}
            if required - set(payload.files):
                raise ValueError(f"OOF 文件字段不完整: {path}")
            rows = {key: np.asarray(payload[key]) for key in required}
        size = rows["clip_id"].size
        if any(value.shape != (size,) for value in rows.values()):
            raise ValueError(f"OOF 文件字段 shape 不一致: {path}")
        for offset in range(size):
            clip_id = str(rows["clip_id"][offset])
            if clip_id not in expected:
                raise ValueError(f"OOF 出现未知 clip: {clip_id}")
            index = expected[clip_id]
            if np.isfinite(scores[index]):
                raise ValueError(f"OOF clip 重复预测: {clip_id}")
            if str(rows["group_id"][offset]) != template_group(clip_id):
                raise ValueError(f"OOF template group 不匹配: {clip_id}")
            if bool(rows["label"][offset]) != bool(clips[index]["has_fall"]):
                raise ValueError(f"OOF label 不匹配: {clip_id}")
            scores[index] = float(rows["score"][offset])
            folds[index] = int(rows["fold"][offset])
        sources.append({"path": str(path.resolve()), "sha256": _sha256(path)})
    if np.any(~np.isfinite(scores)) or np.any(folds < 0):
        missing = int(np.count_nonzero(~np.isfinite(scores)))
        raise ValueError(f"OOF 未完整覆盖 train clips，缺失 {missing}")
    group_fold: dict[str, int] = {}
    for index, clip in enumerate(clips):
        group = template_group(str(clip["clip_id"]))
        previous = group_fold.setdefault(group, int(folds[index]))
        if previous != int(folds[index]):
            raise ValueError(f"template group 跨 folds: {group}")
    return scores, folds, sources


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--val-sidecar", type=Path, required=True)
    parser.add_argument(
        "--member",
        nargs=4,
        action="append",
        metavar=("NAME", "OOF_GLOB", "RUN", "CHECKPOINT"),
        required=True,
    )
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("stacking 输出已存在，拒绝覆盖")
    if not 2 <= len(args.member) <= 12:
        raise ValueError("stacking 需要 2 到 12 个成员")
    train = load_window_cache(args.train_cache, verify_hashes=True)
    val = load_window_cache(args.val_cache, verify_hashes=True)
    if train.metadata.get("split") != "train" or val.metadata.get("split") != "val":
        raise ValueError("stacking 只允许 train OOF 拟合和 val 单次评估")
    clips = train.metadata["clips"]
    names: list[str] = []
    oof_scores: list[np.ndarray] = []
    reference_folds: np.ndarray | None = None
    member_sources = []
    for name, pattern, run_text, checkpoint_text in args.member:
        if name in names:
            raise ValueError(f"stacking member 名称重复: {name}")
        scores, folds, sources = _load_member_oof(pattern, clips)
        if reference_folds is None:
            reference_folds = folds
        elif not np.array_equal(reference_folds, folds):
            raise ValueError("stacking 成员的 OOF fold assignment 不一致")
        names.append(name)
        oof_scores.append(scores)
        member_sources.append(
            {
                "name": name,
                "oof": sources,
                "run": str(Path(run_text).resolve()),
                "run_sha256": _sha256(Path(run_text)),
                "checkpoint": str(Path(checkpoint_text).resolve()),
                "checkpoint_sha256": _sha256(Path(checkpoint_text)),
            }
        )
    labels = np.asarray(
        [float(bool(clip["has_fall"])) for clip in clips], dtype=np.float32
    )
    oof_features = np.column_stack([_logit(score) for score in oof_scores])
    weights, intercept, mean, scale = fit_nonnegative_l2(
        oof_features, labels, l2=args.l2
    )
    fused_oof = apply_nonnegative_l2(
        oof_features, weights, intercept, mean, scale
    )
    teacher_target_path = args.output.with_name(
        f"{args.output.stem}_oof_teacher_targets.npz"
    )
    if teacher_target_path.exists():
        raise FileExistsError("OOF teacher target 输出已存在，拒绝覆盖")
    np.savez_compressed(
        teacher_target_path,
        clip_id=np.asarray([str(clip["clip_id"]) for clip in clips]),
        label=labels.astype(np.uint8),
        fold=reference_folds,
        score=fused_oof,
    )
    val_sidecar = load_sidecar(args.val_sidecar, val)
    device = torch.device(args.device)
    val_scores = [
        _checkpoint_scores(
            cache=val,
            sidecar=val_sidecar,
            run=Path(member[2]),
            checkpoint=Path(member[3]),
            device=device,
            batch_size=args.batch_size,
        )
        for member in args.member
    ]
    val_features = np.column_stack([_logit(score) for score in val_scores])
    fused_val = apply_nonnegative_l2(
        val_features, weights, intercept, mean, scale
    )
    assert reference_folds is not None
    report = {
        "protocol": "nonnegative_l2_train_oof_stacking_v1",
        "selection_split": "train template-grouped OOF only",
        "validation_uses": 1,
        "test_accessed": False,
        "oof_teacher_targets": {
            "path": str(teacher_target_path.resolve()),
            "sha256": _sha256(teacher_target_path),
        },
        "members": member_sources,
        "fit": {
            "l2": args.l2,
            "weights": dict(zip(names, weights.tolist(), strict=True)),
            "intercept": intercept,
            "feature_mean": mean.tolist(),
            "feature_scale": scale.tolist(),
            "fold_assignment_sha256": hashlib.sha256(
                reference_folds.tobytes()
            ).hexdigest(),
        },
        "train_oof": {
            "members": dict(
                zip(
                    names,
                    [_metrics(clips, score) for score in oof_scores],
                    strict=True,
                )
            ),
            "fused": _metrics(clips, fused_oof),
        },
        "validation_once": {
            "members": dict(
                zip(
                    names,
                    [_metrics(val.metadata["clips"], score) for score in val_scores],
                    strict=True,
                )
            ),
            "fused": _metrics(val.metadata["clips"], fused_val),
        },
        "cache_signatures": {
            "train": train.metadata.get("signature_sha256"),
            "val": val.metadata.get("signature_sha256"),
        },
    }
    _atomic_json(args.output, report)
    print(json.dumps(report["validation_once"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
