"""Create a fixed equal-logit fusion from identity-matched OOF predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools.train_edgefall_f1 import metric_result
from tools.train_tcn import _atomic_json, _sha256_file


def equal_logit_fusion(scores: list[np.ndarray]) -> np.ndarray:
    if len(scores) < 2:
        raise ValueError("至少需要两个 OOF 成员")
    arrays = [np.asarray(score, dtype=np.float64) for score in scores]
    if any(value.ndim != 1 or value.shape != arrays[0].shape for value in arrays):
        raise ValueError("OOF score 必须为同形一维")
    clipped = [np.clip(value, 1e-6, 1.0 - 1e-6) for value in arrays]
    logits = [np.log(value / (1.0 - value)) for value in clipped]
    mean_logit = np.mean(logits, axis=0)
    return (1.0 / (1.0 + np.exp(-mean_logit))).astype(np.float32)


def equal_probability_fusion(scores: list[np.ndarray]) -> np.ndarray:
    if len(scores) < 2:
        raise ValueError("至少需要两个 OOF 成员")
    arrays = [np.asarray(score, dtype=np.float64) for score in scores]
    if any(value.ndim != 1 or value.shape != arrays[0].shape for value in arrays):
        raise ValueError("OOF score 必须为同形一维")
    if any(not np.isfinite(value).all() for value in arrays):
        raise ValueError("OOF score 必须为有限数值")
    return np.mean(arrays, axis=0).astype(np.float32)


def load_member(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        clip_key = "clip_ids" if "clip_ids" in payload.files else "clip_id"
        label_key = "labels" if "labels" in payload.files else "label"
        score_key = (
            "fusion_scores" if "fusion_scores" in payload.files else "score"
        )
        required = {clip_key, label_key, score_key}
        if not required.issubset(payload.files):
            raise ValueError(
                f"融合成员缺少字段: {sorted(required - set(payload.files))}"
            )
        clip_ids = np.asarray(payload[clip_key])
        labels = np.asarray(payload[label_key])
        scores = np.asarray(payload[score_key])
        folds = (
            np.asarray(payload["fold"])
            if "fold" in payload.files
            else np.zeros(clip_ids.size, dtype=np.int64)
        )
        return clip_ids, labels, scores, folds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument(
        "--fusion", choices=("equal_logit", "equal_probability"), default="equal_logit"
    )
    parser.add_argument("--validation-uses", type=int, choices=(0, 1), default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    if len(args.input) < 2:
        raise ValueError("至少提供两个 --input")
    if args.output.exists() or args.summary.exists():
        raise FileExistsError("融合输出已存在，拒绝覆盖")
    members = [load_member(path) for path in args.input]
    clip_ids, labels, _, folds = members[0]
    for member in members[1:]:
        if not (
            np.array_equal(clip_ids, member[0])
            and np.array_equal(labels, member[1])
            and np.array_equal(folds, member[3])
        ):
            raise ValueError("OOF 成员 clip/label/fold 身份不匹配")
    fusion_function = (
        equal_logit_fusion
        if args.fusion == "equal_logit"
        else equal_probability_fusion
    )
    scores = fusion_function([member[2] for member in members])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        clip_ids=clip_ids,
        labels=labels,
        fusion_scores=scores,
        fold=folds,
    )
    summary = {
        "protocol": f"edgefall_fixed_{args.fusion}_prediction_fusion_v1",
        "fusion": args.fusion,
        "inputs": [
            {"path": str(path), "sha256": _sha256_file(path)}
            for path in args.input
        ],
        "metrics": metric_result(
            [str(value) for value in clip_ids], labels, scores
        ),
        "output_sha256": _sha256_file(args.output),
        "validation_uses": args.validation_uses,
        "test_accessed": False,
    }
    _atomic_json(args.summary, summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
