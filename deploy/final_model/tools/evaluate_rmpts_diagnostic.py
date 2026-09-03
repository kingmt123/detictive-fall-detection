"""Diagnose the pre-registered RMPTS grid without selecting a deployment weight.

This tool intentionally labels every result as same-OOF diagnostic evidence.
It cannot qualify a non-zero delta; that requires outer-isolated base members
and the full nested grouped protocol.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from models.edgefall_ensemble import RMPTS_GRID, rmpts_fuse_numpy_logits
from tools.audit_paired_clip_predictions import (
    metrics_from_arrays,
    paired_group_bootstrap,
)
from tools.train_tcn import _atomic_json, _sha256_file

EXPECTED_MEMBERS = (
    "F1-320-seed1",
    "F1-640-seed1",
    "F1-320-seed2",
    "F1-640-seed2",
    "F1-320-seed3",
    "F1-640-seed3",
)


def _load_score(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        clip_key = "clip_ids" if "clip_ids" in payload.files else "clip_id"
        label_key = "labels" if "labels" in payload.files else "label"
        score_key = "fusion_scores" if "fusion_scores" in payload.files else "score"
        required = {clip_key, label_key, score_key}
        if not required.issubset(payload.files):
            raise ValueError(f"RMPTS 成员缺少字段: {sorted(required - set(payload.files))}")
        return (
            np.asarray(payload[clip_key]).astype(str),
            np.asarray(payload[label_key], dtype=np.uint8),
            np.asarray(payload[score_key], dtype=np.float64),
        )


def _logit(probability: np.ndarray) -> np.ndarray:
    values = np.asarray(probability, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("RMPTS score 必须为有限一维数组")
    clipped = np.clip(values, 1e-7, 1.0 - 1e-7)
    return np.log(clipped / (1.0 - clipped))


def load_member_matrix(
    member_report: Path, reference_ids: np.ndarray, reference_labels: np.ndarray
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    report = json.loads(member_report.read_text(encoding="utf-8"))
    members = report.get("members")
    if not isinstance(members, list) or tuple(
        member.get("name") for member in members if isinstance(member, dict)
    ) != EXPECTED_MEMBERS:
        raise ValueError("RMPTS 六个短窗成员名称或顺序不符合冻结协议")
    columns = []
    evidence = []
    for member in members:
        rows = member.get("oof")
        if not isinstance(rows, list) or len(rows) != 5:
            raise ValueError("每个 RMPTS 短窗成员必须提供五个 OOF fold")
        scores_by_id: dict[str, float] = {}
        paths = []
        for row in rows:
            path = Path(row["path"])
            clip_ids, labels, scores = _load_score(path)
            for clip_id, label, score in zip(clip_ids, labels, scores, strict=True):
                if clip_id in scores_by_id:
                    raise ValueError("RMPTS 成员跨 fold 出现重复 clip")
                index = np.flatnonzero(reference_ids == clip_id)
                if index.size != 1 or reference_labels[int(index[0])] != label:
                    raise ValueError("RMPTS 成员 clip/label 与 dense universe 不一致")
                scores_by_id[clip_id] = float(score)
            paths.append({"path": str(path), "sha256": _sha256_file(path)})
        if set(scores_by_id) != set(reference_ids.tolist()):
            raise ValueError("RMPTS 成员未完整覆盖 clip universe")
        columns.append(_logit(np.asarray([scores_by_id[item] for item in reference_ids])))
        evidence.append({"name": member["name"], "oof": paths})
    return np.stack(columns, axis=1), evidence


def diagnostic_grid(
    clip_ids: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    short_logits: np.ndarray,
    dense_logits: np.ndarray,
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    if short_logits.shape != (clip_ids.size, 6):
        raise ValueError("RMPTS short logits 必须为 (N,6)")
    baseline_scores: np.ndarray | None = None
    rows = []
    for delta in RMPTS_GRID:
        fused_logit = rmpts_fuse_numpy_logits(
            short_logits[:, (0, 2, 4)],
            short_logits[:, (1, 3, 5)],
            dense_logits,
            delta=delta,
        )
        scores = 1.0 / (1.0 + np.exp(-fused_logit))
        if delta.numerator == 0:
            baseline_scores = scores
        rows.append(
            {
                "delta": f"{delta.numerator}/{delta.denominator}",
                "metrics": metrics_from_arrays(labels, scores),
                "per_fold": [
                    {
                        "fold": int(fold),
                        **metrics_from_arrays(labels[folds == fold], scores[folds == fold]),
                    }
                    for fold in sorted(set(folds.tolist()))
                ],
                "_scores": scores,
            }
        )
    if baseline_scores is None:
        raise AssertionError("RMPTS 预注册网格缺少 delta=0")
    for row in rows:
        scores = row.pop("_scores")
        row["paired_group_bootstrap_delta_vs_zero_95ci"] = paired_group_bootstrap(
            clip_ids,
            labels,
            baseline_scores,
            scores,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--member-report", type=Path, required=True)
    parser.add_argument("--dense-oof", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260829)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("RMPTS diagnostic 输出已存在，拒绝覆盖")
    if args.bootstrap_replicates < 100:
        raise ValueError("RMPTS bootstrap 至少需要 100 次")
    with np.load(args.dense_oof, allow_pickle=False) as payload:
        required = {"clip_id", "label", "fold", "dense48_score"}
        if not required.issubset(payload.files):
            raise ValueError("dense OOF 缺少 RMPTS 所需字段")
        clip_ids = np.asarray(payload["clip_id"]).astype(str)
        labels = np.asarray(payload["label"], dtype=np.uint8)
        folds = np.asarray(payload["fold"], dtype=np.int64)
        dense_logits = _logit(np.asarray(payload["dense48_score"]))
    if len(set(clip_ids.tolist())) != clip_ids.size or len(set(folds.tolist())) != 5:
        raise ValueError("RMPTS 需要唯一 clip 与完整五折 OOF")
    short_logits, member_evidence = load_member_matrix(
        args.member_report, clip_ids, labels
    )
    report = {
        "protocol": "edgefall_rmpts_same_oof_diagnostic_v1",
        "qualification": "diagnostic_only_nested_outer_isolation_required",
        "selected_delta": None,
        "incumbent_delta": "0/1",
        "clips": int(clip_ids.size),
        "member_report_sha256": _sha256_file(args.member_report),
        "dense_oof_sha256": _sha256_file(args.dense_oof),
        "members": member_evidence,
        "grid": diagnostic_grid(
            clip_ids,
            labels,
            folds,
            short_logits,
            dense_logits,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
        ),
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.bootstrap_seed,
        "test_accessed": False,
        "limitations": [
            "delta candidates are inspected on the same OOF used to establish members",
            "no non-zero delta may replace the incumbent from this report",
            "formal promotion requires outer-isolated base training and nested grouped selection",
        ],
    }
    _atomic_json(args.output, report)
    print(json.dumps({"output": str(args.output.resolve()), "clips": clip_ids.size}))


if __name__ == "__main__":
    main()
