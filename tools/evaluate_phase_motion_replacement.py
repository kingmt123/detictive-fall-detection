"""Evaluate a Phase-Motion fold canary as a one-for-one seven-head replacement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from tools.audit_paired_clip_predictions import (
    _operating_point,
    metrics_from_arrays,
    paired_group_bootstrap,
)
from tools.audit_probability_rmpts_errors import error_transitions
from tools.train_tcn import _atomic_json, _sha256_file


def equal_logit_fusion(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("member scores 必须为至少两列")
    values = np.clip(values, 1e-7, 1.0 - 1e-7)
    logits = np.log(values / (1.0 - values)).mean(axis=1)
    return 1.0 / (1.0 + np.exp(-logits))


def _load_aligned(path: Path, reference_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        ids = np.asarray(payload["clip_id"]).astype(str)
        labels = np.asarray(payload["label"], dtype=np.uint8)
        scores = np.asarray(payload["score"], dtype=np.float64)
    index = {clip_id: offset for offset, clip_id in enumerate(ids)}
    if set(index) != set(reference_ids.tolist()):
        raise ValueError(f"OOF clip 集合不一致: {path}")
    order = np.asarray([index[clip_id] for clip_id in reference_ids], dtype=np.int64)
    return labels[order], scores[order]


def _r95_transitions(
    clip_ids: np.ndarray,
    labels: np.ndarray,
    baseline: np.ndarray,
    challenger: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    baseline_point = _operating_point(clip_ids, labels, baseline, 0.95)
    challenger_point = _operating_point(clip_ids, labels, challenger, 0.95)
    transitions = error_transitions(
        clip_ids,
        labels,
        np.zeros(len(labels), dtype=np.int64),
        baseline >= float(baseline_point["threshold"]),
        challenger >= float(challenger_point["threshold"]),
    )
    return baseline_point, challenger_point, transitions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--member-report", type=Path, required=True)
    parser.add_argument("--dense-oof", type=Path, required=True)
    parser.add_argument("--phase-oof", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions-output", type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Phase-Motion replacement 输出已存在，拒绝覆盖")
    if args.predictions_output is not None and args.predictions_output.exists():
        raise FileExistsError("Phase-Motion replacement predictions 已存在，拒绝覆盖")
    member_report = json.loads(args.member_report.read_text(encoding="utf-8"))
    members = member_report.get("members")
    if not isinstance(members, list) or len(members) != 6:
        raise ValueError("member report 必须包含六个短窗成员")
    if not 0 <= args.fold < 5:
        raise ValueError("fold 必须位于 [0,5)")
    reference_path = Path(members[0]["oof"][args.fold]["path"])
    with np.load(reference_path, allow_pickle=False) as payload:
        clip_ids = np.asarray(payload["clip_id"]).astype(str)
        labels = np.asarray(payload["label"], dtype=np.uint8)
    short_scores = []
    member_metrics = {}
    global_member_metrics = {}
    for member in members:
        path = Path(member["oof"][args.fold]["path"])
        candidate_labels, scores = _load_aligned(path, clip_ids)
        if not np.array_equal(candidate_labels, labels):
            raise ValueError("短窗 member 标签不一致")
        short_scores.append(scores)
        member_metrics[str(member["name"])] = metrics_from_arrays(labels, scores)
        global_labels = []
        global_scores = []
        for fold_entry in member["oof"]:
            with np.load(Path(fold_entry["path"]), allow_pickle=False) as payload:
                global_labels.append(np.asarray(payload["label"], dtype=np.uint8))
                global_scores.append(np.asarray(payload["score"], dtype=np.float64))
        global_member_metrics[str(member["name"])] = metrics_from_arrays(
            np.concatenate(global_labels), np.concatenate(global_scores)
        )
    dense_labels, dense_scores = _load_aligned(args.dense_oof, clip_ids)
    phase_labels, phase_scores = _load_aligned(args.phase_oof, clip_ids)
    if not np.array_equal(dense_labels, labels) or not np.array_equal(phase_labels, labels):
        raise ValueError("Dense/Phase-Motion 标签不一致")
    short_matrix = np.stack(short_scores, axis=1)
    weakest_index = min(
        range(len(members)),
        key=lambda index: (
            global_member_metrics[str(members[index]["name"])]["clip_map_percent"],
            index,
        ),
    )
    baseline_matrix = np.column_stack((short_matrix, dense_scores))
    replacement_matrix = baseline_matrix.copy()
    replacement_matrix[:, weakest_index] = phase_scores
    baseline = equal_logit_fusion(baseline_matrix)
    challenger = equal_logit_fusion(replacement_matrix)
    baseline_metrics = metrics_from_arrays(labels, baseline)
    challenger_metrics = metrics_from_arrays(labels, challenger)
    metric_names = ("clip_map_percent", "clip_p_at_r90", "clip_p_at_r95")
    baseline_point, challenger_point, transitions = _r95_transitions(
        clip_ids, labels, baseline, challenger
    )
    report = {
        "protocol": "phase_motion_native48_fold_replacement_audit_v1",
        "fold": args.fold,
        "clips": len(clip_ids),
        "phase_motion": {
            "metrics": metrics_from_arrays(labels, phase_scores),
            "parameters": 5607,
            "score_pearson_by_member": {
                str(member["name"]): float(np.corrcoef(phase_scores, short_matrix[:, index])[0, 1])
                for index, member in enumerate(members)
            },
            "error_pearson_by_member": {
                str(member["name"]): float(
                    np.corrcoef(phase_scores - labels, short_matrix[:, index] - labels)[0, 1]
                )
                for index, member in enumerate(members)
            },
        },
        "short_member_metrics": member_metrics,
        "global_short_member_metrics_used_for_fixed_selection": global_member_metrics,
        "replacement": {
            "rule": "replace_global_fivefold_weakest_short_member_by_single_head_clip_map",
            "removed_member": str(members[weakest_index]["name"]),
            "baseline": baseline_metrics,
            "challenger": challenger_metrics,
            "delta": {
                name: challenger_metrics[name] - baseline_metrics[name]
                for name in metric_names
            },
            "r95_operating_points": {
                "baseline": baseline_point,
                "challenger": challenger_point,
            },
            "r95_transitions": transitions,
            "paired_template_group_bootstrap_delta_95ci": paired_group_bootstrap(
                clip_ids,
                labels,
                baseline,
                challenger,
                replicates=args.bootstrap_replicates,
                seed=20260829,
            ),
        },
        "inputs": {
            "member_report_sha256": _sha256_file(args.member_report),
            "dense_oof_sha256": _sha256_file(args.dense_oof),
            "phase_oof_sha256": _sha256_file(args.phase_oof),
        },
        "test_accessed": False,
    }
    if args.predictions_output is not None:
        args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.predictions_output,
            clip_id=clip_ids,
            label=labels,
            baseline=baseline,
            challenger=challenger,
            fold=np.full(len(clip_ids), args.fold, dtype=np.int64),
        )
        report["predictions_sha256"] = _sha256_file(args.predictions_output)
    _atomic_json(args.output, report)
    print(json.dumps(report["replacement"], sort_keys=True))


if __name__ == "__main__":
    main()
