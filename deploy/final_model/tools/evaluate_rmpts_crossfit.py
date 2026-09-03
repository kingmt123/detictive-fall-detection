"""Cross-fit RMPTS delta selection with grouped 1-SE simplification.

The default CLI consumes the existing first-layer OOF table and is therefore
only a stability surrogate: base models for an inner fold may have seen the
current outer fold.  The selection functions are also used by the future fully
outer-isolated protocol, where that limitation does not apply.
"""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from models.edgefall_ensemble import RMPTS_GRID, rmpts_fuse_numpy_logits
from tools.audit_paired_clip_predictions import (
    metrics_from_arrays,
    paired_group_bootstrap,
)
from tools.evaluate_rmpts_diagnostic import _logit, load_member_matrix
from tools.train_long_context_event_oracle import template_group
from tools.train_tcn import _atomic_json, _sha256_file


def _rmpts_scores(short_logits: np.ndarray, dense_logits: np.ndarray) -> dict[Fraction, np.ndarray]:
    if short_logits.ndim != 2 or short_logits.shape[1] != 6:
        raise ValueError("RMPTS short logits 必须为 (N,6)")
    if dense_logits.shape != (short_logits.shape[0],):
        raise ValueError("RMPTS dense logits 必须为 (N,)")
    return {
        delta: 1.0
        / (
            1.0
            + np.exp(
                -rmpts_fuse_numpy_logits(
                    short_logits[:, (0, 2, 4)],
                    short_logits[:, (1, 3, 5)],
                    dense_logits,
                    delta=delta,
                )
            )
        )
        for delta in RMPTS_GRID
    }


def _paired_map_standard_error(
    clip_ids: np.ndarray,
    labels: np.ndarray,
    best_scores: np.ndarray,
    candidate_scores: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> float:
    groups = np.asarray([template_group(str(item)) for item in clip_ids])
    group_rows = [np.flatnonzero(groups == group) for group in sorted(set(groups.tolist()))]
    if len(group_rows) < 2:
        raise ValueError("RMPTS 1-SE 需要至少两个 template group")
    rng = np.random.default_rng(seed)
    deltas = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        indices = np.concatenate(
            [group_rows[index] for index in rng.integers(0, len(group_rows), len(group_rows))]
        )
        deltas[replicate] = (
            metrics_from_arrays(labels[indices], best_scores[indices])["clip_map_percent"]
            - metrics_from_arrays(labels[indices], candidate_scores[indices])["clip_map_percent"]
        )
    return float(np.std(deltas, ddof=1))


def select_rmpts_delta(
    clip_ids: np.ndarray,
    labels: np.ndarray,
    scores_by_delta: dict[Fraction, np.ndarray],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> tuple[Fraction, list[dict[str, Any]]]:
    """Apply dual working-point guards, then the grouped 1-SE simplicity rule."""
    if tuple(scores_by_delta) != RMPTS_GRID:
        raise ValueError("RMPTS selector 必须收到完整且有序的预注册网格")
    if bootstrap_replicates < 100:
        raise ValueError("RMPTS selector bootstrap 至少需要 100 次")
    labels = np.asarray(labels, dtype=np.uint8)
    clip_ids = np.asarray(clip_ids).astype(str)
    zero_metrics = metrics_from_arrays(labels, scores_by_delta[Fraction(0, 1)])
    candidates: list[dict[str, Any]] = []
    for delta, scores in scores_by_delta.items():
        metrics = metrics_from_arrays(labels, scores)
        eligible = (
            metrics["clip_p_at_r90"] >= zero_metrics["clip_p_at_r90"]
            and metrics["clip_p_at_r95"] >= zero_metrics["clip_p_at_r95"]
        )
        candidates.append(
            {
                "delta": delta,
                "metrics": metrics,
                "passes_working_point_guards": eligible,
            }
        )
    eligible = [item for item in candidates if item["passes_working_point_guards"]]
    provisional = max(
        eligible,
        key=lambda item: (
            item["metrics"]["clip_map_percent"],
            item["metrics"]["clip_p_at_r95"],
            item["metrics"]["clip_p_at_r90"],
        ),
    )
    best_delta = provisional["delta"]
    best_scores = scores_by_delta[best_delta]
    tied = []
    for offset, item in enumerate(eligible):
        delta = item["delta"]
        gap = provisional["metrics"]["clip_map_percent"] - item["metrics"]["clip_map_percent"]
        standard_error = _paired_map_standard_error(
            clip_ids,
            labels,
            best_scores,
            scores_by_delta[delta],
            replicates=bootstrap_replicates,
            seed=bootstrap_seed + offset,
        )
        item["map_gap_from_provisional_best"] = gap
        item["paired_map_gap_standard_error"] = standard_error
        item["within_one_standard_error"] = gap <= standard_error + 1e-12
        if item["within_one_standard_error"]:
            tied.append(item)
    selected = min(
        tied,
        key=lambda item: (
            abs(item["delta"]),
            -item["metrics"]["clip_p_at_r95"],
            -item["metrics"]["clip_p_at_r90"],
            0 if item["delta"] < 0 else 1,
        ),
    )
    serializable = [
        {
            **item,
            "delta": f"{item['delta'].numerator}/{item['delta'].denominator}",
        }
        for item in candidates
    ]
    return selected["delta"], serializable


def surrogate_crossfit(
    clip_ids: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    short_logits: np.ndarray,
    dense_logits: np.ndarray,
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> tuple[np.ndarray, list[dict[str, Any]], Fraction]:
    """Select on four existing OOF folds and apply to the fifth, for diagnosis only."""
    scores = _rmpts_scores(short_logits, dense_logits)
    crossfit = np.full(labels.size, np.nan, dtype=np.float64)
    selections = []
    selected_deltas = []
    for outer_fold in sorted(set(folds.tolist())):
        inner = folds != outer_fold
        held = folds == outer_fold
        selected, candidates = select_rmpts_delta(
            clip_ids[inner],
            labels[inner],
            {delta: value[inner] for delta, value in scores.items()},
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=bootstrap_seed + int(outer_fold) * 100,
        )
        crossfit[held] = scores[selected][held]
        selected_deltas.append(selected)
        baseline_metrics = metrics_from_arrays(labels[held], scores[Fraction(0, 1)][held])
        selected_metrics = metrics_from_arrays(labels[held], scores[selected][held])
        selections.append(
            {
                "outer_fold": int(outer_fold),
                "selected_delta": f"{selected.numerator}/{selected.denominator}",
                "inner_candidates": candidates,
                "outer_metrics": selected_metrics,
                "outer_delta_vs_zero": {
                    name: selected_metrics[name] - baseline_metrics[name]
                    for name in baseline_metrics
                },
            }
        )
    if not np.isfinite(crossfit).all():
        raise AssertionError("RMPTS crossfit 未覆盖全部 outer rows")
    positive = sum(delta > 0 for delta in selected_deltas)
    negative = sum(delta < 0 for delta in selected_deltas)
    if positive >= 4 or negative >= 4:
        final_delta = sorted(selected_deltas)[len(selected_deltas) // 2]
    else:
        final_delta = Fraction(0, 1)
    return crossfit, selections, final_delta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--member-report", type=Path, required=True)
    parser.add_argument("--dense-oof", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260829)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("RMPTS crossfit 输出已存在，拒绝覆盖")
    with np.load(args.dense_oof, allow_pickle=False) as payload:
        required = {"clip_id", "label", "fold", "dense48_score"}
        if not required.issubset(payload.files):
            raise ValueError("dense OOF 缺少 RMPTS crossfit 字段")
        clip_ids = np.asarray(payload["clip_id"]).astype(str)
        labels = np.asarray(payload["label"], dtype=np.uint8)
        folds = np.asarray(payload["fold"], dtype=np.int64)
        dense_logits = _logit(np.asarray(payload["dense48_score"]))
    short_logits, _ = load_member_matrix(args.member_report, clip_ids, labels)
    crossfit, selections, final_delta = surrogate_crossfit(
        clip_ids,
        labels,
        folds,
        short_logits,
        dense_logits,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    zero_scores = _rmpts_scores(short_logits, dense_logits)[Fraction(0, 1)]
    report = {
        "protocol": "edgefall_rmpts_leave_one_outer_fold_out_surrogate_v1",
        "qualification": "stability_surrogate_only_base_members_not_outer_isolated",
        "selected_delta_for_deployment": None,
        "direction_stable_median_delta": (
            f"{final_delta.numerator}/{final_delta.denominator}"
        ),
        "selections": selections,
        "baseline_metrics": metrics_from_arrays(labels, zero_scores),
        "crossfit_metrics": metrics_from_arrays(labels, crossfit),
        "paired_group_bootstrap_delta_95ci": paired_group_bootstrap(
            clip_ids,
            labels,
            zero_scores,
            crossfit,
            replicates=args.bootstrap_replicates,
            seed=args.bootstrap_seed,
        ),
        "member_report_sha256": _sha256_file(args.member_report),
        "dense_oof_sha256": _sha256_file(args.dense_oof),
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.bootstrap_seed,
        "test_accessed": False,
        "limitations": [
            "inner base predictions were not trained with the current outer fold excluded",
            "this result may authorize nested training but cannot qualify deployment",
        ],
    }
    _atomic_json(args.output, report)
    print(json.dumps({"output": str(args.output.resolve()), "final_delta": str(final_delta)}))


if __name__ == "__main__":
    main()
