"""Evaluate one globally fitted temperature for Probability-RMPTS nested folds.

The temperature for outer fold ``k`` is fitted by BCE on the four inner OOF
parts whose seven base members excluded ``k``.  It is then applied to the
frozen seven-member predictions for outer fold ``k`` at fixed delta 1/7.
"""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from tools.audit_paired_clip_predictions import (
    metrics_from_arrays,
    paired_group_bootstrap,
)
from tools.evaluate_probability_rmpts_calibration_crossfit import (
    calibrated_probability_rmpts,
    fit_temperature,
)
from tools.evaluate_rmpts_diagnostic import _logit, load_member_matrix
from tools.evaluate_rmpts_nested import load_nested_inner, parse_outer_roots
from tools.train_tcn import _atomic_json, _sha256_file

FIXED_DELTA = Fraction(1, 7)


def fit_inner_global_temperature(
    labels: np.ndarray,
    short_probabilities: np.ndarray,
    dense_probability: np.ndarray,
) -> float:
    if short_probabilities.shape != (labels.size, 6):
        raise ValueError("nested temperature short probability shape 不匹配")
    if dense_probability.shape != (labels.size,):
        raise ValueError("nested temperature dense probability shape 不匹配")
    all_probabilities = np.column_stack((short_probabilities, dense_probability))
    if not np.isfinite(all_probabilities).all() or np.any(
        (all_probabilities < 0.0) | (all_probabilities > 1.0)
    ):
        raise ValueError("nested temperature probability 无效")
    clipped = np.clip(all_probabilities, 1e-7, 1.0 - 1e-7)
    logits = np.log(clipped / (1.0 - clipped))
    return fit_temperature(logits, labels)


def metric_delta(
    baseline: dict[str, float], challenger: dict[str, float]
) -> dict[str, float]:
    return {key: challenger[key] - baseline[key] for key in baseline}


def promotion_gates(
    baseline: dict[str, float],
    challenger: dict[str, float],
    fold_deltas: list[dict[str, float]],
    bootstrap: dict[str, list[float]],
) -> dict[str, bool]:
    return {
        "global_map_improves": (
            challenger["clip_map_percent"] > baseline["clip_map_percent"]
        ),
        "global_p90_nonnegative": (
            challenger["clip_p_at_r90"] >= baseline["clip_p_at_r90"]
        ),
        "global_p95_nonnegative": (
            challenger["clip_p_at_r95"] >= baseline["clip_p_at_r95"]
        ),
        "map_bootstrap_lower_positive": bootstrap["clip_map_percent"][0] > 0.0,
        "at_least_four_fold_map_nonnegative": (
            sum(row["clip_map_percent"] >= 0.0 for row in fold_deltas) >= 4
        ),
        "no_fold_map_regression_over_half_point": (
            min(row["clip_map_percent"] for row in fold_deltas) >= -0.5
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outer-root", action="append", required=True)
    parser.add_argument("--member-report", type=Path, required=True)
    parser.add_argument("--dense-oof", type=Path, required=True)
    parser.add_argument("--primary-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260829)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("nested temperature 输出已存在，拒绝覆盖")
    if args.bootstrap_replicates < 100:
        raise ValueError("nested temperature bootstrap 至少需要 100 次")
    primary = json.loads(args.primary_result.read_text(encoding="utf-8"))
    if primary.get("deployment_candidate_if_confirmed") != "probability:1/7":
        raise ValueError("nested temperature activation gate 未通过")
    roots = parse_outer_roots(args.outer_root)
    with np.load(args.dense_oof, allow_pickle=False) as payload:
        required = {"clip_id", "label", "fold", "dense48_score"}
        if not required.issubset(payload.files):
            raise ValueError("dense outer OOF 缺少 nested temperature 字段")
        outer_ids = np.asarray(payload["clip_id"]).astype(str)
        outer_labels = np.asarray(payload["label"], dtype=np.uint8)
        outer_folds = np.asarray(payload["fold"], dtype=np.int64)
        outer_dense_probability = np.asarray(payload["dense48_score"], dtype=np.float64)
    outer_short_logits, member_evidence = load_member_matrix(
        args.member_report, outer_ids, outer_labels
    )
    outer_dense_logits = _logit(outer_dense_probability)
    baseline_scores = calibrated_probability_rmpts(
        outer_short_logits,
        outer_dense_logits,
        np.ones(7, dtype=np.float64),
    )
    nested_scores = np.full(outer_labels.size, np.nan, dtype=np.float64)
    fold_rows: list[dict[str, Any]] = []
    fold_deltas = []
    all_inner_ids = []
    all_inner_labels = []
    all_inner_short = []
    all_inner_dense = []
    for outer_fold in range(5):
        inner_parts = [
            load_nested_inner(
                roots[outer_fold], outer_fold=outer_fold, inner_fold=inner_fold
            )
            for inner_fold in range(5)
            if inner_fold != outer_fold
        ]
        inner_ids = np.concatenate([part[0] for part in inner_parts])
        inner_labels = np.concatenate([part[1] for part in inner_parts])
        inner_short = np.concatenate([part[2] for part in inner_parts])
        inner_dense = np.concatenate([part[3] for part in inner_parts])
        if len(set(inner_ids.tolist())) != inner_ids.size:
            raise ValueError("nested temperature inner folds 出现重复 clip")
        temperature = fit_inner_global_temperature(
            inner_labels, inner_short, inner_dense
        )
        held = outer_folds == outer_fold
        held_scores = calibrated_probability_rmpts(
            outer_short_logits[held],
            outer_dense_logits[held],
            np.full(7, temperature, dtype=np.float64),
        )
        nested_scores[held] = held_scores
        baseline_metrics = metrics_from_arrays(outer_labels[held], baseline_scores[held])
        challenger_metrics = metrics_from_arrays(outer_labels[held], held_scores)
        delta = metric_delta(baseline_metrics, challenger_metrics)
        fold_deltas.append(delta)
        fold_rows.append(
            {
                "outer_fold": outer_fold,
                "temperature": temperature,
                "baseline_metrics": baseline_metrics,
                "challenger_metrics": challenger_metrics,
                "delta": delta,
                "inner_evidence": [part[4] for part in inner_parts],
            }
        )
        all_inner_ids.append(inner_ids)
        all_inner_labels.append(inner_labels)
        all_inner_short.append(inner_short)
        all_inner_dense.append(inner_dense)
    if not np.isfinite(nested_scores).all():
        raise AssertionError("nested temperature 未覆盖全部 outer OOF")
    baseline_metrics = metrics_from_arrays(outer_labels, baseline_scores)
    challenger_metrics = metrics_from_arrays(outer_labels, nested_scores)
    bootstrap = paired_group_bootstrap(
        outer_ids,
        outer_labels,
        baseline_scores,
        nested_scores,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    gates = promotion_gates(
        baseline_metrics, challenger_metrics, fold_deltas, bootstrap
    )
    final_temperature = fit_inner_global_temperature(
        np.concatenate(all_inner_labels),
        np.concatenate(all_inner_short),
        np.concatenate(all_inner_dense),
    )
    report = {
        "protocol": "edgefall_probability_rmpts_global_temperature_nested_v1",
        "qualification": "conditional_retrospective_nested_confirmation",
        "fixed_delta": "1/7",
        "temperature_objective": "unweighted_binary_cross_entropy",
        "temperature_scope": "one_shared_temperature_for_all_seven_members",
        "baseline_metrics": baseline_metrics,
        "nested_temperature_metrics": challenger_metrics,
        "delta": metric_delta(baseline_metrics, challenger_metrics),
        "paired_group_bootstrap_delta_95ci": bootstrap,
        "folds": fold_rows,
        "promotion_gates": gates,
        "passes_all_promotion_gates": all(gates.values()),
        "final_temperature_if_confirmed": final_temperature,
        "primary_result_sha256": _sha256_file(args.primary_result),
        "member_report_sha256": _sha256_file(args.member_report),
        "dense_oof_sha256": _sha256_file(args.dense_oof),
        "outer_member_evidence": member_evidence,
        "test_accessed": False,
        "limitations": [
            "temperature calibration was proposed after inspecting historical OOF results",
            "this is conditional retrospective nested confirmation, not blind validation",
            "sealed OF-Syn test and reserved URFD test remain inaccessible",
        ],
    }
    _atomic_json(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "passes": report["passes_all_promotion_gates"],
            }
        )
    )


if __name__ == "__main__":
    main()
