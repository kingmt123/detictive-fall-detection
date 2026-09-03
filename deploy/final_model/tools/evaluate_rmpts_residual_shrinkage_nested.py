"""Nested evaluation of logit-residual shrinkage toward equal-logit fusion."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from tools.audit_paired_clip_predictions import (
    metrics_from_arrays,
    paired_group_bootstrap,
)
from tools.evaluate_rmpts_crossfit import _paired_map_standard_error
from tools.evaluate_rmpts_diagnostic import load_member_matrix
from tools.evaluate_rmpts_nested import (
    fusion_candidate_scores,
    load_nested_inner,
    parse_outer_roots,
)
from tools.train_tcn import _atomic_json, _sha256_file

ALPHAS = (Fraction(0, 1), Fraction(1, 8), Fraction(1, 4), Fraction(1, 2), Fraction(1, 1))
METRICS = ("clip_map_percent", "clip_p_at_r90", "clip_p_at_r95")


def _logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    return np.log(clipped / (1.0 - clipped))


def residual_shrinkage_scores(
    equal_logit: np.ndarray, probability_rmpts: np.ndarray, alpha: Fraction
) -> np.ndarray:
    if equal_logit.shape != probability_rmpts.shape or equal_logit.ndim != 1:
        raise ValueError("residual shrinkage score shape 不匹配")
    if not np.isfinite(equal_logit).all() or not np.isfinite(probability_rmpts).all():
        raise ValueError("residual shrinkage scores 必须有限")
    weight = float(alpha)
    blended = _logit(equal_logit) + weight * (
        _logit(probability_rmpts) - _logit(equal_logit)
    )
    return 1.0 / (1.0 + np.exp(-np.clip(blended, -30.0, 30.0)))


def candidate_scores(
    short_probabilities: np.ndarray, dense_probability: np.ndarray
) -> dict[Fraction, np.ndarray]:
    fusion = fusion_candidate_scores(short_probabilities, dense_probability)
    equal_logit = fusion[("logit", Fraction(0, 1))]
    probability_rmpts = fusion[("probability", Fraction(1, 7))]
    return {
        alpha: residual_shrinkage_scores(equal_logit, probability_rmpts, alpha)
        for alpha in ALPHAS
    }


def select_alpha(
    clip_ids: np.ndarray,
    labels: np.ndarray,
    scores: dict[Fraction, np.ndarray],
    *,
    bootstrap_replicates: int,
    seed: int,
) -> tuple[Fraction, list[dict[str, Any]]]:
    if tuple(scores) != ALPHAS:
        raise ValueError("residual shrinkage alpha grid 不匹配")
    baseline = metrics_from_arrays(labels, scores[Fraction(0, 1)])
    rows = []
    for alpha, values in scores.items():
        metrics = metrics_from_arrays(labels, values)
        rows.append(
            {
                "alpha": alpha,
                "metrics": metrics,
                "passes_working_point_guards": (
                    metrics["clip_p_at_r90"] >= baseline["clip_p_at_r90"]
                    and metrics["clip_p_at_r95"] >= baseline["clip_p_at_r95"]
                ),
            }
        )
    eligible = [row for row in rows if row["passes_working_point_guards"]]
    provisional = max(
        eligible,
        key=lambda row: (
            row["metrics"]["clip_map_percent"],
            row["metrics"]["clip_p_at_r95"],
            row["metrics"]["clip_p_at_r90"],
        ),
    )
    tied = []
    for offset, row in enumerate(eligible):
        gap = (
            provisional["metrics"]["clip_map_percent"]
            - row["metrics"]["clip_map_percent"]
        )
        standard_error = _paired_map_standard_error(
            clip_ids,
            labels,
            scores[provisional["alpha"]],
            scores[row["alpha"]],
            replicates=bootstrap_replicates,
            seed=seed + offset,
        )
        row["map_gap_from_best"] = gap
        row["paired_map_gap_standard_error"] = standard_error
        row["within_one_standard_error"] = gap <= standard_error + 1e-12
        if row["within_one_standard_error"]:
            tied.append(row)
    selected = min(tied, key=lambda row: row["alpha"])["alpha"]
    serializable = [
        {
            **row,
            "alpha": f"{row['alpha'].numerator}/{row['alpha'].denominator}",
        }
        for row in rows
    ]
    return selected, serializable


def _delta(base: dict[str, float], candidate: dict[str, float]) -> dict[str, float]:
    return {name: candidate[name] - base[name] for name in METRICS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--outer-root", action="append", required=True)
    parser.add_argument("--member-report", type=Path, required=True)
    parser.add_argument("--dense-oof", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260831)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("residual shrinkage nested 输出已存在，拒绝覆盖")
    lock = json.loads(args.protocol_lock.read_text(encoding="utf-8"))
    if (
        lock.get("protocol") != "edgefall_rmpts_residual_shrinkage_nested_v1"
        or lock.get("status") != "locked_before_evaluation"
        or lock.get("alpha_grid") != ["0/1", "1/8", "1/4", "1/2", "1/1"]
    ):
        raise ValueError("residual shrinkage nested protocol lock 无效")
    for path_text, expected in lock.get("artifact_sha256", {}).items():
        path = Path(path_text)
        if not path.is_file() or _sha256_file(path) != expected:
            raise ValueError(f"residual shrinkage artifact hash 不匹配: {path}")
    roots = parse_outer_roots(args.outer_root)
    with np.load(args.dense_oof, allow_pickle=False) as payload:
        required = {"clip_id", "label", "fold", "dense48_score"}
        if not required.issubset(payload.files):
            raise ValueError("residual shrinkage dense OOF 字段缺失")
        outer_ids = np.asarray(payload["clip_id"]).astype(str)
        outer_labels = np.asarray(payload["label"], dtype=np.uint8)
        outer_folds = np.asarray(payload["fold"], dtype=np.int64)
        outer_dense = np.asarray(payload["dense48_score"], dtype=np.float64)
    outer_short_logits, _ = load_member_matrix(
        args.member_report, outer_ids, outer_labels
    )
    outer_short = 1.0 / (1.0 + np.exp(-np.clip(outer_short_logits, -30.0, 30.0)))
    outer_scores = candidate_scores(outer_short, outer_dense)
    incumbent = outer_scores[Fraction(0, 1)]
    nested_scores = np.full(outer_labels.size, np.nan, dtype=np.float64)
    selections = []
    selected_alphas = []
    outer_deltas = []
    for outer_fold in range(5):
        parts = [
            load_nested_inner(
                roots[outer_fold], outer_fold=outer_fold, inner_fold=inner_fold
            )
            for inner_fold in range(5)
            if inner_fold != outer_fold
        ]
        inner_ids = np.concatenate([part[0] for part in parts])
        inner_labels = np.concatenate([part[1] for part in parts])
        inner_short = np.concatenate([part[2] for part in parts])
        inner_dense = np.concatenate([part[3] for part in parts])
        if len(set(inner_ids.tolist())) != inner_ids.size:
            raise ValueError("residual shrinkage inner folds clip 重复")
        selected, candidates = select_alpha(
            inner_ids,
            inner_labels,
            candidate_scores(inner_short, inner_dense),
            bootstrap_replicates=args.bootstrap_replicates,
            seed=args.bootstrap_seed + outer_fold * 100,
        )
        held = outer_folds == outer_fold
        nested_scores[held] = outer_scores[selected][held]
        base_metrics = metrics_from_arrays(outer_labels[held], incumbent[held])
        selected_metrics = metrics_from_arrays(outer_labels[held], nested_scores[held])
        outer_deltas.append(_delta(base_metrics, selected_metrics))
        selected_alphas.append(selected)
        selections.append(
            {
                "outer_fold": outer_fold,
                "selected_alpha": f"{selected.numerator}/{selected.denominator}",
                "inner_candidates": candidates,
                "outer_delta": outer_deltas[-1],
            }
        )
    if not np.isfinite(nested_scores).all():
        raise ValueError("residual shrinkage nested OOF 覆盖不完整")
    counts = Counter(f"{alpha.numerator}/{alpha.denominator}" for alpha in selected_alphas)
    consensus_name, consensus_count = counts.most_common(1)[0]
    consensus = consensus_count >= 4 and consensus_name != "0/1"
    base_metrics = metrics_from_arrays(outer_labels, incumbent)
    nested_metrics = metrics_from_arrays(outer_labels, nested_scores)
    delta = _delta(base_metrics, nested_metrics)
    bootstrap = paired_group_bootstrap(
        outer_ids,
        outer_labels,
        incumbent,
        nested_scores,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    gates = {
        "exact_nonzero_four_of_five_consensus": consensus,
        "global_map_improves": delta["clip_map_percent"] > 0.0,
        "global_p90_nonnegative": delta["clip_p_at_r90"] >= 0.0,
        "global_p95_nonnegative": delta["clip_p_at_r95"] >= 0.0,
        "map_bootstrap_lower_positive": bootstrap["clip_map_percent"][0] > 0.0,
        "at_least_four_outer_map_nonnegative": sum(
            row["clip_map_percent"] >= 0.0 for row in outer_deltas
        )
        >= 4,
        "no_outer_map_regression_over_half_point": min(
            row["clip_map_percent"] for row in outer_deltas
        )
        >= -0.5,
    }
    report = {
        "protocol": "edgefall_rmpts_residual_shrinkage_nested_v1",
        "qualification": "training_side_nested_candidate_only",
        "anchor": "seven_member_equal_logit",
        "target": "uncalibrated_probability_rmpts_delta_1_7",
        "blend_domain": "logit_residual",
        "alpha_grid": [f"{alpha.numerator}/{alpha.denominator}" for alpha in ALPHAS],
        "selection_counts": dict(sorted(counts.items())),
        "selections": selections,
        "incumbent_metrics": base_metrics,
        "nested_metrics": nested_metrics,
        "delta": delta,
        "paired_group_bootstrap_delta_95ci": bootstrap,
        "promotion_gates": gates,
        "passes_all_promotion_gates": all(gates.values()),
        "decision": (
            f"research_candidate_alpha_{consensus_name}"
            if all(gates.values())
            else "retain_equal_logit"
        ),
        "inputs": {
            "protocol_lock_sha256": _sha256_file(args.protocol_lock),
            "member_report_sha256": _sha256_file(args.member_report),
            "dense_oof_sha256": _sha256_file(args.dense_oof),
        },
        "test_accessed": False,
        "limitations": [
            "this reuses retrospective nested predictions and is not a new blind estimate",
            "GMDCSA24 and CAUCAFall predictions are excluded from fitting and selection",
            "even a passing result would require a future untouched external dataset before deployment",
        ],
    }
    _atomic_json(args.output, report)
    print(json.dumps({"output": str(args.output), "decision": report["decision"]}))


if __name__ == "__main__":
    main()
