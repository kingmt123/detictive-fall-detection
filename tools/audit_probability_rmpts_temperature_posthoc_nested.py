"""Post-hoc nested audit for fixed Probability-RMPTS plus shared temperature.

This audit is intentionally separate from the conditionally locked confirmation
tool.  It is used only after that tool's activation gate failed, so its result
must never be described as preregistered or blind confirmation.
"""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from models.edgefall_ensemble import rmpts_fuse_numpy_logits
from tools.audit_paired_clip_predictions import (
    metrics_from_arrays,
    paired_group_bootstrap,
)
from tools.evaluate_probability_rmpts_calibration_crossfit import (
    calibrated_probability_rmpts,
)
from tools.evaluate_probability_rmpts_temperature_nested import (
    fit_inner_global_temperature,
    metric_delta,
    promotion_gates,
)
from tools.evaluate_rmpts_diagnostic import _logit, load_member_matrix
from tools.evaluate_rmpts_nested import load_nested_inner, parse_outer_roots
from tools.train_tcn import _atomic_json, _sha256_file

FIXED_DELTA = Fraction(1, 7)
INCUMBENT_DELTA = Fraction(0, 1)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))


def _comparison(
    clip_ids: np.ndarray,
    labels: np.ndarray,
    baseline_scores: np.ndarray,
    challenger_scores: np.ndarray,
    fold_deltas: list[dict[str, float]],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    baseline = metrics_from_arrays(labels, baseline_scores)
    challenger = metrics_from_arrays(labels, challenger_scores)
    bootstrap = paired_group_bootstrap(
        clip_ids,
        labels,
        baseline_scores,
        challenger_scores,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    gates = promotion_gates(baseline, challenger, fold_deltas, bootstrap)
    return {
        "baseline_metrics": baseline,
        "challenger_metrics": challenger,
        "delta": metric_delta(baseline, challenger),
        "paired_group_bootstrap_delta_95ci": bootstrap,
        "promotion_gates": gates,
        "passes_all_promotion_gates": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outer-root", action="append", required=True)
    parser.add_argument("--member-report", type=Path, required=True)
    parser.add_argument("--dense-oof", type=Path, required=True)
    parser.add_argument("--primary-result", type=Path, required=True)
    parser.add_argument("--audit-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260829)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("post-hoc nested audit 输出已存在，拒绝覆盖")
    if args.bootstrap_replicates < 100:
        raise ValueError("post-hoc nested audit bootstrap 至少需要 100 次")
    primary = json.loads(args.primary_result.read_text(encoding="utf-8"))
    if primary.get("deployment_candidate_if_confirmed") == "probability:1/7":
        raise ValueError("主激活门已通过，应使用条件确认工具而非 post-hoc audit")
    audit_lock = json.loads(args.audit_lock.read_text(encoding="utf-8"))
    if audit_lock.get("qualification") != "posthoc_retrospective_nested_audit":
        raise ValueError("post-hoc nested audit lock 签名无效")
    locked_artifacts = audit_lock.get("artifact_sha256")
    if not isinstance(locked_artifacts, dict) or not locked_artifacts:
        raise ValueError("post-hoc nested audit lock 缺少 artifact hashes")
    for path_text, expected_hash in locked_artifacts.items():
        path = Path(path_text)
        if not path.is_file() or _sha256_file(path) != expected_hash:
            raise ValueError(f"post-hoc nested audit artifact hash 不匹配: {path}")

    roots = parse_outer_roots(args.outer_root)
    with np.load(args.dense_oof, allow_pickle=False) as payload:
        required = {"clip_id", "label", "fold", "dense48_score"}
        if not required.issubset(payload.files):
            raise ValueError("dense outer OOF 缺少 post-hoc nested audit 字段")
        outer_ids = np.asarray(payload["clip_id"]).astype(str)
        outer_labels = np.asarray(payload["label"], dtype=np.uint8)
        outer_folds = np.asarray(payload["fold"], dtype=np.int64)
        outer_dense_probability = np.asarray(payload["dense48_score"], dtype=np.float64)
    outer_short_logits, member_evidence = load_member_matrix(
        args.member_report, outer_ids, outer_labels
    )
    outer_dense_logits = _logit(outer_dense_probability)
    incumbent_scores = _sigmoid(
        rmpts_fuse_numpy_logits(
            outer_short_logits[:, (0, 2, 4)],
            outer_short_logits[:, (1, 3, 5)],
            outer_dense_logits,
            delta=INCUMBENT_DELTA,
        )
    )
    probability_scores = calibrated_probability_rmpts(
        outer_short_logits,
        outer_dense_logits,
        np.ones(7, dtype=np.float64),
    )
    nested_scores = np.full(outer_labels.size, np.nan, dtype=np.float64)
    fold_rows: list[dict[str, Any]] = []
    fold_deltas_vs_incumbent: list[dict[str, float]] = []
    fold_deltas_vs_probability: list[dict[str, float]] = []
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
            raise ValueError("post-hoc nested audit inner folds 出现重复 clip")
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
        incumbent_metrics = metrics_from_arrays(outer_labels[held], incumbent_scores[held])
        probability_metrics = metrics_from_arrays(
            outer_labels[held], probability_scores[held]
        )
        challenger_metrics = metrics_from_arrays(outer_labels[held], held_scores)
        delta_incumbent = metric_delta(incumbent_metrics, challenger_metrics)
        delta_probability = metric_delta(probability_metrics, challenger_metrics)
        fold_deltas_vs_incumbent.append(delta_incumbent)
        fold_deltas_vs_probability.append(delta_probability)
        fold_rows.append(
            {
                "outer_fold": outer_fold,
                "temperature": temperature,
                "incumbent_metrics": incumbent_metrics,
                "uncalibrated_probability_rmpts_metrics": probability_metrics,
                "challenger_metrics": challenger_metrics,
                "delta_vs_incumbent": delta_incumbent,
                "delta_vs_uncalibrated_probability_rmpts": delta_probability,
                "inner_evidence": [part[4] for part in inner_parts],
            }
        )
        all_inner_labels.append(inner_labels)
        all_inner_short.append(inner_short)
        all_inner_dense.append(inner_dense)

    if not np.isfinite(nested_scores).all():
        raise AssertionError("post-hoc nested audit 未覆盖全部 outer OOF")
    versus_incumbent = _comparison(
        outer_ids,
        outer_labels,
        incumbent_scores,
        nested_scores,
        fold_deltas_vs_incumbent,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    versus_probability = _comparison(
        outer_ids,
        outer_labels,
        probability_scores,
        nested_scores,
        fold_deltas_vs_probability,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed + 1000,
    )
    passes = (
        versus_incumbent["passes_all_promotion_gates"]
        and versus_probability["passes_all_promotion_gates"]
    )
    final_temperature = fit_inner_global_temperature(
        np.concatenate(all_inner_labels),
        np.concatenate(all_inner_short),
        np.concatenate(all_inner_dense),
    )
    report = {
        "protocol": "edgefall_probability_rmpts_shared_temperature_posthoc_nested_audit_v1",
        "qualification": "posthoc_retrospective_nested_audit",
        "deployment_eligible": False,
        "fixed_delta": "1/7",
        "temperature_objective": "unweighted_binary_cross_entropy",
        "temperature_scope": "one_shared_temperature_for_all_seven_members",
        "comparison_vs_equal_logit_incumbent": versus_incumbent,
        "comparison_vs_uncalibrated_probability_rmpts": versus_probability,
        "folds": fold_rows,
        "passes_all_posthoc_audit_gates": passes,
        "final_temperature_for_future_preregistered_validation": final_temperature,
        "primary_result_sha256": _sha256_file(args.primary_result),
        "audit_lock_sha256": _sha256_file(args.audit_lock),
        "member_report_sha256": _sha256_file(args.member_report),
        "dense_oof_sha256": _sha256_file(args.dense_oof),
        "outer_member_evidence": member_evidence,
        "test_accessed": False,
        "limitations": [
            "the candidate and this audit were requested after historical OOF and primary nested results were visible",
            "this is post-hoc retrospective nested evidence, not preregistered or blind confirmation",
            "deployment eligibility remains false regardless of the numerical result",
            "sealed OF-Syn test and reserved URFD test remain inaccessible",
        ],
    }
    _atomic_json(args.output, report)
    print(json.dumps({"output": str(args.output.resolve()), "passes": passes}))


if __name__ == "__main__":
    main()
