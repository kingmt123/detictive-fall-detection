"""Nested LOSO development audit for a zero-sum residual fusion head."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from tools.audit_paired_clip_predictions import (
    metrics_from_arrays,
    paired_group_bootstrap,
)
from tools.evaluate_rmpts_crossfit import _paired_map_standard_error
from tools.train_tcn import _atomic_json, _sha256_file

RIDGE_GRID = (0.03, 0.1, 0.3, 1.0)
CANDIDATES = ("anchor", "ridge:1", "ridge:0.3", "ridge:0.1", "ridge:0.03")
METRICS = ("clip_map_percent", "clip_p_at_r90", "clip_p_at_r95")


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))


def fit_zero_sum_ridge(
    member_logits: np.ndarray,
    labels: np.ndarray,
    *,
    ridge: float,
    max_iterations: int = 100,
) -> np.ndarray:
    logits = np.asarray(member_logits, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[1] != 7 or targets.shape != (logits.shape[0],):
        raise ValueError("zero-sum head 输入 shape 无效")
    if ridge <= 0.0 or not np.isfinite(logits).all() or not np.isin(targets, (0, 1)).all():
        raise ValueError("zero-sum head 输入值无效")
    anchor = logits.mean(axis=1)
    disagreement = logits - anchor[:, None]
    beta = np.zeros(7, dtype=np.float64)

    def objective(values: np.ndarray) -> float:
        fused = anchor + disagreement @ values
        bce = np.mean(np.logaddexp(0.0, fused) - targets * fused)
        return float(bce + 0.5 * ridge * np.dot(values, values))

    for _ in range(max_iterations):
        fused = anchor + disagreement @ beta
        probability = _sigmoid(fused)
        gradient = disagreement.T @ (probability - targets) / targets.size + ridge * beta
        curvature = probability * (1.0 - probability)
        hessian = (
            (disagreement.T * curvature) @ disagreement / targets.size
            + ridge * np.eye(7)
        )
        step = np.linalg.solve(hessian, gradient)
        if np.linalg.norm(step) <= 1e-10:
            break
        current = objective(beta)
        scale = 1.0
        while scale >= 1e-8:
            candidate = beta - scale * step
            candidate -= candidate.mean()
            if objective(candidate) <= current:
                beta = candidate
                break
            scale *= 0.5
        else:
            break
    beta -= beta.mean()
    return beta


def zero_sum_scores(member_logits: np.ndarray, beta: np.ndarray) -> np.ndarray:
    logits = np.asarray(member_logits, dtype=np.float64)
    values = np.asarray(beta, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[1] != 7 or values.shape != (7,):
        raise ValueError("zero-sum score shape 无效")
    if abs(values.sum()) > 1e-8:
        raise ValueError("zero-sum residual weights 之和必须为零")
    anchor = logits.mean(axis=1)
    return _sigmoid(anchor + (logits - anchor[:, None]) @ values)


def _ridge_name(ridge: float) -> str:
    return f"ridge:{ridge:g}"


def inner_loso_scores(
    logits: np.ndarray, labels: np.ndarray, groups: np.ndarray
) -> dict[str, np.ndarray]:
    result = {candidate: np.full(labels.size, np.nan) for candidate in CANDIDATES}
    result["anchor"] = _sigmoid(logits.mean(axis=1))
    for held_group in sorted(set(groups.tolist())):
        held = groups == held_group
        train = ~held
        for ridge in RIDGE_GRID:
            beta = fit_zero_sum_ridge(logits[train], labels[train], ridge=ridge)
            result[_ridge_name(ridge)][held] = zero_sum_scores(logits[held], beta)
    if any(not np.isfinite(values).all() for values in result.values()):
        raise ValueError("zero-sum inner LOSO 覆盖不完整")
    return result


def select_candidate(
    clip_ids: np.ndarray,
    labels: np.ndarray,
    scores: dict[str, np.ndarray],
    *,
    bootstrap_replicates: int,
    seed: int,
) -> tuple[str, list[dict[str, Any]]]:
    if tuple(scores) != CANDIDATES:
        raise ValueError("zero-sum candidate grid 不匹配")
    baseline = metrics_from_arrays(labels, scores["anchor"])
    rows = []
    for name, values in scores.items():
        metrics = metrics_from_arrays(labels, values)
        rows.append(
            {
                "candidate": name,
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
        gap = provisional["metrics"]["clip_map_percent"] - row["metrics"]["clip_map_percent"]
        standard_error = _paired_map_standard_error(
            clip_ids,
            labels,
            scores[provisional["candidate"]],
            scores[row["candidate"]],
            replicates=bootstrap_replicates,
            seed=seed + offset,
        )
        row["map_gap_from_best"] = gap
        row["paired_map_gap_standard_error"] = standard_error
        row["within_one_standard_error"] = gap <= standard_error + 1e-12
        if row["within_one_standard_error"]:
            tied.append(row)
    priority = {name: index for index, name in enumerate(CANDIDATES)}
    selected = min(tied, key=lambda row: priority[row["candidate"]])["candidate"]
    return selected, rows


def _delta(base: dict[str, float], candidate: dict[str, float]) -> dict[str, float]:
    return {name: candidate[name] - base[name] for name in METRICS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260831)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("zero-sum head 输出已存在，拒绝覆盖")
    lock = json.loads(args.protocol_lock.read_text(encoding="utf-8"))
    if (
        lock.get("protocol") != "caucafall_subject_robust_zero_sum_head_v1"
        or lock.get("status") != "locked_before_development_evaluation"
        or lock.get("candidates") != list(CANDIDATES)
    ):
        raise ValueError("zero-sum head protocol lock 无效")
    for path_text, expected in lock.get("artifact_sha256", {}).items():
        path = Path(path_text)
        if not path.is_file() or _sha256_file(path) != expected:
            raise ValueError(f"zero-sum head artifact hash 不匹配: {path}")
    with np.load(args.predictions, allow_pickle=False) as payload:
        required = {"clip_id", "label", "group_id", "member_logits", "incumbent_score"}
        if not required.issubset(payload.files):
            raise ValueError("zero-sum head prediction 字段缺失")
        clip_ids = np.asarray(payload["clip_id"]).astype(str)
        labels = np.asarray(payload["label"], dtype=np.uint8)
        groups = np.asarray(payload["group_id"]).astype(str)
        logits = np.asarray(payload["member_logits"], dtype=np.float64)
        incumbent = np.asarray(payload["incumbent_score"], dtype=np.float64)
    if (
        logits.shape != (100, 7)
        or labels.sum() != 50
        or len(set(groups.tolist())) != 10
        or not np.allclose(incumbent, _sigmoid(logits.mean(axis=1)), atol=1e-6)
    ):
        raise ValueError("zero-sum head CAUCAFall identity 无效")

    crossfit = np.full(labels.size, np.nan)
    selections = []
    weight_rows = []
    outer_deltas = []
    for outer_index, held_group in enumerate(sorted(set(groups.tolist()))):
        held = groups == held_group
        train = ~held
        inner_scores = inner_loso_scores(logits[train], labels[train], groups[train])
        selected, candidates = select_candidate(
            clip_ids[train],
            labels[train],
            inner_scores,
            bootstrap_replicates=args.bootstrap_replicates,
            seed=args.bootstrap_seed + outer_index * 100,
        )
        if selected == "anchor":
            beta = np.zeros(7, dtype=np.float64)
        else:
            ridge = float(selected.partition(":")[2])
            beta = fit_zero_sum_ridge(logits[train], labels[train], ridge=ridge)
        crossfit[held] = zero_sum_scores(logits[held], beta)
        base_metrics = metrics_from_arrays(labels[held], incumbent[held])
        candidate_metrics = metrics_from_arrays(labels[held], crossfit[held])
        outer_deltas.append(_delta(base_metrics, candidate_metrics))
        selections.append(
            {
                "held_subject": held_group,
                "selected_candidate": selected,
                "inner_candidates": candidates,
                "held_delta": outer_deltas[-1],
            }
        )
        weight_rows.append(
            {
                "held_subject": held_group,
                "residual_beta": beta.tolist(),
                "final_weights": (np.full(7, 1.0 / 7.0) + beta).tolist(),
            }
        )
    if not np.isfinite(crossfit).all():
        raise ValueError("zero-sum outer LOSO 覆盖不完整")
    counts = Counter(row["selected_candidate"] for row in selections)
    consensus_name, consensus_count = counts.most_common(1)[0]
    consensus = consensus_name != "anchor" and consensus_count >= 8
    base_metrics = metrics_from_arrays(labels, incumbent)
    candidate_metrics = metrics_from_arrays(labels, crossfit)
    delta = _delta(base_metrics, candidate_metrics)
    bootstrap = paired_group_bootstrap(
        clip_ids,
        labels,
        incumbent,
        crossfit,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    subject_map = [row["clip_map_percent"] for row in outer_deltas]
    gates = {
        "exact_nonanchor_eight_of_ten_consensus": consensus,
        "global_map_improves": delta["clip_map_percent"] > 0.0,
        "global_p90_nonnegative": delta["clip_p_at_r90"] >= 0.0,
        "global_p95_nonnegative": delta["clip_p_at_r95"] >= 0.0,
        "map_bootstrap_lower_positive": bootstrap["clip_map_percent"][0] > 0.0,
        "at_least_eight_subject_map_nonnegative": sum(value >= 0 for value in subject_map)
        >= 8,
        "no_subject_map_regression_over_two_points": min(subject_map) >= -2.0,
    }
    final_model = None
    if consensus:
        ridge = float(consensus_name.partition(":")[2])
        beta = fit_zero_sum_ridge(logits, labels, ridge=ridge)
        final_model = {
            "ridge": ridge,
            "residual_beta": beta.tolist(),
            "final_weights": (np.full(7, 1.0 / 7.0) + beta).tolist(),
        }
    report = {
        "protocol": "caucafall_subject_robust_zero_sum_head_v1",
        "qualification": "development_nested_loso_only",
        "data_role": "CAUCAFall_reclassified_as_development_after_frozen_external_audit",
        "architecture": "equal_logit_anchor_plus_zero_sum_member_disagreement_residual",
        "candidates": list(CANDIDATES),
        "selection_counts": dict(sorted(counts.items())),
        "selections": selections,
        "weights": weight_rows,
        "incumbent_metrics": base_metrics,
        "candidate_metrics": candidate_metrics,
        "delta": delta,
        "paired_subject_bootstrap_delta_95ci": bootstrap,
        "promotion_gates": gates,
        "passes_all_development_gates": all(gates.values()),
        "decision": "future_external_candidate" if all(gates.values()) else "stop_zero_sum_head",
        "final_model": final_model,
        "inputs": {
            "protocol_lock_sha256": _sha256_file(args.protocol_lock),
            "predictions_sha256": _sha256_file(args.predictions),
        },
        "test_accessed": False,
        "limitations": [
            "CAUCAFall is now development data and cannot provide further external confirmation",
            "GMDCSA24 and all reserved tests are excluded",
            "a passing development result still requires a new untouched external dataset",
        ],
    }
    _atomic_json(args.output, report)
    print(json.dumps({"output": str(args.output), "decision": report["decision"]}))


if __name__ == "__main__":
    main()
