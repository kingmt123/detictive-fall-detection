"""Nested LOSO audit of a compact PCA backbone-representation residual adapter."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from tools.audit_paired_clip_predictions import metrics_from_arrays
from tools.train_tcn import _atomic_json, _sha256_file

CANDIDATES = (
    "anchor",
    "pca:4:ridge:1",
    "pca:8:ridge:1",
    "pca:8:ridge:0.3",
)
METRICS = ("clip_map_percent", "clip_p_at_r90", "clip_p_at_r95")


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))


def fit_pca_projection(
    train: np.ndarray, test: np.ndarray, *, components: int
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Fit standardization and PCA strictly on train, then transform train/test."""
    train = np.asarray(train, dtype=np.float64)
    test = np.asarray(test, dtype=np.float64)
    if (
        train.ndim != 2
        or test.ndim != 2
        or train.shape[1] != test.shape[1]
        or components <= 0
        or components > min(train.shape)
        or not np.isfinite(train).all()
        or not np.isfinite(test).all()
    ):
        raise ValueError("PCA residual adapter 输入无效")
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale = np.where(scale >= 1e-6, scale, 1.0)
    standardized_train = (train - mean) / scale
    standardized_test = (test - mean) / scale
    _, _, right = np.linalg.svd(standardized_train, full_matrices=False)
    basis = right[:components]
    projected_train = standardized_train @ basis.T
    projected_test = standardized_test @ basis.T
    projected_scale = projected_train.std(axis=0)
    projected_scale = np.where(projected_scale >= 1e-6, projected_scale, 1.0)
    return (
        projected_train / projected_scale,
        projected_test / projected_scale,
        {
            "mean": mean,
            "scale": scale,
            "basis": basis,
            "projected_scale": projected_scale,
        },
    )


def fit_anchored_ridge(
    features: np.ndarray,
    anchor_logits: np.ndarray,
    labels: np.ndarray,
    *,
    ridge: float,
    max_iterations: int = 100,
) -> np.ndarray:
    """Fit a no-intercept residual logistic head around frozen anchor logits."""
    features = np.asarray(features, dtype=np.float64)
    anchor = np.asarray(anchor_logits, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.float64)
    if (
        features.ndim != 2
        or anchor.shape != (features.shape[0],)
        or targets.shape != anchor.shape
        or ridge <= 0.0
        or not np.isfinite(features).all()
        or not np.isfinite(anchor).all()
        or not np.isin(targets, (0, 1)).all()
    ):
        raise ValueError("anchored ridge 输入无效")
    beta = np.zeros(features.shape[1], dtype=np.float64)

    def objective(values: np.ndarray) -> float:
        logits = anchor + features @ values
        bce = np.mean(np.logaddexp(0.0, logits) - targets * logits)
        return float(bce + 0.5 * ridge * np.dot(values, values))

    for _ in range(max_iterations):
        logits = anchor + features @ beta
        probability = _sigmoid(logits)
        gradient = features.T @ (probability - targets) / targets.size + ridge * beta
        curvature = probability * (1.0 - probability)
        hessian = (
            (features.T * curvature) @ features / targets.size
            + ridge * np.eye(features.shape[1])
        )
        step = np.linalg.solve(hessian, gradient)
        if np.linalg.norm(step) <= 1e-10:
            break
        current = objective(beta)
        scale = 1.0
        while scale >= 1e-8:
            candidate = beta - scale * step
            if objective(candidate) <= current:
                beta = candidate
                break
            scale *= 0.5
        else:
            break
    return beta


def _parse_candidate(candidate: str) -> tuple[int, float]:
    _, components, _, ridge = candidate.split(":")
    return int(components), float(ridge)


def _adapter_scores(
    train_features: np.ndarray,
    test_features: np.ndarray,
    train_anchor: np.ndarray,
    test_anchor: np.ndarray,
    train_labels: np.ndarray,
    candidate: str,
) -> tuple[np.ndarray, dict[str, np.ndarray] | None]:
    if candidate == "anchor":
        return _sigmoid(test_anchor), None
    components, ridge = _parse_candidate(candidate)
    projected_train, projected_test, projection = fit_pca_projection(
        train_features, test_features, components=components
    )
    beta = fit_anchored_ridge(
        projected_train, train_anchor, train_labels, ridge=ridge
    )
    projection["beta"] = beta
    return _sigmoid(test_anchor + projected_test @ beta), projection


def inner_loso_scores(
    features: np.ndarray,
    anchor: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
) -> dict[str, np.ndarray]:
    result = {candidate: np.full(labels.size, np.nan) for candidate in CANDIDATES}
    for held_group in sorted(set(groups.tolist())):
        held = groups == held_group
        train = ~held
        for candidate in CANDIDATES:
            result[candidate][held], _ = _adapter_scores(
                features[train],
                features[held],
                anchor[train],
                anchor[held],
                labels[train],
                candidate,
            )
    if any(not np.isfinite(values).all() for values in result.values()):
        raise ValueError("PCA residual inner LOSO 覆盖不完整")
    return result


def _group_bootstrap_deltas(
    labels: np.ndarray,
    baseline: np.ndarray,
    challenger: np.ndarray,
    groups: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> np.ndarray:
    group_rows = [
        np.flatnonzero(groups == group) for group in sorted(set(groups.tolist()))
    ]
    rng = np.random.default_rng(seed)
    deltas = np.empty((replicates, len(METRICS)), dtype=np.float64)
    for replicate in range(replicates):
        indices = np.concatenate(
            [group_rows[index] for index in rng.integers(0, len(group_rows), len(group_rows))]
        )
        base = metrics_from_arrays(labels[indices], baseline[indices])
        candidate = metrics_from_arrays(labels[indices], challenger[indices])
        deltas[replicate] = [candidate[name] - base[name] for name in METRICS]
    return deltas


def select_candidate(
    labels: np.ndarray,
    groups: np.ndarray,
    scores: dict[str, np.ndarray],
    *,
    bootstrap_replicates: int,
    seed: int,
) -> tuple[str, list[dict[str, Any]]]:
    if tuple(scores) != CANDIDATES:
        raise ValueError("PCA residual candidate grid 不匹配")
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
        deltas = _group_bootstrap_deltas(
            labels,
            scores[row["candidate"]],
            scores[provisional["candidate"]],
            groups,
            replicates=bootstrap_replicates,
            seed=seed + offset,
        )[:, 0]
        gap = provisional["metrics"]["clip_map_percent"] - row["metrics"][
            "clip_map_percent"
        ]
        standard_error = float(deltas.std(ddof=1))
        row["map_gap_from_best"] = gap
        row["paired_map_gap_standard_error"] = standard_error
        row["within_one_standard_error"] = gap <= standard_error + 1e-12
        if row["within_one_standard_error"]:
            tied.append(row)
    priority = {name: index for index, name in enumerate(CANDIDATES)}
    selected = min(tied, key=lambda row: priority[row["candidate"]])["candidate"]
    return selected, rows


def _metric_delta(
    baseline: dict[str, float], candidate: dict[str, float]
) -> dict[str, float]:
    return {name: candidate[name] - baseline[name] for name in METRICS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260831)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("PCA residual adapter 输出已存在，拒绝覆盖")
    if args.bootstrap_replicates < 100:
        raise ValueError("PCA residual adapter bootstrap 次数过少")
    lock = json.loads(args.protocol_lock.read_text(encoding="utf-8"))
    if (
        lock.get("protocol") != "caucafall_pca_residual_adapter_nested_v1"
        or lock.get("status") != "locked_before_development_evaluation"
        or lock.get("candidates") != list(CANDIDATES)
    ):
        raise ValueError("PCA residual adapter protocol lock 无效")
    for path_text, expected in lock.get("artifact_sha256", {}).items():
        path = Path(path_text)
        if not path.is_file() or _sha256_file(path) != expected:
            raise ValueError(f"PCA residual adapter artifact hash 不匹配: {path}")
    with np.load(args.embeddings, allow_pickle=False) as payload:
        required = {
            "clip_id",
            "label",
            "group_id",
            "member_logits",
            "short_embedding",
            "dense_embedding",
        }
        if not required.issubset(payload.files):
            raise ValueError("PCA residual adapter embedding 字段缺失")
        clip_ids = np.asarray(payload["clip_id"]).astype(str)
        labels = np.asarray(payload["label"], dtype=np.uint8)
        groups = np.asarray(payload["group_id"]).astype(str)
        logits = np.asarray(payload["member_logits"], dtype=np.float64)
        features = np.concatenate(
            [payload["short_embedding"], payload["dense_embedding"]], axis=1
        ).astype(np.float64)
    if (
        clip_ids.shape != (100,)
        or logits.shape != (100, 7)
        or features.shape != (100, 768)
        or labels.sum() != 50
        or len(set(groups.tolist())) != 10
        or not np.isfinite(features).all()
    ):
        raise ValueError("PCA residual adapter CAUCAFall identity 无效")
    anchor_logits = logits.mean(axis=1)
    incumbent = _sigmoid(anchor_logits)
    crossfit = np.full(labels.size, np.nan)
    selections = []
    outer_deltas = []
    models = []
    for outer_index, held_group in enumerate(sorted(set(groups.tolist()))):
        held = groups == held_group
        train = ~held
        inner_scores = inner_loso_scores(
            features[train], anchor_logits[train], labels[train], groups[train]
        )
        selected, candidates = select_candidate(
            labels[train],
            groups[train],
            inner_scores,
            bootstrap_replicates=args.bootstrap_replicates,
            seed=args.bootstrap_seed + outer_index * 100,
        )
        crossfit[held], fitted = _adapter_scores(
            features[train],
            features[held],
            anchor_logits[train],
            anchor_logits[held],
            labels[train],
            selected,
        )
        baseline_metrics = metrics_from_arrays(labels[held], incumbent[held])
        candidate_metrics = metrics_from_arrays(labels[held], crossfit[held])
        outer_delta = _metric_delta(baseline_metrics, candidate_metrics)
        outer_deltas.append(outer_delta)
        selections.append(
            {
                "held_subject": held_group,
                "selected_candidate": selected,
                "inner_candidates": candidates,
                "held_delta": outer_delta,
            }
        )
        models.append(
            {
                "held_subject": held_group,
                "selected_candidate": selected,
                "beta": None if fitted is None else fitted["beta"].tolist(),
            }
        )
    if not np.isfinite(crossfit).all():
        raise ValueError("PCA residual adapter outer LOSO 覆盖不完整")
    counts = Counter(row["selected_candidate"] for row in selections)
    consensus_name, consensus_count = counts.most_common(1)[0]
    consensus = consensus_name != "anchor" and consensus_count >= 8
    baseline_metrics = metrics_from_arrays(labels, incumbent)
    candidate_metrics = metrics_from_arrays(labels, crossfit)
    delta = _metric_delta(baseline_metrics, candidate_metrics)
    bootstrap_deltas = _group_bootstrap_deltas(
        labels,
        incumbent,
        crossfit,
        groups,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    bootstrap = {
        name: [
            float(np.quantile(bootstrap_deltas[:, index], 0.025)),
            float(np.quantile(bootstrap_deltas[:, index], 0.975)),
        ]
        for index, name in enumerate(METRICS)
    }
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
    if consensus and all(gates.values()):
        _, fitted = _adapter_scores(
            features,
            features,
            anchor_logits,
            anchor_logits,
            labels,
            consensus_name,
        )
        assert fitted is not None
        final_model = {
            "candidate": consensus_name,
            "mean": fitted["mean"].tolist(),
            "scale": fitted["scale"].tolist(),
            "basis": fitted["basis"].tolist(),
            "projected_scale": fitted["projected_scale"].tolist(),
            "beta": fitted["beta"].tolist(),
        }
    report = {
        "protocol": "caucafall_pca_residual_adapter_nested_v1",
        "qualification": "development_nested_loso_only",
        "architecture": "equal_logit_anchor_plus_joint_short_dense_pca_residual",
        "pca_fitted_inside_every_inner_and_outer_training_split": True,
        "candidates": list(CANDIDATES),
        "selection_counts": dict(sorted(counts.items())),
        "selections": selections,
        "outer_models": models,
        "incumbent_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "delta": delta,
        "paired_subject_bootstrap_delta_95ci": bootstrap,
        "promotion_gates": gates,
        "passes_all_development_gates": all(gates.values()),
        "decision": (
            "future_external_candidate" if all(gates.values()) else "stop_pca_residual_adapter"
        ),
        "final_model": final_model,
        "inputs": {
            "protocol_lock_sha256": _sha256_file(args.protocol_lock),
            "embeddings_sha256": _sha256_file(args.embeddings),
        },
        "test_accessed": False,
        "limitations": [
            "CAUCAFall is development data and cannot provide external confirmation",
            "GMDCSA24 and all reserved tests are excluded",
            "a passing result still requires a new untouched external dataset",
        ],
    }
    _atomic_json(args.output, report)
    print(json.dumps({"output": str(args.output), "decision": report["decision"]}))


if __name__ == "__main__":
    main()
