"""Evaluate logit- and probability-domain RMPTS with nested grouped isolation.

The inner predictions for outer fold ``k`` must come from models that excluded
both ``k`` and the current inner fold.  The selected fusion rule is then applied
to the already frozen seven-member OOF predictions for outer fold ``k``.
"""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from models.edgefall_ensemble import (
    RMPTS_GRID,
    rmpts_fuse_numpy_logits,
    rmpts_fuse_numpy_probabilities,
)
from tools.audit_paired_clip_predictions import (
    metrics_from_arrays,
    paired_group_bootstrap,
)
from tools.evaluate_rmpts_crossfit import _paired_map_standard_error
from tools.evaluate_rmpts_diagnostic import _load_score, _logit, load_member_matrix
from tools.train_tcn import _atomic_json, _sha256_file

DOMAINS = ("logit", "probability")
INNER_MEMBER_DIRS = (
    "short320_seed1",
    "short640_seed1",
    "short320_headseed20260829",
    "short640_headseed20260829",
    "short320_headseed20260830",
    "short640_headseed20260830",
    "dense48_320",
)
Candidate = tuple[str, Fraction]


def candidate_name(candidate: Candidate) -> str:
    domain, delta = candidate
    return f"{domain}:{delta.numerator}/{delta.denominator}"


def fusion_candidate_scores(
    short_probabilities: np.ndarray, dense_probability: np.ndarray
) -> dict[Candidate, np.ndarray]:
    if short_probabilities.ndim != 2 or short_probabilities.shape[1] != 6:
        raise ValueError("nested RMPTS 要求六个短窗 probability")
    if dense_probability.shape != (short_probabilities.shape[0],):
        raise ValueError("nested RMPTS dense probability shape 不匹配")
    if not np.isfinite(short_probabilities).all() or not np.isfinite(
        dense_probability
    ).all():
        raise ValueError("nested RMPTS probability 必须有限")
    clipped_short = np.clip(short_probabilities, 1e-7, 1.0 - 1e-7)
    short_logits = np.log(clipped_short / (1.0 - clipped_short))
    dense_logits = _logit(dense_probability)
    result: dict[Candidate, np.ndarray] = {}
    for domain in DOMAINS:
        for delta in RMPTS_GRID:
            if domain == "logit":
                fused_logit = rmpts_fuse_numpy_logits(
                    short_logits[:, (0, 2, 4)],
                    short_logits[:, (1, 3, 5)],
                    dense_logits,
                    delta=delta,
                )
                result[(domain, delta)] = 1.0 / (
                    1.0 + np.exp(-np.clip(fused_logit, -30.0, 30.0))
                )
            else:
                result[(domain, delta)] = rmpts_fuse_numpy_probabilities(
                    short_probabilities[:, (0, 2, 4)],
                    short_probabilities[:, (1, 3, 5)],
                    dense_probability,
                    delta=delta,
                )
    return result


def select_nested_candidate(
    clip_ids: np.ndarray,
    labels: np.ndarray,
    scores: dict[Candidate, np.ndarray],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> tuple[Candidate, list[dict[str, Any]]]:
    expected = tuple((domain, delta) for domain in DOMAINS for delta in RMPTS_GRID)
    if tuple(scores) != expected:
        raise ValueError("nested RMPTS 候选网格或顺序不匹配")
    if bootstrap_replicates < 100:
        raise ValueError("nested RMPTS bootstrap 至少需要 100 次")
    incumbent: Candidate = ("logit", Fraction(0, 1))
    baseline = metrics_from_arrays(labels, scores[incumbent])
    rows = []
    for candidate, values in scores.items():
        metrics = metrics_from_arrays(labels, values)
        rows.append(
            {
                "candidate": candidate,
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
    best_scores = scores[provisional["candidate"]]
    tied = []
    for offset, row in enumerate(eligible):
        gap = (
            provisional["metrics"]["clip_map_percent"]
            - row["metrics"]["clip_map_percent"]
        )
        standard_error = _paired_map_standard_error(
            clip_ids,
            labels,
            best_scores,
            scores[row["candidate"]],
            replicates=bootstrap_replicates,
            seed=bootstrap_seed + offset,
        )
        row["map_gap_from_provisional_best"] = gap
        row["paired_map_gap_standard_error"] = standard_error
        row["within_one_standard_error"] = gap <= standard_error + 1e-12
        if row["within_one_standard_error"]:
            tied.append(row)
    selected = min(
        tied,
        key=lambda row: (
            row["candidate"][0] != "logit",
            abs(row["candidate"][1]),
            -row["metrics"]["clip_p_at_r95"],
            -row["metrics"]["clip_p_at_r90"],
        ),
    )["candidate"]
    serializable = []
    for row in rows:
        serializable.append(
            {
                **row,
                "candidate": candidate_name(row["candidate"]),
            }
        )
    return selected, serializable


def load_nested_inner(
    outer_root: Path, *, outer_fold: int, inner_fold: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, str]]]:
    base = outer_root / f"outer{outer_fold}" / f"inner{inner_fold}"
    reference_ids: np.ndarray | None = None
    reference_labels: np.ndarray | None = None
    short_columns = []
    dense = None
    evidence = []
    for offset, directory in enumerate(INNER_MEMBER_DIRS):
        member = base / directory
        summary_path = member / "summary.json"
        prediction_path = member / "oof_predictions.npz"
        if not summary_path.is_file() or not prediction_path.is_file():
            raise FileNotFoundError(f"nested inner 成员不完整: {member}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("test_accessed") is not False:
            raise ValueError("nested inner 成员 test_accessed 签名无效")
        if summary.get("config", {}).get("fold") != inner_fold:
            raise ValueError("nested inner 成员 fold 签名不匹配")
        excluded = summary.get("fold_assignment", {}).get(
            "nested_outer_excluded_folds"
        )
        if excluded != [outer_fold]:
            raise ValueError("nested inner 成员未严格排除 outer fold")
        clip_ids, labels, probabilities = _load_score(prediction_path)
        if reference_ids is None:
            reference_ids, reference_labels = clip_ids, labels
        elif not np.array_equal(clip_ids, reference_ids) or not np.array_equal(
            labels, reference_labels
        ):
            raise ValueError("nested inner 七成员 clip/label 不一致")
        if offset < 6:
            short_columns.append(probabilities)
        else:
            dense = probabilities
        evidence.append(
            {
                "member": directory,
                "summary_sha256": _sha256_file(summary_path),
                "predictions_sha256": _sha256_file(prediction_path),
            }
        )
    assert reference_ids is not None and reference_labels is not None and dense is not None
    return (
        reference_ids,
        reference_labels,
        np.stack(short_columns, axis=1),
        dense,
        evidence,
    )


def parse_outer_roots(values: list[str]) -> dict[int, Path]:
    roots = {}
    for value in values:
        fold_text, separator, path_text = value.partition("=")
        if not separator or not fold_text.isdigit() or not path_text:
            raise ValueError("--outer-root 必须为 FOLD=PATH")
        fold = int(fold_text)
        if not 0 <= fold < 5 or fold in roots:
            raise ValueError("--outer-root fold 必须位于 [0,4] 且唯一")
        roots[fold] = Path(path_text)
    if set(roots) != set(range(5)):
        raise ValueError("nested 评估必须提供全部五个 outer root")
    return roots


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outer-root", action="append", required=True)
    parser.add_argument("--member-report", type=Path, required=True)
    parser.add_argument("--dense-oof", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260829)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("nested RMPTS 输出已存在，拒绝覆盖")
    roots = parse_outer_roots(args.outer_root)
    with np.load(args.dense_oof, allow_pickle=False) as payload:
        required = {"clip_id", "label", "fold", "dense48_score"}
        if not required.issubset(payload.files):
            raise ValueError("dense outer OOF 缺少 nested RMPTS 字段")
        outer_ids = np.asarray(payload["clip_id"]).astype(str)
        outer_labels = np.asarray(payload["label"], dtype=np.uint8)
        outer_folds = np.asarray(payload["fold"], dtype=np.int64)
        outer_dense = np.asarray(payload["dense48_score"], dtype=np.float64)
    outer_short_logits, member_evidence = load_member_matrix(
        args.member_report, outer_ids, outer_labels
    )
    outer_short = 1.0 / (1.0 + np.exp(-np.clip(outer_short_logits, -30.0, 30.0)))
    outer_candidates = fusion_candidate_scores(outer_short, outer_dense)
    incumbent = outer_candidates[("logit", Fraction(0, 1))]
    nested_scores = np.full(outer_labels.size, np.nan, dtype=np.float64)
    selections = []
    selected_candidates = []
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
            raise ValueError("nested inner folds 出现重复 clip")
        selected, candidates = select_nested_candidate(
            inner_ids,
            inner_labels,
            fusion_candidate_scores(inner_short, inner_dense),
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed + outer_fold * 100,
        )
        heldout = outer_folds == outer_fold
        nested_scores[heldout] = outer_candidates[selected][heldout]
        selected_candidates.append(selected)
        selections.append(
            {
                "outer_fold": outer_fold,
                "selected_candidate": candidate_name(selected),
                "inner_candidates": candidates,
                "inner_evidence": [part[4] for part in inner_parts],
                "outer_metrics": metrics_from_arrays(
                    outer_labels[heldout], nested_scores[heldout]
                ),
                "outer_incumbent_metrics": metrics_from_arrays(
                    outer_labels[heldout], incumbent[heldout]
                ),
            }
        )
    if not np.isfinite(nested_scores).all():
        raise AssertionError("nested RMPTS 未覆盖全部 outer OOF")
    counts = {
        candidate: selected_candidates.count(candidate)
        for candidate in set(selected_candidates)
    }
    consensus = max(counts, key=counts.get)
    deployment_candidate = consensus if counts[consensus] >= 4 else ("logit", Fraction(0, 1))
    report = {
        "protocol": "edgefall_dual_domain_rmpts_retrospective_nested_v1",
        "qualification": "retrospective_nested_confirmation_not_blind_validation",
        "candidate_grid": [
            candidate_name((domain, delta)) for domain in DOMAINS for delta in RMPTS_GRID
        ],
        "selections": selections,
        "selection_counts": {candidate_name(key): value for key, value in counts.items()},
        "deployment_candidate_if_confirmed": candidate_name(deployment_candidate),
        "incumbent_metrics": metrics_from_arrays(outer_labels, incumbent),
        "nested_metrics": metrics_from_arrays(outer_labels, nested_scores),
        "paired_group_bootstrap_delta_95ci": paired_group_bootstrap(
            outer_ids,
            outer_labels,
            incumbent,
            nested_scores,
            replicates=args.bootstrap_replicates,
            seed=args.bootstrap_seed,
        ),
        "member_report_sha256": _sha256_file(args.member_report),
        "dense_oof_sha256": _sha256_file(args.dense_oof),
        "outer_member_evidence": member_evidence,
        "test_accessed": False,
        "limitations": [
            "probability-domain fusion was proposed after inspecting the historical OOF",
            "the nested result is retrospective confirmation, not a new blind estimate",
            "sealed OF-Syn test and reserved URFD test remain inaccessible",
        ],
    }
    _atomic_json(args.output, report)
    print(json.dumps({"output": str(args.output.resolve()), "selected": candidate_name(deployment_candidate)}))


if __name__ == "__main__":
    main()
