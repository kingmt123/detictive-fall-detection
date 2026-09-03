"""Convert rich GPT visual labels into an aligned review-confirmation draft."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from tools.audit_edgefall_oof_errors import FN_TAXONOMY, FP_TAXONOMY


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV 缺少表头: {path}")
        return list(reader.fieldnames), list(reader)


def prepare_draft(
    review_rows: list[dict[str, str]],
    visual_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Join exact clip identities and preserve the rich GPT visual taxonomy."""
    visual_by_clip = {row.get("clip_id", ""): row for row in visual_rows}
    if len(visual_by_clip) != len(visual_rows) or "" in visual_by_clip:
        raise ValueError("GPT visual labels 存在空或重复 clip_id")
    output = []
    for review in review_rows:
        clip_id = review.get("clip_id", "")
        if clip_id not in visual_by_clip:
            raise ValueError(f"GPT visual labels 缺少 clip: {clip_id}")
        visual = visual_by_clip[clip_id]
        review_set = review.get("review_set", "")
        expected_type = (
            "FP"
            if review_set == "high_score_false_positive"
            else "FN"
            if review_set == "low_score_false_negative"
            else ""
        )
        if not expected_type or visual.get("error_type") != expected_type:
            raise ValueError(f"GPT visual error type 与 review_set 不一致: {clip_id}")
        label = visual.get("primary_visual_class", "").strip()
        allowed = set(FP_TAXONOMY if expected_type == "FP" else FN_TAXONOMY)
        confidence = visual.get("confidence", "").strip()
        if label not in allowed or confidence not in {"high", "medium", "low"}:
            raise ValueError(f"GPT visual label/confidence 无效: {clip_id}")
        flags = visual.get("quality_flags", "none").strip() or "none"
        annotation_risk = visual.get("annotation_risk", "").strip()
        notes = visual.get("notes", "").strip()
        output.append(
            {
                "image": review.get("image", ""),
                "review_set": review_set,
                "clip_id": clip_id,
                "ai_label": label,
                "confidence": confidence,
                "ai_review_notes": (
                    f"GPT-5.6视觉初标；quality_flags={flags}；"
                    f"annotation_risk={annotation_risk}；{notes}"
                ),
            }
        )
    if len(output) != len(visual_rows):
        raise ValueError("review 与 GPT visual labels 行数不一致")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-labels", type=Path, required=True)
    parser.add_argument("--visual-labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"GPT review draft 已存在，拒绝覆盖: {args.output}")
    _, review_rows = _read_csv(args.review_labels)
    _, visual_rows = _read_csv(args.visual_labels)
    rows = prepare_draft(review_rows, visual_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"prepared {len(rows)} GPT visual review rows: {args.output.resolve()}")


if __name__ == "__main__":
    main()
