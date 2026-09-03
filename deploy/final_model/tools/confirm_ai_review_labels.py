"""Promote user-approved AI review drafts to immutable confirmed label files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from tools.audit_edgefall_oof_errors import FN_TAXONOMY, FP_TAXONOMY


class ConfirmationError(ValueError):
    """Raised when a label draft cannot safely be promoted."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ConfirmationError(f"CSV 缺少表头: {path}")
        return list(reader.fieldnames), list(reader)


def confirmed_rows(
    review_rows: list[dict[str, str]], ai_rows: list[dict[str, str]]
) -> list[dict[str, str]]:
    """Return user-confirmed labels after exact row and taxonomy validation."""
    if len(review_rows) != len(ai_rows):
        raise ConfirmationError("人工表和 AI 草稿行数不一致")
    confirmed: list[dict[str, str]] = []
    for review, draft in zip(review_rows, ai_rows, strict=True):
        keys = ("image", "review_set", "clip_id")
        if any(review.get(key) != draft.get(key) for key in keys):
            raise ConfirmationError("人工表和 AI 草稿的行顺序或 clip_id 不一致")
        if review.get("human_review_label", "").strip():
            raise ConfirmationError("拒绝覆盖已有人工确认标签")
        review_set = review.get("review_set")
        allowed = (
            set(FP_TAXONOMY)
            if review_set == "high_score_false_positive"
            else set(FN_TAXONOMY)
            if review_set == "low_score_false_negative"
            else set()
        )
        label = draft.get("ai_label", "").strip()
        confidence = draft.get("confidence", "").strip()
        note = draft.get("ai_review_notes", "").strip()
        if label not in allowed or confidence not in {"high", "medium", "low"}:
            raise ConfirmationError("AI 草稿包含无效 taxonomy 或置信度")
        value = dict(review)
        value["human_review_label"] = label
        value["reviewer_notes"] = (
            f"用户确认 AI 初标（置信度: {confidence}）: {note}"
        )
        confirmed.append(value)
    return confirmed


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    if path.exists():
        raise FileExistsError(f"确认标签输出已存在，拒绝覆盖: {path}")
    temporary = path.with_suffix(path.suffix + ".partial")
    if temporary.exists():
        raise FileExistsError(f"确认标签临时文件已存在，拒绝覆盖: {temporary}")
    try:
        with temporary.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-labels", type=Path, required=True)
    parser.add_argument("--ai-labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    if args.summary.exists():
        raise FileExistsError(f"确认摘要已存在，拒绝覆盖: {args.summary}")
    fields, review_rows = _read_csv(args.review_labels)
    _, ai_rows = _read_csv(args.ai_labels)
    required = {
        "image", "review_set", "clip_id", "human_review_label", "reviewer_notes"
    }
    if not required.issubset(fields):
        raise ConfirmationError("人工审阅 CSV 缺少必要列")
    rows = confirmed_rows(review_rows, ai_rows)
    _write_csv(args.output, fields, rows)
    payload: dict[str, Any] = {
        "protocol": "edgefall_user_confirmed_ai_assisted_review_v1",
        "rows": len(rows),
        "review_labels_sha256": _sha256_file(args.review_labels),
        "ai_labels_sha256": _sha256_file(args.ai_labels),
        "confirmed_labels_sha256": _sha256_file(args.output),
        "output": str(args.output.resolve()),
        "test_accessed": False,
    }
    temporary_summary = args.summary.with_suffix(args.summary.suffix + ".partial")
    with temporary_summary.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_summary, args.summary)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
