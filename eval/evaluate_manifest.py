"""按 manifest 批量运行视频推理并计算严格 clip-level 指标。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import numpy as np

from eval.metrics import competition_map
from pipeline.video_source import LocalSource, VideoSourceResolver, parse_video_source

EvaluationMode = Literal["clip"]


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_bundle_sha256(paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        normalized = path.read_text(encoding="utf-8")
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(normalized.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _run_config(
    engine: Any,
    *,
    manifest_path: Path,
    dataset: str,
    split: str,
    mode: EvaluationMode,
) -> dict[str, object]:
    model_path = getattr(engine, "model_path", None)
    cache_signature = getattr(engine, "cache_signature", None)
    inference_signature = cache_signature() if callable(cache_signature) else None
    eval_root = Path(__file__).resolve().parent
    project_root = eval_root.parent
    return {
        "cache_schema": 2,
        "dataset": dataset,
        "split": split,
        "mode": mode,
        "manifest_sha256": _sha256(manifest_path),
        "evaluation_code_sha256": _source_bundle_sha256(
            (
                Path(__file__).resolve(),
                eval_root / "metrics.py",
                project_root / "pipeline" / "video_source.py",
            )
        ),
        "inference_signature": inference_signature,
        "model": model_path,
        "model_sha256": _sha256(Path(model_path)) if model_path else None,
        "device": getattr(engine, "device", None),
        "image_size": getattr(engine, "image_size", None),
        "confidence": getattr(engine, "confidence", None),
    }


def _run_id(config: dict[str, object]) -> str:
    encoded = json.dumps(config, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _replace_bytes_atomically(path: Path, content: bytes) -> None:
    temporary = path.with_name(f".{path.name}.repair.tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def _read_existing(path: Path, expected_run_id: str) -> dict[str, dict]:
    latest_success: dict[str, dict] = {}
    if not path.exists():
        return latest_success

    raw_lines = path.read_bytes().splitlines(keepends=True)
    nonempty_indices = [index for index, raw in enumerate(raw_lines) if raw.strip()]
    last_nonempty = nonempty_indices[-1] if nonempty_indices else -1
    repaired = False
    for index, raw_line in enumerate(raw_lines):
        if not raw_line.strip():
            continue
        try:
            line = raw_line.decode("utf-8").rstrip("\r\n")
            record = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            complete_line = raw_line.endswith((b"\n", b"\r"))
            if index == last_nonempty and not complete_line:
                _replace_bytes_atomically(path, b"".join(raw_lines[:index]))
                repaired = True
                break
            raise ValueError(f"JSONL 第 {index + 1} 行损坏") from exc
        if record.get("run_id") != expected_run_id:
            raise ValueError("已有 JSONL 的模型或评测配置与当前运行不一致")
        if record.get("status") == "ok":
            latest_success[record["clip_id"]] = record

    # 允许人工生成的最后一条完整 JSON 无换行；规范化后再安全 append。
    if (
        not repaired
        and last_nonempty == len(raw_lines) - 1
        and raw_lines
        and not raw_lines[-1].endswith((b"\n", b"\r"))
    ):
        _replace_bytes_atomically(path, b"".join(raw_lines) + b"\n")
    return latest_success


def _manifest_root(manifest_path: Path) -> Path:
    # 正式 manifest 位于 <project>/data/manifest.csv，路径相对项目根目录。
    return manifest_path.parent.parent if manifest_path.parent.name == "data" else Path.cwd()


def _resolve_video_source(video_path: str, manifest_path: Path) -> str | Path:
    parsed = parse_video_source(video_path)
    if isinstance(parsed, LocalSource):
        if parsed.path.is_absolute():
            return parsed.path
        return _manifest_root(manifest_path) / parsed.path

    archive_path = parsed.archive_path
    if not archive_path.is_absolute():
        archive_path = _manifest_root(manifest_path) / archive_path
    return f"tar://{archive_path}!/{parsed.member}"


@contextmanager
def _engine_source(
    resolver: VideoSourceResolver, source: str | Path
) -> Iterator[Path]:
    parsed = parse_video_source(source)
    if isinstance(parsed, LocalSource):
        yield parsed.path
        return
    with resolver.materialize(source) as materialized:
        yield materialized.local_path


def _metric_payload(labels: dict[str, bool], scores: dict[str, float]) -> dict:
    metrics = competition_map(labels, scores, mode="clip")
    metrics["curve"] = [asdict(point) for point in metrics["curve"]]
    return metrics


def _latency_across_clips(records: dict[str, dict]) -> dict[str, dict[str, float]]:
    stages = sorted(
        {
            stage
            for record in records.values()
            for stage in record.get("stage_latency_ms", {})
        }
    )
    summary: dict[str, dict[str, float]] = {}
    for stage in stages:
        stage_rows = [
            record["stage_latency_ms"][stage]
            for record in records.values()
            if stage in record.get("stage_latency_ms", {})
        ]
        summary[stage] = {
            "median_clip_mean": float(np.median([row["mean"] for row in stage_rows])),
            "median_clip_p50": float(np.median([row["p50"] for row in stage_rows])),
            "median_clip_p95": float(np.median([row["p95"] for row in stage_rows])),
            "worst_clip_p95": float(max(row["p95"] for row in stage_rows)),
        }
    return summary


def _test_seal_identity(run_id: str, output_jsonl: Path) -> dict[str, str]:
    return {
        "run_id": run_id,
        "predictions": str(output_jsonl.resolve()),
    }


def _canonical_test_seal(manifest_path: Path) -> Path:
    project_root = (
        manifest_path.parent.parent
        if manifest_path.parent.name == "data"
        else manifest_path.parent
    )
    return project_root / "runs" / "eval" / "test_split.seal.json"


def _acquire_test_seal(path: Path, run_id: str, output_jsonl: Path) -> None:
    identity = _test_seal_identity(run_id, output_jsonl)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump({**identity, "state": "started"}, handle, ensure_ascii=False)
            handle.flush()
    except FileExistsError:
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"test seal 损坏，拒绝读取 test: {path}") from exc
        if existing.get("state") == "completed":
            raise ValueError(f"test split 已完成并封存: {path}")
        if any(existing.get(key) != value for key, value in identity.items()):
            raise ValueError(f"test split 已被另一评测运行占用: {path}")


def _complete_test_seal(path: Path, run_id: str, output_jsonl: Path) -> None:
    payload = {
        **_test_seal_identity(run_id, output_jsonl),
        "state": "completed",
    }
    _replace_bytes_atomically(
        path, (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    )


def evaluate_manifest(
    manifest_path: Path,
    *,
    engine: Any,
    dataset: str,
    split: str,
    mode: EvaluationMode,
    output_jsonl: Path,
    allow_test_once: bool = False,
    temp_root: Path | None = None,
) -> dict:
    """评测显式 dataset/split；失败 clip 会记录错误并使整次运行失败。"""
    if mode != "clip":
        raise ValueError("当前批量入口仅支持具有 clip 标签的数据集")
    if not dataset or not split:
        raise ValueError("dataset 和 split 必须显式指定")
    if split == "test" and not allow_test_once:
        raise ValueError("test split 已封存；最终冻结后需显式 allow_test_once=True")

    manifest_path = Path(manifest_path)
    output_jsonl = Path(output_jsonl)
    temp_root = (
        Path(temp_root)
        if temp_root is not None
        else output_jsonl.parent / ".video-source-temp"
    )
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required_fields = {"dataset", "split", "clip_id", "video_path", "has_fall"}
        missing_fields = required_fields - set(reader.fieldnames or [])
        if missing_fields:
            raise ValueError(f"manifest 缺少字段: {sorted(missing_fields)}")
        selected = [
            row
            for row in reader
            if row.get("dataset") == dataset and row.get("split") == split
        ]
    if not selected:
        raise ValueError(f"manifest 中没有 dataset={dataset!r}, split={split!r}")

    clip_ids = [row.get("clip_id", "") for row in selected]
    if any(not clip_id for clip_id in clip_ids):
        raise ValueError("clip_id 必须是非空字符串")
    duplicates = sorted(
        clip_id for clip_id, count in Counter(clip_ids).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"重复 clip_id: {duplicates}")
    for row in selected:
        if row["has_fall"] not in {"0", "1"}:
            raise ValueError(
                f"clip {row['clip_id']} 的 has_fall 非法: {row['has_fall']!r}"
            )

    config = _run_config(
        engine,
        manifest_path=manifest_path,
        dataset=dataset,
        split=split,
        mode=mode,
    )
    run_id = _run_id(config)
    successful = _read_existing(output_jsonl, run_id)
    if split == "test":
        test_seal_path = _canonical_test_seal(manifest_path)
        if test_seal_path.resolve() == output_jsonl.resolve():
            raise ValueError("test_seal_path 不能与 output_jsonl 相同")
        _acquire_test_seal(test_seal_path, run_id, output_jsonl)
    resumed_ids = set(successful) & set(clip_ids)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []

    with (
        VideoSourceResolver(temp_root) as resolver,
        output_jsonl.open("a", encoding="utf-8") as output,
    ):
        for row in selected:
            clip_id = row["clip_id"]
            if clip_id in successful:
                continue
            source = _resolve_video_source(row["video_path"], manifest_path)
            source_kind = (
                "local"
                if isinstance(parse_video_source(source), LocalSource)
                else "tar"
            )
            source_prepare_seconds: float | None = None
            base_record = {
                "run_id": run_id,
                "dataset": dataset,
                "split": split,
                "mode": mode,
                "clip_id": clip_id,
                "video_path": row["video_path"],
                "source_kind": source_kind,
            }
            try:
                prepare_started = time.perf_counter()
                with _engine_source(resolver, source) as engine_source:
                    source_prepare_seconds = time.perf_counter() - prepare_started
                    payload = engine.analyze(engine_source, render=False)
                score = float(payload["clip_score"])
                if not np.isfinite(score) or not 0.0 <= score <= 1.0:
                    raise ValueError(f"非法 clip_score: {score}")
                record = {
                    **base_record,
                    "status": "ok",
                    "label": row["has_fall"],
                    "score": score,
                    "events": payload.get("events", []),
                    "processed_frames": payload.get("processed_frames"),
                    "fps": payload.get("fps"),
                    "stage_latency_ms": payload.get("stage_latency_ms", {}),
                    "source_prepare_seconds": source_prepare_seconds,
                }
                successful[clip_id] = record
            except Exception as exc:  # noqa: BLE001 - 逐 clip 记录后统一失败
                record = {
                    **base_record,
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "source_prepare_seconds": source_prepare_seconds,
                }
                failures.append(clip_id)
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()

    if failures:
        raise RuntimeError(f"以下 clip 推理失败，未计为 0 分: {', '.join(failures)}")

    labels: dict[str, bool] = {}
    scores: dict[str, float] = {}
    for row in selected:
        label = row["has_fall"]
        labels[row["clip_id"]] = label == "1"
        scores[row["clip_id"]] = float(successful[row["clip_id"]]["score"])

    summary = {
        "run_id": run_id,
        "run_config": config,
        "manifest": str(manifest_path.resolve()),
        "predictions": str(output_jsonl.resolve()),
        "clips_total": len(selected),
        "clips_succeeded": len(scores),
        "clips_resumed": len(resumed_ids),
        "metrics": _metric_payload(labels, scores),
        "latency_across_clips": _latency_across_clips(successful),
    }
    if split == "test":
        test_seal_path = _canonical_test_seal(manifest_path)
        _complete_test_seal(test_seal_path, run_id, output_jsonl)
    return summary


def main() -> None:
    from pipeline.inference_engine import InferenceEngine

    parser = argparse.ArgumentParser(description="manifest 批量跌倒评测")
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--mode", choices=["clip"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--model", default="yolo11n-pose.pt")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument(
        "--temp-root",
        type=Path,
        default=Path("runs/tmp/evaluate_manifest"),
        help="tar 视频成员的显式临时目录",
    )
    parser.add_argument(
        "--allow-test-once",
        action="store_true",
        help="仅在最终模型冻结后允许一次 test 运行",
    )

    args = parser.parse_args()

    engine = InferenceEngine(
        model_path=args.model,
        device=args.device,
        image_size=args.imgsz,
        confidence=args.conf,
    )
    summary = evaluate_manifest(
        args.manifest,
        engine=engine,
        dataset=args.dataset,
        split=args.split,
        mode=args.mode,
        output_jsonl=args.output,
        allow_test_once=args.allow_test_once,
        temp_root=args.temp_root,
    )
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
