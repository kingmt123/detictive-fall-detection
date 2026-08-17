"""manifest 批量评测、断点续跑与失败语义测试。"""
from __future__ import annotations

import csv
import io
import json
import tarfile
from pathlib import Path

import pytest

from eval.evaluate_manifest import evaluate_manifest


class _FakeEngine:
    def __init__(
        self,
        scores: dict[str, float],
        fail: set[str] | None = None,
        algorithm_version: str = "v1",
    ):
        self.scores = scores
        self.fail = fail or set()
        self.algorithm_version = algorithm_version
        self.calls: list[str] = []

    def cache_signature(self):
        return {"algorithm_version": self.algorithm_version}

    def analyze(self, source: Path, *, render: bool = False):
        clip_id = source.stem
        self.calls.append(clip_id)
        if clip_id in self.fail:
            raise ValueError(f"broken {clip_id}")
        return {
            "clip_score": self.scores[clip_id],
            "events": [],
            "rendered": render,
            "processed_frames": 2,
            "fps": 30.0,
            "stage_latency_ms": {
                "frame_end_to_end": {"mean": 10.0, "p50": 9.0, "p95": 12.0}
            },
        }


def _write_manifest(path: Path) -> None:
    rows = [
        {
            "dataset": "urfd",
            "split": "val",
            "clip_id": "fall-a",
            "video_path": str(path.parent / "fall-a.mp4"),
            "has_fall": "1",
            "events_json": "[]",
        },
        {
            "dataset": "urfd",
            "split": "val",
            "clip_id": "adl-a",
            "video_path": str(path.parent / "adl-a.mp4"),
            "has_fall": "0",
            "events_json": "[]",
        },
        {
            "dataset": "urfd",
            "split": "train",
            "clip_id": "train-only",
            "video_path": str(path.parent / "train-only.mp4"),
            "has_fall": "1",
            "events_json": "[]",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def test_batch_evaluation_uses_explicit_split_and_resumes(tmp_path: Path):
    manifest = tmp_path / "manifest.csv"
    output = tmp_path / "predictions.jsonl"
    _write_manifest(manifest)
    engine = _FakeEngine({"fall-a": 0.9, "adl-a": 0.1})

    first = evaluate_manifest(
        manifest,
        engine=engine,
        dataset="urfd",
        split="val",
        mode="clip",
        output_jsonl=output,
    )
    assert engine.calls == ["fall-a", "adl-a"]
    assert first["clips_total"] == first["clips_succeeded"] == 2
    assert first["metrics"]["map_percent"] == 100.0
    assert first["latency_across_clips"]["frame_end_to_end"]["worst_clip_p95"] == 12.0

    resumed_engine = _FakeEngine({"fall-a": 0.2, "adl-a": 0.8})
    second = evaluate_manifest(
        manifest,
        engine=resumed_engine,
        dataset="urfd",
        split="val",
        mode="clip",
        output_jsonl=output,
    )
    assert resumed_engine.calls == []
    assert second["clips_resumed"] == 2
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert {record["clip_id"] for record in records} == {"fall-a", "adl-a"}


def test_failed_clip_is_recorded_and_raises_instead_of_becoming_zero(tmp_path: Path):
    manifest = tmp_path / "manifest.csv"
    output = tmp_path / "predictions.jsonl"
    _write_manifest(manifest)
    engine = _FakeEngine({"fall-a": 0.9, "adl-a": 0.1}, fail={"fall-a"})

    with pytest.raises(RuntimeError, match="fall-a"):
        evaluate_manifest(
            manifest,
            engine=engine,
            dataset="urfd",
            split="val",
            mode="clip",
            output_jsonl=output,
        )

    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    failed = next(record for record in records if record["clip_id"] == "fall-a")
    assert failed["status"] == "error"
    assert "score" not in failed


def test_duplicate_clip_ids_are_rejected(tmp_path: Path):
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest)
    rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
    rows[1]["clip_id"] = rows[0]["clip_id"]
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="重复 clip_id"):
        evaluate_manifest(
            manifest,
            engine=_FakeEngine({"fall-a": 0.9}),
            dataset="urfd",
            split="val",
            mode="clip",
            output_jsonl=tmp_path / "predictions.jsonl",
        )


def test_manifest_schema_is_validated_before_inference(tmp_path: Path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "dataset,split,clip_id,video_path\nurfd,val,a,a.mp4\n",
        encoding="utf-8",
    )
    engine = _FakeEngine({"a": 0.9})

    with pytest.raises(ValueError, match="has_fall"):
        evaluate_manifest(
            manifest,
            engine=engine,
            dataset="urfd",
            split="val",
            mode="clip",
            output_jsonl=tmp_path / "predictions.jsonl",
        )
    assert engine.calls == []


def test_resume_rejects_changed_manifest(tmp_path: Path):
    manifest = tmp_path / "manifest.csv"
    output = tmp_path / "predictions.jsonl"
    _write_manifest(manifest)
    evaluate_manifest(
        manifest,
        engine=_FakeEngine({"fall-a": 0.9, "adl-a": 0.1}),
        dataset="urfd",
        split="val",
        mode="clip",
        output_jsonl=output,
    )
    manifest.write_text(
        manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="配置.*不一致"):
        evaluate_manifest(
            manifest,
            engine=_FakeEngine({"fall-a": 0.9, "adl-a": 0.1}),
            dataset="urfd",
            split="val",
            mode="clip",
            output_jsonl=output,
        )


def test_test_split_requires_explicit_one_time_override(tmp_path: Path):
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest)
    text = manifest.read_text(encoding="utf-8")
    manifest.write_text(
        text.replace("urfd,train,train-only", "urfd,test,train-only"),
        encoding="utf-8",
    )
    engine = _FakeEngine({"train-only": 0.9})

    with pytest.raises(ValueError, match="allow_test_once"):
        evaluate_manifest(
            manifest,
            engine=engine,
            dataset="urfd",
            split="test",
            mode="clip",
            output_jsonl=tmp_path / "test_predictions.jsonl",
        )
    assert engine.calls == []

    summary = evaluate_manifest(
        manifest,
        engine=engine,
        dataset="urfd",
        split="test",
        mode="clip",
        output_jsonl=tmp_path / "test_predictions.jsonl",
        allow_test_once=True,
    )
    assert engine.calls == ["train-only"]
    assert summary["clips_succeeded"] == 1

    with pytest.raises(ValueError, match="已完成"):
        evaluate_manifest(
            manifest,
            engine=engine,
            dataset="urfd",
            split="test",
            mode="clip",
            output_jsonl=tmp_path / "test_predictions.jsonl",
            allow_test_once=True,
        )
    assert engine.calls == ["train-only"]


def test_resume_rejects_changed_inference_algorithm(tmp_path: Path):
    manifest = tmp_path / "manifest.csv"
    output = tmp_path / "predictions.jsonl"
    _write_manifest(manifest)
    evaluate_manifest(
        manifest,
        engine=_FakeEngine({"fall-a": 0.9, "adl-a": 0.1}),
        dataset="urfd",
        split="val",
        mode="clip",
        output_jsonl=output,
    )

    with pytest.raises(ValueError, match="配置.*不一致"):
        evaluate_manifest(
            manifest,
            engine=_FakeEngine(
                {"fall-a": 0.9, "adl-a": 0.1}, algorithm_version="v2"
            ),
            dataset="urfd",
            split="val",
            mode="clip",
            output_jsonl=output,
        )


def test_resume_truncates_only_incomplete_trailing_jsonl_record(tmp_path: Path):
    manifest = tmp_path / "manifest.csv"
    output = tmp_path / "predictions.jsonl"
    _write_manifest(manifest)
    engine = _FakeEngine({"fall-a": 0.9, "adl-a": 0.1})
    evaluate_manifest(
        manifest,
        engine=engine,
        dataset="urfd",
        split="val",
        mode="clip",
        output_jsonl=output,
    )
    with output.open("a", encoding="utf-8") as handle:
        handle.write('{"run_id":')

    resumed = evaluate_manifest(
        manifest,
        engine=_FakeEngine({"fall-a": 0.2, "adl-a": 0.8}),
        dataset="urfd",
        split="val",
        mode="clip",
        output_jsonl=output,
    )
    assert resumed["clips_resumed"] == 2
    assert all(
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
        if line
    )


def test_resume_rejects_corrupt_middle_jsonl_record(tmp_path: Path):
    manifest = tmp_path / "manifest.csv"
    output = tmp_path / "predictions.jsonl"
    _write_manifest(manifest)
    engine = _FakeEngine({"fall-a": 0.9, "adl-a": 0.1})
    evaluate_manifest(
        manifest,
        engine=engine,
        dataset="urfd",
        split="val",
        mode="clip",
        output_jsonl=output,
    )
    lines = output.read_text(encoding="utf-8").splitlines()
    output.write_text(f"{lines[0]}\nnot-json\n{lines[1]}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="JSONL"):
        evaluate_manifest(
            manifest,
            engine=_FakeEngine({"fall-a": 0.9, "adl-a": 0.1}),
            dataset="urfd",
            split="val",
            mode="clip",
            output_jsonl=output,
        )


def test_test_seal_allows_same_run_resume_after_failure(tmp_path: Path):
    manifest = tmp_path / "manifest.csv"
    output = tmp_path / "test_predictions.jsonl"
    seal = tmp_path / "runs" / "eval" / "test_split.seal.json"
    _write_manifest(manifest)
    text = manifest.read_text(encoding="utf-8")
    manifest.write_text(
        text.replace("urfd,train,train-only", "urfd,test,train-only"),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="train-only"):
        evaluate_manifest(
            manifest,
            engine=_FakeEngine({"train-only": 0.9}, fail={"train-only"}),
            dataset="urfd",
            split="test",
            mode="clip",
            output_jsonl=output,
            allow_test_once=True,
        )
    assert json.loads(seal.read_text(encoding="utf-8"))["state"] == "started"

    summary = evaluate_manifest(
        manifest,
        engine=_FakeEngine({"train-only": 0.9}),
        dataset="urfd",
        split="test",
        mode="clip",
        output_jsonl=output,
        allow_test_once=True,
    )
    assert summary["clips_succeeded"] == 1
    assert json.loads(seal.read_text(encoding="utf-8"))["state"] == "completed"


def test_tar_manifest_source_is_materialized_for_engine_and_cleaned(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    archive = data_dir / "videos.tar"
    video_bytes = b"synthetic-video-bytes"
    with tarfile.open(archive, "w") as tar:
        info = tarfile.TarInfo("clips/fall-a.mp4")
        info.size = len(video_bytes)
        tar.addfile(info, io.BytesIO(video_bytes))

    manifest = data_dir / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dataset", "split", "clip_id", "video_path", "has_fall"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "dataset": "of-syn",
                "split": "val",
                "clip_id": "fall-a",
                "video_path": "tar://data/videos.tar!/clips/fall-a.mp4",
                "has_fall": "1",
            }
        )

    class _ContentEngine:
        def __init__(self):
            self.materialized_paths: list[Path] = []

        def cache_signature(self):
            return {"algorithm_version": "tar-tracer-v1"}

        def analyze(self, source: Path, *, render: bool = False):
            assert render is False
            source = Path(source)
            assert source.read_bytes() == video_bytes
            self.materialized_paths.append(source)
            return {"clip_score": 0.9, "events": []}

    engine = _ContentEngine()
    temp_root = tmp_path / "materialized"
    output = tmp_path / "predictions.jsonl"
    summary = evaluate_manifest(
        manifest,
        engine=engine,
        dataset="of-syn",
        split="val",
        mode="clip",
        output_jsonl=output,
        temp_root=temp_root,
    )

    assert summary["clips_succeeded"] == 1
    assert len(engine.materialized_paths) == 1
    assert not engine.materialized_paths[0].exists()
    assert list(temp_root.iterdir()) == []
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["source_kind"] == "tar"
    assert record["source_prepare_seconds"] >= 0.0


def test_tar_materialized_source_is_cleaned_when_engine_fails(tmp_path: Path):
    archive = tmp_path / "videos.tar"
    video_bytes = b"broken-video"
    with tarfile.open(archive, "w") as tar:
        info = tarfile.TarInfo("broken.mp4")
        info.size = len(video_bytes)
        tar.addfile(info, io.BytesIO(video_bytes))

    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dataset", "split", "clip_id", "video_path", "has_fall"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "dataset": "of-syn",
                "split": "val",
                "clip_id": "broken",
                "video_path": f"tar://{archive}!/broken.mp4",
                "has_fall": "0",
            }
        )

    class _FailingEngine:
        def __init__(self):
            self.materialized_path: Path | None = None

        def cache_signature(self):
            return {"algorithm_version": "tar-failure-v1"}

        def analyze(self, source: Path, *, render: bool = False):
            assert render is False
            self.materialized_path = Path(source)
            assert self.materialized_path.read_bytes() == video_bytes
            raise ValueError("decoder failed")

    engine = _FailingEngine()
    temp_root = tmp_path / "materialized"
    output = tmp_path / "predictions.jsonl"
    with pytest.raises(RuntimeError, match="broken"):
        evaluate_manifest(
            manifest,
            engine=engine,
            dataset="of-syn",
            split="val",
            mode="clip",
            output_jsonl=output,
            temp_root=temp_root,
        )

    assert engine.materialized_path is not None
    assert not engine.materialized_path.exists()
    assert list(temp_root.iterdir()) == []
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["status"] == "error"
    assert record["error_type"] == "ValueError"
