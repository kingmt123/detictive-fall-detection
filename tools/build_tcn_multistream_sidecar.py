"""Build an auditable per-window bbox sidecar for the frozen 51-D cache.

The sidecar contains six causal, per-frame values::

    [bbox_cx, bbox_cy, bbox_w, bbox_h, clipped_log_aspect, observed]

It deliberately re-scans the already frozen pose NPZ files and never runs a
pose detector.  Every regenerated window is checked against the base window
cache before its sidecar row is written to an atomic temporary directory.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from models.tcn_dataset import (
    SEMANTIC_TO_CODE,
    WindowMemmapCache,
    load_window_cache,
)
from models.tcn_window import (
    label_for_semantics,
    normalize_pose,
    parse_activity_intervals,
    semantics_at_time,
)
from pipeline.pose_cache import pose_cache_path, read_pose_cache

SIDECAR_SCHEMA = 1
SIDECAR_FEATURE_DIM = 6
SIDECAR_SHAPE = (SIDECAR_FEATURE_DIM,)
SIDECAR_FILENAME = "sidecar.bin"
SIDECAR_DTYPE = np.dtype("<f4")
SIDECAR_FEATURE_NAMES = (
    "bbox_cx",
    "bbox_cy",
    "bbox_w",
    "bbox_h",
    "clip_log_aspect",
    "observed",
)
PROTOCOL = "tcn_multistream_bbox_sidecar_v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _load_rows(manifest_path: Path, *, dataset: str, split: str) -> list[dict[str, str]]:
    if split == "test":
        raise ValueError("sidecar 构建禁止读取 test split")
    with Path(manifest_path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"dataset", "split", "clip_id", "has_fall", "events_json"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"manifest 缺少字段: {sorted(missing)}")
        rows = [
            row
            for row in reader
            if row.get("dataset") == dataset and row.get("split") == split
        ]
    if not rows:
        raise ValueError(f"manifest 中没有 dataset={dataset!r}, split={split!r}")
    identities = [(row.get("dataset"), row.get("split"), row.get("clip_id")) for row in rows]
    if any(not all(identity) for identity in identities):
        raise ValueError("manifest 中存在空身份字段")
    if len(set(identities)) != len(identities):
        raise ValueError("manifest 存在重复 clip 身份")
    if any(row.get("has_fall") not in {"0", "1"} for row in rows):
        raise ValueError("manifest has_fall 必须为 0 或 1")
    return rows


def _load_audit(audit_path: Path, manifest_sha256: str) -> dict[str, Any]:
    try:
        audit = json.loads(Path(audit_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("audit 不是合法 JSON") from exc
    if audit.get("manifest_sha256") != manifest_sha256:
        raise ValueError("audit 与 manifest SHA-256 不匹配")
    if audit.get("errors"):
        raise ValueError(f"audit 未通过: {audit['errors']}")
    if audit.get("valid") != audit.get("selected"):
        raise ValueError("audit valid/selected 不一致")
    signatures = audit.get("extractor_signatures")
    if not isinstance(signatures, list) or len(signatures) != 1 or not isinstance(signatures[0], dict):
        raise ValueError("audit 必须包含唯一 extractor signature")
    return audit


def _track_bbox_features(record: Any, track_id: int) -> np.ndarray:
    """Materialize one track's bbox features once for the complete clip."""
    height, width = (float(value) for value in record.frame_size)
    result = np.zeros(
        (record.frame_indices.size, SIDECAR_FEATURE_DIM), dtype="<f4"
    )
    for frame_index in range(record.frame_indices.size):
        columns = np.flatnonzero(
            record.valid_mask[frame_index]
            & (record.track_ids[frame_index] == track_id)
        )
        if columns.size == 0:
            continue
        if columns.size != 1:
            raise ValueError("同一帧出现重复 track_id")
        x1, y1, x2, y2 = (float(value) for value in record.bboxes[frame_index, int(columns[0])])
        box_w = max(x2 - x1, 1.0)
        box_h = max(y2 - y1, 1.0)
        result[frame_index] = (
            ((x1 + x2) * 0.5) / width,
            ((y1 + y2) * 0.5) / height,
            box_w / width,
            box_h / height,
            float(np.clip(math.log(box_w / box_h) / 3.0, -1.0, 1.0)),
            1.0,
        )
    return result


def _track_pose_features(record: Any, track_id: int) -> np.ndarray:
    """Materialize normalized joints once instead of once per overlap window."""
    result = np.zeros(
        (record.frame_indices.size, 17, 3), dtype=np.float32
    )
    for frame_index in range(record.frame_indices.size):
        columns = np.flatnonzero(
            record.valid_mask[frame_index]
            & (record.track_ids[frame_index] == track_id)
        )
        if columns.size == 0:
            continue
        if columns.size != 1:
            raise ValueError("同一帧出现重复 track_id")
        column = int(columns[0])
        result[frame_index] = normalize_pose(
            record.keypoints[frame_index, column],
            record.bboxes[frame_index, column],
        )
    return result


def _causal_slice(
    sequence: np.ndarray,
    *,
    end_frame: int,
    window_size: int,
) -> np.ndarray:
    result = np.zeros((window_size, *sequence.shape[1:]), dtype=sequence.dtype)
    logical_start = end_frame - window_size + 1
    source_start = max(0, logical_start)
    left_pad = source_start - logical_start
    result[left_pad:] = sequence[source_start : end_frame + 1]
    return result


def _eligible_window_identities(
    record: Any,
    *,
    window_size: int,
    stride: int,
    min_observed_frames: int,
    causal_left_pad: bool,
) -> list[tuple[int, int]]:
    """Rebuild only the deterministic (track_id, end_frame) window index."""
    identities: list[tuple[int, int]] = []
    track_ids = sorted({int(value) for value in record.track_ids[record.valid_mask]})
    first_end_frame = 0 if causal_left_pad else window_size - 1
    for track_id in track_ids:
        observed = np.any(
            record.valid_mask & (record.track_ids == track_id), axis=1
        )
        for end_frame in range(first_end_frame, record.frame_indices.size, stride):
            if not observed[end_frame]:
                continue
            start_frame = max(0, end_frame - window_size + 1)
            if int(observed[start_frame : end_frame + 1].sum()) < min_observed_frames:
                continue
            identities.append((track_id, end_frame))
    return identities


def _bbox_features(
    record: Any,
    window: Any,
    *,
    track_features: np.ndarray | None = None,
) -> np.ndarray:
    """Reconstruct bbox values in the exact causal window frame order."""
    if track_features is None:
        track_features = _track_bbox_features(record, int(window.track_id))
    result = np.zeros((window.features.shape[0], SIDECAR_FEATURE_DIM), dtype="<f4")
    source_start_frame = max(0, int(window.start_frame))
    left_pad = source_start_frame - int(window.start_frame)
    result[left_pad:] = track_features[
        source_start_frame : window.end_frame + 1
    ]
    return result


def _base_signature(base_cache: WindowMemmapCache) -> dict[str, Any]:
    signature = base_cache.metadata.get("signature")
    if not isinstance(signature, dict):
        raise TypeError("base cache 缺少 signature")
    required = {"dataset", "split", "manifest_sha256", "audit_sha256", "extractor_signature", "window_config"}
    if not required.issubset(signature):
        raise ValueError("base cache signature 字段不完整")
    if signature["split"] == "test" or base_cache.metadata.get("split") == "test":
        raise ValueError("sidecar 构建禁止读取 test split")
    if signature["dataset"] != base_cache.metadata.get("dataset") or signature["split"] != base_cache.metadata.get("split"):
        raise ValueError("base cache signature 与 metadata 身份不一致")
    return signature


def _validate_base_window(
    *,
    base_features: np.ndarray,
    base_label: int,
    base_track_id: int,
    base_end_time: float,
    base_semantic_code: int,
    window: Any,
    semantic_code: int,
) -> None:
    if not np.allclose(base_features, window.features, rtol=0.0, atol=1e-6):
        raise ValueError("base joint features 与重扫窗口不一致")
    if base_track_id != int(window.track_id):
        raise ValueError("base track_id 与重扫窗口不一致")
    if np.float32(base_end_time) != np.float32(window.end_time):
        raise ValueError("base end_time 与重扫窗口不一致")
    if base_label != int(window.label):
        raise ValueError("base label 与重扫窗口不一致")
    if base_semantic_code != semantic_code:
        raise ValueError("base semantic code 与重扫窗口不一致")


def load_sidecar(
    root: Path,
    base_cache: WindowMemmapCache,
    *,
    verify_hashes: bool = True,
) -> np.memmap:
    """Load a read-only sidecar after validating shape and base identity."""
    root = Path(root)
    try:
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("sidecar metadata 无效") from exc
    if metadata.get("sidecar_schema") != SIDECAR_SCHEMA:
        raise ValueError("不支持的 sidecar schema")
    recorded_signature = metadata.get("signature_sha256")
    if not isinstance(recorded_signature, str):
        raise TypeError("sidecar 缺少 signature_sha256")
    signature_payload = {
        key: value for key, value in metadata.items() if key != "signature_sha256"
    }
    expected_signature = hashlib.sha256(
        _canonical_json(signature_payload).encode("utf-8")
    ).hexdigest()
    if recorded_signature != expected_signature:
        raise ValueError("sidecar metadata signature 不匹配")
    if metadata.get("feature_dim") != SIDECAR_FEATURE_DIM or metadata.get("shape", [])[-1:] != [SIDECAR_FEATURE_DIM]:
        raise ValueError("sidecar shape/feature_dim 不匹配")
    if metadata.get("base_signature_sha256") != base_cache.metadata.get("signature_sha256"):
        raise ValueError("sidecar 与 base cache signature 不匹配")
    if metadata.get("dataset") != base_cache.metadata.get("dataset") or metadata.get("split") != base_cache.metadata.get("split"):
        raise ValueError("sidecar 与 base cache 身份不匹配")
    base_metadata = base_cache.root / "metadata.json"
    if metadata.get("base_metadata_sha256") != _sha256_file(base_metadata):
        raise ValueError("sidecar 与 base cache metadata 不匹配")
    files = metadata.get("files")
    sidecar_spec = files.get("sidecar") if isinstance(files, dict) else None
    if not isinstance(sidecar_spec, dict) or sidecar_spec.get("name") != SIDECAR_FILENAME:
        raise ValueError("sidecar 文件清单无效")
    sample_count = int(metadata.get("sample_count", 0))
    window_size = int(metadata.get("window_size", 0))
    shape = tuple(int(value) for value in metadata.get("shape", []))
    if sample_count <= 0 or window_size <= 0 or shape != (sample_count, window_size, SIDECAR_FEATURE_DIM):
        raise ValueError("sidecar shape 元数据无效")
    path = root / SIDECAR_FILENAME
    expected_size = int(np.prod(shape)) * SIDECAR_DTYPE.itemsize
    if not path.is_file() or path.stat().st_size != expected_size:
        raise ValueError("sidecar 文件大小不匹配")
    if verify_hashes and _sha256_file(path) != sidecar_spec.get("sha256"):
        raise ValueError("sidecar SHA-256 不匹配")
    return np.memmap(path, mode="r", dtype=SIDECAR_DTYPE, shape=shape)


def build_multistream_sidecar(
    base_cache: Path | WindowMemmapCache,
    pose_cache_root: Path,
    manifest_path: Path,
    audit_path: Path,
    output_root: Path,
) -> Path:
    """Build ``sidecar.bin`` from a frozen base cache and pose NPZ files."""
    base = (
        base_cache
        if isinstance(base_cache, WindowMemmapCache)
        else load_window_cache(Path(base_cache), verify_hashes=True)
    )
    signature = _base_signature(base)
    dataset = str(base.metadata["dataset"])
    split = str(base.metadata["split"])
    manifest_sha256 = _sha256_file(manifest_path)
    if signature["manifest_sha256"] != manifest_sha256:
        raise ValueError("base cache 与 manifest SHA-256 不匹配")
    if signature["audit_sha256"] != _sha256_file(audit_path):
        raise ValueError("base cache 与 audit SHA-256 不匹配")
    audit = _load_audit(audit_path, manifest_sha256)
    if audit["extractor_signatures"][0] != signature["extractor_signature"]:
        raise ValueError("audit extractor signature 与 base cache 不匹配")
    rows = _load_rows(manifest_path, dataset=dataset, split=split)
    clips = base.metadata.get("clips")
    if not isinstance(clips, list) or len(clips) != len(rows):
        raise ValueError("base clips 与 manifest 行数不匹配")
    window_config = signature["window_config"]
    try:
        window_size = int(window_config["window_size"])
        stride = int(window_config["stride"])
        min_observed_frames = int(window_config["min_observed_frames"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("base window_config 无效") from exc
    base_arrays = {key: base.array(key) for key in ("features", "labels", "clip_indices", "track_ids", "end_times", "semantic_codes")}
    if any(array.shape[0] != base.sample_count for array in base_arrays.values()):
        raise ValueError("base array 长度不一致")

    output_root = Path(output_root)
    if output_root.exists():
        raise ValueError(f"输出目录必须不存在: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    sidecar_path = temp_root / SIDECAR_FILENAME
    digest = hashlib.sha256()
    sample_count = 0
    try:
        with sidecar_path.open("wb") as handle:
            for clip_index, (row, clip_meta) in enumerate(zip(rows, clips, strict=True)):
                expected_clip_id = row["clip_id"]
                if clip_meta.get("clip_id") != expected_clip_id:
                    raise ValueError("base clip 顺序/identity 与 manifest 不一致")
                if "has_fall" not in clip_meta or bool(clip_meta["has_fall"]) != (row["has_fall"] == "1"):
                    raise ValueError("base clip label 与 manifest 不一致")
                window_start = int(clip_meta["window_start"])
                window_count = int(clip_meta["window_count"])
                if window_start != sample_count or window_count < 0:
                    raise ValueError("base clip window_start/window_count 错位")
                record = read_pose_cache(
                    pose_cache_path(Path(pose_cache_root), dataset, split, expected_clip_id)
                )
                if (record.dataset, record.split, record.clip_id) != (dataset, split, expected_clip_id):
                    raise ValueError("pose cache identity 与 manifest 不一致")
                if record.extractor_signature != signature["extractor_signature"]:
                    raise ValueError("pose cache extractor signature 与 base 不一致")
                causal_left_pad = bool(
                    window_config.get("causal_left_pad", False)
                )
                identities = _eligible_window_identities(
                    record,
                    window_size=window_size,
                    stride=stride,
                    min_observed_frames=min_observed_frames,
                    causal_left_pad=causal_left_pad,
                )
                if len(identities) != window_count:
                    raise ValueError("重扫 window_count 与 base 不一致")
                unique_track_ids = {track_id for track_id, _ in identities}
                bbox_features = {
                    track_id: _track_bbox_features(record, track_id)
                    for track_id in unique_track_ids
                }
                pose_features = {
                    track_id: _track_pose_features(record, track_id)
                    for track_id in unique_track_ids
                }
                intervals = parse_activity_intervals(row["events_json"])
                for local_index, (track_id, end_frame) in enumerate(identities):
                    index = window_start + local_index
                    end_time = float(record.timestamps[end_frame])
                    event_semantics = semantics_at_time(end_time, intervals)
                    semantic_code = SEMANTIC_TO_CODE.get(event_semantics)
                    if semantic_code is None:
                        raise ValueError(f"未知 event semantics: {event_semantics}")
                    pose_row = _causal_slice(
                        pose_features[track_id],
                        end_frame=end_frame,
                        window_size=window_size,
                    )
                    if not np.allclose(
                        np.asarray(base_arrays["features"][index]),
                        pose_row,
                        rtol=0.0,
                        atol=1e-6,
                    ):
                        raise ValueError("base joint features 与重扫窗口不一致")
                    if int(base_arrays["track_ids"][index]) != track_id:
                        raise ValueError("base track_id 与重扫窗口不一致")
                    if np.float32(base_arrays["end_times"][index]) != np.float32(
                        end_time
                    ):
                        raise ValueError("base end_time 与重扫窗口不一致")
                    if int(base_arrays["labels"][index]) != int(
                        label_for_semantics(event_semantics)
                    ):
                        raise ValueError("base label 与重扫窗口不一致")
                    if int(base_arrays["semantic_codes"][index]) != semantic_code:
                        raise ValueError("base semantic code 与重扫窗口不一致")
                    if int(base_arrays["clip_indices"][index]) != clip_index:
                        raise ValueError("base clip_indices 与 metadata clips 不一致")
                    sidecar_row = _causal_slice(
                        bbox_features[track_id],
                        end_frame=end_frame,
                        window_size=window_size,
                    )
                    observed = sidecar_row[:, 5] > 0.5
                    if (
                        not observed[-1]
                        or int(observed.sum()) < min_observed_frames
                    ):
                        raise ValueError("sidecar observed mask 与窗口资格不一致")
                    payload = np.asarray(sidecar_row, dtype=SIDECAR_DTYPE).tobytes(order="C")
                    handle.write(payload)
                    digest.update(payload)
                    sample_count += 1
                if sample_count != window_start + window_count:
                    raise ValueError("sidecar clip 写入范围错位")
                if (clip_index + 1) % 100 == 0 or clip_index + 1 == len(rows):
                    print(
                        json.dumps(
                            {
                                "stage": "build_sidecar",
                                "split": split,
                                "clips": clip_index + 1,
                                "total": len(rows),
                                "windows": sample_count,
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                        flush=True,
                    )
            handle.flush()
            os.fsync(handle.fileno())
        if sample_count != base.sample_count:
            raise ValueError("sidecar sample_count 与 base 不一致")
        shape = (sample_count, window_size, SIDECAR_FEATURE_DIM)
        metadata_core = {
            "protocol": PROTOCOL,
            "sidecar_schema": SIDECAR_SCHEMA,
            "feature_names": list(SIDECAR_FEATURE_NAMES),
            "feature_dim": SIDECAR_FEATURE_DIM,
            "shape": list(shape),
            "sample_count": sample_count,
            "window_size": window_size,
            "dataset": dataset,
            "split": split,
            "base_signature_sha256": base.metadata["signature_sha256"],
            "base_metadata_sha256": _sha256_file(base.root / "metadata.json"),
            "base_files_sha256": base.metadata.get("files"),
            "manifest_sha256": manifest_sha256,
            "audit_sha256": _sha256_file(audit_path),
            "code_sha256": _sha256_file(Path(__file__)),
            "files": {"sidecar": {"name": SIDECAR_FILENAME, "sha256": digest.hexdigest()}},
            "clips": [
                {
                    "clip_id": clip["clip_id"],
                    "window_start": int(clip["window_start"]),
                    "window_count": int(clip["window_count"]),
                }
                for clip in clips
            ],
        }
        metadata = dict(metadata_core)
        metadata["signature_sha256"] = hashlib.sha256(
            _canonical_json(metadata_core).encode("utf-8")
        ).hexdigest()
        metadata_path = temp_root / "metadata.json"
        with metadata_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(metadata, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_root, output_root)
        return output_root
    except Exception:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-cache", type=Path, required=True)
    parser.add_argument("--pose-cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = build_multistream_sidecar(
        args.base_cache,
        args.pose_cache_root,
        args.manifest,
        args.audit,
        args.output,
    )
    print(f"wrote={output}")


if __name__ == "__main__":
    main()
