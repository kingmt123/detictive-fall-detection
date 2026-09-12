"""Build a three-token event-aligned view from an existing 16-token ROI cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from models.tcn_dataset import load_window_cache
from tools.train_event_aligned_rgb_oracle import crop_intervals
from tools.train_tcn import _sha256_file


@torch.inference_mode()
def downward_window_peaks(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return strongest visible downward centre motion and its frame per window."""
    if features.ndim != 4 or features.shape[1:] != (48, 17, 3):
        raise ValueError("event ROI 需要 (B,48,17,3) pose windows")
    coordinates = features[..., :2]
    visible = features[..., 2] > 0
    weights = visible.to(features.dtype).unsqueeze(-1)
    centre = (coordinates * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1.0)
    valid_pair = visible[:, 1:].any(dim=2) & visible[:, :-1].any(dim=2)
    downward = (centre[:, 1:, 1] - centre[:, :-1, 1]).clamp_min(0.0)
    downward = downward * valid_pair.to(downward.dtype)
    peak, frame = downward.max(dim=1)
    return peak, frame + 1


def event_token_index(
    event_time: float, interval: tuple[float, float], *, tokens: int = 16
) -> int:
    left, right = map(float, interval)
    if tokens < 3 or not np.isfinite([event_time, left, right]).all() or right <= left:
        raise ValueError("event token interval 无效")
    fraction = np.clip((event_time - left) / (right - left), 0.0, 1.0)
    return int(np.clip(round(float(fraction) * (tokens - 1)), 1, tokens - 2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--source-roi", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("event ROI 输出目录必须为空")
    if args.batch_size < 1:
        raise ValueError("batch size 必须为正数")

    cache = load_window_cache(args.cache, verify_hashes=True)
    if cache.metadata.get("dataset") != "of-syn" or cache.metadata.get("split") != "train":
        raise ValueError("event ROI 只接受 OF-Syn train cache")
    if cache.metadata.get("window_config") != {
        "window_size": 48,
        "stride": 1,
        "min_observed_frames": 24,
        "causal_left_pad": True,
    }:
        raise ValueError("event ROI 需要锁定的 native dense-48 cache")
    clips = cache.metadata["clips"]
    clip_ids = [str(clip["clip_id"]) for clip in clips]
    with np.load(args.source_roi, allow_pickle=False) as payload:
        source = {key: np.asarray(payload[key]) for key in payload.files}
    if source.get("tokens", np.empty(0)).shape != (len(clips), 16, 2, 192):
        raise ValueError("source ROI 必须为 (clips,16,2,192)")
    if [str(value) for value in source.get("clip_ids", [])] != clip_ids:
        raise ValueError("source ROI clip 身份不匹配")
    expected_labels = np.asarray([bool(clip["has_fall"]) for clip in clips], dtype=np.float32)
    if not np.array_equal(source.get("labels"), expected_labels):
        raise ValueError("source ROI labels 不匹配")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("请求 CUDA，但当前不可用")
    features = cache.array("features")
    peak_values: list[np.ndarray] = []
    peak_frames: list[np.ndarray] = []
    for start in range(0, features.shape[0], args.batch_size):
        batch = torch.from_numpy(
            np.array(
                features[start : start + args.batch_size],
                dtype=np.float32,
                copy=True,
            )
        ).to(device)
        peak, frame = downward_window_peaks(batch)
        peak_values.append(peak.cpu().numpy())
        peak_frames.append(frame.cpu().numpy())
    peaks = np.concatenate(peak_values)
    frames = np.concatenate(peak_frames)
    end_times = np.asarray(cache.array("end_times"), dtype=np.float64)
    intervals = crop_intervals(cache, context_seconds=1.0)

    selected_indices = np.empty(len(clips), dtype=np.int64)
    event_times = np.empty(len(clips), dtype=np.float64)
    for clip_index, (clip, interval) in enumerate(zip(clips, intervals, strict=True)):
        start = int(clip["window_start"])
        count = int(clip["window_count"])
        stop = start + count
        local = int(np.argmax(peaks[start:stop]))
        window = start + local
        times = np.unique(end_times[start:stop])
        differences = np.diff(times)
        positive = differences[differences > 1e-6]
        frame_seconds = float(np.median(positive)) if positive.size else 1.0 / 30.0
        event_time = float(end_times[window] - (47 - int(frames[window])) * frame_seconds)
        if interval is None:
            interval = (0.0, float(end_times[start:stop].max()))
        selected_indices[clip_index] = event_token_index(event_time, interval)
        event_times[clip_index] = event_time

    token_positions = np.stack(
        (
            np.zeros(len(clips), dtype=np.int64),
            selected_indices,
            np.full(len(clips), 15, dtype=np.int64),
        ),
        axis=1,
    )
    rows = np.arange(len(clips))[:, None]
    event_tokens = source["tokens"][rows, token_positions]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "roi_tokens.npz"
    np.savez_compressed(
        output,
        tokens=event_tokens,
        quality=source["quality"],
        clip_ids=source["clip_ids"],
        labels=source["labels"],
        track_ids=source["track_ids"],
        source_token_indices=token_positions,
        event_times=event_times,
    )
    histogram = np.bincount(selected_indices, minlength=16)
    summary = {
        "protocol": "dense48_existing_roi_earliest_peak_descent_current_v1",
        "source_roi_sha256": _sha256_file(args.source_roi),
        "cache_signature_sha256": cache.metadata.get("signature_sha256"),
        "clips": len(clips),
        "token_shape": list(event_tokens.shape),
        "event_index_histogram": histogram.tolist(),
        "output_sha256": _sha256_file(output),
        "additional_yolo_passes": 0,
        "uses_labels_for_event_selection": False,
        "inherits_source_event_crop": True,
        "test_accessed": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
