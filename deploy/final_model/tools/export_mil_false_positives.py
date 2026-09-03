"""导出多流 MIL 验证集在冻结高召回阈值下的全部假阳性 clip。"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from models.tcn_dataset import SEMANTICS, WindowMemmapCache, load_window_cache
from tools.build_tcn_multistream_sidecar import load_sidecar
from tools.evaluate_hard_negative_cohort import _model_from_run
from tools.train_tcn import _atomic_json


@torch.inference_mode()
def _window_probabilities(
    model: torch.nn.Module,
    cache: WindowMemmapCache,
    sidecar: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    values: list[np.ndarray] = []
    features = cache.array("features")
    for start in range(0, cache.sample_count, batch_size):
        end = min(start + batch_size, cache.sample_count)
        pose = torch.from_numpy(np.array(features[start:end], dtype=np.float32, copy=True))
        geometry = torch.from_numpy(np.array(sidecar[start:end], dtype=np.float32, copy=True))
        batch = torch.cat((pose.flatten(2), geometry), dim=-1).to(device)
        values.append(torch.sigmoid(model(batch)).cpu().numpy())
    return np.concatenate(values).astype(np.float32, copy=False)


def false_positive_rows(
    cache: WindowMemmapCache, probabilities: np.ndarray, threshold: float
) -> list[dict[str, Any]]:
    """Return all negative clips whose maximum window probability reaches threshold."""
    if probabilities.shape != (cache.sample_count,):
        raise ValueError("窗口概率与 cache 样本数不一致")
    clips = cache.metadata.get("clips")
    if not isinstance(clips, list):
        raise TypeError("窗口缓存缺少 clips metadata")
    clip_indices = np.asarray(cache.array("clip_indices"), dtype=np.int64)
    best_scores = np.full(len(clips), -np.inf, dtype=np.float32)
    best_windows = np.full(len(clips), -1, dtype=np.int64)
    for index, (clip_index, score) in enumerate(zip(clip_indices, probabilities, strict=True)):
        if score > best_scores[clip_index]:
            best_scores[clip_index] = score
            best_windows[clip_index] = index
    end_times = cache.array("end_times")
    track_ids = cache.array("track_ids")
    semantic_codes = cache.array("semantic_codes")
    rows: list[dict[str, Any]] = []
    for clip_index, clip in enumerate(clips):
        if bool(clip.get("has_fall")) or best_scores[clip_index] < threshold:
            continue
        clip_id = clip.get("clip_id")
        if not isinstance(clip_id, str) or "/" not in clip_id:
            raise ValueError("clip_id 必须为 activity/name")
        window_index = int(best_windows[clip_index])
        code = int(semantic_codes[window_index])
        if not 0 <= code < len(SEMANTICS):
            raise ValueError("semantic code 超出定义范围")
        rows.append(
            {
                "clip_id": clip_id,
                "activity": clip_id.split("/", maxsplit=1)[0],
                "clip_score": float(best_scores[clip_index]),
                "threshold": threshold,
                "margin": float(best_scores[clip_index] - threshold),
                "peak_window_index": window_index,
                "peak_end_time_seconds": float(end_times[window_index]),
                "track_id": int(track_ids[window_index]),
                "peak_semantic": SEMANTICS[code],
            }
        )
    rows.sort(key=lambda item: (-float(item["clip_score"]), str(item["clip_id"])))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()
    if args.output_csv.exists() or args.output_json.exists():
        raise FileExistsError("FP 输出已存在，拒绝覆盖")
    cache = load_window_cache(args.cache)
    if cache.metadata.get("split") == "test":
        raise ValueError("禁止导出 test split 错误")
    run = json.loads(args.run.read_text(encoding="utf-8"))
    model, _, threshold = _model_from_run(
        model_kind="multiscale_multistream_tcn",
        run=run,
        checkpoint=args.checkpoint,
        device=torch.device(args.device),
    )
    rows = false_positive_rows(
        cache,
        _window_probabilities(
            model, cache, load_sidecar(args.sidecar, cache),
            device=torch.device(args.device), batch_size=args.batch_size,
        ),
        threshold,
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["rank", "clip_id"])
        writer.writeheader()
        writer.writerows(rows)
    _atomic_json(
        args.output_json,
        {
            "protocol": "multistream_mil_val_false_positive_export_v1",
            "split": cache.metadata.get("split"),
            "threshold": threshold,
            "false_positive_count": len(rows),
            "checkpoint": str(args.checkpoint.resolve()),
            "rows": rows,
        },
    )
    print(json.dumps({"false_positive_count": len(rows), "output_csv": str(args.output_csv.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
