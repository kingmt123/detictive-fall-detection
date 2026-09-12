"""Evaluate each train-OOF-selected clip-head blend without validation selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from models.tcn_dataset import load_window_cache
from tools.evaluate_stage_s1_clip_heads import ClipHead, _score_metrics
from tools.train_tcn import _atomic_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("输出已存在，拒绝覆盖")
    report = json.loads((args.run_dir / "report.json").read_text(encoding="utf-8"))
    evidence = np.load(args.run_dir / "clip_evidence.npz", allow_pickle=False)
    train_x, val_x = evidence["train_x"], evidence["val_x"]
    train_base, val_base = evidence["train_base"], evidence["val_base"]
    heads = torch.load(args.run_dir / "heads.pt", map_location="cpu", weights_only=False)
    clips = load_window_cache(args.val_cache, verify_hashes=True).metadata["clips"]
    device = torch.device(args.device)
    base_standard = (val_base - train_base.mean()) / max(float(train_base.std()), 1e-6)
    rows = []
    for oof_row in report["oof_results"]:
        spec = oof_row["spec"]
        payload = heads[spec["name"]]
        mean, scale = payload["scaler"]
        model = ClipHead(
            train_x.shape[1], spec["kind"], seed=0, gamma=float(spec["gamma"])
        ).to(device)
        model.load_state_dict(payload["state"], strict=True)
        model.eval()
        with torch.inference_mode():
            train_head = model(
                torch.from_numpy(((train_x - mean) / scale).astype(np.float32)).to(device)
            ).cpu().numpy()
            val_head = model(
                torch.from_numpy(((val_x - mean) / scale).astype(np.float32)).to(device)
            ).cpu().numpy()
        head_standard = (val_head - train_head.mean()) / max(float(train_head.std()), 1e-6)
        chosen = max(oof_row["oof_blends"], key=lambda row: row["objective"])
        alpha = float(chosen["alpha"])
        rows.append(
            {
                "spec": spec,
                "alpha_selected_by_train_oof": alpha,
                "train_oof_blend": chosen["metrics"],
                "validation_head": _score_metrics(clips, val_head),
                "validation_blend": _score_metrics(
                    clips, base_standard + alpha * head_standard
                ),
            }
        )
    output = {
        "protocol": "stage_s1_per_head_train_oof_alpha_single_val_diagnostic_v1",
        "selection_split": "train grouped head-OOF only",
        "test_accessed": False,
        "base_validation": report["base_validation"],
        "rows": rows,
    }
    _atomic_json(args.output, output)
    print(json.dumps(output, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
