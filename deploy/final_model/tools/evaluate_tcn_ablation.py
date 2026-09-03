"""在冻结 OF-Syn val pose cache 上执行 rule/TCN/融合消融。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from eval.tcn_ablation import (
    ABLATION_PROTOCOL,
    aligned_rule_scores,
    atomic_json,
    configure_reproducible_inference,
    evaluate_aligned_ablation,
    evaluation_signature,
    load_trained_tcn,
    predict_window_probabilities,
)
from models.tcn_dataset import load_window_cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-cache-root", type=Path, required=True)
    parser.add_argument("--val-window-cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", default="of-syn")
    parser.add_argument("--split", default="val")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--weight-step", type=float, default=0.05)
    args = parser.parse_args()
    if args.split == "test":
        raise ValueError("消融阶段禁止读取 test split")
    if args.split != "val":
        raise ValueError("消融只允许 val split")
    if not 0.0 < args.weight_step <= 1.0:
        raise ValueError("weight-step 必须位于 (0,1]")
    if args.output.exists():
        raise ValueError("输出已存在，拒绝覆盖")

    cache = load_window_cache(args.val_window_cache, verify_hashes=True)
    if cache.metadata["dataset"] != args.dataset or cache.metadata["split"] != args.split:
        raise ValueError("窗口 cache dataset/split 不匹配")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("请求 CUDA 消融，但 CUDA 不可用")
    configure_reproducible_inference()
    model, run, checkpoint = load_trained_tcn(
        args.checkpoint, args.run_json, device=device
    )
    expected_val_signature = run["signature"]["val_window_signature"]
    if cache.metadata["signature_sha256"] != expected_val_signature:
        raise ValueError("val 窗口 cache 与训练 run 不匹配")

    print(json.dumps({"stage": "tcn_inference"}, sort_keys=True), flush=True)
    window_probabilities = predict_window_probabilities(
        model, cache, device=device, batch_size=args.batch_size
    )
    print(json.dumps({"stage": "rule_inference"}, sort_keys=True), flush=True)
    rule_scores, rule_window_scores, fallback_rule_scores = aligned_rule_scores(
        args.pose_cache_root,
        cache,
        progress=lambda completed, total: print(
            json.dumps(
                {
                    "stage": "rule_progress",
                    "completed": completed,
                    "total": total,
                },
                sort_keys=True,
            ),
            flush=True,
        ),
    )
    steps = round(1.0 / args.weight_step)
    if not np.isclose(steps * args.weight_step, 1.0):
        raise ValueError("weight-step 必须能整除 1.0")
    weights = np.linspace(0.0, 1.0, steps + 1).tolist()
    clips = cache.metadata["clips"]
    result = evaluate_aligned_ablation(
        [clip["clip_id"] for clip in clips],
        [bool(clip["has_fall"]) for clip in clips],
        rule_scores,
        rule_window_scores,
        fallback_rule_scores,
        window_probabilities,
        cache.array("clip_indices"),
        tcn_weights=weights,
    )
    signature_sha256, signature = evaluation_signature(
        checkpoint_path=args.checkpoint,
        run_path=args.run_json,
        cache=cache,
        weights=weights,
    )
    payload = {
        "protocol": ABLATION_PROTOCOL,
        "signature_sha256": signature_sha256,
        "signature": signature,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "dataset": args.dataset,
        "split": args.split,
        "clip_count": len(clips),
        "positive_clips": sum(bool(clip["has_fall"]) for clip in clips),
        **result,
        "limitations": [
            "fusion is aligned by causal track/window endpoint with single-branch fallback",
            "weights and thresholds are selected on val only; test remains sealed",
            "metrics follow the repository-local clip protocol pending organizer clarification",
        ],
    }
    atomic_json(args.output, payload)
    print(
        json.dumps(
            {
                "stage": "complete",
                "output": str(args.output.resolve()),
                "rule_map": payload["rule_only"]["map"],
                "tcn_map": payload["tcn_only"]["map"],
                "fusion": payload["selected_fusion"],
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
