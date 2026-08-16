"""模型参数、权重体积和 warm-up 后推理时延基准。"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO


def benchmark_model(
    model_path: str,
    image_size: int = 640,
    device: str = "0",
    warmup: int = 20,
    runs: int = 100,
) -> dict:
    model = YOLO(model_path)
    parameter_count = sum(parameter.numel() for parameter in model.model.parameters())
    dummy = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    for _ in range(warmup):
        model.predict(dummy, imgsz=image_size, device=device, verbose=False)
    if torch.cuda.is_available() and device != "cpu":
        torch.cuda.synchronize()

    timings = []
    for _ in range(runs):
        start = time.perf_counter()
        model.predict(dummy, imgsz=image_size, device=device, verbose=False)
        if torch.cuda.is_available() and device != "cpu":
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000.0)
    return {
        "model": model_path,
        "parameters": parameter_count,
        "parameters_million": parameter_count / 1e6,
        "fp32_weight_mb": parameter_count * 4 / 1e6,
        "image_size": image_size,
        "device": device,
        "warmup_runs": warmup,
        "measured_runs": runs,
        "latency_ms": {
            "mean": float(np.mean(timings)),
            "p50": float(np.percentile(timings, 50)),
            "p95": float(np.percentile(timings, 95)),
            "min": float(np.min(timings)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo11n-pose.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("reports/benchmark.json"))
    args = parser.parse_args()
    result = benchmark_model(args.model, args.imgsz, args.device, args.warmup, args.runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
