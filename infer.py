"""端到端纯视觉跌倒检测命令行入口。

批量调用请复用 :class:`pipeline.inference_engine.InferenceEngine`，避免每个视频
重新加载姿态模型。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline.inference_engine import (
    ALERT_THRESHOLD,
    CropMode,
    InferenceEngine,
    crop_frame,
    sanitize_fps,
)

__all__ = [
    "ALERT_THRESHOLD",
    "CropMode",
    "InferenceEngine",
    "analyze_video",
    "crop_frame",
    "sanitize_fps",
]


def analyze_video(
    source: Path,
    output_video: Path,
    output_events: Path,
    model_path: str = "yolo11n-pose.pt",
    device: str = "0",
    image_size: int = 640,
    confidence: float = 0.10,
    crop: CropMode = "auto",
    max_frames: int | None = None,
) -> dict:
    """兼容原单视频 API；批量任务应直接复用一个 ``InferenceEngine``。"""
    engine = InferenceEngine(
        model_path=model_path,
        device=device,
        image_size=image_size,
        confidence=confidence,
    )
    return engine.analyze(
        source,
        render=True,
        output_video=output_video,
        output_events=output_events,
        crop=crop,
        max_frames=max_frames,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="端到端视觉跌倒检测基线")
    parser.add_argument("source", type=Path)
    parser.add_argument("--output-video", type=Path, default=Path("reports/demo.mp4"))
    parser.add_argument("--output-events", type=Path, default=Path("reports/events.json"))
    parser.add_argument("--model", default="yolo11n-pose.pt")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument(
        "--crop", choices=["auto", "none", "left", "right"], default="auto"
    )
    parser.add_argument("--max-frames", type=int)
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="跳过骨架绘制与 MP4 编码，仅生成事件 JSON",
    )
    args = parser.parse_args()

    engine = InferenceEngine(
        model_path=args.model,
        device=args.device,
        image_size=args.imgsz,
        confidence=args.conf,
    )
    payload = engine.analyze(
        args.source,
        render=not args.no_render,
        output_video=None if args.no_render else args.output_video,
        output_events=args.output_events,
        crop=args.crop,
        max_frames=args.max_frames,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()