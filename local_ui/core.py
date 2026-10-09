"""Thin local UI adapter. Does not change the frozen seven-head inference."""
from __future__ import annotations

import atexit
import html
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "weights/edgefall_seven_head_label_blind_v1.pt"
POSE_WEIGHTS = ROOT / "weights/yolo11n-pose.pt"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/local_ui"
MAX_UPLOAD_BYTES = 250 * 1024 * 1024
MAX_VIDEO_SECONDS = 180.0


def inspect_video(path: str | Path, *, max_bytes: int = MAX_UPLOAD_BYTES,
                  max_seconds: float = MAX_VIDEO_SECONDS) -> dict[str, float]:
    """Validate a locally uploaded video before expensive GPU inference."""
    import cv2

    source = Path(path)
    if not source.is_file():
        raise ValueError("视频文件不存在，请重新上传。")
    if source.stat().st_size <= 0 or source.stat().st_size > max_bytes:
        raise ValueError(f"视频大小必须在 0 到 {max_bytes // (1024 * 1024)} MB 之间。")
    capture = cv2.VideoCapture(str(source))
    try:
        if not capture.isOpened():
            raise ValueError("无法解码该视频，请转换为 H.264 MP4 后重试。")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0 or frames < 2:
            raise ValueError("视频帧率或总帧数无效。")
        duration = frames / fps
        if duration > max_seconds:
            raise ValueError(f"当前测试界面限制视频不超过 {max_seconds:g} 秒。")
        ok, _ = capture.read()
        if not ok:
            raise ValueError("视频首帧无法解码。")
        return {"fps": fps, "frames": float(frames), "duration": duration}
    finally:
        capture.release()


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def timeline_html(result: dict[str, Any]) -> str:
    """Display a pose-motion proposal, not a frame-level fall prediction."""
    duration = max(0.001, _finite(result.get("frames")) / max(0.001, _finite(result.get("fps"))))
    event = result.get("proposal") or {}
    start = min(duration, max(0.0, _finite(event.get("start_time"))))
    end = min(duration, max(start, _finite(event.get("end_time"))))
    peak = min(duration, max(start, _finite(event.get("peak_time"))))
    left = 100 * start / duration
    width = 100 * (end - start) / duration
    marker = 100 * peak / duration
    track = html.escape(str(event.get("track_id", "—")), quote=True)
    return f"""
    <section style="font-family:system-ui,sans-serif;padding:10px 0;color:#23344b">
      <p style="margin:0 0 8px;font-weight:600">姿态运动候选区间（非逐帧跌倒判定）</p>
      <div role="img" aria-label="视频时长 {duration:.2f} 秒，运动候选区间 {start:.2f} 至 {end:.2f} 秒"
           style="position:relative;height:30px;background:#e8eef5;border-radius:5px;overflow:hidden">
        <div style="position:absolute;left:{left:.3f}%;width:{width:.3f}%;height:100%;background:#77aaba"></div>
        <div style="position:absolute;left:{marker:.3f}%;top:0;height:100%;border-left:3px solid #bd644e"></div>
      </div>
      <div style="display:flex;justify-content:space-between;font-size:12px;margin-top:5px">
        <span>0.00 秒</span><span>{duration:.2f} 秒</span>
      </div>
      <p style="font-size:13px;margin:8px 0 0">Track {track} · 候选 {start:.2f}–{end:.2f} 秒 · 运动峰值 {peak:.2f} 秒</p>
      <p style="font-size:12px;color:#5d6876;margin:5px 0 0">该区间由姿态运动产生；分类器只给出整段视频评分。</p>
    </section>"""


def summary_markdown(result: dict[str, Any], *, device: str) -> str:
    score = _finite(result.get("score"), float("nan"))
    score_text = f"{score:.4f}" if math.isfinite(score) else "不可用"
    n = len(result.get("member_logits") or [])
    return (
        f"### 视频级评分：{score_text}\n"
        f"推理设备：{device} · 七头成员输出：{n} 个\n\n"
        "**注意：该分数不是经过校准的真实跌倒概率。** "
        "当前部署不使用测试数据调节阈值，也不自动给出已确认跌倒的结论。"
    )


def latency_rows(result: dict[str, Any]) -> list[list[Any]]:
    stages = (
        ("pose", "姿态提取"), ("skeleton", "姿态特征"),
        ("roi", "RGB ROI 特征"), ("heads_and_fusion", "七头融合"),
        ("total", "总耗时"), ("per_frame", "平均每帧"),
    )
    latencies = result.get("latency_ms") or {}
    return [[label, round(_finite(latencies.get(key)), 2)] for key, label in stages]


def render_annotated_video(source: Path, target: Path, result: dict[str, Any]) -> Path:
    """Render a timestamped *motion proposal*. Never draw fictitious detection boxes."""
    import cv2

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError("无法打开待渲染视频。")
    raw_path = target.with_name("preview_mp4v.mp4")
    writer = None
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if fps <= 0 or width <= 0 or height <= 0:
            raise ValueError("视频元数据不支持可视化编码。")
        scale = min(1.0, 1280 / width)
        size = (max(2, int(width * scale) // 2 * 2), max(2, int(height * scale) // 2 * 2))
        writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        if not writer.isOpened():
            raise RuntimeError("无法创建 MP4 视频编码器。")
        event = result.get("proposal") or {}
        start = _finite(event.get("start_time"))
        end = _finite(event.get("end_time"))
        score = _finite(result.get("score"), float("nan"))
        duration = max(0.001, _finite(result.get("frames")) / max(0.001, _finite(result.get("fps"))))
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if (frame.shape[1], frame.shape[0]) != size:
                frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            t = index / fps
            within = start <= t <= end
            # Explicitly distinguish the motion proposal from a confirmed fall.
            cv2.rectangle(frame, (0, 0), (size[0], 76), (20, 32, 46), -1)
            cv2.putText(frame, f"VIDEO score: {score:.4f} (not calibrated)",
                        (14, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            message = "MOTION PROPOSAL (not a fall label)" if within else "Outside motion proposal"
            cv2.putText(frame, message, (14, 57), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (180, 220, 240) if within else (225, 225, 225), 2)
            y = size[1] - 15
            cv2.rectangle(frame, (12, y), (size[0] - 12, y + 5), (180, 180, 180), -1)
            x_start = 12 + int((size[0] - 24) * max(0, min(1, start / duration)))
            x_end = 12 + int((size[0] - 24) * max(0, min(1, end / duration)))
            cv2.rectangle(frame, (x_start, y), (max(x_start + 1, x_end), y + 5), (170, 190, 90), -1)
            x_current = 12 + int((size[0] - 24) * max(0, min(1, t / duration)))
            cv2.line(frame, (x_current, y - 5), (x_current, y + 9), (50, 130, 240), 2)
            writer.write(frame)
            index += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    if not raw_path.is_file() or raw_path.stat().st_size == 0:
        raise RuntimeError("结果视频编码失败。")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(raw_path),
                   "-c:v", "libx264", "-preset", "veryfast", "-crf", "25",
                   "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target)]
        try:
            subprocess.run(command, check=True, timeout=180, capture_output=True)
            raw_path.unlink(missing_ok=True)
            return target
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
            target.unlink(missing_ok=True)
    raw_path.rename(target)
    return target


class FallModelRunner:
    """One model per local process, sequential inference, isolated run artifacts."""

    def __init__(self, *, model_factory: Callable[..., Any] | None = None,
                 output_root: Path = DEFAULT_OUTPUT_ROOT, device: str | None = None):
        self.model_factory = model_factory
        self.output_root = Path(output_root)
        selected = device or os.environ.get("EDGEFALL_DEVICE", "auto")
        if selected == "auto":
            import torch
            selected = "cuda:0" if torch.cuda.is_available() else "cpu"
        if selected not in ("cuda:0", "cpu"):
            raise ValueError("EDGEFALL_DEVICE 仅支持 auto、cuda:0 或 cpu。")
        self.device = selected
        self._model: Any = None
        self._lock = threading.Lock()
        atexit.register(self.close)

    def _load(self) -> Any:
        if self._model is None:
            if self.model_factory is None:
                from models.edgefall_seven_head_runtime import SevenHeadVideoModel
                self.model_factory = SevenHeadVideoModel
            self._model = self.model_factory(ARTIFACT, POSE_WEIGHTS, device=self.device)
        return self._model

    def run(self, video: str | Path) -> dict[str, Any]:
        source = Path(video)
        inspect_video(source)
        self.output_root.mkdir(parents=True, exist_ok=True)
        # Serialize both lazy model initialization and GPU inference.
        with self._lock:
            result = self._load().predict(source)
        if not isinstance(result, dict):
            raise TypeError("模型未返回 JSON 字典。")
        run_dir = Path(tempfile.mkdtemp(prefix="run-", dir=self.output_root))
        json_path = run_dir / "result.json"
        json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                             encoding="utf-8")
        video_path: str | None = None
        warning = ""
        try:
            video_path = str(render_annotated_video(source, run_dir / "preview.mp4", result))
        except Exception as exc:
            warning = f"可视化视频生成失败，但原始推理 JSON 已保存：{type(exc).__name__}: {exc}"
        return {
            "summary": summary_markdown(result, device=self.device),
            "timeline": timeline_html(result),
            "latency": latency_rows(result),
            "raw": result,
            "json_path": str(json_path),
            "video_path": video_path,
            "warning": warning,
        }

    def close(self) -> None:
        with self._lock:
            if self._model is not None:
                self._model.close()
                self._model = None
