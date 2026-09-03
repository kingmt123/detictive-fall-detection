"""Create the anonymous Gate 5 demonstration MP4 from validation-only evidence."""

from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "deliverables" / "fall_detection_demo.mp4"
ANNOTATED = ROOT / "runs" / "tmp" / "gate5_demo" / "urfd_val_annotated.mp4"
EVENTS = ROOT / "runs" / "tmp" / "gate5_demo" / "urfd_val_events.json"

WIDTH, HEIGHT, FPS = 1280, 720, 30
NAVY = "#172B4D"
BLUE = "#2E74B5"
LIGHT_BLUE = "#EAF2F8"
PALE = "#F4F6F9"
MUTED = "#5B6573"
WHITE = "#FFFFFF"
BLACK = "#111827"
GREEN = "#13795B"
RED = "#A61B1B"
FONT_PATH = Path("C:/Windows/Fonts/msyh.ttc")
FONT_BOLD_PATH = Path("C:/Windows/Fonts/msyhbd.ttc")


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_BOLD_PATH if bold and FONT_BOLD_PATH.exists() else FONT_PATH
    return ImageFont.truetype(str(path), size=size)


def wrap_text(draw: ImageDraw.ImageDraw, text: str, text_font, max_width: int) -> list[str]:
    lines: list[str] = []
    current = ""
    tokens = re.findall(
        r"[A-Za-z0-9_@./:+-]+(?:\s+)?|[\u4e00-\u9fff]|[^\u4e00-\u9fffA-Za-z0-9_@./:+-]",
        text,
    )
    for token in tokens:
        if token in "，。；：！？、,.!?:;" and current:
            current += token
            continue
        candidate = current + token
        if draw.textbbox((0, 0), candidate, font=text_font)[2] <= max_width or not current:
            current = candidate
        else:
            lines.append(current.rstrip())
            current = token.lstrip()
    if current:
        lines.append(current.rstrip())
    return lines


def base_slide(kicker: str, title: str, subtitle: str | None = None) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (WIDTH, HEIGHT), WHITE)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 18, HEIGHT), fill=BLUE)
    draw.text((72, 52), kicker, font=font(22, bold=True), fill=BLUE)
    draw.text((72, 100), title, font=font(42, bold=True), fill=NAVY)
    if subtitle:
        draw.text((74, 162), subtitle, font=font(21), fill=MUTED)
    draw.text((72, 676), "实时跌倒检测系统 · 匿名提交", font=font(16), fill=MUTED)
    draw.text((1110, 676), "Gate 5", font=font(16, bold=True), fill=BLUE)
    return image, draw


def add_bullets(draw: ImageDraw.ImageDraw, items: list[str], *, x: int = 94, y: int = 225, width: int = 1080) -> None:
    body_font = font(25)
    for item in items:
        draw.ellipse((x, y + 12, x + 10, y + 22), fill=BLUE)
        lines = wrap_text(draw, item, body_font, width - 36)
        for index, line in enumerate(lines):
            draw.text((x + 30, y + index * 38), line, font=body_font, fill=BLACK)
        y += max(58, len(lines) * 38 + 20)


def add_metric_cards(draw: ImageDraw.ImageDraw, cards: list[tuple[str, str, str]], *, top: int = 230) -> None:
    gap = 24
    card_width = (1120 - gap * (len(cards) - 1)) // len(cards)
    for index, (value, label, color) in enumerate(cards):
        x = 80 + index * (card_width + gap)
        draw.rounded_rectangle((x, top, x + card_width, top + 185), radius=18, fill=PALE, outline="#D5DCE5", width=2)
        draw.text((x + 24, top + 36), value, font=font(36, bold=True), fill=color)
        label_lines = wrap_text(draw, label, font(20), card_width - 48)
        for line_index, line in enumerate(label_lines):
            draw.text((x + 24, top + 105 + line_index * 30), line, font=font(20), fill=MUTED)


def title_slide() -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), WHITE)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, WIDTH, 16), fill=BLUE)
    draw.text((WIDTH // 2, 145), "实时视觉智能 · 匿名提交", anchor="mm", font=font(24, bold=True), fill=BLUE)
    draw.text((WIDTH // 2, 250), "面向低算力平台的", anchor="mm", font=font(54, bold=True), fill=NAVY)
    draw.text((WIDTH // 2, 330), "实时跌倒检测", anchor="mm", font=font(64, bold=True), fill=NAVY)
    draw.text((WIDTH // 2, 420), "YOLO11n-pose  +  因果 TCN  +  事件聚合", anchor="mm", font=font(26), fill=MUTED)
    draw.rounded_rectangle((214, 500, 1066, 588), radius=16, fill=LIGHT_BLUE)
    draw.text((WIDTH // 2, 544), "3.000M 参数  |  V100 P95 18.28ms  |  test MAP 42.93%", anchor="mm", font=font(24, bold=True), fill=NAVY)
    draw.text((WIDTH // 2, 660), "演示使用公开 URFD validation 片段，不包含任何封存 test 视频", anchor="mm", font=font(17), fill=MUTED)
    return image


def architecture_slide() -> Image.Image:
    image, draw = base_slide("系统设计", "五阶段因果推理链", "只使用当前与历史帧，输出可追溯事件")
    labels = [
        ("视频输入", "RGB / 灰度"),
        ("姿态", "17 关键点"),
        ("跟踪", "track_id"),
        ("FallTCN", "16 帧因果窗"),
        ("事件聚合", "起止时间 / 分数"),
    ]
    x_positions = [60, 300, 540, 780, 1020]
    for index, ((title, detail), x) in enumerate(zip(labels, x_positions, strict=True)):
        draw.rounded_rectangle((x, 260, x + 200, 430), radius=18, fill=PALE, outline="#CBD5E1", width=2)
        draw.text((x + 100, 315), title, anchor="mm", font=font(25, bold=True), fill=NAVY)
        draw.text((x + 100, 375), detail, anchor="mm", font=font(18), fill=MUTED)
        if index < len(labels) - 1:
            draw.line((x + 205, 345, x + 230, 345), fill=BLUE, width=5)
            draw.polygon([(x + 230, 337), (x + 242, 345), (x + 230, 353)], fill=BLUE)
    draw.rounded_rectangle((80, 495, 1200, 610), radius=14, fill=LIGHT_BLUE)
    draw.text((110, 525), "工程闭环", font=font(22, bold=True), fill=BLUE)
    draw.text((110, 566), "签名化缓存 · 断点恢复 · 失败隔离 · SHA-256 冻结 · 一次性 test seal", font=font(24), fill=NAVY)
    return image


def event_slide(event_payload: dict) -> Image.Image:
    image, draw = base_slide("演示结果", "从逐帧姿态到事件告警", "冻结 TCN-only epoch 7；validation 演示不参与调参")
    cards = [
        (str(event_payload["processed_frames"]), "处理帧数", NAVY),
        (f"{event_payload['pose_coverage']['frame_fraction'] * 100:.1f}%", "含姿态帧比例", BLUE),
        (f"{event_payload['clip_score']:.3f}", "TCN clip score", GREEN),
        (str(len(event_payload["events"])), "聚合事件数", RED),
    ]
    add_metric_cards(draw, cards, top=225)
    draw.rounded_rectangle((80, 455, 1200, 610), radius=16, fill=PALE)
    draw.text((110, 485), "输出契约", font=font(23, bold=True), fill=BLUE)
    draw.text((110, 530), "track_id  ·  t_start  ·  t_end  ·  score", font=font(30, bold=True), fill=NAVY)
    draw.text((110, 575), "轻量跟踪负责身份连续性，事件聚合负责平滑、双阈值与短间隔合并。", font=font(20), fill=MUTED)
    return image


def metrics_slide() -> Image.Image:
    image, draw = base_slide("冻结证据", "准确率、体积与时延", "测试结果只用于报告，测试后不再改变模型或阈值")
    add_metric_cards(
        draw,
        [
            ("42.93%", "OF-Syn test MAP", NAVY),
            ("45.66%", "P@R90", BLUE),
            ("40.19%", "P@R95", BLUE),
            ("18.28ms", "V100 1080P P95", GREEN),
        ],
        top=220,
    )
    draw.rounded_rectangle((80, 450, 1200, 610), radius=16, fill=LIGHT_BLUE)
    draw.text((110, 480), "工程硬门", font=font(22, bold=True), fill=BLUE)
    draw.text((110, 525), "总参数 3,000,165  ·  FP32 参数体积 12.00066MB  ·  207 tests + Ruff", font=font(25, bold=True), fill=NAVY)
    draw.text((110, 570), "V100：Tesla V100-SXM2-16GB，1920×1080，300 帧端到端。", font=font(20), fill=MUTED)
    return image


def robustness_slide() -> Image.Image:
    image, draw = base_slide("模型评估", "鲁棒性与主要失效模式", "固定 100-clip validation 子集，阈值来自完整 clean validation")
    rows = [
        ("clean", "39.71%", "94.1%"),
        ("grayscale", "46.30%", "94.1%"),
        ("lowlight", "34.17%", "94.1%"),
        ("Gaussian noise", "34.31%", "94.1%"),
        ("center occlusion", "17.49%", "82.4%"),
    ]
    draw.rounded_rectangle((80, 215, 700, 585), radius=14, fill=PALE, outline="#D5DCE5", width=2)
    draw.text((115, 242), "条件", font=font(21, bold=True), fill=NAVY)
    draw.text((405, 242), "MAP", font=font(21, bold=True), fill=NAVY)
    draw.text((545, 242), "R95 Recall", font=font(21, bold=True), fill=NAVY)
    for index, (condition, map_value, recall) in enumerate(rows):
        y = 300 + index * 52
        draw.text((115, y), condition, font=font(19), fill=BLACK)
        draw.text((405, y), map_value, font=font(19, bold=True), fill=RED if "occlusion" in condition else NAVY)
        draw.text((555, y), recall, font=font(19), fill=BLACK)
    draw.rounded_rectangle((735, 215, 1200, 585), radius=14, fill=LIGHT_BLUE)
    draw.text((770, 252), "结论", font=font(25, bold=True), fill=BLUE)
    add_bullets(
        draw,
        [
            "相似动作 lie_down / lying 是主要误报来源。",
            "中心遮挡会同时破坏 pose、track 与 TCN。",
            "灰度仅是红外代理，尚无真实热红外专项验证。",
        ],
        x=770,
        y=315,
        width=390,
    )
    return image


def engineering_slide() -> Image.Image:
    image, draw = base_slide("工程落地", "可复现、可移植、可审计", "运行包不包含数据集、测试预测、seal 或用户绝对路径")
    add_bullets(
        draw,
        [
            "单视频与 manifest 批处理共用推理引擎，支持无渲染模式和稳定 JSON/JSONL 输出。",
            "pose cache、窗口 cache 与运行配置均有签名；损坏、错配或跨平台路径异常会 fail closed。",
            "匿名 runtime ZIP 约 7.16MB，包含冻结 YOLO/TCN 权重、配置、运行说明和第三方许可。",
            "clean-tree 验收已通过：CLI、CPU checkpoint 加载、16 帧 validation 黑色视频批推理。",
        ],
        y=220,
    )
    return image


def limitations_slide() -> Image.Image:
    image, draw = base_slide("诚实边界", "已知限制与后续方向", "当前提交候选已永久冻结；改进只属于新的独立版本")
    draw.rounded_rectangle((80, 215, 600, 600), radius=16, fill="#FFF4F4", outline="#E8B4B4", width=2)
    draw.text((115, 250), "当前限制", font=font(26, bold=True), fill=RED)
    add_bullets(
        draw,
        [
            "OF-Syn 是 random clip split，存在跨划分泛化落差。",
            "多人交叉与长遮挡下，无 ReID tracker 可能产生 ID 碎片。",
            "NPU runtime/峰值存储与真实热红外尚未实测。",
        ],
        x=115,
        y=320,
        width=440,
    )
    draw.rounded_rectangle((640, 215, 1200, 600), radius=16, fill=LIGHT_BLUE, outline="#B7D3E9", width=2)
    draw.text((675, 250), "独立版本方向", font=font(26, bold=True), fill=BLUE)
    add_bullets(
        draw,
        [
            "补充真实红外、遮挡与跨场景数据。",
            "比较固定时间采样、observed mask 与轻量骨架图网络。",
            "仅在 pose/ID 证据充分时升级 RTMO/RTMPose 或 ReID。",
        ],
        x=675,
        y=320,
        width=470,
    )
    return image


def closing_slide() -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), NAVY)
    draw = ImageDraw.Draw(image)
    draw.text((WIDTH // 2, 235), "轻量、实时、可审计", anchor="mm", font=font(58, bold=True), fill=WHITE)
    draw.text((WIDTH // 2, 330), "以冻结证据交付，而不是以测试集调参", anchor="mm", font=font(30), fill="#DDE8F2")
    draw.rounded_rectangle((275, 430, 1005, 525), radius=18, fill=BLUE)
    draw.text((WIDTH // 2, 477), "3.000M 参数  |  18.28ms P95  |  test MAP 42.93%", anchor="mm", font=font(25, bold=True), fill=WHITE)
    draw.text((WIDTH // 2, 640), "匿名项目视频 · 2026", anchor="mm", font=font(18), fill="#B7C7D9")
    return image


def write_static(writer: cv2.VideoWriter, image: Image.Image, seconds: float) -> None:
    frame = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    for _ in range(round(seconds * FPS)):
        writer.write(frame)


def write_annotated_demo(writer: cv2.VideoWriter) -> None:
    capture = cv2.VideoCapture(str(ANNOTATED))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open annotated validation video: {ANNOTATED}")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            resized = cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
            overlay = resized.copy()
            cv2.rectangle(overlay, (0, 0), (WIDTH, 62), (23, 43, 77), -1)
            cv2.addWeighted(overlay, 0.86, resized, 0.14, 0, resized)
            banner = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(banner)
            draw.text((28, 16), "URFD validation 可视化  |  冻结 TCN-only  |  非 test", font=font(22, bold=True), fill=WHITE)
            writer.write(cv2.cvtColor(np.asarray(banner), cv2.COLOR_RGB2BGR))
    finally:
        capture.release()


def main() -> None:
    if not ANNOTATED.exists() or not EVENTS.exists():
        raise FileNotFoundError("validation-only annotated demo artifacts are missing")
    event_payload = json.loads(EVENTS.read_text(encoding="utf-8"))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(OUTPUT),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (WIDTH, HEIGHT),
    )
    if not writer.isOpened():
        raise RuntimeError("cannot create demonstration MP4")
    try:
        write_static(writer, title_slide(), 7)
        slide, draw = base_slide("项目目标", "低算力场景中的高召回跌倒告警", "纯视觉输入，不依赖深度相机、穿戴设备或云端服务")
        add_bullets(draw, ["参数 ≤20M、FP32 ≤80MB、端到端 P95 ≤100ms。", "在相似日常动作、低照度和遮挡下维持可解释的高召回。", "输出事件而非单帧标签，支持复核、统计与下游联动。"], y=235)
        write_static(writer, slide, 12)
        write_static(writer, architecture_slide(), 15)
        slide, draw = base_slide("运行演示", "冻结模型处理 validation 视频", "绿色骨架与轨迹分数来自在线因果路径")
        add_metric_cards(draw, [("300", "帧", NAVY), ("1080P", "输入", BLUE), ("TCN-only", "冻结模式", GREEN)], top=260)
        write_static(writer, slide, 5)
        write_annotated_demo(writer)
        write_static(writer, event_slide(event_payload), 12)
        write_static(writer, metrics_slide(), 15)
        write_static(writer, robustness_slide(), 16)
        write_static(writer, engineering_slide(), 16)
        write_static(writer, limitations_slide(), 16)
        write_static(writer, closing_slide(), 8)
    finally:
        writer.release()
    capture = cv2.VideoCapture(str(OUTPUT))
    if not capture.isOpened():
        raise RuntimeError("written demonstration MP4 cannot be reopened")
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    print(json.dumps({"frames": frames, "fps": fps, "duration_s": frames / fps, "width": width, "height": height, "bytes": OUTPUT.stat().st_size}, ensure_ascii=False))


if __name__ == "__main__":
    main()
