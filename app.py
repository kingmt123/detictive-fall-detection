"""Local-only Gradio test UI for the frozen EdgeFall seven-head model."""
from __future__ import annotations

import os

from local_ui.core import FallModelRunner


def build_app(runner: FallModelRunner | None = None):
    import gradio as gr

    runner = runner or FallModelRunner()

    def on_detect(video_path: str | None):
        if not video_path:
            raise gr.Error("请先上传一个视频。")
        try:
            data = runner.run(video_path)
        except Exception as exc:
            raise gr.Error(f"检测失败：{type(exc).__name__}: {exc}") from exc
        return (data["summary"], data["timeline"], data["latency"], data["raw"],
                data["json_path"], data["video_path"], data["warning"])

    with gr.Blocks(title="EdgeFall · 本地跌倒检测测试台") as demo:
        gr.Markdown("# EdgeFall · 本地跌倒检测测试台\n"
                    "上传视频，查看七头模型的视频级评分、候选运动区间和推理耗时。"
                    "本页面**不提供医学级报警判断或逐帧跌倒分类**。")
        with gr.Row():
            original = gr.Video(label="上传并预览原始视频", sources=["upload"], format="mp4")
            annotated = gr.Video(label="结果预览（运动候选区间提示）", interactive=False)
        detect = gr.Button("开始检测", variant="primary")
        summary = gr.Markdown("上传视频后点击「开始检测」。")
        timeline = gr.HTML(label="候选事件时间轴")
        warning = gr.Markdown()
        with gr.Row():
            timings = gr.Dataframe(headers=["推理阶段", "毫秒"], datatype=["str", "number"],
                                   label="各阶段耗时", interactive=False)
            raw = gr.JSON(label="原始七头推理结果")
        json_file = gr.File(label="下载 JSON 结果", interactive=False)
        detect.click(on_detect, inputs=[original],
                     outputs=[summary, timeline, timings, raw, json_file, annotated, warning],
                     concurrency_limit=1)
        gr.Markdown("**提示：** 候选时间段由姿态运动提出，并不等于模型逐帧确认的跌倒。"
                    "程序默认仅监听本机 127.0.0.1；推理输出保存在 outputs/local_ui/。")
    return demo


if __name__ == "__main__":
    host = "127.0.0.1"
    port = int(os.environ.get("EDGEFALL_PORT", "7860"))
    build_app().queue(default_concurrency_limit=1).launch(server_name=host, server_port=port,
                                                         share=False, show_error=True)
