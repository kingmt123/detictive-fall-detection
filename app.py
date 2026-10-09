"""Local EdgeFall operator console: video analysis, evidence and human review."""
from __future__ import annotations

import os
from pathlib import Path

from local_ui.core import DEFAULT_OUTPUT_ROOT, FallModelRunner
from local_ui.dashboard import (
    THEME_CSS, head_chart_html, hero_html, latency_chart_html,
    metric_cards, section_html, status_html, timeline_html,
)
from local_ui.operations import (
    REVIEW_STATES, list_cases, load_case, register_case, save_review,
)


def _history(root: Path) -> tuple[list[list[object]], list[str]]:
    records = list_cases(root)
    return (
        [[r["created_at"], r["source_name"], r["score"], r["event_time"],
          r["review_status"], r["case_id"]] for r in records],
        [r["case_id"] for r in records],
    )


def build_app(runner: FallModelRunner | None = None):
    import gradio as gr

    runner = runner or FallModelRunner()
    root = runner.output_root
    previous, ids = _history(root)

    def _display(case: dict, *, video: str | None = None):
        info, result, review = case["metadata"], case["result"], case["review"]
        return (
            status_html(case_id=info["case_id"], review=review["status"],
                        source=info.get("source_name", "")),
            metric_cards(result, device=info.get("device", "—")),
            timeline_html(result),
            head_chart_html(result),
            latency_chart_html(result),
            video,
            case.get("evidence"),
            result,
            str(root / info["case_id"] / "result.json"),
            case.get("review_path"),
            review["status"],
            review["note"],
            info["case_id"],
            info.get("warning", ""),
        )

    def _refresh_choices():
        rows, choices = _history(root)
        return rows, gr.update(choices=choices)

    def on_detect(source: str | None):
        if not source:
            raise gr.Error("请上传视频后开始检测。")
        try:
            data = runner.run(source)
            case = register_case(root, data["json_path"], source, device=runner.device,
                                 video_path=data["video_path"], warning=data["warning"])
        except Exception as exc:
            raise gr.Error(f"检测未完成：{type(exc).__name__}: {exc}") from exc
        displayed = _display(case, video=data["video_path"])
        return (*displayed, *_refresh_choices())

    def on_load(case_id: str | None):
        if not case_id:
            raise gr.Error("请选择需要查看的历史记录。")
        try:
            case = load_case(root, case_id)
        except Exception as exc:
            raise gr.Error(f"读取记录失败：{exc}") from exc
        # Historical upload source is not persisted: avoid showing a stale video.
        return _display(case, video=case.get("video"))

    def on_review(case_id: str | None, verdict: str, note: str):
        if not case_id:
            raise gr.Error("请先分析或加载一条记录。")
        try:
            review = save_review(root, case_id, verdict, note or "")
            case = load_case(root, case_id)
        except Exception as exc:
            raise gr.Error(f"复核保存失败：{exc}") from exc
        status = status_html(case_id=case_id, review=review["status"],
                             source=case["metadata"].get("source_name", ""))
        rows, _ = _history(root)
        return status, case["review_path"], rows, "人工复核已记录（本机操作者未认证）。"

    def on_seek(case_id: str | None):
        if not case_id:
            raise gr.Error("请先生成或打开一条检测记录。")
        case = load_case(root, case_id)
        if not case["video"]:
            raise gr.Error("当前记录没有可播放的结果视频。")
        seconds = float((case["result"].get("proposal") or {}).get("peak_time", 0))
        return gr.update(value=case["video"], playback_position=max(0, seconds))

    with gr.Blocks(
        title="EdgeFall | 工业检测工作台",
        css=THEME_CSS,
    ) as demo:
        gr.HTML(hero_html())
        current_case = gr.State("")

        gr.HTML(section_html("01 / 视频检测工作台", "从原始视频到可追溯的推理报告"))
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=360):
                original = gr.Video(label="输入视频 / Video source",
                                    sources=["upload"], format="mp4", height=320)
                run_button = gr.Button("▶ 开始视频分析", variant="primary", size="lg")
                gr.Markdown("仅支持本机文件分析；最大 **250 MB / 180 秒**。"
                            "完整视频处理可能需要较长时间。", elem_classes="ef-small")
            with gr.Column(scale=1, min_width=360):
                annotated = gr.Video(label="结果回放 / Motion proposal overlay",
                                     interactive=False, height=320)
                seek = gr.Button("定位至候选运动峰值", size="sm", variant="secondary")
                gr.Markdown("视频文字和时间轴仅标示**运动候选区间**，"
                            "不代表模型逐帧确认发生跌倒。", elem_classes="ef-small")

        state = gr.HTML(status_html())
        kpis = gr.HTML(metric_cards())
        gr.HTML(section_html("02 / 事件与证据", "支持候选时间轴、真实视频关键帧查看"))
        timeline = gr.HTML(timeline_html())
        gr.Markdown("**事件前 / 峰值 / 事件后**：以下仅截取原视频画面，"
                    "未绘制不存在的人体检测框或骨架。", elem_classes="ef-small")
        evidence = gr.Image(label="三帧原始证据 / Before · Peak · After",
                            interactive=False, type="filepath")
        warning = gr.Markdown(visible=True)

        gr.HTML(section_html("03 / 模型诊断", "原始七头分数与性能瓶颈分析"))
        with gr.Row():
            heads = gr.HTML(head_chart_html())
            latency = gr.HTML(latency_chart_html())
        with gr.Accordion("原始机器可读记录 / 导出文件", open=False):
            raw = gr.JSON(label="模型原始输出（未经修改）")
            with gr.Row():
                json_file = gr.File(label="下载原始 result.json", interactive=False)
                review_file = gr.File(label="下载人工复核 review.json", interactive=False)

        gr.HTML(section_html("04 / 记录与人工复核", "本地记录 · 人工判断与模型评分分离"))
        with gr.Row():
            with gr.Column(scale=2):
                historic = gr.Dataframe(
                    value=previous,
                    headers=["分析时间", "源文件", "视频级评分", "候选区间", "人工状态", "记录编号"],
                    datatype=["str", "str", "number", "str", "str", "str"],
                    interactive=False, label="最近 100 条本地分析记录",
                    max_height=285, show_search="search")
                with gr.Row():
                    selector = gr.Dropdown(choices=ids, label="选择历史记录编号",
                                           value=None, scale=3)
                    load = gr.Button("加载记录", scale=1)
                    refresh = gr.Button("刷新列表", scale=1)
            with gr.Column(scale=1):
                verdict = gr.Dropdown(choices=list(REVIEW_STATES), value="待复核",
                                      label="人工复核状态")
                note = gr.Textbox(label="复核备注（最多 500 字符）", lines=3,
                                  max_lines=4, placeholder="记录复核依据；不要输入敏感个人身份信息。")
                save = gr.Button("保存人工复核", variant="secondary")
                review_message = gr.Markdown("人工操作仅记录复核结论，不会改变原始模型结果。")

        gr.Markdown("**工程使用边界**：这是单机视频离线分析与复核台，"
                    "并非接入摄像头的实时告警系统；尚未实现账号权限、审计身份认证、"
                    "长期隐私保留策略与安全级联动。"
                    "不能直接用作无人值守的工业安全控制或医疗应急报警。", elem_classes="ef-small")

        display_outputs = [
            state, kpis, timeline, heads, latency, annotated, evidence,
            raw, json_file, review_file, verdict, note, current_case, warning,
        ]
        run_button.click(on_detect, inputs=original,
                         outputs=display_outputs + [historic, selector],
                         concurrency_limit=1)
        load.click(on_load, inputs=selector, outputs=display_outputs,
                   concurrency_limit=1)
        refresh.click(_refresh_choices, outputs=[historic, selector])
        save.click(on_review, inputs=[current_case, verdict, note],
                   outputs=[state, review_file, historic, review_message],
                   concurrency_limit=1)
        seek.click(on_seek, inputs=current_case, outputs=annotated)

    return demo


if __name__ == "__main__":
    import gradio as gr

    host = "127.0.0.1"
    port = int(os.environ.get("EDGEFALL_PORT", "7860"))
    build_app().queue(default_concurrency_limit=1).launch(
        server_name=host, server_port=port, share=False, show_error=True,
        allowed_paths=[str(DEFAULT_OUTPUT_ROOT.resolve())],
    )
