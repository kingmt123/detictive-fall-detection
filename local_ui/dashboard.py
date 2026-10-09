"""Presentation helpers for the local EdgeFall inspection console.

Only renders genuine output fields. Motion proposals are not model fall labels.
"""
from __future__ import annotations

from datetime import datetime
import html
import math
from pathlib import Path
from typing import Any


THEME_CSS = """
:root { --ef-ink:#18283c; --ef-muted:#687c90; --ef-line:#dce5ee;
        --ef-blue:#245d90; --ef-teal:#11877f; --ef-bg:#f4f7fa; }
.gradio-container { max-width:1480px !important; margin:auto !important;
                    background:var(--ef-bg) !important; }
body { background:var(--ef-bg) !important; }
.ef-hero { border:1px solid #dbe5ed; border-radius:16px; background:#12283f;
    color:#fff; padding:25px 30px; display:flex; flex-wrap:wrap;
    justify-content:space-between; align-items:center; gap:12px; margin-bottom:15px; }
.ef-hero h1 { color:#fff; font-size:26px; font-weight:750; letter-spacing:.2px;
    line-height:1.3; margin:5px 0; }
.ef-hero p {color:#c4d4e2;margin:5px 0 0;font-size:13px;}
.ef-eyebrow {font-size:11px;font-weight:760;letter-spacing:1.6px;color:#84ddd5;}
.ef-hero-right {display:flex;gap:9px;flex-wrap:wrap;align-items:center;}
.ef-chip {border:1px solid #47637a;padding:7px 11px;border-radius:7px;
    color:#dce9f4;font-size:12px;font-weight:650;}
.ef-chip-live {border-color:#328f8e;color:#b5e9db;}
.ef-section { display:flex; justify-content:space-between; align-items:flex-end;
    border-bottom:1px solid #dde5ed; margin:20px 0 11px; padding:0 0 9px;gap:10px; }
.ef-section h2 {font-size:17px;font-weight:750;color:#19334a;margin:0;}
.ef-section p {font-size:12px;color:#607386;margin:0;}
.ef-metrics {display:grid;grid-template-columns:repeat(4,minmax(0,1fr));
    gap:12px;margin:8px 0 12px;}
.ef-stat {background:#fff;border:1px solid var(--ef-line);border-radius:11px;
    padding:17px 19px;min-width:0;box-shadow:0 2px 6px rgba(23,55,78,.03);}
.ef-stat .label {font-size:12px;font-weight:650;color:var(--ef-muted);margin-bottom:9px;}
.ef-stat .value {font-size:27px;font-weight:760;line-height:1.2;color:var(--ef-ink);}
.ef-stat .unit {font-size:13px;color:#61768a;font-weight:500;margin-left:5px;}
.ef-stat .hint {font-size:11px;color:#7a8894;margin-top:8px;line-height:1.4;}
.ef-status {background:#e9f3f1;border:1px solid #cfe7e2;border-left:4px solid #358c83;
    border-radius:9px;padding:13px 18px;margin:8px 0 4px;font-size:13px;color:#284f52;}
.ef-status strong {font-weight:760;color:#153e42;}
.ef-status .meta {font-size:12px;color:#50777a;margin-top:5px;word-break:break-word;}
.ef-timeline {background:#fff;border:1px solid #dae4ed;border-radius:11px;
    padding:19px 21px 13px;color:#1a3149;}
.ef-timeline .subtle {color:#657d90;font-size:12px;}
.ef-timeline .track {height:32px;position:relative;background:repeating-linear-gradient(
    90deg,#edf3f7,#edf3f7 calc(25% - 1px),#d4e0e9 25%);
    overflow:hidden;border-radius:6px;margin:15px 0 6px;}
.ef-timeline .region {position:absolute;top:0;bottom:0;background:#83bebc;opacity:.84}
.ef-timeline .peak {position:absolute;top:0;bottom:0;width:3px;background:#c76446;}
.ef-timeline .times {display:flex;justify-content:space-between;font-size:11px;color:#64778a;}
.ef-key {display:flex;gap:18px;flex-wrap:wrap;font-size:12px;color:#566b7e;margin-top:13px;}
.ef-key i {display:inline-block;vertical-align:middle;width:14px;height:8px;
    margin-right:5px;background:#83bebc;border-radius:2px;}
.ef-key .peakkey {background:#c76446;width:4px;height:13px;}
.ef-panel {background:#fff;border:1px solid #dae4ed;border-radius:11px;padding:18px 22px;}
.ef-rowbar {display:grid;grid-template-columns:88px 1fr 75px;gap:12px;align-items:center;
    margin:10px 0;color:#506578;font-size:12px;}
.ef-bar-bg {height:11px;border-radius:3px;background:#e8eef4;position:relative;overflow:hidden;}
.ef-bar {height:100%;background:#3e8b9b;border-radius:3px;}
.ef-zero {position:absolute;left:50%;top:0;bottom:0;border-left:1px solid #8d9baa;}
.ef-note {font-size:12px;line-height:1.6;color:#597083;margin:12px 0 0;}
.ef-empty {border:1px dashed #c6d6e3;border-radius:10px;background:#fff;
    padding:20px;color:#788a9c;font-size:13px;}
.ef-small {font-size:12px !important;color:#627587 !important;}
@media(max-width:950px) {.ef-metrics{grid-template-columns:repeat(2,minmax(0,1fr));}}
@media(max-width:540px) {.ef-metrics{grid-template-columns:1fr;}
  .ef-hero{padding:20px}.ef-hero h1{font-size:21px}.ef-rowbar{grid-template-columns:55px 1fr 65px;gap:6px}}
"""


def _num(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def _text(value: Any) -> str:
    return html.escape(str(value), quote=True)


def hero_html() -> str:
    return """<header class="ef-hero">
    <div><div class="ef-eyebrow">EDGEFALL · INSPECTION CONSOLE</div>
    <h1>跌倒事件检测 · 工程验证工作台</h1>
    <p>本地视频分析 · 事件证据 · 模型诊断 · 人工复核</p></div>
    <div class="ef-hero-right"><span class="ef-chip ef-chip-live">● 本地运行</span>
    <span class="ef-chip">七头视觉基线 · 冻结</span><span class="ef-chip">非医疗报警产品</span></div>
    </header>"""


def section_html(title: str, subtitle: str = "") -> str:
    return f'<div class="ef-section"><h2>{_text(title)}</h2><p>{_text(subtitle)}</p></div>'


def status_html(*, case_id: str = "", review: str = "待复核", source: str = "") -> str:
    if not case_id:
        return ('<div class="ef-status"><strong>等待视频输入</strong>'
                '<div class="meta">上传视频后运行分析；系统不会根据未经校准的评分自动发出告警。</div></div>')
    return ('<div class="ef-status"><strong>✓ 分析完成 · '
            + _text(review) + '</strong><div class="meta">记录 '
            + _text(case_id) + ' · ' + _text(source)
            + ' · 所有复核结论均为人工录入</div></div>')


def metric_cards(result: dict[str, Any] | None = None, device: str = "—") -> str:
    result = result or {}
    score = _num(result.get("score"), float("nan"))
    event = result.get("proposal") or {}
    start = _num(event.get("start_time"), float("nan"))
    end = _num(event.get("end_time"), float("nan"))
    elapsed = _num((result.get("latency_ms") or {}).get("total"), float("nan")) / 1000
    perf = _num((result.get("latency_ms") or {}).get("per_frame"), float("nan"))
    fmt = lambda x, n: f"{x:.{n}f}" if math.isfinite(x) else "—"
    interval = f"{start:.2f}–{end:.2f}" if math.isfinite(start) and math.isfinite(end) and end >= start else "—"
    cards = [
        ("视频级评分", fmt(score, 4), "", "未经概率校准，不等于报警结论"),
        ("候选运动窗口", interval, "秒", "姿态运动提议，非跌倒标签"),
        ("分析总耗时", fmt(elapsed, 2), "秒", "完整视频端到端耗时"),
        ("平均处理耗时", fmt(perf, 2), "ms/帧", "并非逐帧 P95；设备 " + device),
    ]
    return '<div class="ef-metrics">' + ''.join(
        '<div class="ef-stat"><div class="label">' + _text(a) + '</div><div class="value">'
        + _text(b) + '<span class="unit">' + _text(c) + '</span></div><div class="hint">'
        + _text(d) + '</div></div>' for a, b, c, d in cards
    ) + '</div>'


def timeline_html(result: dict[str, Any] | None = None) -> str:
    if not result:
        return '<div class="ef-empty">完成一次视频分析后，这里会展示运动候选窗口和峰值位置。</div>'
    event = result.get("proposal") or {}
    frames = _num(result.get("frames"))
    fps = _num(result.get("fps"))
    duration = frames / fps if fps > 0 else 0
    if duration <= 0:
        return '<div class="ef-empty">视频缺少有效时长信息，无法显示时间轴。</div>'
    start = max(0.0, min(duration, _num(event.get("start_time"))))
    end = max(start, min(duration, _num(event.get("end_time"))))
    peak = max(0.0, min(duration, _num(event.get("peak_time"))))
    track = _text(event.get("track_id", "—"))
    pct = lambda x: f"{100*x/duration:.3f}%"
    return f"""<div class="ef-timeline">
    <div style="display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap">
      <b>时序证据 / Motion proposal</b>
      <span class="subtle">Track {track} · {fps:.1f} FPS · {frames:.0f} 帧</span></div>
    <div class="track" role="img" aria-label="候选运动事件从 {start:.2f} 秒到 {end:.2f} 秒">
       <div class="region" style="left:{pct(start)};width:{pct(end-start)}"></div>
       <div class="peak" style="left:{pct(peak)}"></div>
    </div>
    <div class="times"><span>00:00</span><span>{duration*0.25:.1f}s</span>
    <span>{duration*0.5:.1f}s</span><span>{duration*0.75:.1f}s</span><span>{duration:.1f}s</span></div>
    <div class="ef-key"><span><i></i>运动候选 {start:.2f}–{end:.2f}s</span>
      <span><i class="peakkey"></i>峰值 {peak:.2f}s</span>
      <span>跟踪 ID: {track}</span></div>
    <p class="ef-note">该时间轴来自姿态运动候选算法；<b>不是逐帧跌倒概率曲线</b>。
    七头分类器仅对整段视频输出一个评分。</p></div>"""


def head_chart_html(result: dict[str, Any] | None = None) -> str:
    values = (result or {}).get("member_logits") or []
    if not values:
        return '<div class="ef-empty">模型推理完成后显示七头原始输出。</div>'
    finite = [_num(x) for x in values]
    bound = max(max(abs(x) for x in finite), 0.2)
    rows = []
    for index, value in enumerate(finite, 1):
        length = min(49.0, abs(value) / bound * 46)
        left = 50 if value >= 0 else 50 - length
        rows.append(f'<div class="ef-rowbar"><b>Head {index:02d}</b>'
                    f'<div class="ef-bar-bg"><div class="ef-zero"></div>'
                    f'<div class="ef-bar" style="position:absolute;left:{left:.2f}%;width:{length:.2f}%"></div>'
                    f'</div><span style="text-align:right">{value:+.3f}</span></div>')
    return ('<div class="ef-panel"><b>七头原始 logits 对比</b><p class="ef-note">'
            '横轴中央为 0，各条长度按当前样本最大绝对值缩放。</p>'
            + ''.join(rows)
            + '<p class="ef-note">融合规则：七头 logits 算术平均后取 sigmoid。'
            '各子模型分数不是独立校准概率，不用于直接下达设备安全指令。</p></div>')


def latency_chart_html(result: dict[str, Any] | None = None) -> str:
    lat = (result or {}).get("latency_ms") or {}
    if not lat:
        return '<div class="ef-empty">完成推理后显示各处理阶段耗时。</div>'
    stages = [("pose", "人体姿态"), ("skeleton", "骨架特征"), ("roi", "图像 ROI"),
              ("heads_and_fusion", "七头融合")]
    total = max(sum(_num(lat.get(key)) for key, _ in stages), 0.001)
    rows = []
    for key, name in stages:
        value = _num(lat.get(key))
        width = max(0, min(100, 100 * value / total))
        rows.append(f'<div class="ef-rowbar"><b>{name}</b><div class="ef-bar-bg">'
                    f'<div class="ef-bar" style="width:{width:.2f}%"></div></div>'
                    f'<span style="text-align:right">{value:,.1f} ms</span></div>')
    return ('<div class="ef-panel"><b>推理阶段耗时分布</b><p class="ef-note">'
            '可定位主要计算开销；不包括页面上传及浏览器渲染时间。</p>'
            + ''.join(rows) + '</div>')
