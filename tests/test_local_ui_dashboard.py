"""Local operator console tests: evidence integrity, safe reviews, UI build."""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from local_ui.dashboard import (
    head_chart_html, latency_chart_html, metric_cards, status_html,
    timeline_html,
)
from local_ui.operations import (
    capture_evidence, list_cases, load_case, read_review, register_case, save_review,
)


def _video(path: Path, frames: int = 45):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (120, 90))
    assert writer.isOpened()
    for i in range(frames):
        image = np.full((90, 120, 3), i * 3, dtype=np.uint8)
        writer.write(image)
    writer.release()


def _result():
    return {
        "protocol": "edgefall_seven_head_label_blind_runtime_v1",
        "score": 0.749, "frames": 45, "fps": 15,
        "member_logits": [0.1, -0.3, 0.2, 0.4, 0.5, -0.5, 0.25],
        "proposal": {"start_time": 0.75, "peak_time": 1.3, "end_time": 2.1,
                     "track_id": 2, "motion_score": 0.52, "observed_frames": 39},
        "latency_ms": {"pose": 13, "skeleton": 4, "roi": 15,
                       "heads_and_fusion": 2, "total": 34, "per_frame": 0.75},
    }


def test_dashboard_is_semantically_truthful():
    result = _result()
    cards = metric_cards(result, "cpu")
    assert "未经概率校准" in cards and "0.7490" in cards
    assert "视频级评分" in cards and "候选运动窗口" in cards
    assert "不是逐帧跌倒概率曲线" in timeline_html(result)
    assert "sigmoid" in head_chart_html(result)
    assert "0.5" in head_chart_html(result)
    assert "人体姿态" in latency_chart_html(result)
    assert "等待视频输入" in status_html()
    result["proposal"]["track_id"] = '<img src=x onerror=alert(1)>'
    assert "<img" not in timeline_html(result)


def test_event_evidence_three_real_frames(tmp_path):
    video = tmp_path / "original.mp4"
    _video(video)
    file = capture_evidence(video, tmp_path / "evidence.jpg", _result())
    image = cv2.imread(str(file))
    assert image is not None and image.shape == (260, 1152, 3)


def test_case_and_human_review_preserve_raw_prediction(tmp_path):
    video = tmp_path / "sample.mp4"
    _video(video)
    root = tmp_path / "outputs"
    directory = root / "run-verify12"
    directory.mkdir(parents=True)
    original = _result()
    raw_file = directory / "result.json"
    raw_file.write_text(json.dumps(original), encoding="utf-8")

    case = register_case(root, raw_file, video, device="cpu", video_path=None)
    assert case["metadata"]["model"] == "edgefall_seven_head_label_blind_v1"
    assert Path(case["evidence"]).is_file()
    assert read_review(root, "run-verify12")["status"] == "待复核"
    review = save_review(root, "run-verify12", "人工排除跌倒", "坐下而非跌倒")
    assert review["status"] == "人工排除跌倒"
    assert review["events"][-1]["operator"] == "local-unverified"
    assert json.loads(raw_file.read_text(encoding="utf-8")) == original
    assert load_case(root, "run-verify12")["review"]["status"] == "人工排除跌倒"
    assert list_cases(root)[0]["case_id"] == "run-verify12"


def test_case_review_rejects_invalid_paths_and_content(tmp_path):
    root = tmp_path / "results"
    (root / "run-valid").mkdir(parents=True)
    with pytest.raises(ValueError):
        read_review(root, "../hidden")
    with pytest.raises(ValueError):
        save_review(root, "run-valid", "已确认", "不能伪装模型输出")
    with pytest.raises(ValueError):
        save_review(root, "run-valid", "待复核", "a" * 501)


def test_local_operators_ui_builds_without_gpu(tmp_path):
    from app import build_app
    from local_ui.core import FallModelRunner

    class FakeModel:
        def __init__(self, *args, **kwargs):
            pass
        def close(self):
            pass
    runner = FallModelRunner(model_factory=FakeModel, output_root=tmp_path, device="cpu")
    try:
        demo = build_app(runner)
        assert demo is not None
    finally:
        runner.close()
