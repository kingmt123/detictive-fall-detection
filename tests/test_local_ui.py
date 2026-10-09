"""Fast offline checks for local UI, with no real model weights."""
import json
from pathlib import Path
import cv2
import numpy as np
import pytest
from local_ui.core import FallModelRunner, inspect_video, latency_rows, timeline_html


def sample_video(path: Path, *, frames: int = 30) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (160, 120))
    assert writer.isOpened()
    for n in range(frames):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        cv2.rectangle(frame, (10 + n, 30), (35 + n, 80), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()


class FakeModel:
    def __init__(self, *args, **kwargs):
        self.calls = 0
    def predict(self, source):
        self.calls += 1
        return {
            "protocol": "edgefall_seven_head_label_blind_runtime_v1",
            "source": str(source), "score": 0.749, "member_logits": [0.1] * 7,
            "frames": 30, "fps": 15.0,
            "proposal": {"track_id": 0, "start_time": 0.5, "peak_time": 1.0,
                         "end_time": 1.5, "motion_score": 0.4, "observed_frames": 25},
            "latency_ms": {"pose": 12, "skeleton": 4, "roi": 10,
                           "heads_and_fusion": 3, "total": 29, "per_frame": 0.97},
        }
    def close(self):
        pass


def test_timeline_is_proposal_not_frame_prediction():
    markup = timeline_html({"frames": 100, "fps": 10, "proposal": {
        "track_id": "<script>", "start_time": 3, "peak_time": 4, "end_time": 5}})
    assert "30.000%" in markup and "20.000%" in markup
    assert "逐帧跌倒判定" in markup
    assert "<script>" not in markup


def test_empty_and_oversized_video(tmp_path):
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    with pytest.raises(ValueError):
        inspect_video(empty)
    video = tmp_path / "small.mp4"
    sample_video(video)
    assert inspect_video(video)["frames"] == 30
    with pytest.raises(ValueError):
        inspect_video(video, max_seconds=1)


def test_fake_model_json_and_video_output(tmp_path):
    video = tmp_path / "sample.mp4"
    sample_video(video)
    runner = FallModelRunner(model_factory=FakeModel, output_root=tmp_path / "results", device="cpu")
    try:
        result = runner.run(video)
        exported = json.loads(Path(result["json_path"]).read_text(encoding="utf-8"))
        assert exported["score"] == 0.749
        assert len(exported["member_logits"]) == 7
        assert len(latency_rows(exported)) == 6
        assert Path(result["video_path"]).is_file(), result["warning"]
        cap = cv2.VideoCapture(result["video_path"])
        try:
            assert cap.isOpened() and cap.read()[0]
        finally:
            cap.release()
        assert "不是经过校准" in result["summary"]
    finally:
        runner.close()


def test_gradio_builds_without_loading_weights(tmp_path):
    from app import build_app
    runner = FallModelRunner(model_factory=FakeModel, output_root=tmp_path, device="cpu")
    try:
        demo = build_app(runner)
        assert demo is not None
    finally:
        runner.close()
