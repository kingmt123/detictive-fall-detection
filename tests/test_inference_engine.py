"""可复用批量推理引擎测试。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pipeline.inference_engine import InferenceEngine


class _EmptyResult:
    boxes = None
    keypoints = None


class _FakeModel:
    def __init__(self):
        self.predict_calls = 0

    def predict(self, **kwargs):
        self.predict_calls += 1
        return [_EmptyResult()]


class _FakeCapture:
    def __init__(self, frames: list[np.ndarray], fps: float = 30.0):
        self.frames = frames
        self.fps = fps
        self.index = 0
        self.released = False

    def isOpened(self):
        return True

    def get(self, property_id):
        # OpenCV CAP_PROP_FPS=5, CAP_PROP_FRAME_COUNT=7.
        if property_id == 5:
            return self.fps
        if property_id == 7:
            return len(self.frames)
        return 0.0

    def read(self):
        if self.index >= len(self.frames):
            return False, None
        frame = self.frames[self.index].copy()
        self.index += 1
        return True, frame

    def set(self, property_id, value):
        # OpenCV CAP_PROP_POS_FRAMES=1.
        if property_id == 1:
            self.index = int(value)
        return True

    def release(self):
        self.released = True


def test_engine_reuses_one_model_and_supports_render_false(monkeypatch, tmp_path: Path):
    import pipeline.inference_engine as module

    frames = [np.zeros((48, 64, 3), dtype=np.uint8) for _ in range(2)]
    captures: list[_FakeCapture] = []

    def capture_factory(_source):
        capture = _FakeCapture(frames)
        captures.append(capture)
        return capture

    models: list[_FakeModel] = []

    def model_factory(_path):
        model = _FakeModel()
        models.append(model)
        return model

    monkeypatch.setattr(module.cv2, "VideoCapture", capture_factory)
    engine = InferenceEngine(model_path="fake.pt", model_factory=model_factory)

    first = engine.analyze(tmp_path / "a.mp4", render=False)
    second = engine.analyze(tmp_path / "b.mp4", render=False)

    assert len(models) == 1
    assert models[0].predict_calls == 4
    signature = engine.cache_signature()
    assert signature["protocol"] == "pose_motion_rule_baseline_v2"
    assert len(signature["implementation_sha256"]) == 64
    assert signature["event_aggregator"]["th_hi"] == 0.5
    assert len(captures) == 2 and all(capture.released for capture in captures)
    assert first["processed_frames"] == second["processed_frames"] == 2
    assert first["events"] == second["events"] == []
    assert first["protocol"] == "pose_motion_rule_baseline_v2"
    assert "stage_latency_ms" in first
    assert "frame_end_to_end" in first["stage_latency_ms"]
    assert "cpu_transfer" in first["stage_latency_ms"]
    assert "aggregate" in first["stage_latency_ms"]
    assert first["detector_predict_wallclock_ms"] == first["stage_latency_ms"]["predict"]
    assert not list(tmp_path.glob("*.rendered.mp4"))


def test_render_true_requires_output_video(tmp_path: Path):
    engine = InferenceEngine(model_path="fake.pt", model_factory=lambda _: _FakeModel())
    with pytest.raises(ValueError, match="output_video"):
        engine.analyze(tmp_path / "input.mp4", render=True)
