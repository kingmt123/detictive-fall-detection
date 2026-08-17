"""单模型姿态提取器的序列与生命周期测试。"""
import hashlib
from pathlib import Path

import numpy as np

from pipeline.pose_extractor import PoseExtractor


class _Tensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class _Boxes:
    def __init__(self, boxes, confidence):
        self.xyxy = _Tensor(np.asarray(boxes, dtype=np.float32))
        self.conf = _Tensor(np.asarray(confidence, dtype=np.float32))

    def __len__(self):
        return len(self.xyxy.value)


class _Keypoints:
    def __init__(self, value):
        self.data = _Tensor(np.asarray(value, dtype=np.float32))


class _Result:
    def __init__(self, boxes=None, confidence=None, keypoints=None):
        self.boxes = None if boxes is None else _Boxes(boxes, confidence)
        self.keypoints = None if keypoints is None else _Keypoints(keypoints)


class _FakeModel:
    def __init__(self):
        keypoints = np.zeros((2, 17, 3), dtype=np.float32)
        keypoints[:, :, 2] = 0.9
        self.results = [
            _Result(
                boxes=[[0, 0, 10, 20], [30, 0, 40, 20]],
                confidence=[0.9, 0.8],
                keypoints=keypoints,
            ),
            _Result(),
        ]
        self.predict_calls = 0

    def predict(self, **_kwargs):
        result = self.results[self.predict_calls % len(self.results)]
        self.predict_calls += 1
        return [result]


class _FakeCapture:
    def __init__(self):
        self.frames = [
            np.zeros((48, 64, 3), dtype=np.uint8),
            np.zeros((48, 64, 3), dtype=np.uint8),
        ]
        self.index = 0
        self.released = False

    def isOpened(self):
        return True

    def get(self, property_id):
        if property_id == 5:
            return 16.0
        if property_id == 7:
            return len(self.frames)
        return 0.0

    def read(self):
        if self.index >= len(self.frames):
            return False, None
        frame = self.frames[self.index].copy()
        self.index += 1
        return True, frame

    def release(self):
        self.released = True


def test_extractor_keeps_empty_frames_in_dense_variable_person_sequence(tmp_path: Path):
    model = _FakeModel()
    capture = _FakeCapture()
    extractor = PoseExtractor(
        model_path=tmp_path / "fake.pt",
        model_factory=lambda _path: model,
        capture_factory=lambda _path: capture,
    )

    sequence = extractor.extract(tmp_path / "clip.mp4")

    assert model.predict_calls == 2
    assert capture.released
    assert sequence.frame_indices.tolist() == [0, 1]
    assert sequence.timestamps.tolist() == [0.0, 1 / 16]
    assert sequence.keypoints.shape == (2, 2, 17, 3)
    assert sequence.bboxes.shape == (2, 2, 4)
    assert sequence.track_ids.tolist() == [[1, 2], [-1, -1]]
    assert sequence.valid_mask.tolist() == [[True, True], [False, False]]
    assert sequence.frame_size.tolist() == [48, 64]


def test_extractor_loads_model_lazily_once_and_resets_tracker_per_clip(
    tmp_path: Path,
):
    models: list[_FakeModel] = []
    captures: list[_FakeCapture] = []

    def model_factory(_path: str):
        model = _FakeModel()
        models.append(model)
        return model

    def capture_factory(_path: str):
        capture = _FakeCapture()
        captures.append(capture)
        return capture

    extractor = PoseExtractor(
        model_path=tmp_path / "fake.pt",
        model_factory=model_factory,
        capture_factory=capture_factory,
    )
    assert models == []

    first = extractor.extract(tmp_path / "a.mp4")
    second = extractor.extract(tmp_path / "b.mp4")

    assert len(models) == 1
    assert models[0].predict_calls == 4
    assert first.track_ids[0].tolist() == second.track_ids[0].tolist() == [1, 2]
    assert len(captures) == 2 and all(capture.released for capture in captures)


def test_extractor_signature_binds_model_bytes_code_and_behavior_config(tmp_path: Path):
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"model-v1")
    extractor = PoseExtractor(
        model_path=model_path,
        image_size=320,
        confidence=0.2,
        model_factory=lambda _path: _FakeModel(),
    )

    signature = extractor.cache_signature(crop="auto", max_frames=30)

    assert signature["protocol"] == "pose_extraction_v1"
    assert signature["cache_schema"] == 1
    assert signature["model_sha256"] == hashlib.sha256(b"model-v1").hexdigest()
    assert len(signature["implementation_sha256"]) == 64
    assert signature["image_size"] == 320
    assert signature["confidence"] == 0.2
    assert signature["crop"] == "auto"
    assert signature["max_frames"] == 30

    model_path.write_bytes(b"model-v2")
    assert extractor.cache_signature(crop="auto", max_frames=30) != signature
