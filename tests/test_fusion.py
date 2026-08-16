"""时序规则融合与图像裁剪测试。"""
from infer import crop_frame, sanitize_fps
from pipeline.fusion import TemporalFallScorer


class Feature:
    def __init__(
        self,
        center_y,
        center_velocity=0.0,
        coverage=1.0,
        aspect_ratio=0.5,
        verticality=0.9,
        torso_observed=True,
    ):
        self.center_y = center_y
        self.center_velocity = center_velocity
        self.keypoint_coverage = coverage
        self.aspect_ratio = aspect_ratio
        self.verticality = verticality
        self.torso_observed = torso_observed


def test_temporal_scorer_detects_large_downward_displacement():
    scorer = TemporalFallScorer(window_seconds=1.2)
    scores = []
    centers = [0.10, 0.12, 0.16, 0.24, 0.36, 0.50]
    verticalities = [0.95, 0.90, 0.75, 0.45, 0.25, 0.20]
    for i, (center, verticality) in enumerate(zip(centers, verticalities)):
        scores.append(
            scorer.update(
                i * 0.2,
                Feature(center, aspect_ratio=1.2, verticality=verticality),
            )
        )
    assert max(scores) >= 0.6


def test_temporal_scorer_rejects_walking_toward_camera_while_upright():
    scorer = TemporalFallScorer(window_seconds=1.2)
    scores = []
    for i, center in enumerate([0.10, 0.12, 0.16, 0.24, 0.36, 0.50]):
        scores.append(scorer.update(i * 0.2, Feature(center, verticality=0.95)))
    assert max(scores) < 0.6


def test_temporal_scorer_rejects_slow_small_posture_change():
    scorer = TemporalFallScorer(window_seconds=1.2)
    scores = []
    for i, center in enumerate([0.40, 0.42, 0.44, 0.46, 0.48, 0.50]):
        scores.append(scorer.update(i * 0.2, Feature(center)))
    assert max(scores) < 0.6


def test_temporal_scorer_resets_after_detection_gap():
    scorer = TemporalFallScorer(window_seconds=1.2, reset_gap_seconds=0.5)
    scorer.update(0.0, Feature(0.1))
    scorer.update(0.2, Feature(0.2))
    score = scorer.update(1.0, Feature(0.6))
    assert score < 0.6


def test_temporal_scorer_rejects_low_coverage():
    scorer = TemporalFallScorer(window_seconds=1.2)
    scorer.update(0.0, Feature(0.1))
    assert scorer.update(0.2, Feature(0.6, coverage=0.1)) is None


def test_temporal_scorer_treats_missing_torso_as_unknown_without_history_update():
    scorer = TemporalFallScorer()
    assert scorer.update(0.0, Feature(0.2, torso_observed=False)) is None
    assert list(scorer.history) == []


def test_invalid_fps_falls_back_to_default():
    assert sanitize_fps(float("nan")) == 25.0
    assert sanitize_fps(-1.0) == 25.0
    assert sanitize_fps(0.0) == 25.0
    assert sanitize_fps(30.0) == 30.0


def test_auto_crop_uses_rgb_half_for_urfd_shape():
    import numpy as np

    frame = np.zeros((240, 640, 3), dtype=np.uint8)
    frame[:, 320:] = 255
    crop, offset = crop_frame(frame, "auto")
    assert crop.shape == (240, 320, 3)
    assert offset == (320, 0)
    assert crop.mean() == 255


def test_auto_crop_keeps_normal_video_unchanged():
    import numpy as np

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    crop, offset = crop_frame(frame, "auto")
    assert crop.shape == frame.shape
    assert offset == (0, 0)


def test_auto_crop_accepts_1080p_camera_input_without_splitting():
    import numpy as np

    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    crop, offset = crop_frame(frame, "auto")
    assert crop.shape == (1080, 1920, 3)
    assert offset == (0, 0)
