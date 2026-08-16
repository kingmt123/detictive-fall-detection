"""主目标姿态跟踪与 TCN 输入归一化测试。"""
import numpy as np
import pytest

from pipeline.pose_track import (
    MultiPoseTracker,
    PrimaryPoseTracker,
    normalize_keypoints,
)


def test_multi_tracker_rejects_non_finite_center_gate():
    with pytest.raises(ValueError):
        MultiPoseTracker(max_center_distance=float("nan"))


def test_tracker_keeps_same_person_when_another_has_higher_confidence():
    tracker = PrimaryPoseTracker(max_misses=2)
    boxes1 = np.array([[10, 10, 60, 110], [200, 10, 250, 110]], dtype=np.float32)
    conf1 = np.array([0.9, 0.7], dtype=np.float32)
    keypoints = np.ones((2, 17, 3), dtype=np.float32)
    first = tracker.update(boxes1, conf1, keypoints)
    assert first is not None and first.index == 0
    assert first.track_switched is True

    # 第二帧右侧陌生人置信度更高，仍应保持与前一框 IoU 高的左侧人物。
    boxes2 = np.array([[12, 12, 62, 112], [200, 10, 250, 110]], dtype=np.float32)
    conf2 = np.array([0.6, 0.99], dtype=np.float32)
    second = tracker.update(boxes2, conf2, keypoints)
    assert second is not None and second.index == 0
    assert second.track_switched is False


def test_multi_tracker_keeps_two_people_as_independent_tracks():
    tracker = MultiPoseTracker(max_misses=2, min_iou=0.1)
    keypoints = np.zeros((2, 17, 3), dtype=np.float32)
    first = tracker.update(
        np.array([[0, 0, 10, 20], [50, 0, 60, 20]], dtype=np.float32),
        np.array([0.9, 0.8], dtype=np.float32),
        keypoints,
    )
    assert [pose.track_id for pose in first] == [1, 2]

    second = tracker.update(
        np.array([[51, 0, 61, 20], [1, 0, 11, 20]], dtype=np.float32),
        np.array([0.99, 0.5], dtype=np.float32),
        keypoints,
    )
    by_track = {pose.track_id: pose for pose in second}
    assert by_track[1].index == 1
    assert by_track[2].index == 0
    assert not by_track[1].track_switched
    assert not by_track[2].track_switched


def test_multi_tracker_exposes_missed_track_until_lifecycle_expires():
    tracker = MultiPoseTracker(max_misses=1)
    keypoints = np.zeros((1, 17, 3), dtype=np.float32)
    tracker.update(
        np.array([[0, 0, 10, 20]], dtype=np.float32),
        np.array([0.9], dtype=np.float32),
        keypoints,
    )
    empty_boxes = np.empty((0, 4), dtype=np.float32)
    empty_conf = np.empty(0, dtype=np.float32)
    empty_keypoints = np.empty((0, 17, 3), dtype=np.float32)
    assert tracker.update(empty_boxes, empty_conf, empty_keypoints) == []
    assert tracker.active_track_ids == {1}
    tracker.update(empty_boxes, empty_conf, empty_keypoints)
    assert tracker.active_track_ids == set()


def test_multi_tracker_does_not_transfer_id_across_zero_iou_jump():
    tracker = MultiPoseTracker(max_misses=2, min_iou=0.1, max_center_distance=1.5)
    keypoints = np.zeros((1, 17, 3), dtype=np.float32)
    first = tracker.update(
        np.array([[0, 0, 10, 20]], dtype=np.float32),
        np.array([0.9], dtype=np.float32),
        keypoints,
    )
    second = tracker.update(
        np.array([[30, 0, 40, 20]], dtype=np.float32),
        np.array([0.9], dtype=np.float32),
        keypoints,
    )
    assert first[0].track_id == 1
    assert second[0].track_id == 2
    assert second[0].track_switched is True
    assert tracker.active_track_ids == {1, 2}


def test_multi_tracker_uses_motion_prediction_during_crossing():
    tracker = MultiPoseTracker(max_center_distance=1.5)
    keypoints = np.zeros((2, 17, 3), dtype=np.float32)
    confidence = np.array([0.9, 0.9], dtype=np.float32)
    tracker.update(
        np.array([[0, 0, 20, 20], [40, 0, 60, 20]], dtype=np.float32),
        confidence,
        keypoints,
    )
    tracker.update(
        np.array([[8, 0, 28, 20], [32, 0, 52, 20]], dtype=np.float32),
        confidence,
        keypoints,
    )
    crossed = tracker.update(
        # 检测顺序是“向左的人、向右的人”，ID 应沿运动方向而非旧位置延续。
        np.array([[24, 0, 44, 20], [16, 0, 36, 20]], dtype=np.float32),
        confidence,
        keypoints,
    )
    by_track = {pose.track_id: pose for pose in crossed}
    assert by_track[1].index == 1
    assert by_track[2].index == 0


def test_multi_tracker_does_not_inherit_nearby_replacement_identity():
    tracker = MultiPoseTracker(max_misses=2, min_iou=0.1)
    keypoints = np.zeros((1, 17, 3), dtype=np.float32)
    first = tracker.update(
        np.array([[0, 0, 10, 20]], dtype=np.float32),
        np.array([0.9], dtype=np.float32),
        keypoints,
    )
    replacement = tracker.update(
        np.array([[20, 0, 30, 20]], dtype=np.float32),
        np.array([0.9], dtype=np.float32),
        keypoints,
    )
    assert first[0].track_id == 1
    assert replacement[0].track_id == 2
    assert replacement[0].track_switched is True


def test_tracker_marks_low_iou_reacquisition_as_track_switch():
    tracker = PrimaryPoseTracker(max_misses=2)
    keypoints = np.ones((1, 17, 3), dtype=np.float32)
    first = tracker.update(
        np.array([[10, 10, 60, 110]], dtype=np.float32),
        np.array([0.9], dtype=np.float32),
        keypoints,
    )
    second = tracker.update(
        np.array([[220, 10, 270, 110]], dtype=np.float32),
        np.array([0.95], dtype=np.float32),
        keypoints,
    )
    assert first is not None and first.track_switched is True
    assert second is not None and second.track_switched is True


def test_tracker_resets_after_consecutive_misses():
    tracker = PrimaryPoseTracker(max_misses=1)
    boxes = np.array([[10, 10, 60, 110]], dtype=np.float32)
    conf = np.array([0.9], dtype=np.float32)
    keypoints = np.ones((1, 17, 3), dtype=np.float32)
    tracker.update(boxes, conf, keypoints)
    assert tracker.update(np.empty((0, 4)), np.empty(0), np.empty((0, 17, 3))) is None
    assert tracker.previous_box is not None
    assert tracker.update(np.empty((0, 4)), np.empty(0), np.empty((0, 17, 3))) is None
    assert tracker.previous_box is None


def test_normalize_keypoints_is_translation_and_scale_invariant():
    keypoints = np.zeros((17, 3), dtype=np.float32)
    keypoints[:, 0] = np.linspace(100, 200, 17)
    keypoints[:, 1] = np.linspace(50, 150, 17)
    keypoints[:, 2] = 0.8
    bbox = np.array([90, 40, 210, 160], dtype=np.float32)
    normalized = normalize_keypoints(keypoints, bbox)

    moved = keypoints.copy()
    moved[:, :2] = moved[:, :2] * 2 + np.array([300, 100])
    moved_bbox = bbox * 2 + np.array([300, 100, 300, 100])
    normalized_moved = normalize_keypoints(moved, moved_bbox)
    assert normalized.shape == (17, 3)
    assert np.allclose(normalized, normalized_moved, atol=1e-6)
    assert np.all((normalized[:, :2] >= 0) & (normalized[:, :2] <= 1))
