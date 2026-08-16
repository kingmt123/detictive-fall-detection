"""纯函数姿态特征与物理规则测试。"""
import numpy as np

from pipeline.rules import PoseFeatures, compute_pose_features, rule_score


def make_pose(horizontal: bool) -> tuple[np.ndarray, np.ndarray]:
    keypoints = np.zeros((17, 3), dtype=np.float32)
    keypoints[:, 2] = 0.95
    if horizontal:
        # 鼻、肩、髋沿 x 方向展开，身体主轴接近水平。
        keypoints[0, :2] = (80, 100)
        keypoints[5, :2] = (100, 100)
        keypoints[6, :2] = (100, 110)
        keypoints[11, :2] = (140, 100)
        keypoints[12, :2] = (140, 110)
        bbox = np.array([60, 80, 180, 130], dtype=np.float32)
    else:
        keypoints[0, :2] = (100, 40)
        keypoints[5, :2] = (90, 70)
        keypoints[6, :2] = (110, 70)
        keypoints[11, :2] = (92, 130)
        keypoints[12, :2] = (108, 130)
        bbox = np.array([70, 30, 130, 180], dtype=np.float32)
    return keypoints, bbox


def test_horizontal_pose_has_high_aspect_and_low_verticality():
    keypoints, bbox = make_pose(horizontal=True)
    features = compute_pose_features(keypoints, bbox, frame_height=240)
    assert features.aspect_ratio > 2.0
    assert features.verticality < 0.3
    assert features.keypoint_coverage > 0.9


def test_upright_pose_has_low_aspect_and_high_verticality():
    keypoints, bbox = make_pose(horizontal=False)
    features = compute_pose_features(keypoints, bbox, frame_height=240)
    assert features.aspect_ratio < 0.6
    assert features.verticality > 0.8


def test_center_velocity_is_normalized_by_elapsed_seconds():
    keypoints, bbox = make_pose(horizontal=False)
    fast_sample = compute_pose_features(
        keypoints,
        bbox,
        frame_height=240,
        previous_center_y=0.2375,
        delta_seconds=0.20,
    )
    dense_sample = compute_pose_features(
        keypoints,
        bbox,
        frame_height=240,
        previous_center_y=0.3375,
        delta_seconds=0.10,
    )
    assert abs(fast_sample.center_velocity - 1.0) < 1e-9
    assert abs(dense_sample.center_velocity - 1.0) < 1e-9


def test_rule_score_needs_dynamics_not_only_horizontal_posture():
    lying_still = PoseFeatures(2.4, 0.1, 0.75, 0.0, 1.0)
    falling_fast = PoseFeatures(2.4, 0.1, 0.75, 2.5, 1.0)
    assert rule_score(lying_still) < 0.6
    assert rule_score(falling_fast) > rule_score(lying_still)
    assert rule_score(falling_fast) >= 0.6


def test_rule_score_rejects_missing_pose():
    missing = PoseFeatures(3.0, 0.0, 0.8, 0.3, 0.1)
    assert rule_score(missing) == 0.0


def test_missing_shoulder_and_hip_axis_is_marked_unobservable():
    keypoints, bbox = make_pose(horizontal=False)
    keypoints[[5, 6, 11, 12], 2] = 0.0
    features = compute_pose_features(
        keypoints,
        bbox,
        frame_height=240,
    )
    assert features.keypoint_coverage > 0.25
    assert features.torso_observed is False
    assert rule_score(features) == 0.0
