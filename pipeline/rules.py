"""由姿态关键点和检测框计算轻量物理特征与规则分数。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PoseFeatures:
    aspect_ratio: float
    verticality: float
    center_y: float
    center_velocity: float
    keypoint_coverage: float
    torso_observed: bool = True


def compute_pose_features(
    keypoints: np.ndarray,
    bbox_xyxy: np.ndarray,
    frame_height: int,
    previous_center_y: float | None = None,
    delta_seconds: float | None = None,
    confidence_threshold: float = 0.2,
) -> PoseFeatures:
    """计算归一化姿态特征；center_velocity 为每秒归一化向下速度。"""
    x1, y1, x2, y2 = map(float, bbox_xyxy)
    width = max(0.0, x2 - x1)
    height = max(1e-6, y2 - y1)
    aspect_ratio = width / height

    confidence = keypoints[:, 2]
    valid = confidence >= confidence_threshold
    coverage = float(valid.mean())

    shoulder_ids = [5, 6]
    hip_ids = [11, 12]
    shoulders = [keypoints[i, :2] for i in shoulder_ids if valid[i]]
    hips = [keypoints[i, :2] for i in hip_ids if valid[i]]
    if shoulders and hips:
        shoulder = np.mean(shoulders, axis=0)
        hip = np.mean(hips, axis=0)
        dx, dy = float(hip[0] - shoulder[0]), float(hip[1] - shoulder[1])
        verticality = abs(dy) / max((dx * dx + dy * dy) ** 0.5, 1e-6)
        torso_observed = True
    else:
        verticality = 0.5
        torso_observed = False

    center_y = ((y1 + y2) / 2.0) / max(frame_height, 1)
    if previous_center_y is None:
        center_velocity = 0.0
    else:
        if delta_seconds is None or delta_seconds <= 0:
            raise ValueError("计算速度时 delta_seconds 必须大于 0")
        center_velocity = max(0.0, center_y - previous_center_y) / delta_seconds
    return PoseFeatures(
        aspect_ratio=aspect_ratio,
        verticality=verticality,
        center_y=center_y,
        center_velocity=center_velocity,
        keypoint_coverage=coverage,
        torso_observed=torso_observed,
    )


def rule_score(features: PoseFeatures) -> float:
    """物理规则跌倒分数。

    水平姿态只能提供中等分数；需要向下动态才跨过默认告警阈值，避免把
    有意躺卧的静态姿态直接判为跌倒。规则是无训练基线，后续由 TCN 替代主判据。
    """
    if features.keypoint_coverage < 0.25 or not features.torso_observed:
        return 0.0
    horizontal = float(np.clip((features.aspect_ratio - 0.8) / 1.6, 0.0, 1.0))
    tilted = float(np.clip((0.75 - features.verticality) / 0.65, 0.0, 1.0))
    downward = float(np.clip(features.center_velocity / 1.5, 0.0, 1.0))
    ground_level = float(np.clip((features.center_y - 0.45) / 0.35, 0.0, 1.0))
    score = 0.22 * horizontal + 0.18 * tilted + 0.45 * downward + 0.10 * ground_level
    return float(np.clip(score, 0.0, 1.0))
