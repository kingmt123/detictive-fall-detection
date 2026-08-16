"""轻量时序规则融合：累计下坠 + 单帧物理规则。"""
from __future__ import annotations

from collections import deque

import numpy as np

from pipeline.rules import PoseFeatures, rule_score


class TemporalFallScorer:
    """保留短时间姿态历史，用累计向下位移识别快速跌倒。

    单帧质心变化容易受检测框抖动影响，因此用约 1 秒窗口内的净下坠作为
    主动态证据；单帧物理规则仅作为辅助。
    """

    def __init__(
        self,
        window_seconds: float = 1.2,
        reset_gap_seconds: float = 0.5,
        displacement_threshold: float = 0.22,
    ):
        self.window_seconds = window_seconds
        self.reset_gap_seconds = reset_gap_seconds
        self.displacement_threshold = displacement_threshold
        self.history: deque[tuple[float, float]] = deque()
        self.last_timestamp: float | None = None

    def reset(self) -> None:
        self.history.clear()
        self.last_timestamp = None

    def update(self, timestamp: float, features: PoseFeatures) -> float | None:
        if features.keypoint_coverage < 0.25 or not features.torso_observed:
            return None
        if (
            self.last_timestamp is not None
            and timestamp - self.last_timestamp > self.reset_gap_seconds
        ):
            self.history.clear()
        self.last_timestamp = timestamp
        self.history.append((timestamp, float(features.center_y)))
        while self.history and timestamp - self.history[0][0] > self.window_seconds:
            self.history.popleft()

        baseline = min(center_y for _, center_y in self.history)
        displacement = max(0.0, float(features.center_y) - baseline)
        dynamic = float(
            np.clip(
                (displacement - 0.08) / max(self.displacement_threshold - 0.08, 1e-6),
                0.0,
                1.0,
            )
        )
        posture_change = float(
            np.clip((0.80 - features.verticality) / 0.60, 0.0, 1.0)
        )
        physical = rule_score(features)
        # 只有“快速下坠 + 躯干由竖直转为倾斜”共同出现才给高分；
        # 单纯走向摄像机也会造成 2D 质心下移，但人体仍保持竖直。
        score = 0.72 * dynamic * posture_change + 0.28 * physical
        return float(np.clip(score, 0.0, 1.0))
