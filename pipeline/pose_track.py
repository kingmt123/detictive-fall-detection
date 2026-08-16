"""轻量主目标姿态跟踪与关键点归一化。"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class TrackedPose:
    index: int
    box: np.ndarray
    confidence: float
    keypoints: np.ndarray
    track_switched: bool
    track_id: int = 0


@dataclass
class _TrackState:
    box: np.ndarray
    misses: int = 0
    velocity: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=np.float32)
    )


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = np.maximum(a[:2], b[:2])
    x2, y2 = np.minimum(a[2:], b[2:])
    intersection = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _box_center(box: np.ndarray) -> np.ndarray:
    return (box[:2] + box[2:]) / 2.0


def _shift_box(box: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    shifted = box.astype(np.float32, copy=True)
    shifted[[0, 2]] += velocity[0]
    shifted[[1, 3]] += velocity[1]
    return shifted


def _normalized_center_distance(a: np.ndarray, b: np.ndarray) -> float:
    diagonal_a = float(np.linalg.norm(a[2:] - a[:2]))
    diagonal_b = float(np.linalg.norm(b[2:] - b[:2]))
    scale = max(diagonal_a, diagonal_b, 1e-6)
    return float(np.linalg.norm(_box_center(a) - _box_center(b)) / scale)


class PrimaryPoseTracker:
    """IoU 优先、置信度兜底的单主目标跟踪器。"""

    def __init__(self, max_misses: int = 10, min_iou: float = 0.1):
        self.max_misses = max_misses
        self.min_iou = min_iou
        self.previous_box: np.ndarray | None = None
        self.misses = 0

    def update(
        self,
        boxes: np.ndarray,
        confidences: np.ndarray,
        keypoints: np.ndarray,
    ) -> TrackedPose | None:
        if len(boxes) == 0:
            self.misses += 1
            if self.misses > self.max_misses:
                self.previous_box = None
            return None

        if self.previous_box is None:
            index = int(np.argmax(confidences))
            track_switched = True
        else:
            ious = np.array([box_iou(self.previous_box, box) for box in boxes])
            best = int(np.argmax(ious))
            track_switched = bool(ious[best] < self.min_iou)
            index = best if not track_switched else int(np.argmax(confidences))

        self.previous_box = boxes[index].astype(np.float32, copy=True)
        self.misses = 0
        return TrackedPose(
            index=index,
            box=self.previous_box,
            confidence=float(confidences[index]),
            keypoints=keypoints[index].astype(np.float32, copy=True),
            track_switched=track_switched,
        )


class MultiPoseTracker:
    """IoU + 常速度中心预测的轻量多人轨迹生命周期管理器。"""

    def __init__(
        self,
        max_misses: int = 10,
        min_iou: float = 0.1,
        max_center_distance: float = 1.5,
    ):
        if max_misses < 0:
            raise ValueError("max_misses 不能为负数")
        if not 0.0 <= min_iou <= 1.0:
            raise ValueError("min_iou 必须在 [0,1] 内")
        if not np.isfinite(max_center_distance) or max_center_distance <= 0.0:
            raise ValueError("max_center_distance 必须大于 0")
        self.max_misses = max_misses
        self.min_iou = min_iou
        self.max_center_distance = max_center_distance
        self._tracks: dict[int, _TrackState] = {}
        self._next_track_id = 1

    @property
    def active_track_ids(self) -> set[int]:
        return set(self._tracks)

    def update(
        self,
        boxes: np.ndarray,
        confidences: np.ndarray,
        keypoints: np.ndarray,
    ) -> list[TrackedPose]:
        if not (len(boxes) == len(confidences) == len(keypoints)):
            raise ValueError("boxes/confidences/keypoints 数量必须一致")

        candidates = []
        for track_id, state in self._tracks.items():
            predicted_box = _shift_box(state.box, state.velocity)
            for detection_i in range(len(boxes)):
                iou = box_iou(predicted_box, boxes[detection_i])
                distance = _normalized_center_distance(
                    predicted_box, boxes[detection_i]
                )
                # 中心距离仅作为运动预测后的附加门限，不能在零 IoU 时单独
                # 继承身份；否则附近新人物会静默接管旧人的时序状态。
                if iou >= self.min_iou and distance <= self.max_center_distance:
                    candidates.append(
                        (distance - iou, -iou, distance, track_id, detection_i)
                    )
        candidates.sort()
        track_to_detection: dict[int, int] = {}
        assigned_detections: set[int] = set()
        for _, _, _, track_id, detection_i in candidates:
            if track_id in track_to_detection or detection_i in assigned_detections:
                continue
            track_to_detection[track_id] = detection_i
            assigned_detections.add(detection_i)

        for track_id in list(self._tracks):
            if track_id in track_to_detection:
                detection_i = track_to_detection[track_id]
                state = self._tracks[track_id]
                next_box = boxes[detection_i].astype(np.float32, copy=True)
                state.velocity = (_box_center(next_box) - _box_center(state.box)).astype(
                    np.float32
                )
                state.box = next_box
                state.misses = 0
            else:
                self._tracks[track_id].misses += 1
                if self._tracks[track_id].misses > self.max_misses:
                    del self._tracks[track_id]

        new_track_ids: set[int] = set()
        for detection_i in range(len(boxes)):
            if detection_i in assigned_detections:
                continue
            track_id = self._next_track_id
            self._next_track_id += 1
            self._tracks[track_id] = _TrackState(
                box=boxes[detection_i].astype(np.float32, copy=True)
            )
            track_to_detection[track_id] = detection_i
            new_track_ids.add(track_id)

        observed = []
        for track_id, detection_i in sorted(track_to_detection.items()):
            observed.append(
                TrackedPose(
                    index=detection_i,
                    box=self._tracks[track_id].box.copy(),
                    confidence=float(confidences[detection_i]),
                    keypoints=keypoints[detection_i].astype(np.float32, copy=True),
                    track_switched=track_id in new_track_ids,
                    track_id=track_id,
                )
            )
        return observed


def normalize_keypoints(keypoints: np.ndarray, bbox_xyxy: np.ndarray) -> np.ndarray:
    """把关键点坐标归一化到人体检测框，置信度保持不变。"""
    x1, y1, x2, y2 = map(float, bbox_xyxy)
    width, height = max(x2 - x1, 1e-6), max(y2 - y1, 1e-6)
    normalized = keypoints.astype(np.float32, copy=True)
    normalized[:, 0] = np.clip((normalized[:, 0] - x1) / width, 0.0, 1.0)
    normalized[:, 1] = np.clip((normalized[:, 1] - y1) / height, 0.0, 1.0)
    return normalized
