"""FallTCN checkpoint 加载与逐帧因果在线窗口状态。"""
from __future__ import annotations

import hashlib
import json
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch

from models.tcn import FallTCN
from models.tcn_window import normalize_pose


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repository_path(project_root: Path, relative_path: str) -> Path:
    normalized_parts = relative_path.replace("\\", "/").split("/")
    if not normalized_parts or any(
        part in {"", ".", ".."} for part in normalized_parts
    ):
        raise ValueError(f"TCN 训练代码路径非法: {relative_path}")
    return project_root.joinpath(*normalized_parts)


class OnlineFallTCNScorer:
    """逐帧维护每个 track 的窗口；只在当前人物可见且满足观测数时预测。"""

    def __init__(
        self,
        model: FallTCN,
        *,
        device: torch.device,
        window_size: int = 16,
        min_observed_frames: int = 8,
    ) -> None:
        if window_size < 1 or not 1 <= min_observed_frames <= window_size:
            raise ValueError("在线 TCN 窗口参数无效")
        self.model = model.to(device).eval()
        self.device = device
        self.window_size = window_size
        self.min_observed_frames = min_observed_frames
        self.features: dict[int, deque[np.ndarray]] = {}
        self.observed: dict[int, deque[bool]] = {}
        self.next_frame_index = 0

    @classmethod
    def from_artifacts(
        cls,
        checkpoint_path: Path,
        run_path: Path,
        *,
        device: torch.device,
    ) -> OnlineFallTCNScorer:
        try:
            run = json.loads(Path(run_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("TCN run.json 无效") from exc
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        if checkpoint.get("run_signature_sha256") != run.get("signature_sha256"):
            raise ValueError("TCN checkpoint 与 run signature 不匹配")
        if run.get("pilot"):
            raise ValueError("在线推理拒绝 pilot checkpoint")
        project_root = Path(__file__).parent.parent
        for relative_path, expected in run["signature"]["code_sha256"].items():
            # run.json 可能由 Windows 训练机生成，而正式 V100 环境通常是 Linux。
            # 签名中的逻辑仓库路径统一使用 POSIX 分隔语义，避免把反斜杠当文件名。
            path = _repository_path(project_root, relative_path)
            if not path.is_file() or _sha256_file(path) != expected:
                raise ValueError(f"TCN 训练代码哈希漂移: {relative_path}")
        config = run["signature"]["config"]
        model = FallTCN(
            channels=tuple(config["channels"]),
            kernel=int(config["kernel"]),
            dropout=float(config["dropout"]),
        )
        model.load_state_dict(checkpoint["model_state"])
        return cls(
            model,
            device=device,
            window_size=int(config["window_size"]),
            min_observed_frames=int(config["min_observed_frames"]),
        )

    def reset(self) -> None:
        self.features.clear()
        self.observed.clear()
        self.next_frame_index = 0

    @torch.inference_mode()
    def score_frame(
        self,
        frame_index: int,
        observations: dict[int, tuple[np.ndarray, np.ndarray]],
        *,
        active_track_ids: set[int],
    ) -> dict[int, float | None]:
        if frame_index != self.next_frame_index:
            raise ValueError(
                f"TCN frame index 必须连续: expected={self.next_frame_index}, "
                f"actual={frame_index}"
            )
        self.next_frame_index += 1
        zero = np.zeros((17, 3), dtype=np.float32)
        for track_id in list(self.features):
            self.features[track_id].append(zero.copy())
            self.observed[track_id].append(False)
        for track_id, (keypoints, bbox) in observations.items():
            if track_id not in self.features:
                prior = min(frame_index, self.window_size - 1)
                self.features[track_id] = deque(
                    (zero.copy() for _ in range(prior)), maxlen=self.window_size
                )
                self.observed[track_id] = deque(
                    (False for _ in range(prior)), maxlen=self.window_size
                )
                self.features[track_id].append(zero.copy())
                self.observed[track_id].append(False)
            self.features[track_id][-1] = normalize_pose(keypoints, bbox)
            self.observed[track_id][-1] = True
        keep_ids = active_track_ids | set(observations)
        for track_id in set(self.features) - keep_ids:
            del self.features[track_id]
            del self.observed[track_id]

        result = {track_id: None for track_id in observations}
        candidates = [
            track_id
            for track_id in sorted(observations)
            if len(self.features[track_id]) == self.window_size
            and sum(self.observed[track_id]) >= self.min_observed_frames
        ]
        if not candidates:
            return result
        batch = np.stack(
            [np.stack(self.features[track_id]) for track_id in candidates]
        ).astype(np.float32, copy=False)
        probabilities = torch.sigmoid(
            self.model(torch.from_numpy(batch).to(self.device))
        ).cpu().numpy()
        for track_id, probability in zip(candidates, probabilities, strict=True):
            result[track_id] = float(probability)
        return result

    def artifact_signature(self) -> dict[str, Any]:
        return {
            "window_size": self.window_size,
            "min_observed_frames": self.min_observed_frames,
            "device": str(self.device),
        }
