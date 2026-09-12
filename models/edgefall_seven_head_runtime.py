"""Label-blind raw-video runtime for the frozen seven-head EdgeFall model."""
from __future__ import annotations

import hashlib
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from models.event_proposal import PROTOCOL as EVENT_PROPOSAL_PROTOCOL
from models.event_proposal import propose_motion_event
from models.tcn_window import build_tcn_windows
from pipeline.pose_cache import PoseCacheRecord
from pipeline.pose_extractor import PoseExtractor
from tools.benchmark_edgefall_seven_head_tail import _head
from tools.build_tcn_multistream_sidecar import _bbox_features
from tools.build_yolo_roi_token_cache import (
    SharedYoloRoiEncoder,
    YoloRoiCacheConfig,
    decode_yolo_frames,
)
from tools.train_edgefall_f1 import F1Config, _state_dict_sha256, build_skeleton_model

PROTOCOL = "edgefall_seven_head_label_blind_runtime_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pose_record(sequence: Any, source: Path) -> PoseCacheRecord:
    """Adapt an in-memory pose sequence without adding labels or event metadata."""
    return PoseCacheRecord(
        clip_id=source.stem,
        dataset="deployment",
        split="unlabelled",
        source_identity={"path": source.name},
        source_content_sha256="0" * 64,
        extractor_signature={"protocol": "in_memory"},
        fps=float(sequence.fps),
        frame_indices=sequence.frame_indices,
        timestamps=sequence.timestamps,
        keypoints=sequence.keypoints,
        bboxes=sequence.bboxes,
        track_ids=sequence.track_ids,
        valid_mask=sequence.valid_mask,
        frame_size=sequence.frame_size,
    )


def skeleton_windows(record: PoseCacheRecord, *, size: int, minimum: int) -> torch.Tensor:
    windows = build_tcn_windows(
        record,
        (),
        window_size=size,
        stride=1,
        min_observed_frames=minimum,
        causal_left_pad=size == 48,
    )
    if not windows:
        raise ValueError(f"视频没有满足条件的 {size} 帧姿态窗口")
    pose = torch.from_numpy(np.stack([window.features for window in windows]))
    bbox = np.stack([_bbox_features(record, window) for window in windows])
    return torch.cat((pose.flatten(2), torch.from_numpy(bbox)), dim=-1)


def equal_logit_score(logits: torch.Tensor) -> torch.Tensor:
    if logits.shape != (7,) or not torch.isfinite(logits).all():
        raise ValueError("七头 logits 必须为有限的 [7]")
    return torch.sigmoid(logits.mean())


class SevenHeadVideoModel:
    """Load once and score multiple unlabelled videos with fixed equal-logit fusion."""

    def __init__(
        self,
        artifact: Path,
        yolo_checkpoint: Path,
        *,
        device: str = "cuda:0",
        pose_image_size: int = 640,
        pose_confidence: float = 0.10,
    ) -> None:
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("请求 CUDA，但当前不可用")
        payload = torch.load(artifact, map_location="cpu", weights_only=False)
        if payload.get("protocol") != PROTOCOL:
            raise ValueError("七头部署 artifact protocol 不匹配")
        if payload.get("event_proposal_protocol") != EVENT_PROPOSAL_PROTOCOL:
            raise ValueError("七头部署 event proposal protocol 不匹配")
        if _sha256(yolo_checkpoint) != payload.get("yolo_sha256"):
            raise ValueError("YOLO checkpoint hash 不匹配")
        config = F1Config(**payload["config"])
        config.validate()
        if _state_dict_sha256(payload["skeleton_model_state"]) != payload.get(
            "skeleton_state_sha256"
        ):
            raise ValueError("七头部署 skeleton hash 不匹配")
        self.skeleton = build_skeleton_model(config).to(self.device).eval()
        self.skeleton.load_state_dict(payload["skeleton_model_state"], strict=True)
        members = payload.get("members")
        if not isinstance(members, list) or len(members) != 7:
            raise ValueError("七头部署 artifact 必须包含七个成员")
        self.heads = []
        for member in members:
            checkpoint = {
                "fusion_state": member["fusion_state"],
                "roi_token_dim": member["roi_token_dim"],
                "roi_token_layout": member["roi_token_layout"],
                "roi_temporal_pooling": member["roi_temporal_pooling"],
            }
            self.heads.append(_head(checkpoint).to(self.device).eval())
        pose_device = str(self.device.index or 0) if self.device.type == "cuda" else "cpu"
        self.pose = PoseExtractor(
            yolo_checkpoint,
            device=pose_device,
            image_size=pose_image_size,
            confidence=pose_confidence,
        )
        self.roi_configs = {
            320: YoloRoiCacheConfig(input_size=320),
            640: YoloRoiCacheConfig(input_size=640),
        }
        self.roi_encoder: SharedYoloRoiEncoder | None = None
        self.artifact = Path(artifact)
        self.yolo_checkpoint = Path(yolo_checkpoint)

    def _roi(self, source: Path, record: PoseCacheRecord, size: int) -> torch.Tensor:
        proposal = propose_motion_event(record, context_seconds=2.0)
        config = self.roi_configs[size]
        frames, boxes, _ = decode_yolo_frames(
            source,
            record=record,
            track_id=proposal.track_id,
            interval=(proposal.start_time, proposal.end_time),
            config=config,
        )
        if self.roi_encoder is None:
            shared_model = self.pose._get_model().model
            self.roi_encoder = SharedYoloRoiEncoder(
                self.yolo_checkpoint,
                device=self.device,
                config=self.roi_configs[320],
                model=shared_model,
            )
        tokens = self.roi_encoder.encode(frames, boxes, config=config)
        return tokens[:, 0].unsqueeze(0).to(self.device)

    @torch.inference_mode()
    def predict(self, source: Path) -> dict[str, Any]:
        source = Path(source)
        started = time.perf_counter()
        sequence = self.pose.extract(source, crop="auto")
        pose_finished = time.perf_counter()
        record = pose_record(sequence, source)
        short = skeleton_windows(record, size=16, minimum=8).to(self.device)
        dense = skeleton_windows(record, size=48, minimum=24).to(self.device)
        short_embedding = self.skeleton.encode(short).amax(0, keepdim=True)
        dense_embedding = self.skeleton.encode(dense).amax(0, keepdim=True)
        skeleton_finished = time.perf_counter()
        roi_320 = self._roi(source, record, 320)
        roi_640 = self._roi(source, record, 640)
        roi_finished = time.perf_counter()
        logits = []
        for index in range(6):
            roi = roi_320 if index % 2 == 0 else roi_640
            logits.append(self.heads[index](short_embedding, roi)[0].squeeze(0))
        logits.append(self.heads[6](dense_embedding, roi_320)[0].squeeze(0))
        values = torch.stack(logits)
        score = equal_logit_score(values)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        finished = time.perf_counter()
        proposal = propose_motion_event(record, context_seconds=2.0)
        return {
            "protocol": PROTOCOL,
            "source": str(source),
            "score": float(score.cpu()),
            "member_logits": [float(value) for value in values.cpu()],
            "proposal": asdict(proposal),
            "selection_is_label_blind": True,
            "frames": int(sequence.frame_indices.size),
            "fps": float(sequence.fps),
            "latency_ms": {
                "pose": (pose_finished - started) * 1000.0,
                "skeleton": (skeleton_finished - pose_finished) * 1000.0,
                "roi": (roi_finished - skeleton_finished) * 1000.0,
                "heads_and_fusion": (finished - roi_finished) * 1000.0,
                "total": (finished - started) * 1000.0,
                "per_frame": (finished - started) * 1000.0 / sequence.frame_indices.size,
            },
        }

    def close(self) -> None:
        if self.roi_encoder is not None:
            self.roi_encoder.close()
        self.pose.close()
