"""Consolidate the frozen skeleton and seven classifier heads for deployment."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from models.edgefall_seven_head_runtime import PROTOCOL, _sha256
from models.event_proposal import PROTOCOL as EVENT_PROPOSAL_PROTOCOL
from tools.benchmark_edgefall_seven_head_tail import _config, _load_checkpoint
from tools.train_edgefall_f1 import _state_dict_sha256


def export_artifact(
    checkpoints: list[Path], yolo_checkpoint: Path, output: Path
) -> dict[str, object]:
    if len(checkpoints) != 7 or output.exists():
        raise ValueError("需要七个 checkpoint，且输出不得已存在")
    payloads = [_load_checkpoint(path) for path in checkpoints]
    configs = [_config(payload) for payload in payloads]
    if any(config != configs[0] for config in configs[1:]):
        raise ValueError("七个 checkpoint config 不一致")
    hashes = {_state_dict_sha256(payload["skeleton_model_state"]) for payload in payloads}
    if len(hashes) != 1:
        raise ValueError("七个 checkpoint 未共享同一 skeleton")
    members = []
    for path, payload in zip(checkpoints, payloads, strict=True):
        members.append(
            {
                "source_sha256": _sha256(path),
                "fusion_state": payload["fusion_state"],
                "roi_token_dim": int(payload.get("roi_token_dim", 192)),
                "roi_token_layout": str(payload.get("roi_token_layout", "direct")),
                "roi_temporal_pooling": str(payload.get("roi_temporal_pooling", "mean_max")),
            }
        )
    artifact = {
        "protocol": PROTOCOL,
        "event_proposal_protocol": EVENT_PROPOSAL_PROTOCOL,
        "fusion": "mean(seven raw logits) then sigmoid",
        "config": asdict(configs[0]),
        "skeleton_state_sha256": next(iter(hashes)),
        "skeleton_model_state": payloads[0]["skeleton_model_state"],
        "members": members,
        "yolo_sha256": _sha256(yolo_checkpoint),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(artifact, temporary)
    temporary.replace(output)
    report = {"path": str(output), "sha256": _sha256(output), "bytes": output.stat().st_size}
    print(json.dumps(report, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--yolo-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export_artifact(args.checkpoint, args.yolo_checkpoint, args.output)


if __name__ == "__main__":
    main()
