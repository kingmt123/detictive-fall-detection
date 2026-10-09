"""Local case evidence and human review records.

Human review does not alter immutable model result.json.
This is a single-workstation prototype; it does not authenticate operators.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import tempfile
from typing import Any


REVIEW_STATES = ("待复核", "人工确认跌倒", "人工排除跌倒", "无法判断")
_CASE_ID = re.compile(r"^run-[A-Za-z0-9_-]+$")


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    """Avoid half-written status records if the process exits during a write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=".draft-",
        suffix=".json", delete=False
    ) as f:
        temporary = Path(f.name)
        f.write(payload)
        f.flush()
    temporary.replace(path)


def _case_dir(root: Path, case_id: str) -> Path:
    if not _CASE_ID.fullmatch(str(case_id)):
        raise ValueError("记录编号格式无效。")
    base = Path(root).resolve()
    path = (base / case_id).resolve()
    if path.parent != base or not path.is_dir():
        raise ValueError("记录不存在。")
    return path


def capture_evidence(source: Path, destination: Path, result: dict[str, Any]) -> Path:
    """Extract 3 actual video frames. Never infer or draw a human bounding box."""
    import cv2
    import numpy as np

    event = result.get("proposal") or {}
    fps = float(result.get("fps") or 0)
    frames = int(result.get("frames") or 0)
    if fps <= 0 or frames <= 0:
        raise ValueError("视频 fps/frames 无效。")
    duration = (frames - 1) / fps
    start = max(0.0, min(duration, float(event.get("start_time", 0))))
    peak = max(0.0, min(duration, float(event.get("peak_time", 0))))
    end = max(0.0, min(duration, float(event.get("end_time", 0))))
    points = [("BEFORE", max(0.0, start - 1.0)),
              ("PEAK", peak), ("AFTER", min(duration, end + 1.0))]
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError("无法重新解码原视频生成事件快照。")
    width, height = 384, 216
    snapshots = []
    try:
        for name, second in points:
            capture.set(cv2.CAP_PROP_POS_FRAMES, round(second * fps))
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"无法获取视频位置 {second:.2f}s 的原始帧。")
            h, w = frame.shape[:2]
            scale = min(width / w, height / h)
            img = cv2.resize(frame, (max(1, round(w * scale)), max(1, round(h * scale))))
            canvas = np.full((height + 44, width, 3), (28, 43, 59), dtype=np.uint8)
            y = (height - img.shape[0]) // 2
            x = (width - img.shape[1]) // 2
            canvas[y:y + img.shape[0], x:x + img.shape[1]] = img
            cv2.putText(canvas, f"{name} / {second:.2f}s", (12, height + 29),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (236, 243, 249), 2)
            snapshots.append(canvas)
    finally:
        capture.release()
    sheet = np.concatenate(snapshots, axis=1)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), sheet):
        raise RuntimeError("无法写入事件快照。")
    return destination


def register_case(root: Path, json_path: str | Path, source: str | Path, *,
                  device: str, video_path: str | None, warning: str = "") -> dict[str, Any]:
    base = Path(root).resolve()
    raw_path = Path(json_path).resolve()
    case_id = raw_path.parent.name
    case_dir = _case_dir(base, case_id)
    if raw_path != case_dir / "result.json":
        raise ValueError("模型结果位置与当前记录不一致。")
    result = json.loads(raw_path.read_text(encoding="utf-8"))
    metadata = {
        "case_id": case_id, "created_at": _now(), "source_name": Path(source).name,
        "device": device, "model": "edgefall_seven_head_label_blind_v1",
        "protocol": result.get("protocol", ""), "preview_ready": bool(video_path),
        "warning": warning,
    }
    _write_json(case_dir / "case.json", metadata)
    evidence = None
    try:
        evidence = str(capture_evidence(Path(source), case_dir / "evidence.jpg", result))
    except Exception as exc:
        metadata["warning"] = (metadata["warning"] + "；" if metadata["warning"] else "") \
                              + f"事件帧提取失败：{type(exc).__name__}: {exc}"
        _write_json(case_dir / "case.json", metadata)
    return {"metadata": metadata, "result": result, "evidence": evidence,
            "review": read_review(base, case_id)}


def read_review(root: Path, case_id: str) -> dict[str, Any]:
    location = _case_dir(root, case_id) / "review.json"
    if not location.is_file():
        return {"status": "待复核", "note": "", "updated_at": "", "events": []}
    obj = json.loads(location.read_text(encoding="utf-8"))
    if obj.get("status") not in REVIEW_STATES:
        raise ValueError("复核状态数据不合法。")
    return obj


def load_case(root: Path, case_id: str) -> dict[str, Any]:
    folder = _case_dir(root, case_id)
    metadata = json.loads((folder / "case.json").read_text(encoding="utf-8"))
    result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
    image = folder / "evidence.jpg"
    movie = folder / "preview.mp4"
    return {"metadata": metadata, "result": result, "review": read_review(root, case_id),
            "evidence": str(image) if image.is_file() else None,
            "video": str(movie) if movie.is_file() else None,
            "json_path": str(folder / "result.json"),
            "review_path": str(folder / "review.json") if (folder / "review.json").is_file() else None}


def list_cases(root: Path, limit: int = 100) -> list[dict[str, Any]]:
    base = Path(root)
    if not base.is_dir():
        return []
    rows = []
    for path in sorted(base.glob("run-*/case.json"), reverse=True):
        if len(rows) >= limit:
            break
        try:
            case_id = path.parent.name
            case = load_case(base, case_id)
            meta = case["metadata"]
            result = case["result"]
            event = result.get("proposal") or {}
            rows.append({
                "case_id": case_id, "created_at": meta.get("created_at", ""),
                "source_name": meta.get("source_name", ""),
                "score": result.get("score"),
                "event_time": f'{float(event.get("start_time",0)):.2f}–{float(event.get("end_time",0)):.2f}s',
                "review_status": case["review"]["status"],
            })
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            continue
    rows.sort(key=lambda x: x["created_at"], reverse=True)
    return rows


def save_review(root: Path, case_id: str, status: str, note: str) -> dict[str, Any]:
    if status not in REVIEW_STATES:
        raise ValueError("请选择合法的人工复核状态。")
    if len(note) > 500:
        raise ValueError("复核备注不得超过 500 字符。")
    previous = read_review(root, case_id)
    now = _now()
    event = {"at": now, "status": status, "note": note,
             "operator": "local-unverified"}
    previous["events"].append(event)
    previous.update({"status": status, "note": note, "updated_at": now})
    _write_json(_case_dir(root, case_id) / "review.json", previous)
    return previous
