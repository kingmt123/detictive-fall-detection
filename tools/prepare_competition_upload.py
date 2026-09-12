"""Prepare the four competition upload artifacts with official filenames."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DELIVERABLES = ROOT / "deliverables"
RELEASE_MANIFEST = DELIVERABLES / "release_manifest.json"
MAX_AUXILIARY_BYTES = 200 * 1024 * 1024
MAX_VIDEO_BYTES = 200 * 1024 * 1024

SOURCE_TO_SUFFIX = {
    "fall_detection_project_brief.pdf": "参赛作品简介.pdf",
    "fall_detection_project_report.pdf": "项目文档.pdf",
    "fall_detection_demo.mp4": "项目视频.mp4",
    "fall_detection_runtime.zip": "其他.zip",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_name_component(value: str, label: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{label}不能为空")
    if normalized in {".", ".."} or any(char in normalized for char in '<>:"/\\|?*'):
        raise ValueError(f"{label}包含 Windows 文件名非法字符")
    if normalized.endswith((" ", ".")):
        raise ValueError(f"{label}不能以空格或句点结尾")
    return normalized


def verify_sources(deliverables: Path, manifest_path: Path) -> dict[str, dict[str, object]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise TypeError("release manifest 缺少 artifacts")
    verified: dict[str, dict[str, object]] = {}
    for source_name in SOURCE_TO_SUFFIX:
        expected = artifacts.get(source_name)
        if not isinstance(expected, dict):
            raise TypeError(f"release manifest 缺少 {source_name}")
        source = deliverables / source_name
        if not source.is_file():
            raise FileNotFoundError(f"提交源文件不存在: {source}")
        actual_size = source.stat().st_size
        actual_hash = sha256_file(source)
        if actual_size != int(expected["bytes"]) or actual_hash != expected["sha256"]:
            raise ValueError(f"提交源文件与 release manifest 不匹配: {source_name}")
        verified[source_name] = {"bytes": actual_size, "sha256": actual_hash}
    if verified["fall_detection_demo.mp4"]["bytes"] > MAX_VIDEO_BYTES:
        raise ValueError("项目视频超过 200 MiB")
    if verified["fall_detection_runtime.zip"]["bytes"] > MAX_AUXILIARY_BYTES:
        raise ValueError("其他辅助材料 ZIP 超过 200 MiB")
    return verified


def prepare_upload(
    *,
    team: str,
    project: str,
    output_dir: Path,
    deliverables: Path = DELIVERABLES,
    manifest_path: Path = RELEASE_MANIFEST,
) -> dict[str, object]:
    team = validate_name_component(team, "团队名称")
    project = validate_name_component(project, "项目名称")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("上传目录必须为空，避免混入旧版本")
    verified = verify_sources(deliverables, manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, dict[str, object]] = {}
    for source_name, suffix in SOURCE_TO_SUFFIX.items():
        destination_name = f"{team}_{project}_{suffix}"
        destination = output_dir / destination_name
        shutil.copy2(deliverables / source_name, destination)
        if sha256_file(destination) != verified[source_name]["sha256"]:
            raise RuntimeError(f"复制后哈希不一致: {destination_name}")
        copied[destination_name] = verified[source_name]
    upload_manifest = {
        "schema": 1,
        "team": team,
        "project": project,
        "source_release_manifest_sha256": sha256_file(manifest_path),
        "files": copied,
        "model_status": "competition_compliant_gate4a_fallback",
        "research_champion_excluded": (
            "CLIP event-semantic residual exceeds the 20M/80MB model limits and "
            "its current evaluation uses annotation-derived event crops"
        ),
    }
    (output_dir / "UPLOAD_MANIFEST.json").write_text(
        json.dumps(upload_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return upload_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--team", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = prepare_upload(team=args.team, project=args.project, output_dir=args.output_dir)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
