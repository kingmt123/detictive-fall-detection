"""Build a hash-manifested, source-whitelisted seven-head runtime ZIP."""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = ("models", "pipeline", "tools", "eval", "data")
ROOT_FILES = ("requirements.txt",)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def package_runtime(*, artifact: Path, yolo: Path, output: Path) -> dict[str, object]:
    if output.exists():
        raise FileExistsError("部署 ZIP 已存在，拒绝覆盖")
    if not artifact.is_file() or not yolo.is_file():
        raise FileNotFoundError("部署权重不完整")
    entries: dict[str, bytes] = {}
    for root_name in SOURCE_ROOTS:
        for source in sorted((ROOT / root_name).rglob("*.py")):
            if "__pycache__" not in source.parts:
                entries[source.relative_to(ROOT).as_posix()] = source.read_bytes()
    for name in ROOT_FILES:
        entries[name] = (ROOT / name).read_bytes()
    entries["README.md"] = (ROOT / "deployment/SEVEN_HEAD_README.md").read_bytes()
    entries["weights/edgefall_seven_head_label_blind_v1.pt"] = artifact.read_bytes()
    entries["weights/yolo11n-pose.pt"] = yolo.read_bytes()
    manifest = {
        "schema": 1,
        "status": "deployment_candidate_pending_v100_p95",
        "selection_is_label_blind": True,
        "fusion": "mean(seven raw logits) then sigmoid",
        "files": {
            name: {"bytes": len(value), "sha256": _sha256_bytes(value)}
            for name, value in sorted(entries.items())
        },
    }
    entries["MANIFEST.json"] = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in sorted(entries.items()):
            archive.writestr(name, value)
    temporary.replace(output)
    result = {
        "path": str(output),
        "bytes": output.stat().st_size,
        "sha256": _sha256_bytes(output.read_bytes()),
        "files": len(entries),
        "status": manifest["status"],
    }
    print(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--yolo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    package_runtime(artifact=args.artifact, yolo=args.yolo, output=args.output)


if __name__ == "__main__":
    main()
