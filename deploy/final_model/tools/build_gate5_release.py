"""Assemble the Gate 5 runtime release from the immutable pre-seal package."""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
BASE_ZIP = ROOT / "runs" / "submission" / "gate4a_submission_dryrun.zip"
OUTPUT_DIR = ROOT / "deliverables"
OUTPUT_ZIP = OUTPUT_DIR / "fall_detection_runtime.zip"
OUTPUT_MANIFEST = OUTPUT_DIR / "release_manifest.json"

EXPECTED_BASE_SHA256 = "4faf3c58620e7ee7fdbe2dbbf9eef2fc0898ccf656678c7bb87c8c1cd735bd03"
EXPECTED_CANDIDATE_ID = "0e4c16bcd88bf33124bfb9a3089212396b27023b94b58794fbefcc06486f7186"
MAX_BYTES = 200 * 1024 * 1024
FORBIDDEN_PREFIXES = (
    "data/",
    "runs/",
    "reports/",
    "tests/",
    ".git/",
    ".venv/",
    "docs/research/_raw/",
)
SENSITIVE_PATTERNS = (
    re.compile(r"LTAI[A-Za-z0-9]+"),
    re.compile(r"AKIA[A-Z0-9]+"),
    re.compile(r"(?i)access[_ -]?key[_ -]?secret\s*[:=]\s*\S+"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._-]{12,}"),
    re.compile(r"(?i)(password|passwd)\s*[:=]\s*\S+"),
    re.compile(r"(?i)[A-Z]:\\Users\\"),
    re.compile(r"/home/[^/\s]+"),
    re.compile(r"/root/"),
)
TEXT_SUFFIXES = {".md", ".txt", ".py", ".json", ".yaml", ".yml", ".csv"}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    return info


def validate_name(name: str) -> None:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError(f"unsafe ZIP path: {name}")
    if name.startswith(FORBIDDEN_PREFIXES):
        raise ValueError(f"forbidden ZIP path: {name}")


def scan_text(name: str, data: bytes) -> None:
    if PurePosixPath(name).suffix.lower() not in TEXT_SUFFIXES:
        return
    text: str | None = None
    for encoding in ("utf-8", "utf-16"):
        try:
            text = data.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise ValueError(f"text file cannot be decoded: {name}")
    for pattern in SENSITIVE_PATTERNS:
        if pattern.search(text):
            raise ValueError(f"sensitive text in {name}: {pattern.pattern}")


def build_runtime_zip() -> dict[str, object]:
    if sha256_file(BASE_ZIP) != EXPECTED_BASE_SHA256:
        raise ValueError("immutable base ZIP hash mismatch")

    with zipfile.ZipFile(BASE_ZIP) as source:
        base_names = source.namelist()
        if len(base_names) != len(set(base_names)):
            raise ValueError("duplicate entry in base ZIP")
        candidate = json.loads(source.read("configs/gate4a_candidate.json"))
        if candidate["candidate_id"] != EXPECTED_CANDIDATE_ID:
            raise ValueError("candidate id mismatch")
        entries = {
            name: source.read(name)
            for name in base_names
            if name != "SUBMISSION_README.md"
        }

    entries["SUBMISSION_README.md"] = (ROOT / "SUBMISSION_README.md").read_bytes()
    entries["THIRD_PARTY_NOTICES.md"] = (ROOT / "THIRD_PARTY_NOTICES.md").read_bytes()
    runtime_manifest = {
        "schema": 1,
        "release": "gate5_anonymous_runtime",
        "candidate_id": EXPECTED_CANDIDATE_ID,
        "immutable_base_zip_sha256": EXPECTED_BASE_SHA256,
        "test_status": {
            "state": "completed",
            "run_id": "2c75283d1eec565a",
            "map": 0.429262883235486,
            "p_at_r90": 0.45662100456621,
            "p_at_r95": 0.40190476190476193,
            "summary_sha256": "380d4caaebb7755948f69d544781774fd2bb72337ee1eff770a7160850b96fce",
            "policy": "report-only; no rerun or tuning",
        },
        "v100_gate4a": {
            "p50_ms": 16.56,
            "p95_ms": 18.28,
            "parameters": 3_000_165,
            "fp32_parameter_mb": 12.00066,
            "report_sha256": "f72ede1c3bdccdc876d9878baa28afdbbd982325dbab7333f0c6a8030aacd4a3",
        },
        "files": {
            name: {"bytes": len(data), "sha256": sha256_bytes(data)}
            for name, data in sorted(entries.items())
        },
    }
    entries["RELEASE_MANIFEST.json"] = (
        json.dumps(runtime_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")

    for name, data in entries.items():
        validate_name(name)
        scan_text(name, data)

    temporary = OUTPUT_ZIP.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", allowZip64=True) as archive:
        for name, data in sorted(entries.items()):
            archive.writestr(zip_info(name), data)
    temporary.replace(OUTPUT_ZIP)

    if OUTPUT_ZIP.stat().st_size > MAX_BYTES:
        raise ValueError("runtime ZIP exceeds 200MB")
    with zipfile.ZipFile(OUTPUT_ZIP) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or archive.testzip() is not None:
            raise ValueError("runtime ZIP integrity failure")
        for name in names:
            validate_name(name)
            scan_text(name, archive.read(name))

    return {
        "bytes": OUTPUT_ZIP.stat().st_size,
        "entries": len(entries),
        "sha256": sha256_file(OUTPUT_ZIP),
    }


def build_external_manifest(runtime: dict[str, object]) -> None:
    artifacts = {}
    for name in (
        "fall_detection_project_brief.docx",
        "fall_detection_project_brief.pdf",
        "fall_detection_project_report.docx",
        "fall_detection_project_report.pdf",
    ):
        path = OUTPUT_DIR / name
        artifacts[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    demo = OUTPUT_DIR / "fall_detection_demo.mp4"
    if demo.exists():
        artifacts[demo.name] = {"bytes": demo.stat().st_size, "sha256": sha256_file(demo)}
    artifacts[OUTPUT_ZIP.name] = runtime
    manifest = {
        "schema": 1,
        "release_date": "2026-08-18",
        "candidate_id": EXPECTED_CANDIDATE_ID,
        "test_seal": "completed",
        "artifacts": artifacts,
        "remaining_gate5_item": "official team/project filenames and template confirmation",
    }
    OUTPUT_MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    runtime = build_runtime_zip()
    build_external_manifest(runtime)
    print(json.dumps(runtime, ensure_ascii=False, sort_keys=True))
    print(OUTPUT_MANIFEST)


if __name__ == "__main__":
    main()
