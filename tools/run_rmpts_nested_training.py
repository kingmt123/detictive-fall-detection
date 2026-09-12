"""Run the resumable outer-isolated RMPTS base-member training matrix."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools.train_tcn import _atomic_json, _sha256_file

HEAD_SEEDS = (None, 20260829, 20260830)


@dataclass(frozen=True)
class NestedJob:
    outer_fold: int
    inner_fold: int
    family: str
    head_seed: int | None
    output_dir: Path
    cache: Path
    sidecar: Path
    roi_cache: Path
    fold_map: Path
    reuse_backbone: Path | None
    embedding_cache: Path | None
    save_embedding_cache: bool

    @property
    def name(self) -> str:
        seed = "seed1" if self.head_seed is None else f"headseed{self.head_seed}"
        return f"outer{self.outer_fold}_inner{self.inner_fold}_{self.family}_{seed}"


def build_jobs(
    output_root: Path,
    *,
    outer_folds: tuple[int, ...],
    short_cache: Path,
    short_sidecar: Path,
    dense_cache: Path,
    dense_sidecar: Path,
    roi_320: Path,
    roi_640: Path,
    short_fold_map: Path,
    dense_fold_map: Path,
) -> list[NestedJob]:
    if not outer_folds or len(set(outer_folds)) != len(outer_folds):
        raise ValueError("outer folds 必须非空且唯一")
    if any(not 0 <= fold < 5 for fold in outer_folds):
        raise ValueError("outer fold 必须位于 [0,4]")
    jobs = []
    for outer in outer_folds:
        for inner in (fold for fold in range(5) if fold != outer):
            base = output_root / f"outer{outer}" / f"inner{inner}"
            short_backbone = base / "short320_seed1" / "last.pt"
            short_embeddings = base / "short320_seed1" / "clip_embeddings.npz"
            for head_seed in HEAD_SEEDS:
                seed_name = "seed1" if head_seed is None else f"headseed{head_seed}"
                for resolution, roi_cache in ((320, roi_320), (640, roi_640)):
                    output = base / f"short{resolution}_{seed_name}"
                    first = head_seed is None and resolution == 320
                    jobs.append(
                        NestedJob(
                            outer_fold=outer,
                            inner_fold=inner,
                            family=f"short{resolution}",
                            head_seed=head_seed,
                            output_dir=output,
                            cache=short_cache,
                            sidecar=short_sidecar,
                            roi_cache=roi_cache,
                            fold_map=short_fold_map,
                            reuse_backbone=None if first else short_backbone,
                            embedding_cache=short_embeddings,
                            save_embedding_cache=first,
                        )
                    )
            jobs.append(
                NestedJob(
                    outer_fold=outer,
                    inner_fold=inner,
                    family="dense48_320",
                    head_seed=None,
                    output_dir=base / "dense48_320",
                    cache=dense_cache,
                    sidecar=dense_sidecar,
                    roi_cache=roi_320,
                    fold_map=dense_fold_map,
                    reuse_backbone=None,
                    embedding_cache=None,
                    save_embedding_cache=False,
                )
            )
    return jobs


def _expected_summary(job: NestedJob) -> dict[str, Any]:
    return {
        "fold": job.inner_fold,
        "excluded": [job.outer_fold],
        "head_seed": job.head_seed,
        "roi_sha256": _sha256_file(job.roi_cache),
        "fold_map_sha256": _sha256_file(job.fold_map),
        "cache_metadata_sha256": _sha256_file(job.cache / "metadata.json"),
        "sidecar_metadata_sha256": _sha256_file(job.sidecar / "metadata.json"),
    }


def completed_job_is_valid(job: NestedJob) -> bool:
    summary_path = job.output_dir / "summary.json"
    checkpoint_path = job.output_dir / "last.pt"
    predictions_path = job.output_dir / "oof_predictions.npz"
    present = [path.exists() for path in (summary_path, checkpoint_path, predictions_path)]
    if not any(present):
        if job.output_dir.exists() and any(job.output_dir.iterdir()):
            raise ValueError(f"nested job 存在未知部分输出: {job.output_dir}")
        return False
    if not all(present):
        raise ValueError(f"nested job 输出不完整，拒绝覆盖: {job.output_dir}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = _expected_summary(job)
    if summary.get("config", {}).get("fold") != expected["fold"]:
        raise ValueError(f"nested job fold 签名不匹配: {job.name}")
    assignment = summary.get("fold_assignment", {})
    if assignment.get("nested_outer_excluded_folds") != expected["excluded"]:
        raise ValueError(f"nested job outer exclusion 签名不匹配: {job.name}")
    if summary.get("head_seed") != expected["head_seed"]:
        raise ValueError(f"nested job head seed 签名不匹配: {job.name}")
    if summary.get("roi_cache_sha256") != expected["roi_sha256"]:
        raise ValueError(f"nested job ROI 签名不匹配: {job.name}")
    if assignment.get("sha256") != expected["fold_map_sha256"]:
        raise ValueError(f"nested job fold map 签名不匹配: {job.name}")
    if summary.get("checkpoint_sha256") != _sha256_file(checkpoint_path):
        raise ValueError(f"nested job checkpoint 哈希不匹配: {job.name}")
    return True


def _command(job: NestedJob, args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "tools.train_edgefall_f1",
        "--cache",
        str(job.cache),
        "--sidecar",
        str(job.sidecar),
        "--roi-cache",
        str(job.roi_cache),
        "--joint-pretrain",
        str(args.joint_pretrain),
        "--fold-map",
        str(job.fold_map),
        "--output-dir",
        str(job.output_dir),
        "--device",
        args.device,
        "--fold",
        str(job.inner_fold),
        "--exclude-fold",
        str(job.outer_fold),
        "--backbone-epochs",
        "12",
        "--head-epochs",
        "12",
        "--seed",
        "20260825",
        "--use-transformer-encoder",
        "--transformer-joint-only",
        "--embedding-batch-size",
        "1024",
    ]
    if job.head_seed is not None:
        command.extend(("--head-seed", str(job.head_seed)))
    if job.reuse_backbone is not None:
        if not job.reuse_backbone.is_file():
            raise FileNotFoundError(f"复用 backbone 尚未完成: {job.reuse_backbone}")
        command.extend(("--reuse-backbone-checkpoint", str(job.reuse_backbone)))
    if job.save_embedding_cache:
        assert job.embedding_cache is not None
        command.extend(("--save-embedding-cache", str(job.embedding_cache)))
    elif job.embedding_cache is not None:
        if not job.embedding_cache.is_file():
            raise FileNotFoundError(f"复用 embedding cache 尚未完成: {job.embedding_cache}")
        command.extend(("--reuse-embedding-cache", str(job.embedding_cache)))
    return command


def _run_job(job: NestedJob, args: argparse.Namespace) -> None:
    job.output_dir.parent.mkdir(parents=True, exist_ok=True)
    command = _command(job, args)
    log_path = job.output_dir.parent / f"{job.output_dir.name}.runner.log"
    with log_path.open("a", encoding="utf-8") as log:
        log.write(json.dumps({"job": job.name, "command": command}) + "\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=Path.cwd(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"nested job 失败: {job.name}, exit={return_code}")
    if not completed_job_is_valid(job):
        raise RuntimeError(f"nested job 完成后校验失败: {job.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--outer-fold", type=int, action="append", required=True)
    parser.add_argument("--short-cache", type=Path, required=True)
    parser.add_argument("--short-sidecar", type=Path, required=True)
    parser.add_argument("--dense-cache", type=Path, required=True)
    parser.add_argument("--dense-sidecar", type=Path, required=True)
    parser.add_argument("--roi-320", type=Path, required=True)
    parser.add_argument("--roi-640", type=Path, required=True)
    parser.add_argument("--short-fold-map", type=Path, required=True)
    parser.add_argument("--dense-fold-map", type=Path, required=True)
    parser.add_argument("--joint-pretrain", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    paths = (
        args.short_cache / "metadata.json",
        args.short_sidecar / "metadata.json",
        args.dense_cache / "metadata.json",
        args.dense_sidecar / "metadata.json",
        args.roi_320,
        args.roi_640,
        args.short_fold_map,
        args.dense_fold_map,
        args.joint_pretrain,
    )
    if any(not path.is_file() for path in paths):
        missing = [str(path) for path in paths if not path.is_file()]
        raise FileNotFoundError(f"nested runner 输入缺失: {missing}")
    jobs = build_jobs(
        args.output_root,
        outer_folds=tuple(args.outer_fold),
        short_cache=args.short_cache,
        short_sidecar=args.short_sidecar,
        dense_cache=args.dense_cache,
        dense_sidecar=args.dense_sidecar,
        roi_320=args.roi_320,
        roi_640=args.roi_640,
        short_fold_map=args.short_fold_map,
        dense_fold_map=args.dense_fold_map,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "protocol": "edgefall_rmpts_outer_isolated_base_training_v1",
        "outer_folds": args.outer_fold,
        "jobs": [
            {
                "name": job.name,
                "output_dir": str(job.output_dir),
                **_expected_summary(job),
            }
            for job in jobs
        ],
        "inputs": {str(path): _sha256_file(path) for path in paths},
        "test_accessed": False,
    }
    manifest_path = args.output_root / "training_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError("nested training manifest 已存在且内容不匹配")
    else:
        _atomic_json(manifest_path, manifest)
    print(json.dumps({"jobs": len(jobs), "manifest": str(manifest_path.resolve())}))
    if args.dry_run:
        return
    for index, job in enumerate(jobs, start=1):
        if completed_job_is_valid(job):
            print(json.dumps({"job": job.name, "status": "skipped_valid"}), flush=True)
            continue
        print(json.dumps({"job": job.name, "status": "starting", "index": index, "total": len(jobs)}), flush=True)
        _run_job(job, args)


if __name__ == "__main__":
    main()
