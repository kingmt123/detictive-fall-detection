#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python -m pytest -q
python tools/build_manifest.py
python tools/prepare_omnifall_events.py
python eval/benchmark.py --warmup 10 --runs 30 --output reports/benchmark_rtx4060.json
python infer.py data/sources/urfd/fall-01-cam1.mp4 \
  --output-video reports/demo_fall01_final.mp4 \
  --output-events reports/demo_fall01_final.json

echo "Reproduction completed: tests, manifests, benchmark, and fall demo."
