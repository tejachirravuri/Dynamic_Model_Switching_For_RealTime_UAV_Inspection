#!/usr/bin/env bash
# run_pipeline.sh — run the full DMS evaluation pipeline on one video + model pair.
#
# Usage:
#   bash experiments/run_pipeline.sh <video> <fast_weights> <accurate_weights> [device] [max_frames]
#
#   device      cpu (default) or cuda
#   max_frames  0 = all frames (default); a small number gives a quick smoke test
#
# Example — 50-frame CPU smoke test:
#   bash experiments/run_pipeline.sh \
#       data/videos/glass.mp4 \
#       data/models/glass_y8n_fast.pt \
#       data/models/glass_y8s_accurate.pt \
#       cpu 50
#
# Steps, in order (a failure at step k stops the run):
#   1. run_reference      inference + timing + metrics sanity
#   2. run_features       per-frame scene proxies
#   3. run_sweep          every policy on the video
#   4. validate_sweep     G1-G5 sanity gates (aborts if any fail)
#   5. build_master_table aggregate results into master_table.csv
set -euo pipefail

VIDEO="${1:?path to input video required}"
FAST="${2:?path to fast (n) weights required}"
ACCURATE="${3:?path to accurate (s) weights required}"
DEVICE="${4:-cpu}"
MAX_FRAMES="${5:-0}"
RUN_ID="${RUN_ID:-pipeline_$(date +%Y%m%d_%H%M%S)}"

cd "$(dirname "$0")/.."   # repo root

echo "=== [1/5] run_reference (run id: $RUN_ID) ==="
python -m experiments.run_reference \
    --video "$VIDEO" --fast-weights "$FAST" --accurate-weights "$ACCURATE" \
    --device "$DEVICE" --max-frames "$MAX_FRAMES" --run-id "$RUN_ID"

echo "=== [2/5] run_features ==="
python -m experiments.run_features \
    --video "$VIDEO" --max-frames "$MAX_FRAMES" --run-id "$RUN_ID"

echo "=== [3/5] run_sweep ==="
python -m experiments.run_sweep \
    --video "$VIDEO" --fast-weights "$FAST" --accurate-weights "$ACCURATE" \
    --device "$DEVICE" --max-frames "$MAX_FRAMES" --run-id "$RUN_ID"

echo "=== [4/5] validate_sweep ==="
python -m experiments.validate_sweep --run-dir "results/stage_sweep/$RUN_ID"

echo "=== [5/5] build_master_table ==="
python -m experiments.build_master_table --results-root results

echo ""
echo "Pipeline complete (run id: $RUN_ID)."
echo "  per-policy summary: results/stage_sweep/$RUN_ID/sweep_summary.json"
echo "  aggregated table:   results/master_table.csv"
