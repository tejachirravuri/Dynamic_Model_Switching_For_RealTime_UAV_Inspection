#!/usr/bin/env bash
# experiments/run_stage3_ablation.sh
#
# Three ablations + housekeeping in one script. Designed to be launched
# under nohup; survives SSH drops and resumes via skip-existing.
#
#   1. Frame-skip ablation     K in {1, 3, 5, 10}
#                              On 2 representative n_l pairs (one
#                              YOLOv8 + one YOLO26) so we see how
#                              policy behaviour degrades with K.
#                              Full-length video; fast at large K.
#
#   2. Proxy-resolution        size in {32, 80, 160, 320}
#      ablation                On the same 2 pairs, capped at 2000
#                              frames so it's bounded.
#                              Goal: T_scene cost vs trigger fidelity.
#
#   3. Per-frame proxy trace   experiments.run_features on each of the
#                              two stage3 videos, full length, default
#                              proxy size 160. Drops sweep_features.csv
#                              so the threshold sensitivity for scene-
#                              feature policies can be replayed offline.
#
#   Plus housekeeping:
#      a. tar all training_runs/<name>/results.csv into one bundle.
#      b. tar the 4 .pt files for the visualised-frame examples.
#
# Calibration / device choice
# ---------------------------
#   DEVICE=cpu     (the platform where the ablation story actually
#                  matters; CUDA is saturated and ablation effects
#                  are sub-noise.)
#
# Usage
# -----
#   nohup bash experiments/run_stage3_ablation.sh > stage3_ablation.log 2>&1 &
#   echo $! > stage3_ablation.pid
#   tail -f stage3_ablation.log
#
# All outputs land under results/stage_sweep/ablation_*; the existing
# stage3_*_cpu / stage3_*_cuda dirs are untouched.
#
set -euo pipefail

DEVICE="${DEVICE:-cpu}"
IMGSZ="${IMGSZ:-640}"
WARMUP="${WARMUP:-3}"
CONF_EMA_C_HIGH="${CONF_EMA_C_HIGH:-0.02}"
CONF_EMA_C_LOW="${CONF_EMA_C_LOW:-0.007}"

ROOT="${ROOT:-${HOME}/dms_work/dms-thesis-clean}"
WEIGHTS_ROOT="${WEIGHTS_ROOT:-${HOME}/training_runs}"
GLASS_VID="${GLASS_VID:-${HOME}/test_videos/glass/20190916-722.mp4}"
PORC_VID="${PORC_VID:-${HOME}/test_videos/porcelain/YUN_0037.MP4}"

cd "${ROOT}"

# ---- pair matrix: family|fast|acc|domain|video -----------------------------
ABLATION_PAIRS=(
  "yolov8|n|l|glass|${GLASS_VID}"
  "yolo26|n|l|porcelain|${PORC_VID}"
)

# ---- 1. Frame-skip ablation ------------------------------------------------
echo
echo "============================================================"
echo "  ABLATION 1: frame-skip K in {1, 3, 5, 10}"
echo "============================================================"

for K in 1 3 5 10; do
  for pair in "${ABLATION_PAIRS[@]}"; do
    IFS='|' read -r FAM FAST ACC DOM VID <<< "${pair}"
    FAST_W="${WEIGHTS_ROOT}/${FAM}_${FAST}_${DOM}/weights/best.pt"
    ACC_W="${WEIGHTS_ROOT}/${FAM}_${ACC}_${DOM}/weights/best.pt"
    RUN_ID="ablation_skip_K${K}_${FAM}_${FAST}_${ACC}_${DOM}_${DEVICE}"
    if [[ -L "results/stage_sweep/${RUN_ID}" ]] \
       && [[ -f "results/stage_sweep/${RUN_ID}/sweep_summary.json" ]]; then
      echo "[skip-ablation] resume: ${RUN_ID} already exists (symlink to stage3)"
      continue
    fi
    if [[ -f "results/stage_sweep/${RUN_ID}/sweep_summary.json" ]]; then
      echo "[skip-ablation] resume: ${RUN_ID} already exists"
      continue
    fi
    if [[ ! -f "${FAST_W}" || ! -f "${ACC_W}" || ! -f "${VID}" ]]; then
      echo "[skip-ablation] SKIP (missing input): ${RUN_ID}"
      continue
    fi
    echo
    echo "----- K=${K}  ${FAM} ${FAST}<->${ACC}/${DOM} -----"
    python -u -m experiments.run_sweep \
      --video             "${VID}" \
      --fast-weights      "${FAST_W}" \
      --accurate-weights  "${ACC_W}" \
      --device            "${DEVICE}" \
      --imgsz             "${IMGSZ}" \
      --max-frames        0 \
      --frame-skip        "${K}" \
      --proxy-size        160 \
      --warmup-frames     "${WARMUP}" \
      --conf-ema-c-high   "${CONF_EMA_C_HIGH}" \
      --conf-ema-c-low    "${CONF_EMA_C_LOW}" \
      --run-id            "${RUN_ID}"
  done
done

# ---- 2. Proxy-resolution ablation ------------------------------------------
echo
echo "============================================================"
echo "  ABLATION 2: proxy-size in {32, 80, 160, 320}"
echo "  (max-frames 1000 cap)"
echo "============================================================"

for SIZE in 32 80 160 320; do
  for pair in "${ABLATION_PAIRS[@]}"; do
    IFS='|' read -r FAM FAST ACC DOM VID <<< "${pair}"
    FAST_W="${WEIGHTS_ROOT}/${FAM}_${FAST}_${DOM}/weights/best.pt"
    ACC_W="${WEIGHTS_ROOT}/${FAM}_${ACC}_${DOM}/weights/best.pt"
    RUN_ID="ablation_proxy_S${SIZE}_${FAM}_${FAST}_${ACC}_${DOM}_${DEVICE}"
    if [[ -f "results/stage_sweep/${RUN_ID}/sweep_summary.json" ]]; then
      echo "[proxy-ablation] resume: ${RUN_ID} already exists"
      continue
    fi
    if [[ ! -f "${FAST_W}" || ! -f "${ACC_W}" || ! -f "${VID}" ]]; then
      echo "[proxy-ablation] SKIP (missing input): ${RUN_ID}"
      continue
    fi
    echo
    echo "----- size=${SIZE}  ${FAM} ${FAST}<->${ACC}/${DOM} -----"
    python -u -m experiments.run_sweep \
      --video             "${VID}" \
      --fast-weights      "${FAST_W}" \
      --accurate-weights  "${ACC_W}" \
      --device            "${DEVICE}" \
      --imgsz             "${IMGSZ}" \
      --max-frames        1000 \
      --proxy-size        "${SIZE}" \
      --warmup-frames     "${WARMUP}" \
      --conf-ema-c-high   "${CONF_EMA_C_HIGH}" \
      --conf-ema-c-low    "${CONF_EMA_C_LOW}" \
      --run-id            "${RUN_ID}"
  done
done

# ---- 3. Per-frame proxy trace (run_features.py) ----------------------------
echo
echo "============================================================"
echo "  3. Per-frame proxy trace via run_features"
echo "============================================================"

for VID in "${GLASS_VID}" "${PORC_VID}"; do
  STEM=$(basename "${VID}" .mp4)
  STEM_LOWER=$(echo "${STEM}" | tr '[:upper:]' '[:lower:]')
  RUN_ID="features_${STEM_LOWER}"
  OUT="results/stage_features/${RUN_ID}"
  if [[ -f "${OUT}/per_frame.csv" ]]; then
    echo "[features] resume: ${RUN_ID} already exists"
    continue
  fi
  echo
  echo "----- features for ${VID} -----"
  python -u -m experiments.run_features \
    --video       "${VID}" \
    --proxy-size  160 \
    --run-id      "${RUN_ID}" \
    --results-root results/stage_features \
    || echo "[features] WARN: run_features failed for ${VID}"
done

# ---- 4. Training stats bundle ----------------------------------------------
echo
echo "============================================================"
echo "  4. Bundling training_runs results.csv files"
echo "============================================================"
PACK_DIR="${ROOT}/results/extras_pack"
mkdir -p "${PACK_DIR}"
TRAIN_TGZ="${PACK_DIR}/training_runs_results.tgz"
if [[ -f "${TRAIN_TGZ}" ]]; then
  echo "[extras] resume: ${TRAIN_TGZ} already exists"
else
  ( cd "${WEIGHTS_ROOT}" && \
    tar czf "${TRAIN_TGZ}" \
      $(find . -mindepth 2 -maxdepth 3 -name 'results.csv' 2>/dev/null | sort) \
      $(find . -mindepth 2 -maxdepth 3 -name 'args.yaml'   2>/dev/null | sort) \
      2>&1 | tail -5
  )
  echo "  -> ${TRAIN_TGZ} ($(du -h "${TRAIN_TGZ}" | cut -f1))"
fi

# ---- 5. .pt pack for visualised frames -------------------------------------
echo
echo "============================================================"
echo "  5. .pt files for visualised frames"
echo "============================================================"
PT_TGZ="${PACK_DIR}/weights_for_viz.tgz"
if [[ -f "${PT_TGZ}" ]]; then
  echo "[extras] resume: ${PT_TGZ} already exists"
else
  ( cd "${WEIGHTS_ROOT}" && \
    tar czf "${PT_TGZ}" \
      yolov8_n_glass/weights/best.pt \
      yolov8_s_glass/weights/best.pt \
      yolo26_n_porcelain/weights/best.pt \
      yolo26_l_porcelain/weights/best.pt \
      2>&1 | tail -3
  )
  echo "  -> ${PT_TGZ} ($(du -h "${PT_TGZ}" | cut -f1))"
fi

# ---- 6. Aggregate ----------------------------------------------------------
echo
echo "============================================================"
echo "  6. Aggregate ablation runs into a master_table_ablation.csv"
echo "============================================================"
python -m experiments.build_master_table \
  --results-root results \
  --out          results/master_table.csv

echo
echo "============================================================"
echo "  STAGE 3 ABLATIONS DONE"
echo "============================================================"
echo "  outputs:"
echo "    results/stage_sweep/ablation_skip_*"
echo "    results/stage_sweep/ablation_proxy_*"
echo "    results/stage_features/features_*"
echo "    ${TRAIN_TGZ}"
echo "    ${PT_TGZ}"
echo
echo "  next: rclone push the ablation dirs + extras_pack to Drive."
