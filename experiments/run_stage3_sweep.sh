#!/usr/bin/env bash
# experiments/run_stage3_sweep.sh
#
# Path B Stage 3 — 8-pair policy sweep on full videos, single platform.
#
#   pairs   = 8     YOLOv8 + YOLO26  ×  {n_s, n_l}  ×  {glass, porcelain}
#   videos  = 2     20190916-722.mp4 (glass), YUN_0037.MP4 (porcelain)
#   policies= 7     n_only, s_only, entropy_only, combined, combined_hyst,
#                   conf_ema (c_high=0.02), multi_proxy
#   imgsz   = 640
#
# Calibration note (do NOT silently change to "default")
# ------------------------------------------------------
# --conf-ema-c-high 0.02 / --conf-ema-c-low 0.007 is the
# **quality-oriented calibration**, not a global default.
# The 1000-frame threshold sweeps on glass and porcelain (run IDs
# remote_diag_glass_cuda1000 / remote_diag_porcelain_cuda1000) showed:
#
#   * On a saturated GPU with no latency cost, iou_match is
#     monotonically improved by lower c_high on BOTH domains;
#     c_high=0.02 wins iou_match cross-domain.
#   * Trigger validity (F1) is INVERTED across domains: glass F1
#     peaks at c_high=0.02, porcelain F1 peaks at c_high=0.12.
#
# This script uses 0.02 because it dominates iou_match cross-domain
# on this hardware. The threshold-sweep CSVs are preserved as the
# evidence of domain-dependent trigger validity (see analysis/).
#
# Usage
# -----
# Foreground (dies if SSH drops):
#   bash experiments/run_stage3_sweep.sh
#
# Detached (survives SSH disconnect — recommended on remote):
#   nohup bash experiments/run_stage3_sweep.sh > stage3.log 2>&1 &
#   echo $! > stage3.pid
#   # then to monitor from any new shell:
#   tail -f stage3.log
#   # check if still running:
#   ps -p "$(cat stage3.pid)" -o pid,etime,cmd
#
# Override the device / paths via env vars if needed:
#   DEVICE=cpu bash experiments/run_stage3_sweep.sh
#
# Resume safety
# -------------
# Pairs whose sweep_summary.json already exists are skipped. To force a
# re-run, set FORCE_RERUN=1 (or delete the run dir).
#
set -euo pipefail

# ---- Configurable knobs (env-overridable) -----------------------------------
DEVICE="${DEVICE:-cuda}"
IMGSZ="${IMGSZ:-640}"
PROXY_SIZE="${PROXY_SIZE:-160}"
WARMUP="${WARMUP:-3}"
CONF_EMA_C_HIGH="${CONF_EMA_C_HIGH:-0.02}"
CONF_EMA_C_LOW="${CONF_EMA_C_LOW:-0.007}"
FORCE_RERUN="${FORCE_RERUN:-0}"

ROOT="${ROOT:-${HOME}/dms_work/dms-thesis-clean}"
WEIGHTS_ROOT="${WEIGHTS_ROOT:-${HOME}/training_runs}"
GLASS_VID="${GLASS_VID:-${HOME}/test_videos/glass/20190916-722.mp4}"
PORC_VID="${PORC_VID:-${HOME}/test_videos/porcelain/YUN_0037.MP4}"
RESULTS_ROOT="${RESULTS_ROOT:-${ROOT}/results}"

cd "${ROOT}"

# ---- Pair matrix (family|fast|accurate|domain|video) ------------------------
# Each line is one (fast, accurate) pair. Domain selects the video; the
# weights file naming follows ${family}_${size}_${domain} under WEIGHTS_ROOT.
PAIRS=(
  "yolov8|n|s|glass|${GLASS_VID}"
  "yolov8|n|l|glass|${GLASS_VID}"
  "yolov8|n|s|porcelain|${PORC_VID}"
  "yolov8|n|l|porcelain|${PORC_VID}"
  "yolo26|n|s|glass|${GLASS_VID}"
  "yolo26|n|l|glass|${GLASS_VID}"
  "yolo26|n|s|porcelain|${PORC_VID}"
  "yolo26|n|l|porcelain|${PORC_VID}"
)

declare -a SKIPPED=()
declare -a FLAGGED=()
declare -a RESUMED=()

# ---- Run loop ---------------------------------------------------------------
for pair in "${PAIRS[@]}"; do
  IFS='|' read -r FAMILY FAST_SIZE ACC_SIZE DOMAIN VID <<< "${pair}"
  FAST_W="${WEIGHTS_ROOT}/${FAMILY}_${FAST_SIZE}_${DOMAIN}/weights/best.pt"
  ACC_W="${WEIGHTS_ROOT}/${FAMILY}_${ACC_SIZE}_${DOMAIN}/weights/best.pt"
  RUN_ID="stage3_${FAMILY}_${FAST_SIZE}_${ACC_SIZE}_${DOMAIN}_${DEVICE}"

  echo
  echo "============================================================"
  echo "  PAIR: ${FAMILY} ${FAST_SIZE}<->${ACC_SIZE}  domain=${DOMAIN}"
  echo "  RUN_ID: ${RUN_ID}"
  echo "============================================================"

  if [[ ! -f "${FAST_W}" || ! -f "${ACC_W}" || ! -f "${VID}" ]]; then
    echo "[stage3] SKIP: missing input"
    [[ ! -f "${FAST_W}" ]] && echo "  fast: ${FAST_W}"
    [[ ! -f "${ACC_W}" ]]  && echo "  acc:  ${ACC_W}"
    [[ ! -f "${VID}" ]]    && echo "  vid:  ${VID}"
    SKIPPED+=("${RUN_ID}")
    continue
  fi

  SUMMARY_PATH="${RESULTS_ROOT}/stage_sweep/${RUN_ID}/sweep_summary.json"
  if [[ -f "${SUMMARY_PATH}" && "${FORCE_RERUN}" != "1" ]]; then
    echo "[stage3] RESUME: ${RUN_ID} already complete, skipping "
    echo "         (set FORCE_RERUN=1 to override, or delete ${SUMMARY_PATH})"
    RESUMED+=("${RUN_ID}")
    continue
  fi

  python -m experiments.run_sweep \
    --video             "${VID}" \
    --fast-weights      "${FAST_W}" \
    --accurate-weights  "${ACC_W}" \
    --device            "${DEVICE}" \
    --imgsz             "${IMGSZ}" \
    --max-frames        0 \
    --proxy-size        "${PROXY_SIZE}" \
    --warmup-frames     "${WARMUP}" \
    --conf-ema-c-high   "${CONF_EMA_C_HIGH}" \
    --conf-ema-c-low    "${CONF_EMA_C_LOW}" \
    --run-id            "${RUN_ID}"

  if ! python -m experiments.validate_sweep \
        --run-dir "results/stage_sweep/${RUN_ID}"; then
    echo "[stage3] WARN: ${RUN_ID} flagged by validate_sweep — keep going"
    FLAGGED+=("${RUN_ID}")
  fi
done

# ---- Aggregate to one master table ------------------------------------------
echo
echo "============================================================"
echo "  AGGREGATING all sweeps into one master table"
echo "============================================================"
python -m experiments.build_master_table \
  --results-root "${RESULTS_ROOT}" \
  --out          "${RESULTS_ROOT}/master_table.csv"

# ---- Final report -----------------------------------------------------------
echo
echo "============================================================"
echo "  STAGE 3 DONE"
echo "============================================================"
n_done=$(( ${#PAIRS[@]} - ${#SKIPPED[@]} - ${#RESUMED[@]} ))
echo "  ran    ${n_done}/${#PAIRS[@]} pairs in this invocation"
echo "  resumed ${#RESUMED[@]}/${#PAIRS[@]} (already had sweep_summary.json)"
if [[ ${#SKIPPED[@]} -gt 0 ]]; then
  echo "  SKIPPED (missing inputs):"
  for s in "${SKIPPED[@]}"; do echo "    - ${s}"; done
fi
if [[ ${#FLAGGED[@]} -gt 0 ]]; then
  echo "  FLAGGED by validate_sweep (inspect before scaling):"
  for f in "${FLAGGED[@]}"; do echo "    - ${f}"; done
fi
echo "  master_table: ${RESULTS_ROOT}/master_table.csv"
echo
echo "  conf_ema calibration in this sweep: c_high=${CONF_EMA_C_HIGH}"
echo "  (quality-oriented; F1-optimal calibration is per-domain)"
