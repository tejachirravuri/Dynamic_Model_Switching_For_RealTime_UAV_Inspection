#!/usr/bin/env bash
# experiments/run_multi_video_sweep.sh
#
# Multi-video Stage-3 sweep that pulls each video from Drive, runs the
# full pair matrix on it, pushes results back, and deletes the local
# copy — so remote disk stays clean and the run survives any SSH drop.
#
# Per video pipeline
# ------------------
#   1. rclone copy <video> from Drive -> ~/scratch_videos/
#   2. for each pair config that matches this video's domain:
#        for each device in {cuda, cpu(n_l-only by default)}:
#          run experiments.run_sweep with the locked Stage-3 config:
#            * --conf-ema-c-high 0.02 / --conf-ema-c-low 0.007
#            * --warmup-frames 3
#            * --imgsz 640 --proxy-size 160
#          run experiments.validate_sweep
#   3. rclone copy results/stage_sweep/<run_id>/ -> Drive
#   4. rm -f the local video to save disk
#   5. delete the per-frame CSV locally (we kept it on Drive); keep the
#      sweep_summary.json locally so the master_table can rebuild
#
# Resume-safety
# -------------
# A video is SKIPPED if every (pair, device) combination already has a
# sweep_summary.json on Drive at:
#   <RESULTS_FOLDER>/<domain>/<video_stem>/<run_id>/sweep_summary.json
# Re-launching the script picks up where the last invocation stopped.
#
# CPU budget (default sane subset)
# ---------------------------------
# CPU is expensive — running every (pair, video) on CPU is 30+ hours.
# By default we run CPU only on the n_l pair (where the latency gap
# matters). To run ALL CPU pairs, set:
#   CPU_PAIRS_FILTER=all
# To skip CPU entirely (GPU only):
#   SKIP_CPU=1
# To skip GPU entirely (rare):
#   SKIP_GPU=1
#
# Detached launch
# ---------------
#   nohup bash experiments/run_multi_video_sweep.sh \
#         > multivideo.log 2>&1 &
#   echo $! > multivideo.pid
#   tail -f multivideo.log
#
# Override knobs (env vars)
# -------------------------
#   VIDEOS_FOLDER       Drive folder rclone-prefix where glass/, porcelain/
#                       subfolders live with the source videos
#                       (default: testvideos folder ID 1tYUH...)
#   RESULTS_FOLDER      Drive folder rclone-prefix where this run's
#                       results land
#   ROOT                Local repo path (default ~/dms_work/dms-thesis-clean)
#   WEIGHTS_ROOT        Local trained-weights path (default ~/training_runs)
#   SCRATCH             Where to download videos to (default ~/scratch_videos)
#   CPU_PAIRS_FILTER    "all" | "n_l" (default "n_l")
#   SKIP_CPU=1          skip every CPU run
#   SKIP_GPU=1          skip every CUDA run
#   FORCE_RERUN=1       ignore the on-Drive resume check
#
set -euo pipefail

VIDEOS_FOLDER="${VIDEOS_FOLDER:-gdrive,root_folder_id=1tYUHzAGnSqhunkeKgNc4BSAtDu4Ofrbz:}"
RESULTS_FOLDER="${RESULTS_FOLDER:-gdrive,root_folder_id=1UcqxSCcMuEtmIq72JS4meQYIQjgRUJn7:dms_thesis_clean/multi_video}"
ROOT="${ROOT:-${HOME}/dms_work/dms-thesis-clean}"
WEIGHTS_ROOT="${WEIGHTS_ROOT:-${HOME}/training_runs}"
SCRATCH="${SCRATCH:-${HOME}/scratch_videos}"
IMGSZ="${IMGSZ:-640}"
PROXY_SIZE="${PROXY_SIZE:-160}"
WARMUP="${WARMUP:-3}"
CONF_EMA_C_HIGH="${CONF_EMA_C_HIGH:-0.02}"
CONF_EMA_C_LOW="${CONF_EMA_C_LOW:-0.007}"
CPU_PAIRS_FILTER="${CPU_PAIRS_FILTER:-n_l}"
SKIP_CPU="${SKIP_CPU:-0}"
SKIP_GPU="${SKIP_GPU:-0}"
FORCE_RERUN="${FORCE_RERUN:-0}"
POLICIES=(
  n_only
  s_only
  entropy_only
  combined
  combined_hyst
  conf_ema
  multi_proxy
  local_contrast_hyst
)

mkdir -p "${SCRATCH}"
cd "${ROOT}"

# Pair matrix
GLASS_PAIRS=(
  "yolov8|n|s|glass"
  "yolov8|n|l|glass"
  "yolo26|n|s|glass"
  "yolo26|n|l|glass"
)
PORCELAIN_PAIRS=(
  "yolov8|n|s|porcelain"
  "yolov8|n|l|porcelain"
  "yolo26|n|s|porcelain"
  "yolo26|n|l|porcelain"
)

# ============================================================================
# Helpers
# ============================================================================
rclone_lsf_safe() {
  rclone lsf --files-only "$1" 2>/dev/null || true
}

result_already_pushed() {
  # $1 = <domain>/<video_stem>/<run_id>
  if [[ "${FORCE_RERUN}" == "1" ]]; then
    return 1
  fi
  local subpath="$1"
  rclone_lsf_safe "${RESULTS_FOLDER}/${subpath}/" \
    | grep -q '^sweep_summary\.json$'
}

run_one_pair() {
  # Args: family fast acc domain device video_path video_stem
  local FAM="$1" FAST="$2" ACC="$3" DOM="$4" DEV="$5" VID="$6" STEM="$7"
  local FAST_W="${WEIGHTS_ROOT}/${FAM}_${FAST}_${DOM}/weights/best.pt"
  local ACC_W="${WEIGHTS_ROOT}/${FAM}_${ACC}_${DOM}/weights/best.pt"
  local RUN_ID="multi_${STEM}_${FAM}_${FAST}_${ACC}_${DOM}_${DEV}"
  local LOCAL_OUT="results/stage_sweep/${RUN_ID}"
  local DRIVE_SUB="${DOM}/${STEM}/${RUN_ID}"

  if result_already_pushed "${DRIVE_SUB}"; then
    echo "[multivideo]   skip (resume): ${RUN_ID}"
    return 0
  fi
  if [[ ! -f "${FAST_W}" || ! -f "${ACC_W}" || ! -f "${VID}" ]]; then
    echo "[multivideo]   SKIP missing input: ${RUN_ID}"
    [[ ! -f "${FAST_W}" ]] && echo "      fast: ${FAST_W}"
    [[ ! -f "${ACC_W}" ]]  && echo "      acc:  ${ACC_W}"
    [[ ! -f "${VID}" ]]    && echo "      vid:  ${VID}"
    return 0
  fi

  echo "[multivideo]   run: ${RUN_ID}"
  python -u -m experiments.run_sweep \
    --video             "${VID}" \
    --fast-weights      "${FAST_W}" \
    --accurate-weights  "${ACC_W}" \
    --device            "${DEV}" \
    --imgsz             "${IMGSZ}" \
    --max-frames        0 \
    --proxy-size        "${PROXY_SIZE}" \
    --warmup-frames     "${WARMUP}" \
    --conf-ema-c-high   "${CONF_EMA_C_HIGH}" \
    --conf-ema-c-low    "${CONF_EMA_C_LOW}" \
    --policies "${POLICIES[@]}" \
    --run-id            "${RUN_ID}"

  if ! python -u -m experiments.validate_sweep \
        --run-dir "${LOCAL_OUT}"; then
    echo "[multivideo]   WARN validate flagged: ${RUN_ID}"
  fi

  python -u -m analysis.check_policy_schema \
    "${LOCAL_OUT}/sweep_per_frame.csv" \
    --policy local_contrast_hyst

  # Push to Drive immediately so SSH drops never lose progress.
  rclone copy "${LOCAL_OUT}" \
    "${RESULTS_FOLDER}/${DRIVE_SUB}/" -v >/dev/null

  # Trim the per-frame CSV locally to save disk; keep summary.
  if [[ -f "${LOCAL_OUT}/sweep_per_frame.csv" ]]; then
    rm -f "${LOCAL_OUT}/sweep_per_frame.csv"
  fi
}

run_one_video() {
  # Args: domain video_filename
  local DOM="$1" VIDEO_FN="$2"
  local STEM
  STEM=$(basename "${VIDEO_FN}" | sed -E 's/\.(mp4|MP4|mov|MOV|avi|AVI)$//')
  local LOCAL_VID="${SCRATCH}/${VIDEO_FN}"

  echo
  echo "============================================================"
  echo "  VIDEO: ${DOM}/${VIDEO_FN}"
  echo "============================================================"

  # Decide which pairs to run.
  local PAIRS_ARR
  if [[ "${DOM}" == "glass" ]]; then
    PAIRS_ARR=("${GLASS_PAIRS[@]}")
  else
    PAIRS_ARR=("${PORCELAIN_PAIRS[@]}")
  fi

  # Enumerate (pair, device) combos we'll run.
  local need_video=0
  for pair in "${PAIRS_ARR[@]}"; do
    IFS='|' read -r FAM FAST ACC DOM2 <<< "${pair}"
    for DEV in cuda cpu; do
      [[ "${DEV}" == "cuda" && "${SKIP_GPU}" == "1" ]] && continue
      [[ "${DEV}" == "cpu"  && "${SKIP_CPU}" == "1" ]] && continue
      if [[ "${DEV}" == "cpu" && "${CPU_PAIRS_FILTER}" != "all" ]]; then
        # Only the n_l pair on CPU by default.
        [[ "${ACC}" != "l" ]] && continue
      fi
      local RUN_ID="multi_${STEM}_${FAM}_${FAST}_${ACC}_${DOM2}_${DEV}"
      local DRIVE_SUB="${DOM2}/${STEM}/${RUN_ID}"
      if ! result_already_pushed "${DRIVE_SUB}"; then
        need_video=1
      fi
    done
  done

  if [[ "${need_video}" == "0" ]]; then
    echo "[multivideo]   nothing to do (all combos already pushed)"
    return 0
  fi

  # Pull video.
  if [[ ! -f "${LOCAL_VID}" ]]; then
    echo "[multivideo]   downloading ${VIDEO_FN}..."
    rclone copy "${VIDEOS_FOLDER}${DOM}/${VIDEO_FN}" "${SCRATCH}/" -v >/dev/null
  fi
  if [[ ! -f "${LOCAL_VID}" ]]; then
    echo "[multivideo]   ERROR: download failed for ${VIDEO_FN}"
    return 1
  fi

  # Run all (pair, device) combos.
  for pair in "${PAIRS_ARR[@]}"; do
    IFS='|' read -r FAM FAST ACC DOM2 <<< "${pair}"
    for DEV in cuda cpu; do
      [[ "${DEV}" == "cuda" && "${SKIP_GPU}" == "1" ]] && continue
      [[ "${DEV}" == "cpu"  && "${SKIP_CPU}" == "1" ]] && continue
      if [[ "${DEV}" == "cpu" && "${CPU_PAIRS_FILTER}" != "all" ]]; then
        [[ "${ACC}" != "l" ]] && continue
      fi
      run_one_pair "${FAM}" "${FAST}" "${ACC}" "${DOM2}" "${DEV}" \
                   "${LOCAL_VID}" "${STEM}"
    done
  done

  # Cleanup local video after all pairs done.
  echo "[multivideo]   cleanup: removing ${LOCAL_VID}"
  rm -f "${LOCAL_VID}"
}

# ============================================================================
# Main
# ============================================================================
echo "============================================================"
echo "  multi-video sweep starting"
echo "  videos folder: ${VIDEOS_FOLDER}"
echo "  results folder: ${RESULTS_FOLDER}"
echo "  CPU pairs: ${CPU_PAIRS_FILTER}   skip_cpu=${SKIP_CPU}   skip_gpu=${SKIP_GPU}"
echo "  conf_ema c_high=${CONF_EMA_C_HIGH}, c_low=${CONF_EMA_C_LOW}"
echo "============================================================"

for DOM in glass porcelain; do
  echo
  echo "============================================================"
  echo "  DOMAIN: ${DOM}"
  echo "============================================================"
  echo "[multivideo] listing ${DOM}/ on Drive..."
  VIDEO_LIST=$(rclone_lsf_safe "${VIDEOS_FOLDER}${DOM}/" \
    | grep -iE '\.(mp4|mov|avi)$' | sort)
  if [[ -z "${VIDEO_LIST}" ]]; then
    echo "[multivideo]   no videos in ${DOM}/ on Drive"
    continue
  fi
  while IFS= read -r v; do
    [[ -z "${v}" ]] && continue
    run_one_video "${DOM}" "${v}" || \
      echo "[multivideo]   WARN: failed on ${DOM}/${v}, continuing"
  done <<< "${VIDEO_LIST}"
done

echo
echo "============================================================"
echo "  multi-video sweep DONE"
echo "============================================================"
echo "  results pushed to ${RESULTS_FOLDER}/<domain>/<video>/<run_id>/"
echo "  local sweep dirs (summaries only) under ${ROOT}/results/stage_sweep/"
echo
echo "  next: pull results locally and rebuild master_table:"
echo "    rclone copy '${RESULTS_FOLDER}/' results/multi_video/ -v"
echo "    python -m experiments.build_master_table --results-root results"
