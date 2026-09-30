#!/usr/bin/env bash
# Review-gated final multi-video DMS sweep.
#
# This script is prepared for the final multi-video CPU/GPU sweep, but it
# refuses to run unless CONFIRM_FULL_SWEEP=1 is set. Use PLAN_ONLY=1 to list
# planned runs without downloading videos or launching inference.
#
# Paste-friendly launch after review:
#   cd ~/dms_work/dms-thesis-clean
#   PLAN_ONLY=1 bash experiments/run_multi_video_sweep_final.sh
#
# Actual launch after review:
#   nohup bash -lc 'CONFIRM_FULL_SWEEP=1 bash experiments/run_multi_video_sweep_final.sh' \
#     > multivideo_final.log 2>&1 &
#   echo $! > multivideo_final.pid
#   tail -f multivideo_final.log
#
set -euo pipefail

VIDEOS_FOLDER="${VIDEOS_FOLDER:-gdrive,root_folder_id=1tYUHzAGnSqhunkeKgNc4BSAtDu4Ofrbz:}"
RESULTS_FOLDER="${RESULTS_FOLDER:-gdrive,root_folder_id=1UcqxSCcMuEtmIq72JS4meQYIQjgRUJn7:dms_thesis_clean/multi_video_final}"
ROOT="${ROOT:-${HOME}/dms_work/dms-thesis-clean}"
WEIGHTS_ROOT="${WEIGHTS_ROOT:-${HOME}/training_runs}"
SCRATCH="${SCRATCH:-${HOME}/scratch_videos}"
IMGSZ="${IMGSZ:-640}"
PROXY_SIZE="${PROXY_SIZE:-160}"
WARMUP="${WARMUP:-3}"
MAX_FRAMES="${MAX_FRAMES:-0}"
CONF_EMA_C_HIGH="${CONF_EMA_C_HIGH:-0.02}"
CONF_EMA_C_LOW="${CONF_EMA_C_LOW:-0.007}"
CPU_PAIRS_FILTER="${CPU_PAIRS_FILTER:-n_l}"
PAIR_FILTER="${PAIR_FILTER:-all}"
VIDEO_LIMIT_PER_DOMAIN="${VIDEO_LIMIT_PER_DOMAIN:-0}"
SKIP_CPU="${SKIP_CPU:-0}"
SKIP_GPU="${SKIP_GPU:-0}"
FORCE_RERUN="${FORCE_RERUN:-0}"
PLAN_ONLY="${PLAN_ONLY:-0}"
CONFIRM_FULL_SWEEP="${CONFIRM_FULL_SWEEP:-0}"
PLANNED_RUNS=0
SKIPPED_EXISTING=0
SKIPPED_MISSING_INPUT=0
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

cd "${ROOT}"
mkdir -p "${SCRATCH}"

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

rclone_lsf_safe() {
  rclone lsf --files-only "$1" 2>/dev/null || true
}

result_already_pushed() {
  local subpath="$1"
  if [[ "${FORCE_RERUN}" == "1" ]]; then
    return 1
  fi
  local summary_path="${RESULTS_FOLDER}/${subpath}/sweep_summary.json"
  rclone lsjson --stat "${summary_path}" >/dev/null 2>&1
}

pair_enabled() {
  local fam="$1"
  local fast="$2"
  local acc="$3"
  local key="${fam}_${fast}_${acc}"
  [[ "${PAIR_FILTER}" == "all" || "${PAIR_FILTER}" == "${key}" ]]
}

verify_upload() {
  local drive_path="$1"
  local listing
  listing="$(rclone_lsf_safe "${drive_path}/")"
  local missing=0

  for name in sweep_per_frame.csv frame_features.csv sweep_summary.json; do
    if ! grep -qx "${name}" <<< "${listing}"; then
      echo "[multivideo] ERROR upload missing ${drive_path}/${name}"
      missing=1
    fi
  done

  if [[ -f "${2:-}/run.log" ]] && ! grep -qx 'run.log' <<< "${listing}"; then
    echo "[multivideo] ERROR upload missing ${drive_path}/run.log"
    missing=1
  fi

  [[ "${missing}" == "0" ]]
}

schema_check() {
  local run_dir="$1"
  RUN_DIR="${run_dir}" python - <<'PY'
import csv
import math
import os
from pathlib import Path

run_dir = Path(os.environ["RUN_DIR"])
required = {
    "t_total_deployed_ms",
    "t_total_policy_relevant_ms",
    "t_total_shared_proxy_ms",
}

for name in ("sweep_per_frame.csv", "frame_features.csv"):
    path = run_dir / name
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = set(reader.fieldnames or [])
    missing = sorted(required - fields)
    if missing:
        raise SystemExit(f"{path}: missing columns {missing}")
    if name == "sweep_per_frame.csv":
        for col in required:
            bad = []
            for i, row in enumerate(rows):
                value = row.get(col, "")
                try:
                    is_bad = value == "" or math.isnan(float(value))
                except ValueError:
                    is_bad = True
                if is_bad:
                    bad.append(i)
            if bad:
                raise SystemExit(f"{path}: bad {col} rows {bad[:5]}")
    print(f"[schema] {name}: rows={len(rows)} cols={len(fields)} OK")
PY
}

run_one_pair() {
  local fam="$1"
  local fast="$2"
  local acc="$3"
  local domain="$4"
  local device="$5"
  local video_path="$6"
  local video_stem="$7"

  local fast_w="${WEIGHTS_ROOT}/${fam}_${fast}_${domain}/weights/best.pt"
  local acc_w="${WEIGHTS_ROOT}/${fam}_${acc}_${domain}/weights/best.pt"
  local pair_type="${fast}_${acc}"
  local run_id="multi_${domain}_${video_stem}_${fam}_${pair_type}_${device}"
  local local_out="results/stage_sweep/${run_id}"
  local drive_sub="${domain}/${video_stem}/${run_id}"
  local drive_out="${RESULTS_FOLDER}/${drive_sub}"
  local run_log="${local_out}/run.log"

  if result_already_pushed "${drive_sub}"; then
    echo "SKIP existing ${run_id}"
    SKIPPED_EXISTING=$((SKIPPED_EXISTING + 1))
    return 0
  fi

  if [[ "${PLAN_ONLY}" == "1" ]]; then
    PLANNED_RUNS=$((PLANNED_RUNS + 1))
    echo
    echo "RUN PLAN: ${run_id}"
    echo "  WOULD RUN      ${run_id}"
    echo "  WOULD VALIDATE ${run_id}"
    echo "  WOULD CHECK    local_contrast_hyst policy schema"
    echo "  WOULD UPLOAD   sweep_per_frame.csv frame_features.csv sweep_summary.json run.log"
    echo "  TO             ${drive_out}/"
    return 0
  fi

  if [[ ! -f "${fast_w}" || ! -f "${acc_w}" || ! -f "${video_path}" ]]; then
    echo "[multivideo] SKIP missing input: ${run_id}"
    [[ ! -f "${fast_w}" ]] && echo "  fast: ${fast_w}"
    [[ ! -f "${acc_w}" ]] && echo "  acc:  ${acc_w}"
    [[ ! -f "${video_path}" ]] && echo "  vid:  ${video_path}"
    SKIPPED_MISSING_INPUT=$((SKIPPED_MISSING_INPUT + 1))
    return 1
  fi

  PLANNED_RUNS=$((PLANNED_RUNS + 1))
  echo "[multivideo] run: ${run_id}"
  rm -rf "${local_out}"
  mkdir -p "${local_out}"

  python -u -m experiments.run_sweep \
    --video="${video_path}" \
    --fast-weights="${fast_w}" \
    --accurate-weights="${acc_w}" \
    --device="${device}" \
    --imgsz="${IMGSZ}" \
    --max-frames="${MAX_FRAMES}" \
    --proxy-size="${PROXY_SIZE}" \
    --warmup-frames="${WARMUP}" \
    --conf-ema-c-high="${CONF_EMA_C_HIGH}" \
    --conf-ema-c-low="${CONF_EMA_C_LOW}" \
    --policies "${POLICIES[@]}" \
    --run-id="${run_id}" \
    --domain="${domain}" \
    --model-family="${fam}" \
    --pair-type="${pair_type}" \
    --video-id="${video_stem}" 2>&1 | tee "${run_log}"

  python -u -m experiments.validate_sweep \
    --run-dir="${local_out}" 2>&1 | tee -a "${run_log}"

  schema_check "${local_out}" 2>&1 | tee -a "${run_log}"
  python -u -m analysis.check_policy_schema \
    "${local_out}/sweep_per_frame.csv" \
    --policy local_contrast_hyst 2>&1 | tee -a "${run_log}"

  rclone copy "${local_out}" "${drive_out}/" -v
  verify_upload "${drive_out}" "${local_out}"

  rm -f "${local_out}/sweep_per_frame.csv"
  rm -f "${local_out}/frame_features.csv"
}

run_one_video() {
  local domain="$1"
  local video_name="$2"
  local video_stem
  video_stem="$(basename "${video_name}" | sed -E 's/\.(mp4|MP4|mov|MOV|avi|AVI)$//')"
  local local_video="${SCRATCH}/${video_name}"
  local drive_video="${VIDEOS_FOLDER}${domain}/${video_name}"

  echo
  echo "============================================================"
  echo "VIDEO ${domain}/${video_name}"
  echo "============================================================"

  local pairs=()
  if [[ "${domain}" == "glass" ]]; then
    pairs=("${GLASS_PAIRS[@]}")
  else
    pairs=("${PORCELAIN_PAIRS[@]}")
  fi

  local total_runs=0
  local existing_runs=0
  local existing_run_ids=()
  local pending_tuples=()
  for pair in "${pairs[@]}"; do
    IFS='|' read -r fam fast acc pair_domain <<< "${pair}"
    pair_enabled "${fam}" "${fast}" "${acc}" || continue
    for device in cuda cpu; do
      [[ "${device}" == "cuda" && "${SKIP_GPU}" == "1" ]] && continue
      [[ "${device}" == "cpu" && "${SKIP_CPU}" == "1" ]] && continue
      if [[ "${device}" == "cpu" && "${CPU_PAIRS_FILTER}" != "all" ]]; then
        [[ "${acc}" != "l" ]] && continue
      fi

      local pair_type="${fast}_${acc}"
      local run_id="multi_${pair_domain}_${video_stem}_${fam}_${pair_type}_${device}"
      local drive_sub="${pair_domain}/${video_stem}/${run_id}"
      total_runs=$((total_runs + 1))

      if result_already_pushed "${drive_sub}"; then
        existing_runs=$((existing_runs + 1))
        existing_run_ids+=("${run_id}")
      else
        pending_tuples+=("${fam}|${fast}|${acc}|${pair_domain}|${device}")
      fi
    done
  done

  if [[ "${total_runs}" == "0" ]]; then
    echo "[multivideo] no enabled runs for ${domain}/${video_name}"
    return 0
  fi

  if [[ "${existing_runs}" == "${total_runs}" ]]; then
    echo "SKIP video ${domain}/${video_name}: all planned runs already exist"
    SKIPPED_EXISTING=$((SKIPPED_EXISTING + existing_runs))
    return 0
  fi

  local run_id
  for run_id in "${existing_run_ids[@]}"; do
    echo "SKIP existing ${run_id}"
    SKIPPED_EXISTING=$((SKIPPED_EXISTING + 1))
  done

  if [[ "${PLAN_ONLY}" == "1" ]]; then
    echo "WOULD DOWNLOAD ${drive_video} -> ${local_video}"
  elif [[ ! -s "${local_video}" ]]; then
    echo "[multivideo] downloading ${drive_video} -> ${local_video}"
    rclone copy "${drive_video}" "${SCRATCH}/" -v
  fi

  if [[ "${PLAN_ONLY}" != "1" && ! -s "${local_video}" ]]; then
    echo "[multivideo] ERROR: missing or empty local video after download: ${local_video}"
    return 1
  fi

  local upload_ok=1
  local tuple
  for tuple in "${pending_tuples[@]}"; do
    IFS='|' read -r fam fast acc pair_domain device <<< "${tuple}"
    run_one_pair "${fam}" "${fast}" "${acc}" "${pair_domain}" \
      "${device}" "${local_video}" "${video_stem}" || upload_ok=0
  done

  if [[ "${PLAN_ONLY}" == "1" ]]; then
    echo "WOULD DELETE ${local_video} after successful upload"
  elif [[ "${upload_ok}" == "1" ]]; then
    echo "[multivideo] cleanup video after successful uploads: ${local_video}"
    rm -f "${local_video}"
  else
    echo "[multivideo] preserving video because at least one run/upload failed: ${local_video}"
    return 1
  fi
}

if [[ "${PLAN_ONLY}" != "1" && "${CONFIRM_FULL_SWEEP}" != "1" ]]; then
  echo "Refusing to launch full sweep without CONFIRM_FULL_SWEEP=1."
  echo "Review first with:"
  echo "  PLAN_ONLY=1 bash experiments/run_multi_video_sweep_final.sh"
  exit 2
fi

echo "============================================================"
echo "FINAL MULTI-VIDEO SWEEP"
echo "videos: ${VIDEOS_FOLDER}"
echo "results: ${RESULTS_FOLDER}"
echo "max_frames: ${MAX_FRAMES}"
echo "cpu_pairs: ${CPU_PAIRS_FILTER}"
echo "pair_filter: ${PAIR_FILTER}"
echo "video_limit_per_domain: ${VIDEO_LIMIT_PER_DOMAIN}"
echo "skip_cpu=${SKIP_CPU} skip_gpu=${SKIP_GPU}"
echo "plan_only=${PLAN_ONLY}"
echo "============================================================"

for domain in glass porcelain; do
  echo
  echo "[multivideo] listing ${domain}/"
  video_list="$(rclone_lsf_safe "${VIDEOS_FOLDER}${domain}/" \
    | grep -iE '\.(mp4|mov|avi)$' | sort || true)"
  if [[ -z "${video_list}" ]]; then
    echo "[multivideo] no videos found for ${domain}"
    continue
  fi
  n_videos=0
  while IFS= read -r video_name; do
    [[ -z "${video_name}" ]] && continue
    if [[ "${VIDEO_LIMIT_PER_DOMAIN}" != "0" && "${n_videos}" -ge "${VIDEO_LIMIT_PER_DOMAIN}" ]]; then
      break
    fi
    run_one_video "${domain}" "${video_name}"
    n_videos=$((n_videos + 1))
  done <<< "${video_list}"
done

echo
echo "DONE. Results path:"
echo "  ${RESULTS_FOLDER}/<domain>/<video>/<run_id>/"
echo "planned_runs=${PLANNED_RUNS}"
echo "skipped_existing=${SKIPPED_EXISTING}"
echo "skipped_missing_input=${SKIPPED_MISSING_INPUT}"
