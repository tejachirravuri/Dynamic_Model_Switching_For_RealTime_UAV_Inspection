"""experiments/run_sweep.py — full policy sweep.

Full policy sweep on a single video. For each frame we:

  1. Compute scene proxies once (one resize, one Sobel, etc.).
  2. Run BOTH models once (n=fast and s=accurate). Their detections
     and per-model latencies are reused across all policies.
  3. For each policy in the canonical list, ask the policy to decide
     'n' or 's' for this frame. The policy's "output" is whichever
     model's detections it picked; the policy's "latency" is whichever
     model's wall-clock it picked, plus the proxy + decision overhead
     (overhead is reported separately so it can be added or omitted).

T_total semantics (logged in `t_total_selected_ms` / `t_total_deployed_ms`)
--------------------------------------------------------------------------
* n_only:                   T_total = T_fast
* s_only:                   T_total = T_accurate
* scene policies            (entropy_only, combined, combined_hyst,
  multi_proxy):             T_total = T_scene + T_ctrl + T_selected
* conf_ema (deployed):      T_total = T_fast + T_ctrl + I[s] * T_accurate
* conf_ema (selected):      T_total =          T_ctrl + T_selected
                            (oracle / counterfactual: cost of the chosen
                             model alone, ignoring conf_ema's mandatory
                             fast pass; useful for comparing against
                             scene policies on equal footing)

Note: `frame_skip` (CLI flag) is VIDEO SUBSAMPLING (process every K-th
raw frame). It is *not* the same as `proxy_stride` (recompute proxies
every K frames, reusing cached values in between) — that's an offline
replay knob and is handled in `analysis/replay_proxy_stride.py`.

Why precompute both models per frame?
-------------------------------------
This is offline analysis, not deployment. Running each policy in its
own pass would cost 7x the inference work and would not change any
metric — the per-frame decision is deterministic given the proxies
and the post-inference signal. By running both models once and
multiplexing across policies, the sweep is cheap and every policy
sees identical per-frame inputs (which is the only fair comparison).

Output
------
results/<stage>/<run_id>/
    sweep_per_frame.csv     long-format: one row per (frame, policy)
    sweep_summary.json      one summary block per policy + run config

Per-frame fields
----------------
    frame_idx
    policy                  one of the 7 canonical names
    choice                  'n' or 's'
    latency_ms              wall-clock of the chosen model
    proxy_ms                proxy compute time (same across policies)
    decision_ms             policy decide() overhead
    n_count, s_count        post-conf-floor det counts (same across policies)
    fast_mean_conf          post-conf-floor mean of fast-model scores;
                            same across all policy rows for a given
                            frame. Diagnostic for conf_ema replay.
    chosen_count            post-conf-floor det count for the chosen model
    iou_match               strict 1:1 vs s_only
    det_coverage            relaxed binary recall
    det_recall              relaxed continuous recall
    count_agree             cheap count-only secondary
    benefit_positive        1 iff there exists an accurate det that the
                            FAST model couldn't match (LOCKED definition;
                            not the policy's chosen-vs-reference)

CLI
---
python -m experiments.run_sweep \
    --video <path> \
    --fast-weights <pt> \
    --accurate-weights <pt> \
    --device cpu \
    --proxy-size 160 \
    --max-frames 100 \
    --policies n_only s_only entropy_only combined combined_hyst \
               conf_ema multi_proxy \
    --run-id sweep_glass_cpu100
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from dms.inference import Detections, UltralyticsYOLOBackend, time_inference
from dms.metrics import (
    benefit_positive_frame,
    count_agree_frame,
    det_coverage_frame,
    det_recall_frame,
    iou_match_frame,
    summarise_run,
)
from dms.policies import (
    CANONICAL_POLICIES,
    FrameFeatures,
    POLICY_REGISTRY,
    PolicyConfig,
    make_policy,
)
from dms.proxies import ProxyConfig, compute_proxies


_NAN = float("nan")

# Scene features computed by `compute_proxies` and logged in the per-row
# schema. The list excludes `brightness_mean` because the runner adds it
# unconditionally so brightness_mean / brightness_std / contrast aliases
# are always populated (see ProxyConfig assembly below).
_SCENE_FEATURES = ("L", "H", "color_entropy", "tenengrad",
                   "edge_density", "local_contrast",
                   "bright_fraction", "hue_std", "brightness_mean")

# Map sweep-row column name -> proxies.compute_proxies key. Lets the
# CSV column "laplacian" / "entropy" point at the existing L / H values
# without renaming the locked feature keys in dms/proxies.py.
_VALUE_ALIASES = (
    ("laplacian", "L"),
    ("entropy", "H"),
    ("brightness_std", "local_contrast"),
    ("contrast", "local_contrast"),
)

# Aggregated per-feature timing decompositions:
#   proxy_ms_brightness = sum of timings for the gray-only intensity
#                          features (brightness_mean, local_contrast,
#                          bright_fraction)
#   proxy_ms_hsv        = sum of timings for the HSV-derived features
#                          (color_entropy, hue_std)
# These are diagnostic decompositions; the authoritative wall-clock
# is `proxy_ms_total`.
_BRIGHTNESS_GROUP = ("brightness_mean", "local_contrast", "bright_fraction")
_HSV_GROUP = ("color_entropy", "hue_std")

# Runtime-accounting policy groups. `proxy_ms_total` remains the measured
# shared proxy-block wall-clock. Policy-relevant proxy cost is an approximate
# feature-minimal estimate: preprocessing/gray conversion once, plus only the
# feature kernels the policy actually needs.
_SCENE_POLICY_PROXY_FEATURES = {
    "entropy_only": ("H",),
    "combined": ("L", "H"),
    "combined_hyst": ("L", "H"),
    "local_contrast_hyst": ("local_contrast",),
    "multi_proxy": ("L", "H", "tenengrad", "color_entropy"),
}


# ===========================================================================
# Per-frame, per-policy row
# ===========================================================================
@dataclass
class SweepRow:
    # ---- Run / frame metadata (broadcast across every policy row) ----
    video_id: str             # short logical id for the source video
    video_filename: str       # basename of the source video file
    domain: str               # 'glass' | 'porcelain' | ''  (free-form)
    device: str               # 'cpu' | 'cuda' | etc. (mirrors --device)
    model_family: str         # 'yolov8' | 'yolo26' | ''  (free-form)
    pair_type: str            # 'n_s' | 'n_l' | ''        (free-form)
    frame_idx: int            # index inside the timed loop (0..N-1)
    raw_video_idx: int        # absolute index in the source video (counts
                              # frame-skip-skipped frames as well)
    timestamp_sec: float      # raw_video_idx / fps_video (NaN if fps==0)
    imgsz: int                # detector inference size
    proxy_size: int           # ProxyConfig.size
    preserve_aspect: int      # 0/1
    conf_floor: float
    iou_threshold: float

    policy: str
    choice: str               # 'n' or 's'
    s_choice: int             # 1 if choice == 's' else 0

    # ---- Latencies (ms) ----
    n_latency_ms: float       # fast model, this frame
    s_latency_ms: float       # accurate model, this frame
    chosen_latency_ms: float  # whichever model the policy picked
    proxy_ms_total: float     # authoritative measured proxy-block wall-clock
    proxy_ms_policy_relevant: float  # approximate feature-minimal proxy cost
    proxy_overhead_ms: float  # resize + colour conversion only
    decision_ms: float        # policy.decide() overhead
    t_total_selected_ms: float
    t_total_deployed_ms: float
    t_total_shared_proxy_ms: float
    t_total_policy_relevant_ms: float

    # ---- Counts ----
    n_count: int
    s_count: int
    chosen_count: int

    # ---- Confidence summaries (post-conf-floor; per-frame; broadcast) ----
    # fast_*/n_* and accurate_*/s_* are aliases of the same values:
    # fast_/n_ refer to the fast (n) model, accurate_/s_ to the accurate
    # (s) model. fast_mean_conf is preserved as the locked column name
    # used by conf_ema's trigger contract.
    fast_mean_conf: float
    fast_max_conf: float
    accurate_mean_conf: float
    accurate_max_conf: float
    n_mean_conf: float        # alias of fast_mean_conf
    n_max_conf: float         # alias of fast_max_conf
    s_mean_conf: float        # alias of accurate_mean_conf
    s_max_conf: float         # alias of accurate_max_conf

    # ---- Scene proxy values (broadcast across policies) ----
    L: float
    H: float
    laplacian: float          # alias of L
    entropy: float            # alias of H
    color_entropy: float
    tenengrad: float
    edge_density: float
    local_contrast: float
    bright_fraction: float
    hue_std: float
    brightness_mean: float
    brightness_std: float     # alias of local_contrast (gray.std())
    contrast: float           # alias of local_contrast
    combined_score: float     # the Combined policy's per-frame C value
                              # (NaN if 'combined' isn't in --policies)

    # ---- Per-feature proxy timing (ms; approximate diagnostic) ----
    # Naming convention matches the spec: proxy_ms_<feature>.
    proxy_ms_laplacian: float
    proxy_ms_entropy: float
    proxy_ms_color_entropy: float
    proxy_ms_tenengrad: float
    proxy_ms_edge_density: float
    proxy_ms_local_contrast: float
    proxy_ms_bright_fraction: float
    proxy_ms_hue_std: float
    proxy_ms_brightness_mean: float
    # Aggregated decompositions (sum of group members):
    proxy_ms_brightness: float   # brightness_mean + local_contrast + bright_fraction
    proxy_ms_hsv: float          # color_entropy + hue_std

    # ---- Policy diagnostics (uniform; NaN where N/A) ----
    policy_score: float
    policy_thresh_low: float   # off-threshold for hysteretic policies
    policy_thresh_high: float  # on-threshold for hysteretic policies
    policy_thresh_mid: float   # single mid threshold (Combined only)
    fast_ema: float            # conf_ema only
    slow_ema: float            # conf_ema only
    conf_drop: float           # conf_ema only (= policy_score for conf_ema)

    # ---- Quality metrics (vs s_only reference) ----
    iou_match: int
    det_coverage: int
    det_recall: float
    count_agree: int
    benefit_positive: int


CSV_FIELDS = list(SweepRow.__dataclass_fields__.keys())


# ---------------------------------------------------------------------------
# Per-frame view (one row per frame, no policy dimension). Lets downstream
# scripts join sweep rows against scene features without redundancy.
# ---------------------------------------------------------------------------
FRAME_FEATURES_FIELDS = (
    # metadata
    "video_id", "video_filename", "domain", "device",
    "model_family", "pair_type",
    "frame_idx", "raw_video_idx", "timestamp_sec",
    "imgsz", "proxy_size", "preserve_aspect",
    "conf_floor", "iou_threshold",
    # detector summaries
    "n_count", "s_count",
    "n_latency_ms", "s_latency_ms",
    "fast_mean_conf", "fast_max_conf",
    "accurate_mean_conf", "accurate_max_conf",
    "n_mean_conf", "n_max_conf", "s_mean_conf", "s_max_conf",
    "fast_ema", "slow_ema", "conf_drop",
    "benefit_positive",
    # proxy values + aliases
    "L", "H", "laplacian", "entropy",
    "color_entropy", "tenengrad", "edge_density",
    "local_contrast", "bright_fraction", "hue_std",
    "brightness_mean", "brightness_std", "contrast",
    "combined_score",
    # proxy timing (per-feature is approximate; total is authoritative)
    "proxy_overhead_ms",
    "proxy_ms_laplacian", "proxy_ms_entropy",
    "proxy_ms_color_entropy", "proxy_ms_tenengrad",
    "proxy_ms_edge_density", "proxy_ms_local_contrast",
    "proxy_ms_bright_fraction", "proxy_ms_hue_std",
    "proxy_ms_brightness_mean",
    "proxy_ms_brightness", "proxy_ms_hsv",
    "proxy_ms_total",
    # policy-specific columns are present for schema stability but blank/NaN
    # in the one-row-per-frame view.
    "proxy_ms_policy_relevant", "decision_ms", "chosen_latency_ms",
    "t_total_selected_ms", "t_total_deployed_ms",
    "t_total_shared_proxy_ms", "t_total_policy_relevant_ms",
    "policy", "choice", "s_choice",
    "policy_score", "policy_thresh_low",
    "policy_thresh_high", "policy_thresh_mid",
    "iou_match", "det_coverage", "det_recall", "count_agree",
)


# ---------------------------------------------------------------------------
# T_total semantics (see module docstring)
# ---------------------------------------------------------------------------
def _t_total(
    policy_name: str,
    choice: str,
    n_ms: float,
    s_ms: float,
    proxy_ms: float,
    decision_ms: float,
) -> Tuple[float, float]:
    """Return (selected, deployed) total wall-clock for one (policy, frame).

    selected:  hypothetical cost if you ran ONLY the chosen pipeline.
    deployed:  realistic cost if you ran the policy in its actual deployment
               topology (conf_ema MUST always run fast to feed its trigger;
               scene policies MUST always compute proxies).
    """
    if policy_name == "n_only":
        return n_ms, n_ms
    if policy_name == "s_only":
        return s_ms, s_ms
    chosen = n_ms if choice == "n" else s_ms
    if policy_name == "conf_ema":
        # selected = chosen + ctrl (oracle / no proxy pipeline).
        # deployed = fast + ctrl + I[s] * accurate (always pay fast).
        sel = chosen + decision_ms
        dep = n_ms + decision_ms + (s_ms if choice == "s" else 0.0)
        return sel, dep
    # scene policies (entropy_only, combined, combined_hyst, multi_proxy)
    total = proxy_ms + decision_ms + chosen
    return total, total


def _proxy_ms_policy_relevant(
    policy_name: str,
    proxy_overhead_ms: float,
    proxy_times: Dict[str, float],
) -> float:
    """Approximate feature-minimal proxy time for one policy.

    This is a diagnostic estimate, not exact hardware profiling. The
    measured `proxy_ms_total` column remains authoritative for the current
    shared proxy implementation.
    """
    feats = _SCENE_POLICY_PROXY_FEATURES.get(policy_name)
    if not feats:
        return 0.0
    vals = [proxy_overhead_ms] + [proxy_times.get(f, _NAN) for f in feats]
    if any(math.isnan(v) for v in vals):
        return _NAN
    return float(sum(vals))


def _t_total_shared_proxy(
    proxy_ms_total: float,
    decision_ms: float,
    chosen_latency_ms: float,
) -> float:
    """Shared implementation formula requested for audit accounting."""
    return float(proxy_ms_total + decision_ms + chosen_latency_ms)


def _t_total_policy_relevant(
    policy_name: str,
    choice: str,
    n_ms: float,
    s_ms: float,
    proxy_ms_policy_relevant: float,
    decision_ms: float,
) -> float:
    """Feature-minimal accounting with conf_ema watchdog semantics."""
    if policy_name == "n_only":
        return float(n_ms)
    if policy_name == "s_only":
        return float(s_ms)
    if policy_name == "conf_ema":
        return float(n_ms + decision_ms + (s_ms if choice == "s" else 0.0))
    chosen = n_ms if choice == "n" else s_ms
    return float(proxy_ms_policy_relevant + decision_ms + chosen)


# ===========================================================================
# Helpers
# ===========================================================================
def _build_features(
    frame_idx: int,
    proxies: Dict[str, float],
    last_mean_conf: float,
) -> FrameFeatures:
    """Map a compute_proxies() dict onto the FrameFeatures dataclass."""
    return FrameFeatures(
        frame_idx=frame_idx,
        L=proxies.get("L"),
        H=proxies.get("H"),
        color_entropy=proxies.get("color_entropy"),
        tenengrad=proxies.get("tenengrad"),
        edge_density=proxies.get("edge_density"),
        local_contrast=proxies.get("local_contrast"),
        bright_fraction=proxies.get("bright_fraction"),
        hue_std=proxies.get("hue_std"),
        last_mean_conf=float(last_mean_conf),
    )


def _mean_conf(d: Detections) -> float:
    if len(d) == 0:
        return 0.0
    return float(d.scores.mean())


def _max_conf(d: Detections) -> float:
    """Post-conf-floor max confidence; 0.0 when there are no detections.

    Pair with `_mean_conf` for a cheap two-number summary of the detector's
    confidence distribution on this frame.
    """
    if len(d) == 0:
        return 0.0
    return float(d.scores.max())


# ===========================================================================
# CLI
# ===========================================================================
def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Full policy sweep on one video.",
    )
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--fast-weights", type=Path, required=True)
    p.add_argument("--accurate-weights", type=Path, required=True)
    p.add_argument("--device", type=str, default="cpu",
                   choices=("cpu", "cuda", "0", "1"))
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--max-frames", type=int, default=100)
    p.add_argument("--start-frame", type=int, default=0,
                   help="Seek to this frame index before the timed loop "
                        "(after warmup). Useful when early frames are "
                        "transit / no insulators in view.")
    p.add_argument("--frame-skip", type=int, default=1,
                   help="VIDEO SUBSAMPLING (Case A). Process every K-th "
                        "raw video frame; cap.read() advances but skipped "
                        "frames are not fed to either backend. K=1 = every "
                        "frame. This is a deployment scenario ('what if "
                        "we can't keep up with the frame rate?'), NOT "
                        "proxy-stride amortization. For proxy-stride (Case "
                        "B: evaluate every frame but recompute proxy every "
                        "K frames, reusing cached proxy values in between) "
                        "see analysis/replay_proxy_stride.py which derives "
                        "it offline from a per-frame features trace.")
    p.add_argument("--warmup-frames", type=int, default=3,
                   help="Per-backend warmup iterations on a discarded "
                        "frame before the timed loop. Default 3 absorbs "
                        "CUDA kernel-compile cost on first call.")
    p.add_argument("--conf-floor", type=float, default=0.25)
    p.add_argument("--iou-threshold", type=float, default=0.5)

    p.add_argument("--proxy-size", type=int, default=160)
    p.add_argument("--preserve-aspect", action="store_true")
    p.add_argument("--proxy-features", type=str, nargs="+",
                   default=["L", "H", "color_entropy",
                            "tenengrad", "edge_density", "local_contrast",
                            "bright_fraction", "hue_std",
                            "brightness_mean"])

    p.add_argument("--policies", type=str, nargs="+",
                   default=list(CANONICAL_POLICIES),
                   help=f"Default: {' '.join(CANONICAL_POLICIES)}")
    # ----- conf_ema overrides ---------------------------------------------
    # PolicyConfig defaults are the locked thesis baseline; CLI overrides
    # let us sweep thresholds (e.g., for a sensitivity analysis) without
    # mutating that baseline. Pass None to keep the default.
    p.add_argument("--conf-ema-c-high", type=float, default=None,
                   help="Override PolicyConfig.conf_ema_c_high "
                        "(default 0.12). Lower to make conf_ema more "
                        "responsive.")
    p.add_argument("--conf-ema-c-low", type=float, default=None,
                   help="Override PolicyConfig.conf_ema_c_low "
                        "(default 0.04). Hysteresis off-threshold; "
                        "must be < c_high.")
    p.add_argument("--conf-ema-fast-beta", type=float, default=None,
                   help="Override PolicyConfig.conf_ema_fast_beta "
                        "(default 0.30). Higher = faster reaction.")
    p.add_argument("--conf-ema-slow-beta", type=float, default=None,
                   help="Override PolicyConfig.conf_ema_slow_beta "
                        "(default 0.02). Lower = longer baseline.")
    p.add_argument("--results-root", type=Path,
                   default=Path(__file__).resolve().parent.parent
                   / "results" / "stage_sweep")
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--use-classes-in-match", action="store_true")

    # ---- Run metadata (logged into every per-row CSV record) -------------
    # Free-form strings; downstream multi-video aggregation uses them as
    # join keys for per-domain / per-pair / per-platform analysis.
    p.add_argument("--video-id", type=str, default=None,
                   help="Logical video id (e.g. '20190916-722'). Defaults "
                        "to the source video's stem.")
    p.add_argument("--domain", type=str, default="",
                   help="'glass' | 'porcelain' | other. Logged per row.")
    p.add_argument("--model-family", type=str, default="",
                   help="'yolov8' | 'yolo26' | other. Logged per row.")
    p.add_argument("--pair-type", type=str, default="",
                   help="'n_s' | 'n_l' | other. Logged per row.")
    return p.parse_args(argv)


# ===========================================================================
# Main
# ===========================================================================
def main(argv: Optional[list] = None) -> int:
    import cv2  # local import

    args = parse_args(argv)

    # Validate policy names early.
    bad = [pn for pn in args.policies if pn not in POLICY_REGISTRY]
    if bad:
        print(f"[run_sweep] ERROR: unknown policies: {bad}. "
              f"valid: {list(POLICY_REGISTRY)}", file=sys.stderr)
        return 2

    if args.run_id:
        run_id = args.run_id
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        run_id = f"{args.video.stem}_{args.device}_p{args.proxy_size}_{ts}"
    out_dir = args.results_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "sweep_per_frame.csv"
    summary_path = out_dir / "sweep_summary.json"

    print(f"[run_sweep] writing to {out_dir}")
    print(f"[run_sweep] policies: {args.policies}")

    # ----- Backends -------------------------------------------------------
    print(f"[run_sweep] loading fast weights:     {args.fast_weights}")
    fast_backend = UltralyticsYOLOBackend(
        weights_path=str(args.fast_weights),
        device=args.device, imgsz=args.imgsz, conf=0.0, iou_nms=0.7,
    )
    print(f"[run_sweep] loading accurate weights: {args.accurate_weights}")
    acc_backend = UltralyticsYOLOBackend(
        weights_path=str(args.accurate_weights),
        device=args.device, imgsz=args.imgsz, conf=0.0, iou_nms=0.7,
    )

    # ----- Policies -------------------------------------------------------
    # Start from thesis defaults; apply per-CLI overrides for the conf_ema
    # knobs only (other policies are not parameterised on the CLI yet).
    cfg_kwargs: Dict[str, float] = {}
    if args.conf_ema_c_high is not None:
        cfg_kwargs["conf_ema_c_high"] = float(args.conf_ema_c_high)
    if args.conf_ema_c_low is not None:
        cfg_kwargs["conf_ema_c_low"] = float(args.conf_ema_c_low)
    if args.conf_ema_fast_beta is not None:
        cfg_kwargs["conf_ema_fast_beta"] = float(args.conf_ema_fast_beta)
    if args.conf_ema_slow_beta is not None:
        cfg_kwargs["conf_ema_slow_beta"] = float(args.conf_ema_slow_beta)
    cfg = PolicyConfig(**cfg_kwargs) if cfg_kwargs else PolicyConfig()
    if cfg_kwargs:
        print(f"[run_sweep] conf_ema overrides: {cfg_kwargs}")
    policies = {pn: make_policy(pn, cfg) for pn in args.policies}

    # Confidence-feedback signal for conf_ema:
    # ALWAYS the previous frame's FAST-model mean confidence, regardless
    # of which model the policy actually picked. Mixing fast and accurate
    # confidence distributions across the EMA history would corrupt the
    # trigger (fast and accurate calibrate to different score scales).
    # Same value broadcast to every policy; only conf_ema reads it.
    last_fast_mean_conf: float = 0.0

    # ----- Proxy config ---------------------------------------------------
    # Always include required schema features so default runs do not emit
    # empty proxy-value columns. brightness_std and contrast are aliases of
    # `local_contrast` (same compute kernel).
    requested_feats = list(args.proxy_features)
    for required_feat in ("bright_fraction", "hue_std", "brightness_mean"):
        if required_feat not in requested_feats:
            requested_feats.append(required_feat)
    proxy_cfg = ProxyConfig(
        size=int(args.proxy_size),
        preserve_aspect=bool(args.preserve_aspect),
        features=tuple(requested_feats),
    )

    # ----- Per-row metadata (constants for the run) -----------------------
    video_id = args.video_id or args.video.stem
    video_filename = args.video.name

    # ----- Video ----------------------------------------------------------
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"[run_sweep] ERROR: cannot open video {args.video}",
              file=sys.stderr)
        return 2
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or -1
    fps_video = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[run_sweep] video: {n_total} frames, {fps_video:.2f} fps, "
          f"{width}x{height}")

    cap_n = args.max_frames if args.max_frames > 0 else n_total

    # ----- GPU warmup ------------------------------------------------------
    # Run each backend on a discarded frame so cuDNN kernel compilation,
    # workspace allocation, and any first-call overhead are paid BEFORE
    # the timed loop. Without this, on CUDA the first call can be
    # 100-1000x steady-state, which inflates the per-policy mean and
    # makes G1 (n_only < s_only) fail spuriously.
    if args.warmup_frames > 0:
        ok, warmup_frame = cap.read()
        if ok and warmup_frame is not None:
            print(f"[run_sweep] warming up backends "
                  f"({args.warmup_frames} iters each)...")
            for _ in range(args.warmup_frames):
                _ = fast_backend.predict(warmup_frame)
                _ = acc_backend.predict(warmup_frame)
        # Always rewind so the timed loop starts at the user-chosen
        # start-frame regardless of warmup success.
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if args.start_frame > 0:
        print(f"[run_sweep] seeking to frame {args.start_frame}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.start_frame))

    if args.frame_skip < 1:
        print(f"[run_sweep] ERROR: --frame-skip must be >= 1 "
              f"(got {args.frame_skip})", file=sys.stderr)
        return 2
    if args.frame_skip > 1:
        print(f"[run_sweep] frame-skip K={args.frame_skip} "
              f"(every {args.frame_skip}-th frame is processed)")

    csv_file = csv_path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    writer.writeheader()
    csv_file.flush()

    # frame_features.csv: one row per frame (no policy dimension). The
    # sweep CSV broadcasts these values across policy rows, but the
    # de-duplicated view is what trigger_validity.py expects.
    frame_features_path = out_dir / "frame_features.csv"
    ff_file = frame_features_path.open("w", newline="", encoding="utf-8")
    ff_writer = csv.DictWriter(ff_file, fieldnames=list(FRAME_FEATURES_FIELDS))
    ff_writer.writeheader()
    ff_file.flush()

    # ----- Per-policy collection for summarise_run ------------------------
    per_policy_records: Dict[str, list] = {pn: [] for pn in args.policies}
    sweep_rows_all: List[SweepRow] = []

    frame_idx = 0
    raw_idx = 0   # absolute video frame index (counts skipped frames too)
    t_run0 = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if cap_n > 0 and frame_idx >= cap_n:
                break

            # Frame-skip: process every K-th raw frame; advance raw_idx
            # on the rest without inference. cap.read() must still happen
            # (sequential decode), but we don't pay any compute.
            if args.frame_skip > 1 and (raw_idx % args.frame_skip) != 0:
                raw_idx += 1
                continue
            raw_idx += 1

            this_raw_idx = raw_idx - 1   # post-increment above; this is
                                         # the absolute video index of
                                         # the frame we just consumed.
            timestamp_sec = (float(this_raw_idx) / fps_video
                             if fps_video > 0.0 else _NAN)

            # ----- Proxies (once per frame) ----------------------------
            prox_out = compute_proxies(frame, proxy_cfg)
            proxy_ms_total = float(prox_out["time_ms"])

            # ----- Both models (once per frame) ------------------------
            n_dets, n_ms = time_inference(fast_backend, frame)
            s_dets, s_ms = time_inference(acc_backend, frame)
            n_dets_f = n_dets.filter_score(args.conf_floor)
            s_dets_f = s_dets.filter_score(args.conf_floor)
            # Post-conf-floor confidence summaries (per-frame; broadcast
            # to every policy row). `this_fast_mean_conf` is the locked
            # name fed to conf_ema's trigger one frame later via
            # `last_fast_mean_conf`.
            this_fast_mean_conf = _mean_conf(n_dets_f)
            this_fast_max_conf = _max_conf(n_dets_f)
            this_acc_mean_conf = _mean_conf(s_dets_f)
            this_acc_max_conf = _max_conf(s_dets_f)

            # Per-feature proxy values (NaN if not in --proxy-features)
            proxy_vals = {f: float(prox_out.get(f, _NAN))
                          for f in _SCENE_FEATURES}
            proxy_times = {
                f: float(prox_out.get(f"time_{f}_ms", _NAN))
                for f in _SCENE_FEATURES
            }
            proxy_overhead_ms = float(prox_out.get("time_overhead_ms",
                                                    _NAN))

            # Aggregated per-group timings (sum over members; NaN if any
            # member is missing). Diagnostic decomposition of
            # proxy_ms_total — total is authoritative.
            def _sum_or_nan(keys):
                vals = [proxy_times.get(k, _NAN) for k in keys]
                if any(math.isnan(v) for v in vals):
                    return _NAN
                return float(sum(vals))
            proxy_ms_brightness = _sum_or_nan(_BRIGHTNESS_GROUP)
            proxy_ms_hsv = _sum_or_nan(_HSV_GROUP)

            # Locked benefit-positive: a property of THE FRAME, not of
            # any policy. Same value broadcast to every policy row.
            kw_match = {}
            if args.use_classes_in_match:
                kw_match["fast_classes"] = n_dets_f.classes
                kw_match["accurate_classes"] = s_dets_f.classes
            ben_frame = benefit_positive_frame(
                n_dets_f.boxes, s_dets_f.boxes,
                iou_threshold=args.iou_threshold,
                **kw_match,
            )

            # ----- Each policy (collect first; defer write so we can
            # broadcast Combined's policy_score as `combined_score`) ---
            frame_rows: List[Tuple] = []
            for pn, policy in policies.items():
                feat = _build_features(
                    frame_idx, prox_out, last_fast_mean_conf,
                )
                t_d0 = time.perf_counter()
                choice = policy.decide(feat)
                decision_ms = (time.perf_counter() - t_d0) * 1000.0
                diag = dict(policy.last_diag)  # snapshot before next call

                if choice == "n":
                    chosen_dets = n_dets_f
                    chosen_lat = n_ms
                else:
                    chosen_dets = s_dets_f
                    chosen_lat = s_ms

                # Compare chosen detections to s_only reference.
                kw_corr = {}
                if args.use_classes_in_match:
                    kw_corr["policy_classes"] = chosen_dets.classes
                    kw_corr["ref_classes"] = s_dets_f.classes
                iou_m = iou_match_frame(
                    chosen_dets.boxes, s_dets_f.boxes,
                    iou_threshold=args.iou_threshold,
                    **kw_corr,
                )
                d_cov = det_coverage_frame(
                    chosen_dets.boxes, s_dets_f.boxes,
                    iou_threshold=args.iou_threshold,
                    **kw_corr,
                )
                d_rec = det_recall_frame(
                    chosen_dets.boxes, s_dets_f.boxes,
                    iou_threshold=args.iou_threshold,
                    **kw_corr,
                )
                cnt_a = count_agree_frame(chosen_dets.boxes, s_dets_f.boxes)

                t_sel, t_dep = _t_total(
                    pn, choice, float(n_ms), float(s_ms),
                    float(proxy_ms_total), float(decision_ms),
                )
                proxy_ms_rel = _proxy_ms_policy_relevant(
                    pn, proxy_overhead_ms, proxy_times,
                )
                t_shared_proxy = _t_total_shared_proxy(
                    float(proxy_ms_total), float(decision_ms),
                    float(chosen_lat),
                )
                t_policy_rel = _t_total_policy_relevant(
                    pn, choice, float(n_ms), float(s_ms),
                    float(proxy_ms_rel), float(decision_ms),
                )
                frame_rows.append((
                    pn, diag, choice, float(decision_ms),
                    float(chosen_lat), float(t_sel),
                    int(bool(iou_m)), int(bool(d_cov)), int(bool(cnt_a)),
                    float(d_rec), len(chosen_dets),
                    float(t_dep), float(proxy_ms_rel),
                    float(t_shared_proxy), float(t_policy_rel),
                ))

            # Snapshot `combined_score` once all policies have decided.
            combined_score = _NAN
            if "combined" in policies:
                combined_score = float(
                    policies["combined"].last_diag.get("policy_score", _NAN))
            conf_diag = {}
            if "conf_ema" in policies:
                conf_diag = dict(policies["conf_ema"].last_diag)

            # Frame-level row.
            ff_writer.writerow({
                "video_id": video_id,
                "video_filename": video_filename,
                "domain": args.domain,
                "device": args.device,
                "model_family": args.model_family,
                "pair_type": args.pair_type,
                "frame_idx": frame_idx,
                "raw_video_idx": int(this_raw_idx),
                "timestamp_sec": float(timestamp_sec),
                "imgsz": int(args.imgsz),
                "proxy_size": int(prox_out.get("proxy_size",
                                                args.proxy_size)),
                "preserve_aspect": int(bool(args.preserve_aspect)),
                "conf_floor": float(args.conf_floor),
                "iou_threshold": float(args.iou_threshold),
                "n_count": len(n_dets_f),
                "s_count": len(s_dets_f),
                "n_latency_ms": float(n_ms),
                "s_latency_ms": float(s_ms),
                "fast_mean_conf": float(this_fast_mean_conf),
                "fast_max_conf": float(this_fast_max_conf),
                "accurate_mean_conf": float(this_acc_mean_conf),
                "accurate_max_conf": float(this_acc_max_conf),
                "n_mean_conf": float(this_fast_mean_conf),
                "n_max_conf": float(this_fast_max_conf),
                "s_mean_conf": float(this_acc_mean_conf),
                "s_max_conf": float(this_acc_max_conf),
                "fast_ema": float(conf_diag.get("fast_ema", _NAN)),
                "slow_ema": float(conf_diag.get("slow_ema", _NAN)),
                "conf_drop": float(conf_diag.get("conf_drop", _NAN)),
                "benefit_positive": int(bool(ben_frame)),
                "L": proxy_vals["L"],
                "H": proxy_vals["H"],
                "laplacian": proxy_vals["L"],
                "entropy": proxy_vals["H"],
                "color_entropy": proxy_vals["color_entropy"],
                "tenengrad": proxy_vals["tenengrad"],
                "edge_density": proxy_vals["edge_density"],
                "local_contrast": proxy_vals["local_contrast"],
                "bright_fraction": proxy_vals["bright_fraction"],
                "hue_std": proxy_vals["hue_std"],
                "brightness_mean": proxy_vals["brightness_mean"],
                "brightness_std": proxy_vals["local_contrast"],
                "contrast": proxy_vals["local_contrast"],
                "combined_score": combined_score,
                "proxy_overhead_ms": proxy_overhead_ms,
                "proxy_ms_laplacian": proxy_times["L"],
                "proxy_ms_entropy": proxy_times["H"],
                "proxy_ms_color_entropy": proxy_times["color_entropy"],
                "proxy_ms_tenengrad": proxy_times["tenengrad"],
                "proxy_ms_edge_density": proxy_times["edge_density"],
                "proxy_ms_local_contrast": proxy_times["local_contrast"],
                "proxy_ms_bright_fraction": proxy_times["bright_fraction"],
                "proxy_ms_hue_std": proxy_times["hue_std"],
                "proxy_ms_brightness_mean": proxy_times["brightness_mean"],
                "proxy_ms_brightness": proxy_ms_brightness,
                "proxy_ms_hsv": proxy_ms_hsv,
                "proxy_ms_total": proxy_ms_total,
                "proxy_ms_policy_relevant": _NAN,
                "decision_ms": _NAN,
                "chosen_latency_ms": _NAN,
                "t_total_selected_ms": _NAN,
                "t_total_deployed_ms": _NAN,
                "t_total_shared_proxy_ms": _NAN,
                "t_total_policy_relevant_ms": _NAN,
                "policy": "",
                "choice": "",
                "s_choice": "",
                "policy_score": _NAN,
                "policy_thresh_low": _NAN,
                "policy_thresh_high": _NAN,
                "policy_thresh_mid": _NAN,
                "iou_match": "",
                "det_coverage": "",
                "det_recall": "",
                "count_agree": "",
            })

            # ----- Now write per-policy rows ---------------------------
            for entry in frame_rows:
                (pn, diag, choice, decision_ms, chosen_lat, t_sel,
                 iou_m_i, d_cov_i, cnt_a_i, d_rec_v, chosen_count_v,
                 t_dep, proxy_ms_rel, t_shared_proxy, t_policy_rel) = entry

                row = SweepRow(
                    video_id=video_id,
                    video_filename=video_filename,
                    domain=args.domain,
                    device=args.device,
                    model_family=args.model_family,
                    pair_type=args.pair_type,
                    frame_idx=frame_idx,
                    raw_video_idx=int(this_raw_idx),
                    timestamp_sec=float(timestamp_sec),
                    imgsz=int(args.imgsz),
                    proxy_size=int(prox_out.get("proxy_size",
                                                 args.proxy_size)),
                    preserve_aspect=int(bool(args.preserve_aspect)),
                    conf_floor=float(args.conf_floor),
                    iou_threshold=float(args.iou_threshold),
                    policy=pn,
                    choice=choice,
                    s_choice=int(choice == "s"),
                    n_latency_ms=float(n_ms),
                    s_latency_ms=float(s_ms),
                    chosen_latency_ms=float(chosen_lat),
                    proxy_ms_total=float(proxy_ms_total),
                    proxy_ms_policy_relevant=float(proxy_ms_rel),
                    proxy_overhead_ms=proxy_overhead_ms,
                    decision_ms=float(decision_ms),
                    t_total_selected_ms=float(t_sel),
                    t_total_deployed_ms=float(t_dep),
                    t_total_shared_proxy_ms=float(t_shared_proxy),
                    t_total_policy_relevant_ms=float(t_policy_rel),
                    n_count=len(n_dets_f),
                    s_count=len(s_dets_f),
                    chosen_count=int(chosen_count_v),
                    fast_mean_conf=float(this_fast_mean_conf),
                    fast_max_conf=float(this_fast_max_conf),
                    accurate_mean_conf=float(this_acc_mean_conf),
                    accurate_max_conf=float(this_acc_max_conf),
                    n_mean_conf=float(this_fast_mean_conf),
                    n_max_conf=float(this_fast_max_conf),
                    s_mean_conf=float(this_acc_mean_conf),
                    s_max_conf=float(this_acc_max_conf),
                    L=proxy_vals["L"],
                    H=proxy_vals["H"],
                    laplacian=proxy_vals["L"],
                    entropy=proxy_vals["H"],
                    color_entropy=proxy_vals["color_entropy"],
                    tenengrad=proxy_vals["tenengrad"],
                    edge_density=proxy_vals["edge_density"],
                    local_contrast=proxy_vals["local_contrast"],
                    bright_fraction=proxy_vals["bright_fraction"],
                    hue_std=proxy_vals["hue_std"],
                    brightness_mean=proxy_vals["brightness_mean"],
                    brightness_std=proxy_vals["local_contrast"],
                    contrast=proxy_vals["local_contrast"],
                    combined_score=combined_score,
                    proxy_ms_laplacian=proxy_times["L"],
                    proxy_ms_entropy=proxy_times["H"],
                    proxy_ms_color_entropy=proxy_times["color_entropy"],
                    proxy_ms_tenengrad=proxy_times["tenengrad"],
                    proxy_ms_edge_density=proxy_times["edge_density"],
                    proxy_ms_local_contrast=proxy_times["local_contrast"],
                    proxy_ms_bright_fraction=proxy_times["bright_fraction"],
                    proxy_ms_hue_std=proxy_times["hue_std"],
                    proxy_ms_brightness_mean=proxy_times["brightness_mean"],
                    proxy_ms_brightness=proxy_ms_brightness,
                    proxy_ms_hsv=proxy_ms_hsv,
                    policy_score=float(diag.get("policy_score", _NAN)),
                    policy_thresh_low=float(diag.get(
                        "policy_thresh_low", _NAN)),
                    policy_thresh_high=float(diag.get(
                        "policy_thresh_high", _NAN)),
                    policy_thresh_mid=float(diag.get(
                        "policy_thresh_mid", _NAN)),
                    fast_ema=float(diag.get("fast_ema", _NAN)),
                    slow_ema=float(diag.get("slow_ema", _NAN)),
                    conf_drop=float(diag.get("conf_drop", _NAN)),
                    iou_match=int(iou_m_i),
                    det_coverage=int(d_cov_i),
                    det_recall=float(d_rec_v),
                    count_agree=int(cnt_a_i),
                    benefit_positive=int(bool(ben_frame)),
                )
                writer.writerow(asdict(row))
                sweep_rows_all.append(row)

                per_policy_records[pn].append(dict(
                    iou_match=row.iou_match,
                    det_coverage=row.det_coverage,
                    det_recall=row.det_recall,
                    count_agree=row.count_agree,
                    benefit=row.benefit_positive,
                    choice=row.choice,
                    time_ms=row.chosen_latency_ms,
                ))

            # Advance conf_ema feedback signal: ALWAYS the fast-model
            # mean conf (post-conf-floor), so every policy sees the
            # same fast-detector state next frame. Reuse the value we
            # already computed and logged for this frame.
            last_fast_mean_conf = this_fast_mean_conf

            csv_file.flush()
            ff_file.flush()

            if (frame_idx + 1) % 25 == 0 or frame_idx == 0:
                elapsed = time.perf_counter() - t_run0
                print(f"  frame {frame_idx + 1:>4d}  "
                      f"prox={proxy_ms_total:5.1f}ms  "
                      f"n={n_ms:6.1f}ms  s={s_ms:6.1f}ms  "
                      f"|n|={len(n_dets_f)}  |s|={len(s_dets_f)}  "
                      f"benefit={int(ben_frame)}  [wall {elapsed:.1f}s]")

            frame_idx += 1
    finally:
        csv_file.close()
        ff_file.close()
        cap.release()

    t_run = time.perf_counter() - t_run0
    print(f"[run_sweep] processed {frame_idx} frames "
          f"x {len(args.policies)} policies in {t_run:.1f}s")

    # ----- Summary --------------------------------------------------------
    summaries = {pn: summarise_run(per_policy_records[pn])
                 for pn in args.policies}
    summary = dict(
        run_id=run_id,
        wall_seconds=float(t_run),
        n_frames=frame_idx,
        host=platform.node(),
        platform=platform.platform(),
        config=dict(
            video=str(args.video),
            fast_weights=str(args.fast_weights),
            accurate_weights=str(args.accurate_weights),
            device=args.device,
            imgsz=args.imgsz,
            max_frames=args.max_frames,
            start_frame=int(args.start_frame),
            warmup_frames=int(args.warmup_frames),
            frame_skip=int(args.frame_skip),
            conf_floor=args.conf_floor,
            iou_threshold=args.iou_threshold,
            proxy_size=int(args.proxy_size),
            preserve_aspect=bool(args.preserve_aspect),
            proxy_features=list(args.proxy_features),
            use_classes_in_match=bool(args.use_classes_in_match),
            policies=list(args.policies),
            conf_ema_params=dict(
                fast_beta=float(cfg.conf_ema_fast_beta),
                slow_beta=float(cfg.conf_ema_slow_beta),
                c_low=float(cfg.conf_ema_c_low),
                c_high=float(cfg.conf_ema_c_high),
                warmup=int(cfg.conf_ema_warmup),
            ),
            video_meta=dict(
                total_frames=n_total, fps=fps_video,
                width=width, height=height,
            ),
        ),
        per_policy=summaries,
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # ----- Console report -------------------------------------------------
    print("\n[run_sweep] summary:")
    print(f"  {'policy':<14s}  "
          f"{'lat_mean':>10s}  {'fps':>6s}  "
          f"{'iou':>6s}  {'cov':>6s}  {'rec':>6s}  {'s%':>6s}  {'ben%':>6s}")
    for pn in args.policies:
        s = summaries[pn]
        if s.get("n_frames", 0) == 0:
            continue
        print(f"  {pn:<14s}  "
              f"{s['latency']['mean']:>10.2f}  "
              f"{s['latency']['fps']:>6.2f}  "
              f"{s['iou_match_rate']:>6.3f}  "
              f"{s['det_coverage_rate']:>6.3f}  "
              f"{s['det_recall_mean']:>6.3f}  "
              f"{s['s_choice_rate']:>6.3f}  "
              f"{s['benefit_rate']:>6.3f}")
    print(f"  per-frame CSV:    {csv_path}")
    print(f"  frame features:   {frame_features_path}")
    print(f"  summary JSON:     {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
