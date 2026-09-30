"""analysis/replay_proxy_stride.py — proxy-stride amortization (offline).

Why this script exists
----------------------
The thesis question "what about k=1, k=3, k=5, k=10 frame skipping?"
was originally answered by ``--frame-skip K`` in run_sweep.py, which
implements **video subsampling** (Case A): skip raw frames entirely,
no inference at all on skipped frames. That answers a different
deployment question ("what if we can't keep up with the frame rate?").

The thesis question is closer to **proxy-stride** (Case B): evaluate
inference on every frame, but only RECOMPUTE proxies every K frames,
reusing the cached values for the K-1 frames in between. This tests
whether scene-feature policies can amortize their trigger compute
without losing decision quality.

This script derives proxy-stride results offline from:
  * a per-frame features trace from ``experiments.run_features``
    (gives proxy values + proxy_ms per frame, no inference cost)
  * a per-frame sweep CSV from ``experiments.run_sweep``
    (gives n / s detection counts, n_iou / s_iou outcomes per frame,
    benefit_positive label per frame, fast_mean_conf per frame)

For each stride K and each scene-feature policy, we:
  1. Build a "stale" proxy stream: at frame t use values from frame
     ``floor(t/K) * K`` (i.e., the most recent refresh).
  2. Replay the policy's decide() on the stale proxy stream.
  3. Substitute n_only / s_only per-frame metrics based on the new
     decision (when policy picks 's', use that frame's s_only result;
     when 'n', use the n_only result). This works because the policy
     is deterministic given features and the per-frame inference
     outcomes are already captured in the sweep CSV.
  4. Aggregate iou_match, det_coverage, det_recall, s_choice_rate,
     T_total = T_scene/K (amortized) + T_ctrl + T_chosen.

Outputs ``analysis/figures/stage3/proxy_stride_replay.csv`` with one
row per (run, policy, stride_K) tuple, ready for a Pareto plot.

Conf_ema is also replayed at each K but its T_total accounting is
**always-watchdog** (T_fast every frame + I[s] * T_acc) so its
proxy-stride is irrelevant — it pays no proxy cost at any K. We
include it anyway for visual comparison on the Pareto plot.

Usage
-----
    python -m analysis.replay_proxy_stride \
        --features-dir results/remote_pull/.../features_<video> \
        --sweep-dir    results/remote_pull/.../stage3_<pair>_<device> \
        --strides 1 3 5 10 \
        --out-dir analysis/figures/stage3
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dms.policies import FrameFeatures, PolicyConfig, make_policy


SCENE_FEATURE_POLICIES = ("entropy_only", "combined", "combined_hyst",
                          "multi_proxy")


def _f(x) -> float:
    if x is None or x == "":
        return float("nan")
    try:
        return float(x)
    except ValueError:
        return float("nan")


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------
def read_features(per_frame_csv: Path) -> List[Dict]:
    """Load per-frame proxy values + proxy_ms from run_features output."""
    with per_frame_csv.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        d = {"frame_idx": int(r["frame_idx"])}
        for k, v in r.items():
            if k == "frame_idx":
                continue
            d[k] = _f(v)
        out.append(d)
    return out


def read_sweep_per_policy(per_frame_csv: Path) -> Dict[str, Dict[int, Dict]]:
    """Index sweep rows by policy -> frame_idx -> row.

    We need to look up, for each frame:
      * what n_only's iou_match / det_coverage / det_recall was
      * what s_only's was (always 1 / 1 / 1.0 / 1)
      * benefit_positive label
      * n_only and s_only latency_ms (= T_fast and T_accurate)
    """
    by_pol: Dict[str, Dict[int, Dict]] = defaultdict(dict)
    with per_frame_csv.open("r", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            by_pol[r["policy"]][int(r["frame_idx"])] = r
    return by_pol


# ---------------------------------------------------------------------------
# Policy rules (must mirror dms.policies decide() exactly)
# ---------------------------------------------------------------------------
def _build_features(frow: Dict, frame_idx: int) -> FrameFeatures:
    """Map a per-frame features dict (from run_features.py CSV) to a
    FrameFeatures object that the live policies can consume.
    """
    return FrameFeatures(
        frame_idx=frame_idx,
        L=frow.get("L"),
        H=frow.get("H"),
        color_entropy=frow.get("color_entropy"),
        tenengrad=frow.get("tenengrad"),
        edge_density=frow.get("edge_density"),
        local_contrast=frow.get("local_contrast"),
        bright_fraction=frow.get("bright_fraction"),
        hue_std=frow.get("hue_std"),
        last_mean_conf=0.0,  # not used by scene-feature policies
    )


# ---------------------------------------------------------------------------
# Replay at stride K
# ---------------------------------------------------------------------------
def replay_one(
    features_rows: List[Dict],
    sweep_by_pol: Dict[str, Dict[int, Dict]],
    policy: str,
    stride_K: int,
    cfg: Dict,
) -> Dict:
    """Replay one (policy, stride_K) tuple and aggregate metrics."""
    n = len(features_rows)
    if n == 0 or "n_only" not in sweep_by_pol:
        return {}
    n_only = sweep_by_pol["n_only"]
    s_only = sweep_by_pol.get("s_only", {})

    s_count = 0
    iou_sum = 0
    cov_sum = 0
    rec_sum = 0.0
    cnt_sum = 0
    ben_sum = 0
    tp = fp = fn = tn = 0

    per_frame_total: List[float] = []
    cached_feat: Optional[Dict] = None

    # Instantiate the actual live policy with default PolicyConfig.
    pol_cfg = PolicyConfig(**cfg)
    live_policy = make_policy(policy, pol_cfg)

    for i, frow in enumerate(features_rows):
        fi = frow["frame_idx"]
        # Refresh proxy values every K frames; use cache otherwise.
        if (i % stride_K) == 0:
            cached_feat = frow
        feat_for_decide = cached_feat or frow
        feat_obj = _build_features(feat_for_decide, fi)
        choice = live_policy.decide(feat_obj)
        is_s = (choice == "s")
        if is_s:
            s_count += 1

        # Pull per-frame outcome from the right reference row.
        # Sweep CSV has every frame_idx for every policy; n_only's
        # row's iou_match / etc. is exactly what the policy would
        # score if it picked 'n', and s_only's is what it would score
        # if it picked 's'.
        n_row = n_only.get(fi)
        s_row = s_only.get(fi)
        if n_row is None or s_row is None:
            continue
        ref_row = s_row if is_s else n_row
        iou_sum += int(ref_row["iou_match"])
        cov_sum += int(ref_row["det_coverage"])
        rec_sum += _f(ref_row["det_recall"])
        cnt_sum += int(ref_row["count_agree"])
        ben = int(n_row["benefit_positive"])
        ben_sum += ben

        if is_s and ben:
            tp += 1
        elif is_s and not ben:
            fp += 1
        elif (not is_s) and ben:
            fn += 1
        else:
            tn += 1

        # T_total: amortized scene + ctrl + chosen-model latency.
        # Proxy is paid ONLY on refresh frames; amortized cost per
        # frame is proxy_ms / K so we charge a fraction (K-1)/K to
        # nothing and 1/K to the full proxy_ms. Equivalent: charge
        # proxy_ms only on (i % K == 0) frames.
        t_scene = _f(frow.get("time_ms", 0.0)) if (i % stride_K) == 0 else 0.0
        t_ctrl = _f(n_row.get("decision_ms", 0.0))   # ~constant
        chosen_lat = (_f(s_row.get("chosen_latency_ms", 0.0)) if is_s
                      else _f(n_row.get("chosen_latency_ms", 0.0)))
        per_frame_total.append(t_scene + t_ctrl + chosen_lat)

    if not per_frame_total:
        return {}

    sorted_total = sorted(per_frame_total)
    p95 = sorted_total[max(0, int(0.95 * (len(sorted_total) - 1)))]
    n_iou = sum(int(r["iou_match"]) for r in n_only.values()) / max(1, len(n_only))
    s_iou = 1.0
    s_rate = s_count / n
    rand_iou = n_iou + s_rate * (s_iou - n_iou)

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec_t = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec_t / (prec + rec_t)) if (prec + rec_t) > 0 else 0.0

    return dict(
        policy=policy,
        stride_K=stride_K,
        n_frames=n,
        s_choice_rate=s_rate,
        iou_match_rate=iou_sum / n,
        det_coverage_rate=cov_sum / n,
        det_recall_mean=rec_sum / n,
        count_agree_rate=cnt_sum / n,
        benefit_rate=ben_sum / n,
        random_iou_at_same_s=rand_iou,
        informed_gain_over_random=(iou_sum / n) - rand_iou,
        trigger_precision=prec,
        trigger_recall=rec_t,
        trigger_f1=f1,
        t_total_mean_ms=sum(per_frame_total) / n,
        t_total_p95_ms=p95,
        t_scene_mean_amortized_ms=sum(
            _f(r.get("time_ms", 0.0)) if (i % stride_K) == 0 else 0.0
            for i, r in enumerate(features_rows)
        ) / n,
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Offline proxy-stride amortization replay.",
    )
    p.add_argument("--features-dir", type=Path, required=True,
                   help="Dir with per_frame.csv from run_features.")
    p.add_argument("--sweep-dir", type=Path, required=True,
                   help="Dir with sweep_per_frame.csv from run_sweep.")
    p.add_argument("--strides", type=int, nargs="+",
                   default=[1, 2, 3, 5, 10, 20])
    p.add_argument("--policies", type=str, nargs="+",
                   default=list(SCENE_FEATURE_POLICIES))
    p.add_argument("--out-csv", type=Path, default=None,
                   help="Where to write the replay table.")
    return p.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    feat_csv = args.features_dir / "per_frame.csv"
    sweep_csv = args.sweep_dir / "sweep_per_frame.csv"
    if not feat_csv.exists():
        print(f"[replay_proxy_stride] missing {feat_csv}", file=sys.stderr)
        return 2
    if not sweep_csv.exists():
        print(f"[replay_proxy_stride] missing {sweep_csv}", file=sys.stderr)
        return 2
    features_rows = read_features(feat_csv)
    sweep_by_pol = read_sweep_per_policy(sweep_csv)
    print(f"[replay_proxy_stride] features: {len(features_rows)} frames, "
          f"sweep: {len(sweep_by_pol)} policies")

    rows: List[Dict] = []
    for policy in args.policies:
        for K in args.strides:
            r = replay_one(features_rows, sweep_by_pol,
                           policy, K, cfg={})
            if r:
                rows.append(r)

    if not rows:
        print("[replay_proxy_stride] no rows produced", file=sys.stderr)
        return 2

    print(f"\n{'policy':<14s} {'K':>3s}  "
          f"{'iou':>6s} {'rand':>6s} {'gain':>6s}  "
          f"{'s%':>5s}  {'T_scene':>8s} {'T_total':>8s}  {'F1':>6s}")
    for r in rows:
        gain = r["informed_gain_over_random"]
        gain_s = f"{gain:+.3f}" if not math.isnan(gain) else "  -- "
        print(f"{r['policy']:<14s} {r['stride_K']:>3d}  "
              f"{r['iou_match_rate']:>6.3f} "
              f"{r['random_iou_at_same_s']:>6.3f} "
              f"{gain_s:>6s}  "
              f"{r['s_choice_rate']:>4.1%}  "
              f"{r['t_scene_mean_amortized_ms']:>8.2f} "
              f"{r['t_total_mean_ms']:>8.2f}  "
              f"{r['trigger_f1']:>6.3f}")

    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\n[replay_proxy_stride] wrote {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
