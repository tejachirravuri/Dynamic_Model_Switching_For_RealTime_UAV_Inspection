"""analysis/sweep_scene_thresholds.py — threshold sensitivity for
scene-feature policies (entropy_only / combined / combined_hyst /
multi_proxy).

Why
---
The thesis question "what if we change the threshold values?"
(handwritten note: "If we are using these manual thresholds...")
deserves the same offline-replay treatment we gave conf_ema. For
scene-feature policies that use rolling-percentile normalisation,
the relevant knob is ``c_mid`` (the percentile rank above which
the scene is considered hard).

This script replays each scene-feature policy across a grid of
threshold values using:
  * a per-frame features trace (run_features.py output)
  * the live policy class from dms.policies (so rolling stats
    and stabilisers behave exactly as in deployment)
  * the per-frame n_only / s_only reference outcomes
    from a sweep_per_frame.csv

For each (policy, threshold) tuple we report iou_match,
s_choice_rate, informed_gain_over_random, trigger_F1, and T_total.

The grid sweeps PolicyConfig overrides:
  entropy_only   : c_mid in {0.30, 0.40, 0.50, 0.60, 0.70}
                   (median-rank threshold)
  combined / combined_hyst : c_mid in {0.30, 0.40, 0.50, 0.60, 0.70}
  multi_proxy    : mp_c_high in {0.04, 0.06, 0.08, 0.10, 0.12, 0.20}

multi_proxy uses an EMA-deviation composite, so its grid is on the
deviation threshold rather than on a percentile.

Usage
-----
    python -m analysis.sweep_scene_thresholds \
        --features-dir <features_*> \
        --sweep-dir    <stage3_*_cpu> \
        --out-csv analysis/figures/stage3/scene_threshold_sweep.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from dms.policies import FrameFeatures, PolicyConfig, make_policy


def _f(x) -> float:
    if x is None or x == "":
        return float("nan")
    try:
        return float(x)
    except ValueError:
        return float("nan")


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------
def read_features(p: Path) -> List[Dict]:
    rows: List[Dict] = []
    with p.open("r", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            d = {"frame_idx": int(r["frame_idx"])}
            for k, v in r.items():
                if k == "frame_idx":
                    continue
                d[k] = _f(v)
            rows.append(d)
    return rows


def read_sweep(p: Path) -> Dict[str, Dict[int, Dict]]:
    by_pol: Dict[str, Dict[int, Dict]] = defaultdict(dict)
    with p.open("r", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            by_pol[r["policy"]][int(r["frame_idx"])] = r
    return by_pol


def _build_features(frow: Dict, frame_idx: int) -> FrameFeatures:
    return FrameFeatures(
        frame_idx=frame_idx,
        L=frow.get("L"), H=frow.get("H"),
        color_entropy=frow.get("color_entropy"),
        tenengrad=frow.get("tenengrad"),
        edge_density=frow.get("edge_density"),
        local_contrast=frow.get("local_contrast"),
        bright_fraction=frow.get("bright_fraction"),
        hue_std=frow.get("hue_std"),
        last_mean_conf=0.0,
    )


# ---------------------------------------------------------------------------
# Replay one (policy, override) tuple
# ---------------------------------------------------------------------------
def replay(
    features: List[Dict],
    sweep: Dict[str, Dict[int, Dict]],
    policy: str,
    overrides: Dict,
    label: str,
) -> Dict:
    n_only = sweep["n_only"]
    s_only = sweep["s_only"]
    pol_cfg = PolicyConfig(**overrides)
    live = make_policy(policy, pol_cfg)

    n = len(features)
    s_count = 0
    iou_sum = 0
    cov_sum = 0
    rec_sum = 0.0
    ben_sum = 0
    tp = fp = fn = tn = 0
    per_frame_total: List[float] = []

    for frow in features:
        fi = frow["frame_idx"]
        feat = _build_features(frow, fi)
        choice = live.decide(feat)
        is_s = (choice == "s")
        if is_s:
            s_count += 1

        n_row = n_only.get(fi)
        s_row = s_only.get(fi)
        if n_row is None or s_row is None:
            continue
        ref_row = s_row if is_s else n_row
        iou_sum += int(ref_row["iou_match"])
        cov_sum += int(ref_row["det_coverage"])
        rec_sum += _f(ref_row["det_recall"])
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

        scene = _f(frow.get("time_ms", 0.0))
        ctrl = _f(n_row.get("decision_ms", 0.0))
        chosen = (_f(s_row["chosen_latency_ms"]) if is_s
                  else _f(n_row["chosen_latency_ms"]))
        per_frame_total.append(scene + ctrl + chosen)

    if not per_frame_total:
        return {}

    s_rate = s_count / n
    n_iou = sum(int(r["iou_match"]) for r in n_only.values()) / max(1, len(n_only))
    s_iou = 1.0
    rand_iou = n_iou + s_rate * (s_iou - n_iou)
    iou_rate = iou_sum / n
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    return dict(
        policy=policy,
        threshold_label=label,
        n_frames=n,
        s_choice_rate=s_rate,
        iou_match_rate=iou_rate,
        det_coverage_rate=cov_sum / n,
        det_recall_mean=rec_sum / n,
        random_iou_at_same_s=rand_iou,
        informed_gain_over_random=iou_rate - rand_iou,
        trigger_precision=prec,
        trigger_recall=rec,
        trigger_f1=f1,
        t_total_mean_ms=sum(per_frame_total) / n,
        **overrides,
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
SCENE_GRID = {
    # entropy_only & combined family use c_mid (rolling-percentile rank).
    # multi_proxy uses mp_c_high (EMA deviation threshold).
    "entropy_only":  [("c_mid=0.30", {"c_mid": 0.30}),
                      ("c_mid=0.40", {"c_mid": 0.40}),
                      ("c_mid=0.50", {"c_mid": 0.50}),  # default
                      ("c_mid=0.60", {"c_mid": 0.60}),
                      ("c_mid=0.70", {"c_mid": 0.70})],
    "combined":      [("c_mid=0.30", {"c_mid": 0.30}),
                      ("c_mid=0.40", {"c_mid": 0.40}),
                      ("c_mid=0.50", {"c_mid": 0.50}),
                      ("c_mid=0.60", {"c_mid": 0.60}),
                      ("c_mid=0.70", {"c_mid": 0.70})],
    "combined_hyst": [("c_mid=0.30", {"c_mid": 0.30}),
                      ("c_mid=0.40", {"c_mid": 0.40}),
                      ("c_mid=0.50", {"c_mid": 0.50}),
                      ("c_mid=0.60", {"c_mid": 0.60}),
                      ("c_mid=0.70", {"c_mid": 0.70})],
    "multi_proxy":   [("mp_c_high=0.04", {"mp_c_high": 0.04}),
                      ("mp_c_high=0.06", {"mp_c_high": 0.06}),
                      ("mp_c_high=0.08", {"mp_c_high": 0.08}),
                      ("mp_c_high=0.10", {"mp_c_high": 0.10}),  # default
                      ("mp_c_high=0.20", {"mp_c_high": 0.20})],
}


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Threshold sensitivity sweep for scene-feature policies."
    )
    p.add_argument("--features-dir", type=Path, required=True)
    p.add_argument("--sweep-dir", type=Path, required=True)
    p.add_argument("--policies", type=str, nargs="+",
                   default=list(SCENE_GRID.keys()))
    p.add_argument("--out-csv", type=Path, default=None)
    return p.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    feat_csv = args.features_dir / "per_frame.csv"
    sweep_csv = args.sweep_dir / "sweep_per_frame.csv"
    if not feat_csv.exists() or not sweep_csv.exists():
        print(f"[sweep_scene] missing input", file=sys.stderr)
        return 2
    features = read_features(feat_csv)
    sweep = read_sweep(sweep_csv)
    print(f"[sweep_scene] features {len(features)}, sweep policies "
          f"{len(sweep)}; sweep dir = {args.sweep_dir.name}")

    rows: List[Dict] = []
    for policy in args.policies:
        for label, ov in SCENE_GRID.get(policy, []):
            r = replay(features, sweep, policy, ov, label)
            if r:
                rows.append(r)

    print()
    print(f"{'policy':<14s} {'threshold':>16s}  "
          f"{'iou':>6s} {'rand':>6s} {'gain':>6s}  "
          f"{'s%':>5s}  {'T_total':>8s} {'F1':>6s}")
    for r in rows:
        gain = r["informed_gain_over_random"]
        gain_s = f"{gain:+.3f}" if not math.isnan(gain) else "  -- "
        print(f"{r['policy']:<14s} {r['threshold_label']:>16s}  "
              f"{r['iou_match_rate']:>6.3f} "
              f"{r['random_iou_at_same_s']:>6.3f} "
              f"{gain_s:>6s}  "
              f"{r['s_choice_rate']:>4.1%}  "
              f"{r['t_total_mean_ms']:>8.2f} "
              f"{r['trigger_f1']:>6.3f}")

    if args.out_csv is not None and rows:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        # Union of keys across rows.
        all_cols = []
        seen: set = set()
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    seen.add(k)
                    all_cols.append(k)
        with args.out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=all_cols)
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c, "") for c in all_cols})
        print(f"\n[sweep_scene] wrote {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
