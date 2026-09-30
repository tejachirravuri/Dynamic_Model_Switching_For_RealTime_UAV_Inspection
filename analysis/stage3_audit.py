"""analysis/stage3_audit.py — honest re-aggregation for thesis defense.

Why this script exists
----------------------
1. The original master_table.csv reports ``lat_mean_ms`` = mean of
   ``latency_ms`` per frame, which is **selected-model wall-clock only**.
   It does NOT include scene-proxy compute time or the policy's
   ``decide()`` overhead. The thesis defense formulation requires
   ``T_total = T_scene + T_ctrl + T_infer``. This script recomputes
   T_total from the per-frame CSVs (which already have the three
   columns) with correct per-policy attribution: only policies that
   actually read scene features pay T_scene.

2. An examiner will ask "why not switch randomly at the same rate?"
   For each policy we report:
       random_iou_at_same_s = n_iou + s_choice_rate * (1 - n_iou)
       informed_gain        = policy_iou - random_iou_at_same_s
   This quantifies the trigger's *additional* lift above what raw
   accurate-model usage would buy.

3. We compute per-policy trigger validity (precision / recall / F1)
   against the locked benefit_positive label, so scene-feature
   policies are reported on the same scale as conf_ema.

Inputs
------
A directory containing one or more ``stage3_*_<device>/`` run dirs,
each with ``sweep_per_frame.csv`` and ``sweep_summary.json``.
Default scans ``results/remote_pull/stage3_combined`` then
``results/remote_pull/stage3_cuda``.

Outputs (under --out-dir, default ``analysis/figures/stage3``)
--------------------------------------------------------------
* ``master_table_stage3_audit.csv`` — one row per (run, policy) with
  the original metrics PLUS:
      t_scene_mean_ms, t_ctrl_mean_ms, t_infer_mean_ms,
      t_total_mean_ms, t_total_p50_ms, t_total_p95_ms, t_total_p99_ms,
      random_iou_at_same_s, informed_gain_over_random,
      trigger_tp, trigger_fp, trigger_fn, trigger_tn,
      trigger_precision, trigger_recall, trigger_f1,
      reads_scene_features (bool)
* ``stage3_summary_audit.csv``   — pivoted summary, T_total flavour.

Scene-feature attribution (locked, matches dms/policies.py)
-----------------------------------------------------------
Policies that READ proxies in decide() and therefore pay T_scene:
    entropy_only, combined, combined_hyst, multi_proxy
Policies that do NOT pay T_scene:
    n_only, s_only, conf_ema

T_scene is the SHARED PROXY EXTRACTION BLOCK cost
-------------------------------------------------
``proxy_ms`` is the wall-clock to compute the FULL 6-feature block
(L, H, color_entropy, tenengrad, edge_density, local_contrast).
It is the same value broadcast across all scene-feature policies
within a frame, even though entropy_only only reads ``H`` and
combined only reads ``L`` + ``H``. This is the conservative cost
attribution: in the deployment the thesis describes, all scene-
feature policies share one extraction pass. A per-feature timing
ablation (entropy-only would actually be much cheaper if isolated)
is flagged as future work.

T_fast watchdog cost for confidence-based switching
---------------------------------------------------
Policies in ``NEEDS_FAST_WATCHDOG`` (currently just conf_ema) read
the previous frame's fast-model mean confidence to decide. In a
deployed system the fast model therefore runs every frame as a
"watchdog" so next-frame trigger has fresh signal. When such a
policy picks 's', the accurate model runs ON TOP of the fast model
(not instead of it):

    T_total[conf_ema, frame i] = T_fast[i] + T_ctrl[i]
                                + (T_acc[i] if choice == 's' else 0)

Earlier audit versions reported only T_chosen for conf_ema, which
under-counted T_total by 14-21% in our Stage 3 runs. Fixed
2026-04-29.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


SCENE_FEATURE_POLICIES = frozenset(
    {
        "entropy_only",
        "combined",
        "combined_hyst",
        "local_contrast_hyst",
        "multi_proxy",
    }
)

# Policies that always need the FAST model to run, even when they pick
# 's', because their next-frame trigger reads the fast model's previous
# confidence. In a deployed system, the fast model is therefore a
# "watchdog" that runs on every frame; switching to 's' adds the
# accurate model on top, it does not replace the fast one.
NEEDS_FAST_WATCHDOG = frozenset({"conf_ema"})


# ===========================================================================
# Run-id parser
# ===========================================================================
@dataclass(frozen=True)
class RunKey:
    family: str
    fast: str
    acc: str
    domain: str
    device: str

    @property
    def pair(self) -> str:
        return f"{self.fast}_{self.acc}"


def parse_run_id(run_id: str) -> Optional[RunKey]:
    """Parse stage3_*, ablation_skip_K*_*, or ablation_proxy_S*_* run_ids."""
    # stage3_<family>_n_<acc>_<domain>_<device>
    m = re.match(
        r"stage3_(yolov8|yolo26)_(n)_(s|l)_(glass|porcelain)_(cpu|cuda)",
        run_id,
    )
    if m:
        fam, fast, acc, dom, dev = m.groups()
        return RunKey(family=fam, fast=fast, acc=acc,
                      domain=dom, device=dev)

    # ablation_skip_K<K>_<family>_n_<acc>_<domain>_<device>
    m = re.match(
        r"ablation_skip_K(\d+)_(yolov8|yolo26)_(n)_(s|l)_"
        r"(glass|porcelain)_(cpu|cuda)",
        run_id,
    )
    if m:
        _, fam, fast, acc, dom, dev = m.groups()
        return RunKey(family=fam, fast=fast, acc=acc,
                      domain=dom, device=dev)

    # ablation_proxy_S<size>_<family>_n_<acc>_<domain>_<device>
    m = re.match(
        r"ablation_proxy_S(\d+)_(yolov8|yolo26)_(n)_(s|l)_"
        r"(glass|porcelain)_(cpu|cuda)",
        run_id,
    )
    if m:
        _, fam, fast, acc, dom, dev = m.groups()
        return RunKey(family=fam, fast=fast, acc=acc,
                      domain=dom, device=dev)

    return None


def parse_ablation_param(run_id: str) -> Tuple[Optional[str], Optional[int]]:
    """Return (kind, value) for ablation runs.

    kind in {'skip', 'proxy', None}
    value = the K (frame-skip) or S (proxy-size) integer parameter.
    Returns (None, None) for non-ablation runs.
    """
    m = re.match(r"ablation_skip_K(\d+)_", run_id)
    if m:
        return "skip", int(m.group(1))
    m = re.match(r"ablation_proxy_S(\d+)_", run_id)
    if m:
        return "proxy", int(m.group(1))
    return None, None


# ===========================================================================
# Per-frame CSV reading
# ===========================================================================
def _f(x) -> float:
    if x is None or x == "":
        return float("nan")
    try:
        return float(x)
    except ValueError:
        return float("nan")


def _percentile(sorted_vals: List[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def aggregate_one_run(
    per_frame_csv: Path,
    run_id: str,
) -> List[Dict]:
    """Re-aggregate one stage3 run by (run, policy).

    Returns one dict per policy with the audit columns.

    T_total deployment semantics (post-2026-04-29 fix):
      * Scene-feature policies: pay T_scene + T_ctrl + T_chosen. They
        decide BEFORE running either model, so only the chosen model
        runs.
      * Watchdog policies (conf_ema): pay T_fast + T_ctrl + (T_acc if
        choice=='s' else 0). The fast model runs every frame regardless
        of policy choice, because next-frame trigger needs current fast
        confidence. Switching to 's' adds the accurate model on top, it
        does not replace the fast one.
      * n_only: pay T_n every frame (no scene, ~zero ctrl).
      * s_only: pay T_s every frame.
    """
    by_policy: Dict[str, List[Dict]] = defaultdict(list)
    with per_frame_csv.open("r", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            by_policy[r["policy"]].append(r)

    out: List[Dict] = []

    # First compute n_only and s_only iou_match for the random baseline.
    def _iou_for(pol: str) -> float:
        if pol not in by_policy:
            return float("nan")
        rows = by_policy[pol]
        return sum(int(r["iou_match"]) for r in rows) / len(rows)

    n_iou = _iou_for("n_only")
    s_iou = _iou_for("s_only")  # always 1.0 by definition

    # Per-frame T_fast and T_acc, indexed by frame_idx, taken from
    # n_only and s_only rows. Used by watchdog-policy T_total
    # accounting (see NEEDS_FAST_WATCHDOG).
    fast_per_frame: Dict[int, float] = {}
    acc_per_frame: Dict[int, float] = {}
    if "n_only" in by_policy:
        for r in by_policy["n_only"]:
            fast_per_frame[int(r["frame_idx"])] = _f(r["chosen_latency_ms"])
    if "s_only" in by_policy:
        for r in by_policy["s_only"]:
            acc_per_frame[int(r["frame_idx"])] = _f(r["chosen_latency_ms"])

    for policy, rows in by_policy.items():
        n = len(rows)
        if n == 0:
            continue

        pays_scene = policy in SCENE_FEATURE_POLICIES
        is_watchdog = policy in NEEDS_FAST_WATCHDOG
        per_frame_total: List[float] = []
        per_frame_scene: List[float] = []
        per_frame_ctrl:  List[float] = []
        per_frame_infer: List[float] = []
        s_choices = 0
        iou_sum = 0
        cov_sum = 0
        rec_sum = 0.0
        cnt_sum = 0
        ben_sum = 0
        tp = fp = fn = tn = 0

        for r in rows:
            fi = int(r["frame_idx"])
            scene = _f(r["proxy_ms"]) if pays_scene else 0.0
            ctrl  = _f(r["decision_ms"])
            chosen_lat = _f(r["chosen_latency_ms"])
            choice_s = (r["choice"] == "s")

            if is_watchdog:
                # Fast watchdog runs every frame; accurate added if 's'.
                t_fast_i = fast_per_frame.get(fi, chosen_lat
                                              if not choice_s else 0.0)
                if choice_s:
                    t_acc_i = acc_per_frame.get(fi, chosen_lat)
                    infer = t_fast_i + t_acc_i
                else:
                    infer = t_fast_i
            else:
                infer = chosen_lat

            total = scene + ctrl + infer
            per_frame_scene.append(scene)
            per_frame_ctrl.append(ctrl)
            per_frame_infer.append(infer)
            per_frame_total.append(total)

            iou = int(r["iou_match"])
            cov = int(r["det_coverage"])
            rec = _f(r["det_recall"])
            cnt = int(r["count_agree"])
            ben = int(r["benefit_positive"])
            choice_s = (r["choice"] == "s")

            iou_sum += iou
            cov_sum += cov
            rec_sum += rec
            cnt_sum += cnt
            ben_sum += ben
            if choice_s:
                s_choices += 1
                if ben:
                    tp += 1
                else:
                    fp += 1
            else:
                if ben:
                    fn += 1
                else:
                    tn += 1

        s_choice_rate = s_choices / n
        iou_match_rate = iou_sum / n

        # Random baseline at same s%.
        if not math.isnan(n_iou) and not math.isnan(s_iou):
            random_iou = n_iou + s_choice_rate * (s_iou - n_iou)
        else:
            random_iou = float("nan")
        informed_gain = iou_match_rate - random_iou

        # Trigger precision / recall / F1.
        trig_p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        trig_r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        trig_f1 = (2 * trig_p * trig_r / (trig_p + trig_r)
                   if (trig_p + trig_r) > 0 else 0.0)

        # T_total percentiles.
        sorted_total = sorted(per_frame_total)
        ablation_kind, ablation_value = parse_ablation_param(run_id)
        out.append(dict(
            run_id=run_id,
            policy=policy,
            n_frames=n,
            ablation_kind=ablation_kind or "",
            ablation_value=(ablation_value if ablation_value is not None
                            else ""),
            reads_scene_features=int(pays_scene),
            needs_fast_watchdog=int(is_watchdog),
            iou_match_rate=iou_match_rate,
            det_coverage_rate=cov_sum / n,
            det_recall_mean=rec_sum / n,
            count_agree_rate=cnt_sum / n,
            benefit_rate=ben_sum / n,
            s_choice_rate=s_choice_rate,
            t_scene_mean_ms=sum(per_frame_scene) / n,
            t_ctrl_mean_ms=sum(per_frame_ctrl) / n,
            t_infer_mean_ms=sum(per_frame_infer) / n,
            t_total_mean_ms=sum(per_frame_total) / n,
            t_total_p50_ms=_percentile(sorted_total, 50),
            t_total_p95_ms=_percentile(sorted_total, 95),
            t_total_p99_ms=_percentile(sorted_total, 99),
            random_iou_at_same_s=random_iou,
            informed_gain_over_random=informed_gain,
            trigger_tp=tp,
            trigger_fp=fp,
            trigger_fn=fn,
            trigger_tn=tn,
            trigger_precision=trig_p,
            trigger_recall=trig_r,
            trigger_f1=trig_f1,
        ))
    return out


# ===========================================================================
# Driver
# ===========================================================================
def find_run_dirs(roots: List[Path]) -> List[Path]:
    """Return every <root>/{stage3,ablation_skip,ablation_proxy}_* dir
    that has both sweep_per_frame.csv and sweep_summary.json.

    Symlinks (e.g. ablation_skip_K1_* -> stage3_*) are followed.
    """
    out: List[Path] = []
    seen: set = set()
    accepted_prefixes = ("stage3_", "ablation_skip_", "ablation_proxy_")
    for root in roots:
        if not root.exists():
            continue
        for d in sorted(root.iterdir()):
            # Allow symlinks to dirs.
            if not (d.is_dir() or d.is_symlink()):
                continue
            if not any(d.name.startswith(p) for p in accepted_prefixes):
                continue
            csv_p = d / "sweep_per_frame.csv"
            json_p = d / "sweep_summary.json"
            if not (csv_p.exists() and json_p.exists()):
                continue
            if d.name in seen:
                continue  # earlier root wins
            seen.add(d.name)
            out.append(d)
    return out


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Re-aggregate Stage 3 with T_total, random baseline, trigger F1.",
    )
    p.add_argument("--roots", type=Path, nargs="+",
                   default=[
                       Path("results/remote_pull/stage3_combined"),
                       Path("results/remote_pull/stage3_cuda"),
                   ],
                   help="Directories that contain stage3_*_<device> subdirs.")
    p.add_argument("--out-dir", type=Path,
                   default=Path("analysis/figures/stage3"))
    return p.parse_args(argv)


def write_long_csv(rows: List[Dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    run_dirs = find_run_dirs(list(args.roots))
    if not run_dirs:
        print("[stage3_audit] no stage3 run dirs found in:",
              args.roots, file=sys.stderr)
        return 2
    print(f"[stage3_audit] aggregating {len(run_dirs)} run dirs")

    all_rows: List[Dict] = []
    for d in run_dirs:
        run_id = d.name
        rows = aggregate_one_run(d / "sweep_per_frame.csv", run_id)
        all_rows.extend(rows)
        print(f"  {run_id}: {len(rows)} policy rows")

    long_path = args.out_dir / "master_table_stage3_audit.csv"
    write_long_csv(all_rows, long_path)
    print(f"[stage3_audit] wrote {long_path} ({len(all_rows)} rows)")

    # Quick summary print: per (device, domain) winner under T_total.
    print()
    print("=== Honest Pareto: T_total mean (ms) and informed_gain_over_random ===")
    by_panel: Dict = defaultdict(list)
    for r in all_rows:
        key = parse_run_id(r["run_id"])
        if key is None:
            continue
        panel = (key.device, key.domain)
        by_panel[panel].append((key, r))

    for (device, domain), entries in sorted(by_panel.items()):
        print(f"\n--- {device.upper()} / {domain} ---")
        print(f"{'family':7s} {'pair':4s}  {'policy':14s} "
              f"{'iou':>6s} {'rand':>6s} {'gain':>6s}  "
              f"{'T_total':>8s} {'p95':>8s}  {'trig_F1':>7s}")
        # group by run, sort policies inside.
        runs: Dict = defaultdict(list)
        for key, r in entries:
            runs[(key.family, key.pair)].append(r)
        for (fam, pair) in sorted(runs.keys()):
            for r in sorted(runs[(fam, pair)],
                            key=lambda x: x["t_total_mean_ms"]):
                gain = r["informed_gain_over_random"]
                gain_s = (f"{gain:+.3f}" if not math.isnan(gain) else "  -- ")
                print(f"{fam:7s} {pair:4s}  {r['policy']:14s} "
                      f"{r['iou_match_rate']:>6.3f} "
                      f"{r['random_iou_at_same_s']:>6.3f} "
                      f"{gain_s:>6s}  "
                      f"{r['t_total_mean_ms']:>8.2f} "
                      f"{r['t_total_p95_ms']:>8.2f}  "
                      f"{r['trigger_f1']:>7.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
