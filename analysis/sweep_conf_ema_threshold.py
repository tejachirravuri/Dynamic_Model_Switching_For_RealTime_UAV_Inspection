"""analysis/sweep_conf_ema_threshold.py — offline c_high sweep.

Why this exists
---------------
conf_ema's choice on a given frame is a deterministic function of
``fast_mean_conf`` history + the params (``fast_beta``, ``slow_beta``,
``c_low``, ``c_high``, ``warmup``, plus the dwell / max-switches
stabiliser settings). The sweep CSV already records:

  * ``fast_mean_conf`` per frame  (drives the EMA)
  * the s_only ``iou_match`` / ``det_coverage`` / ``det_recall`` per frame
    (= what conf_ema *would* score whenever it picks 's')
  * the n_only equivalents per frame
    (= what conf_ema *would* score whenever it picks 'n')
  * the locked ``benefit_positive`` per frame (= ground-truth label)

So we can replay conf_ema across a grid of ``c_high`` values without
re-running on remote, and report:

  * s_choice_rate
  * iou_match_rate, det_coverage_rate, det_recall_mean
  * **trigger precision** = P(benefit_positive | choice=='s')
    -- of the frames the trigger fires on, how many were really hard
  * **trigger recall**    = P(choice=='s' | benefit_positive)
    -- of the hard frames, how many did we catch
  * F1 of the (trigger fire, benefit_positive) confusion

The point is to find a ``c_high`` that maximises trigger F1 (or the
specific operating point the thesis wants), not to guess one.

Usage
-----
    python -m analysis.sweep_conf_ema_threshold \
        --run-dir results/stage_sweep/<run_id> \
        --grid 0.02 0.04 0.06 0.08 0.10 0.12

Reads ``sweep_per_frame.csv`` from the run-dir and prints one row per
threshold. With ``--out-csv`` it also writes the sweep to disk for
later plotting.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from dms.controllers import Hysteresis, SwitchStabiliser
from dms.policies import PolicyConfig

from analysis.inspect_conf_ema import (
    DEFAULT_C_HIGH,
    DEFAULT_C_LOW,
    DEFAULT_FAST_BETA,
    DEFAULT_SLOW_BETA,
    DEFAULT_WARMUP,
)


# ===========================================================================
# CSV ingestion
# ===========================================================================
def _per_frame_table(per_frame_csv: Path) -> List[Dict]:
    """Pull the unique-per-frame columns we need.

    fast_mean_conf, n iou_match (under choice='n' i.e. the n_only row's
    iou_match for that frame), s iou_match (== 1.0 always), and the
    benefit label (broadcast across rows for a given frame).
    """
    with per_frame_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return []
    if "fast_mean_conf" not in rows[0]:
        raise ValueError(
            "fast_mean_conf column missing — re-run experiments.run_sweep "
            "with the latest schema (commit 29d7aef or later)."
        )

    by_frame: Dict[int, Dict[str, dict]] = defaultdict(dict)
    for r in rows:
        fi = int(r["frame_idx"])
        pn = r["policy"]
        by_frame[fi][pn] = r

    out: List[Dict] = []
    for fi in sorted(by_frame.keys()):
        polrow = by_frame[fi]
        # n_only and s_only must both exist; per-frame quantities are
        # broadcast across rows so any policy row would do for the
        # broadcast fields.
        if "n_only" not in polrow or "s_only" not in polrow:
            raise ValueError(
                f"frame {fi}: need both n_only and s_only rows for "
                "offline threshold sweep — re-run with default --policies"
            )
        n_row = polrow["n_only"]
        s_row = polrow["s_only"]
        out.append(dict(
            frame_idx=fi,
            fast_mean_conf=float(n_row["fast_mean_conf"]),
            benefit_positive=int(n_row["benefit_positive"]),
            # under-'n' metrics:
            n_iou_match=int(n_row["iou_match"]),
            n_det_coverage=int(n_row["det_coverage"]),
            n_det_recall=float(n_row["det_recall"]),
            n_count_agree=int(n_row["count_agree"]),
            # under-'s' metrics (all 1.0 by definition of the reference):
            s_iou_match=int(s_row["iou_match"]),
            s_det_coverage=int(s_row["det_coverage"]),
            s_det_recall=float(s_row["det_recall"]),
            s_count_agree=int(s_row["count_agree"]),
            # latencies (per-row, but identical across n_only / s_only
            # columns and we only need them for s% reporting):
            n_latency_ms=float(n_row["chosen_latency_ms"]),
            s_latency_ms=float(s_row["chosen_latency_ms"]),
        ))
    return out


# ===========================================================================
# Replay
# ===========================================================================
def replay_at_threshold(
    table: List[Dict],
    c_high: float,
    c_low: Optional[float] = None,
    fast_beta: float = DEFAULT_FAST_BETA,
    slow_beta: float = DEFAULT_SLOW_BETA,
    warmup: int = DEFAULT_WARMUP,
) -> Dict[str, float]:
    """Replay conf_ema at one (c_high, c_low) and return aggregate metrics.

    Implementation mirrors dms.policies.ConfEMA exactly: same Hysteresis
    + SwitchStabiliser composition. The point of using the live classes
    is that any future change to those classes is automatically picked
    up here too — the test_inspect_conf_ema parity test guards us.
    """
    if c_low is None:
        # Track the live default ratio if user didn't supply one.
        c_low = c_high * (DEFAULT_C_LOW / DEFAULT_C_HIGH)
    if not table:
        return dict(
            n_frames=0, s_choice_rate=0.0, iou_match_rate=0.0,
            det_coverage_rate=0.0, det_recall_mean=0.0,
            count_agree_rate=0.0,
            trigger_precision=0.0, trigger_recall=0.0, trigger_f1=0.0,
        )

    cfg = PolicyConfig()  # for stabiliser params (dwell, max_switches)
    hyst = Hysteresis(c_low, c_high, initial="n")
    stab = SwitchStabiliser(cfg.min_dwell_frames, cfg.max_switches_per_100)

    fast_ema: Optional[float] = None
    slow_ema: Optional[float] = None
    eps = 1e-9

    n = len(table)
    s_choices = 0
    iou_sum = 0
    cov_sum = 0
    rec_sum = 0.0
    cnt_sum = 0

    # Confusion-matrix buckets for trigger validity.
    tp = fp = fn = tn = 0

    for row in table:
        fi = row["frame_idx"]
        x = float(row["fast_mean_conf"])

        if fast_ema is None:
            fast_ema = x
            slow_ema = x
        else:
            fast_ema = fast_beta * x + (1.0 - fast_beta) * fast_ema
            slow_ema = slow_beta * x + (1.0 - slow_beta) * slow_ema

        conf_drop = max(0.0, (slow_ema - fast_ema) / (slow_ema + eps))

        if fi < warmup:
            requested = "n"
        else:
            requested = hyst.step(conf_drop)
        choice = stab.step(fi, requested)

        is_s = (choice == "s")
        bp = bool(row["benefit_positive"])

        if is_s:
            s_choices += 1
            iou_sum += row["s_iou_match"]
            cov_sum += row["s_det_coverage"]
            rec_sum += row["s_det_recall"]
            cnt_sum += row["s_count_agree"]
        else:
            iou_sum += row["n_iou_match"]
            cov_sum += row["n_det_coverage"]
            rec_sum += row["n_det_recall"]
            cnt_sum += row["n_count_agree"]

        if is_s and bp:
            tp += 1
        elif is_s and not bp:
            fp += 1
        elif (not is_s) and bp:
            fn += 1
        else:
            tn += 1

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec_t = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * prec * rec_t / (prec + rec_t)) if (prec + rec_t) > 0 else 0.0

    return dict(
        n_frames=n,
        s_choice_rate=s_choices / n,
        iou_match_rate=iou_sum / n,
        det_coverage_rate=cov_sum / n,
        det_recall_mean=rec_sum / n,
        count_agree_rate=cnt_sum / n,
        trigger_precision=prec,
        trigger_recall=rec_t,
        trigger_f1=f1,
        c_high=c_high,
        c_low=c_low,
    )


# ===========================================================================
# Reporting
# ===========================================================================
def print_table(rows: List[Dict]) -> None:
    print(f"  {'c_high':>7s}  {'c_low':>7s}  {'s%':>6s}  "
          f"{'iou_match':>9s}  {'det_cov':>7s}  {'det_rec':>7s}  "
          f"{'trig_P':>7s}  {'trig_R':>7s}  {'trig_F1':>7s}")
    for r in rows:
        print(f"  {r['c_high']:>7.3f}  {r['c_low']:>7.3f}  "
              f"{r['s_choice_rate']:>5.1%}  "
              f"{r['iou_match_rate']:>9.3f}  "
              f"{r['det_coverage_rate']:>7.3f}  "
              f"{r['det_recall_mean']:>7.3f}  "
              f"{r['trigger_precision']:>7.3f}  "
              f"{r['trigger_recall']:>7.3f}  "
              f"{r['trigger_f1']:>7.3f}")


def write_csv(rows: List[Dict], out_path: Path) -> None:
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ===========================================================================
# CLI
# ===========================================================================
def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Offline c_high sweep over an existing run's CSV.",
    )
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--csv-name", type=str, default="sweep_per_frame.csv")
    p.add_argument("--grid", type=float, nargs="+",
                   default=[0.02, 0.04, 0.06, 0.08, 0.10, 0.12],
                   help="c_high grid (one --grid arg per value or "
                        "space-separated list).")
    p.add_argument("--c-low-ratio", type=float, default=None,
                   help="If set, c_low = c_high * ratio for every grid "
                        "point. Default: keep the live ratio "
                        f"({DEFAULT_C_LOW / DEFAULT_C_HIGH:.3f}).")
    p.add_argument("--out-csv", type=Path, default=None,
                   help="Optional path to write the sweep table.")
    return p.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    csv_path = args.run_dir / args.csv_name
    if not csv_path.exists():
        print(f"[sweep_conf_ema_threshold] missing {csv_path}", file=sys.stderr)
        return 2

    table = _per_frame_table(csv_path)
    if not table:
        print("[sweep_conf_ema_threshold] no rows", file=sys.stderr)
        return 2

    bp_rate = sum(r["benefit_positive"] for r in table) / len(table)
    print(f"[sweep_conf_ema_threshold] frames: {len(table)}  "
          f"benefit_rate: {bp_rate:.3%}")

    rows: List[Dict] = []
    for c_high in args.grid:
        c_low = (c_high * args.c_low_ratio
                 if args.c_low_ratio is not None else None)
        rows.append(replay_at_threshold(table, c_high=c_high, c_low=c_low))

    print()
    print_table(rows)

    if args.out_csv is not None:
        write_csv(rows, args.out_csv)
        print(f"\n[sweep_conf_ema_threshold] wrote {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
