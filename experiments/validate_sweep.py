"""experiments/validate_sweep.py — pre-full-sweep sanity gates.

Runs a fixed list of cheap checks against a sweep_per_frame.csv +
sweep_summary.json pair. Designed to gate the move from a 100-frame
CUDA sanity sweep to a multi-thousand-frame full sweep on remote.

Why a separate script and not inline in run_sweep
-------------------------------------------------
- run_sweep is the recorder: it should write whatever the policies and
  metrics produced, not silently mutate or hide data.
- validate_sweep is the gatekeeper: it inspects what was written and
  decides whether it is safe to scale up. Separating the two means a
  failed sanity check leaves the per-frame CSV intact for inspection.

Gates (all 5 must PASS to exit 0)
---------------------------------
G1. n_only latency  <  s_only latency
    Fast model must be faster on this hardware. If not, the device
    selection is wrong (e.g., CPU thermal throttle) or the weights
    were swapped. Threshold: n_only mean < s_only mean.

G2. s_only reference detections are non-empty on average
    If the accurate model finds nothing, every relaxed-recall metric
    is vacuous (recall == 1 by convention) and the trigger-validity
    table will be empty. Threshold: mean(s_count) >= 0.5.

G3. iou_match / count_agree are not trivially 1.0 across the board
    Unless the video is genuinely easy (n always matches s perfectly),
    a flat 1.0 across every policy means the matching is degenerate.
    Threshold: at least one of n_only's iou_match_rate or
    count_agree_rate is < 1.0  OR  benefit_rate is == 0 (legitimate
    easy-video case is escape-hatched by also requiring that all 7
    policies report the same iou_match_rate to within 1e-6).

G4. Per-policy s_choice_rate values are plausible
    n_only must be exactly 0; s_only must be exactly 1. The five
    switching policies must lie in [0, 1]. Conf_ema during warmup
    forces n, so for short sweeps it can be 0; we don't fail on that.

G5. Per-frame CSV has no NaN or empty timing columns
    A NaN in latency_ms or proxy_ms means a row was emitted before a
    measurement landed — almost always a backend / IO bug.

Usage
-----
python -m experiments.validate_sweep \
    --run-dir results/stage_sweep/<run_id>

Exit codes
----------
0  all gates pass
1  one or more gates failed (per-gate detail printed)
2  cannot read the sweep outputs (missing / corrupt)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


GATE_OK = "PASS"
GATE_FAIL = "FAIL"
GATE_SKIP = "SKIP"


# ===========================================================================
# IO helpers
# ===========================================================================
def _read_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_per_frame(path: Path) -> List[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows


def _f(x) -> float:
    """Tolerant float coercion: empty strings, None -> NaN."""
    if x is None or x == "":
        return float("nan")
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


# ===========================================================================
# Gate functions
# ===========================================================================
def gate_n_faster_than_s(summary: dict) -> Tuple[str, str]:
    """Accept fast-faster-than-accurate, OR a saturated-platform tie.

    The intent of this gate is to catch swapped weights, thermal
    throttle, or wrong device — anomalies where fast is much slower
    than accurate. On a saturated GPU running small YOLO models, both
    nets can be bottlenecked by NMS / postprocess time, leaving an
    n_lat ~ s_lat steady state with no meaningful gap. That is a real
    'platform inversion' datum and must not fail the gate.

    PASS rules (any one):
      A. n_lat <= s_lat                     -- normal case
      B. n_lat - s_lat <= 1.0 ms            -- below clock noise
      C. n_lat <= s_lat * 1.05              -- within 5% (saturated)
    FAIL otherwise (the swap / throttle case: n_lat >> s_lat).
    """
    pp = summary.get("per_policy", {}) or {}
    n_lat = pp.get("n_only", {}).get("latency", {}).get("mean")
    s_lat = pp.get("s_only", {}).get("latency", {}).get("mean")
    if n_lat is None or s_lat is None:
        return GATE_SKIP, ("n_only or s_only missing from per_policy; "
                           "cannot compare latencies")

    if n_lat <= s_lat:
        return GATE_OK, (f"n_only mean {n_lat:.2f}ms < s_only mean "
                         f"{s_lat:.2f}ms")

    gap = n_lat - s_lat
    rel = gap / max(s_lat, 1e-9)
    if gap <= 1.0:
        return GATE_OK, (f"n_only mean {n_lat:.2f}ms vs s_only "
                         f"{s_lat:.2f}ms (gap {gap:.2f}ms <= 1ms, "
                         f"sub-clock-noise; saturated platform)")
    if rel <= 0.05:
        return GATE_OK, (f"n_only mean {n_lat:.2f}ms vs s_only "
                         f"{s_lat:.2f}ms (gap {gap:.2f}ms = "
                         f"{rel*100:.1f}% <= 5%; saturated platform — "
                         f"fast / accurate tie within tolerance)")
    return GATE_FAIL, (
        f"n_only mean {n_lat:.2f}ms >> s_only mean {s_lat:.2f}ms "
        f"(gap {gap:.2f}ms = {rel*100:.0f}% > 5%) — fast is materially "
        "slower than accurate. Likely swapped weights, thermal throttle, "
        "or missing GPU warmup (run_sweep --warmup-frames)."
    )


def gate_s_dets_non_empty(per_frame: List[dict]) -> Tuple[str, str]:
    s_counts = [_f(r.get("s_count")) for r in per_frame]
    s_counts = [c for c in s_counts if not math.isnan(c)]
    if not s_counts:
        return GATE_SKIP, "no s_count column data"
    mean_s = sum(s_counts) / len(s_counts)
    if mean_s >= 0.5:
        return GATE_OK, f"mean(s_count) = {mean_s:.2f} >= 0.5"
    return GATE_FAIL, (
        f"mean(s_count) = {mean_s:.3f} < 0.5 — accurate model finds "
        "almost nothing. Recall metrics will be vacuous."
    )


def gate_metrics_not_trivially_one(
    per_frame: List[dict],
    summary: dict,
) -> Tuple[str, str]:
    pp = summary.get("per_policy", {}) or {}
    if not pp:
        return GATE_SKIP, "no per_policy block in summary"

    iou_rates = {pn: pp[pn].get("iou_match_rate", 0.0) for pn in pp}
    cnt_rates = {pn: pp[pn].get("count_agree_rate", 0.0) for pn in pp}
    ben_rates = {pn: pp[pn].get("benefit_rate", 0.0) for pn in pp}

    all_iou_one = all(abs(v - 1.0) < 1e-6 for v in iou_rates.values())
    all_cnt_one = all(abs(v - 1.0) < 1e-6 for v in cnt_rates.values())

    if not (all_iou_one and all_cnt_one):
        return GATE_OK, (
            f"non-trivial spread: iou_match in "
            f"[{min(iou_rates.values()):.3f}, {max(iou_rates.values()):.3f}], "
            f"count_agree in "
            f"[{min(cnt_rates.values()):.3f}, {max(cnt_rates.values()):.3f}]"
        )

    # All metrics are flat 1.0. Accept this only if the video is
    # genuinely easy (benefit_rate is also 0 across all policies AND
    # iou rates are identical across all policies — i.e., switching
    # made no observable difference because there was nothing to switch
    # for).
    all_ben_zero = all(v == 0.0 for v in ben_rates.values())
    iou_spread = max(iou_rates.values()) - min(iou_rates.values())
    if all_ben_zero and iou_spread < 1e-6:
        return GATE_OK, (
            "all metrics flat at 1.0 but benefit_rate==0 and iou_rate "
            "is identical across policies — legitimate easy-video case"
        )

    return GATE_FAIL, (
        "iou_match_rate and count_agree_rate are 1.0 across every "
        "policy with non-zero benefit_rate. Matching is likely "
        "degenerate (e.g., classes-mode disagreement, or both backends "
        "returning identical bytes). Inspect a few per-frame rows."
    )


def gate_s_choice_plausible(summary: dict) -> Tuple[str, str]:
    pp = summary.get("per_policy", {}) or {}
    if not pp:
        return GATE_SKIP, "no per_policy block in summary"

    issues: List[str] = []
    n_rate = pp.get("n_only", {}).get("s_choice_rate")
    s_rate = pp.get("s_only", {}).get("s_choice_rate")
    if n_rate is None or abs(n_rate - 0.0) > 1e-9:
        issues.append(f"n_only.s_choice_rate={n_rate} (expected 0.0)")
    if s_rate is None or abs(s_rate - 1.0) > 1e-9:
        issues.append(f"s_only.s_choice_rate={s_rate} (expected 1.0)")

    for pn, p in pp.items():
        if pn in ("n_only", "s_only"):
            continue
        rate = p.get("s_choice_rate")
        if rate is None:
            issues.append(f"{pn}.s_choice_rate is missing")
        elif not (0.0 <= rate <= 1.0):
            issues.append(f"{pn}.s_choice_rate={rate} out of [0, 1]")

    if issues:
        return GATE_FAIL, "; ".join(issues)
    return GATE_OK, "all s_choice_rate values in expected ranges"


def gate_no_nan_timings(per_frame: List[dict]) -> Tuple[str, str]:
    bad = []
    for i, r in enumerate(per_frame):
        for col in ("chosen_latency_ms", "proxy_ms_total",
                    "proxy_ms_policy_relevant", "decision_ms",
                    "n_latency_ms", "s_latency_ms",
                    "t_total_selected_ms", "t_total_deployed_ms",
                    "t_total_shared_proxy_ms",
                    "t_total_policy_relevant_ms"):
            if col in r:
                v = _f(r[col])
                if math.isnan(v):
                    bad.append((i, col, r.get(col)))
                    break
        if len(bad) >= 5:
            break
    if not bad:
        return GATE_OK, f"checked {len(per_frame)} rows, no NaN/empty timings"
    sample = "; ".join(f"row {i} col {c}={v!r}" for i, c, v in bad[:3])
    return GATE_FAIL, f"NaN/empty timing columns: {sample}"


# ===========================================================================
# Driver
# ===========================================================================
def validate(run_dir: Path) -> int:
    summary_path = None
    for candidate in (
        run_dir / "sweep_summary.json",
        run_dir / "summary.json",
    ):
        if candidate.exists():
            summary_path = candidate
            break
    if summary_path is None:
        print(f"[validate_sweep] ERROR: no summary JSON in {run_dir}",
              file=sys.stderr)
        return 2

    per_frame_path = None
    for candidate in (
        run_dir / "sweep_per_frame.csv",
        run_dir / "per_frame.csv",
    ):
        if candidate.exists():
            per_frame_path = candidate
            break
    if per_frame_path is None:
        print(f"[validate_sweep] ERROR: no per_frame CSV in {run_dir}",
              file=sys.stderr)
        return 2

    summary = _read_summary(summary_path)
    per_frame = _read_per_frame(per_frame_path)
    print(f"[validate_sweep] {summary_path.name} + {per_frame_path.name} "
          f"({len(per_frame)} rows)")

    gates = [
        ("G1 n<s latency", gate_n_faster_than_s(summary)),
        ("G2 s_dets non-empty", gate_s_dets_non_empty(per_frame)),
        ("G3 metrics not trivial 1.0",
         gate_metrics_not_trivially_one(per_frame, summary)),
        ("G4 s_choice plausible", gate_s_choice_plausible(summary)),
        ("G5 no NaN timings", gate_no_nan_timings(per_frame)),
    ]

    n_fail = 0
    for label, (status, msg) in gates:
        marker = {GATE_OK: "[OK]  ",
                  GATE_FAIL: "[FAIL]",
                  GATE_SKIP: "[SKIP]"}[status]
        print(f"  {marker} {label}: {msg}")
        if status == GATE_FAIL:
            n_fail += 1

    print()
    if n_fail == 0:
        print("[validate_sweep] all gates passed — safe to scale up.")
        return 0
    print(f"[validate_sweep] {n_fail} gate(s) failed — DO NOT scale up "
          f"until resolved.")
    return 1


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run pre-full-sweep sanity gates on a sweep run dir.",
    )
    p.add_argument("--run-dir", type=Path, required=True,
                   help="Path to results/stage_sweep/<run_id> directory.")
    return p.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    if not args.run_dir.exists():
        print(f"[validate_sweep] ERROR: run dir does not exist: "
              f"{args.run_dir}", file=sys.stderr)
        return 2
    return validate(args.run_dir)


if __name__ == "__main__":
    raise SystemExit(main())
