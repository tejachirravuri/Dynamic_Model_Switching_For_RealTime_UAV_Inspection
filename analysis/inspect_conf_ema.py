"""analysis/inspect_conf_ema.py — replay conf_ema's EMA math from a sweep CSV.

Why this exists
---------------
conf_ema's only input is the previous frame's fast-model mean
confidence (audit F4b contract). That signal is now logged as
``fast_mean_conf`` per row in ``sweep_per_frame.csv``. Given that
trace + the locked PolicyConfig defaults, the entire conf_ema
trajectory (``fast_ema``, ``slow_ema``, ``conf_drop``) is
deterministically reproducible offline. So instead of adding more
columns to the CSV (which would make every other policy carry
conf_ema's state), we replay it here.

Classification
--------------
* CASE 1 (TRUE NEGATIVE): ``max(conf_drop) < 0.05``. The video has
  no meaningful confidence dips; conf_ema is correctly silent.
  Thesis line: 'confidence-based switching activates only when
  meaningful detection uncertainty arises.'
* CASE 2 (MISCONFIGURED THRESHOLD): peak of ``conf_drop`` lands in
  ``[0.06, 0.10]`` but never reaches the trigger ``c_high = 0.12``.
  Signal exists, threshold too strict; lower ``c_high``.
* CASE 3 (RARE SPIKES): ``conf_drop`` occasionally exceeds 0.12 but
  the live policy reported s% ~ 0. Hysteresis / smoothing too
  inertial; tune ``fast_beta`` / ``slow_beta``.

Usage
-----
    python -m analysis.inspect_conf_ema \
        --run-dir results/stage_sweep/<run_id>

Optional overrides for sensitivity sweeps:
    --c-high 0.06   (override the live default to see what would have triggered)
    --plot path.png (write a small matplotlib trajectory plot)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

# Live defaults — keep in sync with dms.policies.PolicyConfig.
DEFAULT_FAST_BETA = 0.30
DEFAULT_SLOW_BETA = 0.02
DEFAULT_C_LOW = 0.04
DEFAULT_C_HIGH = 0.12
DEFAULT_WARMUP = 5


# ===========================================================================
# Replay
# ===========================================================================
@dataclass
class ReplayResult:
    fast_mean_conf: List[float]
    fast_ema: List[float]
    slow_ema: List[float]
    conf_drop: List[float]


def replay_conf_ema(
    fast_mean_conf_per_frame: List[float],
    fast_beta: float = DEFAULT_FAST_BETA,
    slow_beta: float = DEFAULT_SLOW_BETA,
) -> ReplayResult:
    """Run the conf_ema EMA recursion offline.

    Matches dms.policies.ConfEMA._conf_drop exactly:
      - First frame: both EMAs initialise to fast_mean_conf[0].
      - Subsequent frames: standard exponential moving average.
      - conf_drop = max(0, (slow_ema - fast_ema) / (slow_ema + 1e-9))
    """
    n = len(fast_mean_conf_per_frame)
    fast_ema = [0.0] * n
    slow_ema = [0.0] * n
    conf_drop = [0.0] * n
    if n == 0:
        return ReplayResult([], [], [], [])

    fast_ema[0] = float(fast_mean_conf_per_frame[0])
    slow_ema[0] = float(fast_mean_conf_per_frame[0])
    conf_drop[0] = 0.0

    for i in range(1, n):
        x = float(fast_mean_conf_per_frame[i])
        fast_ema[i] = fast_beta * x + (1.0 - fast_beta) * fast_ema[i - 1]
        slow_ema[i] = slow_beta * x + (1.0 - slow_beta) * slow_ema[i - 1]
        denom = slow_ema[i] + 1e-9
        conf_drop[i] = max(0.0, (slow_ema[i] - fast_ema[i]) / denom)

    return ReplayResult(
        fast_mean_conf=[float(v) for v in fast_mean_conf_per_frame],
        fast_ema=fast_ema,
        slow_ema=slow_ema,
        conf_drop=conf_drop,
    )


# ===========================================================================
# CSV reader
# ===========================================================================
def read_fast_mean_conf(per_frame_csv: Path) -> List[float]:
    """Extract per-frame fast_mean_conf trace from sweep_per_frame.csv.

    fast_mean_conf is broadcast across all 7 policy rows for a given
    frame, so picking the rows for a single policy gives the unique
    trace. We pick whichever policy is in the file (n_only by
    convention; falls back to the first encountered).
    """
    with per_frame_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return []
    if "fast_mean_conf" not in rows[0]:
        raise ValueError(
            "fast_mean_conf column missing — re-run experiments.run_sweep "
            "with the latest schema."
        )
    target_policy = "n_only" if any(r["policy"] == "n_only" for r in rows) \
        else rows[0]["policy"]
    trace = [float(r["fast_mean_conf"]) for r in rows
             if r["policy"] == target_policy]
    return trace


# ===========================================================================
# Classification
# ===========================================================================
def classify(
    conf_drop: List[float],
    c_high: float = DEFAULT_C_HIGH,
) -> Tuple[str, str]:
    """Return (case_label, explanation) from the conf_drop trajectory."""
    if not conf_drop:
        return "EMPTY", "no frames"
    peak = max(conf_drop)
    if peak < 0.05:
        return "CASE_1_TRUE_NEGATIVE", (
            f"max(conf_drop) = {peak:.4f} < 0.05 — video has no "
            f"meaningful confidence dips. conf_ema is correctly silent. "
            f"Lock c_high = {c_high:.2f} as-is."
        )
    n_above_c_high = sum(1 for v in conf_drop if v >= c_high)
    if n_above_c_high == 0:
        return "CASE_2_MISCONFIGURED", (
            f"max(conf_drop) = {peak:.4f} -- signal exists but never "
            f"reaches c_high = {c_high:.2f}. Lower c_high to ~"
            f"{max(0.05, peak * 0.6):.2f}-{max(0.06, peak * 0.8):.2f} "
            f"and re-run."
        )
    return "CASE_3_RARE_SPIKES", (
        f"conf_drop exceeded c_high = {c_high:.2f} on "
        f"{n_above_c_high}/{len(conf_drop)} frames "
        f"({100 * n_above_c_high / len(conf_drop):.1f}%) "
        f"with peak = {peak:.4f}. If live s% was still ~0, the "
        f"hysteresis or stabiliser is suppressing valid triggers — "
        f"tune fast_beta upward or slow_beta downward."
    )


# ===========================================================================
# Reporting
# ===========================================================================
def percentile(sorted_vals: List[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def report(replay: ReplayResult, c_high: float) -> int:
    n = len(replay.conf_drop)
    if n == 0:
        print("[inspect_conf_ema] no rows", file=sys.stderr)
        return 2

    fmc = sorted(replay.fast_mean_conf)
    cd = sorted(replay.conf_drop)
    print("[inspect_conf_ema] frames:", n)
    print("  fast_mean_conf  "
          f"min={fmc[0]:.4f}  p50={percentile(fmc, 50):.4f}  "
          f"p95={percentile(fmc, 95):.4f}  max={fmc[-1]:.4f}")
    print("  conf_drop       "
          f"min={cd[0]:.4f}  p50={percentile(cd, 50):.4f}  "
          f"p95={percentile(cd, 95):.4f}  max={cd[-1]:.4f}")

    thresholds = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12]
    counts = {t: sum(1 for v in replay.conf_drop if v >= t) for t in thresholds}
    print("  frames above conf_drop threshold:")
    for t in thresholds:
        print(f"    >= {t:.2f}  {counts[t]:>5d}  "
              f"({100 * counts[t] / n:5.1f}%)")

    label, msg = classify(replay.conf_drop, c_high=c_high)
    print()
    print(f"[inspect_conf_ema] verdict: {label}")
    print(f"  {msg}")
    return 0


def maybe_plot(replay: ReplayResult, out_path: Path) -> None:  # pragma: no cover
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[inspect_conf_ema] matplotlib not available; skipping plot",
              file=sys.stderr)
        return
    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    ax[0].plot(replay.fast_mean_conf, lw=0.6, label="fast_mean_conf",
               color="0.4")
    ax[0].plot(replay.fast_ema, lw=0.8, label="fast_ema",
               color="tab:blue")
    ax[0].plot(replay.slow_ema, lw=0.8, label="slow_ema",
               color="tab:orange")
    ax[0].set_ylabel("confidence")
    ax[0].legend(loc="lower right", fontsize=8)
    ax[0].grid(True, alpha=0.3)
    ax[1].plot(replay.conf_drop, lw=0.6, color="tab:red", label="conf_drop")
    ax[1].axhline(DEFAULT_C_HIGH, lw=0.8, ls="--", color="0.5",
                  label=f"c_high={DEFAULT_C_HIGH}")
    ax[1].axhline(DEFAULT_C_LOW, lw=0.5, ls=":", color="0.7",
                  label=f"c_low={DEFAULT_C_LOW}")
    ax[1].set_xlabel("frame")
    ax[1].set_ylabel("conf_drop")
    ax[1].legend(loc="upper right", fontsize=8)
    ax[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"[inspect_conf_ema] plot written to {out_path}")


# ===========================================================================
# CLI
# ===========================================================================
def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Replay conf_ema EMA math from a sweep run.",
    )
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--csv-name", type=str, default="sweep_per_frame.csv")
    p.add_argument("--fast-beta", type=float, default=DEFAULT_FAST_BETA)
    p.add_argument("--slow-beta", type=float, default=DEFAULT_SLOW_BETA)
    p.add_argument("--c-high", type=float, default=DEFAULT_C_HIGH)
    p.add_argument("--plot", type=Path, default=None,
                   help="Optional output path for a trajectory plot.")
    return p.parse_args(argv)


def _load_active_params(run_dir: Path) -> dict:
    """Read summary.config.conf_ema_params if present, else empty dict.

    Lets the diagnostic compare against the c_high / betas the run
    actually used, not the hardcoded library defaults. Without this,
    reports were misleading after a CLI override (e.g., live run with
    --conf-ema-c-high 0.02 was mislabelled as MISCONFIGURED because
    the diagnostic was still reading 0.12).
    """
    for name in ("sweep_summary.json", "summary.json"):
        p = run_dir / name
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return (data.get("config") or {}).get("conf_ema_params") or {}
    return {}


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    csv_path = args.run_dir / args.csv_name
    if not csv_path.exists():
        print(f"[inspect_conf_ema] missing {csv_path}", file=sys.stderr)
        return 2

    # Pull the run's active params; CLI flags take precedence so users
    # can still simulate alternate calibrations against a single CSV.
    active = _load_active_params(args.run_dir)
    fast_beta = (args.fast_beta if args.fast_beta != DEFAULT_FAST_BETA
                 else float(active.get("fast_beta", DEFAULT_FAST_BETA)))
    slow_beta = (args.slow_beta if args.slow_beta != DEFAULT_SLOW_BETA
                 else float(active.get("slow_beta", DEFAULT_SLOW_BETA)))
    c_high = (args.c_high if args.c_high != DEFAULT_C_HIGH
              else float(active.get("c_high", DEFAULT_C_HIGH)))
    if active:
        c_low_active = active.get("c_low")
        c_low_str = f"{float(c_low_active):.4f}" if c_low_active is not None else "?"
        print(f"[inspect_conf_ema] active run params: "
              f"c_high={c_high:.4f} c_low={c_low_str} "
              f"fast_beta={fast_beta:.3f} slow_beta={slow_beta:.3f}")

    trace = read_fast_mean_conf(csv_path)
    replay = replay_conf_ema(trace,
                             fast_beta=fast_beta,
                             slow_beta=slow_beta)
    rc = report(replay, c_high=c_high)
    if args.plot is not None:
        maybe_plot(replay, args.plot)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
