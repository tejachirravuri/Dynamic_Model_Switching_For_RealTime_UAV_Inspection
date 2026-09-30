"""analysis/stage3_pareto.py — Stage 3 figures (T_total + random-baseline).

Reads the audit table written by ``analysis.stage3_audit`` (preferred)
or falls back to a master_table.csv if the audit table is absent.

The audit table has the locked-attribution T_total and the random-
baseline / informed-gain / trigger-F1 columns; without it, the figures
would silently report selected-model latency (a bug in an earlier version).

run_id grammar
--------------
``stage3_{family}_{fast}_{acc}_{domain}_{device}``

Outputs (under --out-dir, default ``analysis/figures/stage3``)
--------------------------------------------------------------
* ``stage3_summary.csv``           pivoted summary
* ``fig_gpu_pareto.png``           2x2 Pareto on CUDA, T_total vs iou
* ``fig_cpu_pareto.png``           2x2 Pareto on CPU, T_total vs iou
* ``fig_platform_inversion.png``   domain-winner CPU vs CUDA, log-x
* ``fig_random_baseline.png``      informed_gain_over_random per policy,
                                   panelled by (device, domain)
* ``fig_p95_tail.png``             p95 tail latency vs iou_match

Design
------
* X axis on Pareto plots is **T_total mean (ms)** for honest
  apples-to-apples comparison.
* A dotted "random switching" reference line connects (n_iou at 0%)
  and (s_iou at 100%) so a reader can see at a glance which policies
  are above the random interpolation.
* Policy colours stable across figures: combined_hyst deep blue
  (glass winner on iou), entropy_only green (porcelain winner on iou).
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Run-id parser
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RunKey:
    family: str    # yolov8 | yolo26
    fast: str      # n
    acc: str       # s | l
    domain: str    # glass | porcelain
    device: str    # cpu | cuda


def parse_run_id(run_id: str) -> Optional[RunKey]:
    """Parse `stage3_yolov8_n_s_glass_cuda` -> RunKey(...).

    Returns None for non-stage3 rows.
    """
    if not run_id.startswith("stage3_"):
        return None
    body = run_id[len("stage3_"):]
    parts = body.split("_")
    if len(parts) != 5:
        return None
    family, fast, acc, domain, device = parts
    return RunKey(family=family, fast=fast, acc=acc,
                  domain=domain, device=device)


# ---------------------------------------------------------------------------
# Master table reader
# ---------------------------------------------------------------------------
# Columns we'll always coerce to float when present.
NUMERIC_COLS = (
    "iou_match_rate", "det_coverage_rate", "det_recall_mean",
    "count_agree_rate", "benefit_rate", "s_choice_rate",
    # legacy (selected-model only):
    "lat_mean_ms", "lat_p50_ms", "lat_p95_ms", "lat_p99_ms", "fps_steady",
    # audit (T_total + breakdown):
    "t_scene_mean_ms", "t_ctrl_mean_ms", "t_infer_mean_ms",
    "t_total_mean_ms", "t_total_p50_ms", "t_total_p95_ms", "t_total_p99_ms",
    # audit (random baseline + trigger F1):
    "random_iou_at_same_s", "informed_gain_over_random",
    "trigger_precision", "trigger_recall", "trigger_f1",
)


def read_stage3(master_csv: Path) -> List[Dict]:
    out: List[Dict] = []
    with master_csv.open("r", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = parse_run_id(r["run_id"])
            if key is None:
                continue
            row = dict(r)
            row["_key"] = key
            for c in NUMERIC_COLS:
                if c in row and row[c] != "":
                    try:
                        row[c] = float(row[c])
                    except ValueError:
                        row[c] = float("nan")
            out.append(row)
    return out


def has_audit_columns(rows: List[Dict]) -> bool:
    """Detect whether rows came from stage3_audit (T_total) vs legacy."""
    if not rows:
        return False
    return "t_total_mean_ms" in rows[0]


# ---------------------------------------------------------------------------
# Summary CSV
# ---------------------------------------------------------------------------
POLICY_ORDER = (
    "n_only", "s_only", "entropy_only", "combined", "combined_hyst",
    "conf_ema", "multi_proxy",
)


def write_summary_csv(rows: List[Dict], out_path: Path) -> None:
    """One row per run, columns = (policy×metric) wide layout."""
    by_run: Dict[Tuple, Dict] = {}
    for r in rows:
        k = r["_key"]
        run_tag = (k.family, f"{k.fast}_{k.acc}", k.domain, k.device)
        if run_tag not in by_run:
            by_run[run_tag] = dict(
                family=k.family, pair=f"{k.fast}_{k.acc}",
                domain=k.domain, device=k.device,
                n_frames=int(float(r.get("n_frames") or 0)),
                benefit_rate=r.get("benefit_rate", float("nan")),
            )
        for metric in ("iou_match_rate", "det_coverage_rate",
                       "det_recall_mean", "s_choice_rate",
                       "lat_mean_ms", "lat_p95_ms"):
            by_run[run_tag][f"{r['policy']}__{metric}"] = r.get(
                metric, float("nan")
            )

    if not by_run:
        return
    fieldnames = list(next(iter(by_run.values())).keys())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for tag in sorted(by_run.keys()):
            w.writerow(by_run[tag])


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
POLICY_COLORS = {
    "n_only":         "#888888",
    "s_only":         "#222222",
    "entropy_only":   "#2ca02c",   # green — porcelain winner
    "combined":       "#1f77b4",
    "combined_hyst":  "#0033cc",   # deep blue — glass winner
    "conf_ema":       "#d62728",
    "multi_proxy":    "#9467bd",
}
PAIR_MARKERS = {"n_s": "o", "n_l": "s"}
DEVICE_FILL = {"cuda": "white", "cpu": None}  # None = use color


def _random_baseline_line(cell_rows: List[Dict], x_col: str
                          ) -> Optional[Tuple[List[float], List[float]]]:
    """Return (xs, ys) for the random-switch interpolation line.

    Builds the line between the n_only point and the s_only point on
    whatever x_axis we're using. If a policy lies BELOW this line, it
    is being beaten by random switching at the same accurate-model
    usage rate (or by the same latency budget on x=T_total).
    """
    by_pol = {r["policy"]: r for r in cell_rows
              if isinstance(r.get(x_col), float)
              and not _isnan(r[x_col])}
    if "n_only" not in by_pol or "s_only" not in by_pol:
        return None
    n = by_pol["n_only"]
    s = by_pol["s_only"]
    x_n = float(n[x_col])
    x_s = float(s[x_col])
    y_n = float(n["iou_match_rate"])
    y_s = float(s["iou_match_rate"])
    if x_n == x_s:
        return [x_n, x_s], [y_n, y_s]
    return [x_n, x_s], [y_n, y_s]


def _isnan(v) -> bool:
    try:
        import math as _m
        return _m.isnan(float(v))
    except Exception:
        return True


def _plot_pareto_grid(
    rows: List[Dict],
    device_filter: str,
    x_col: str,
    x_label: str,
    title: str,
    out_path: Path,
    annotate_winners: bool = True,
) -> None:
    """2x2 Pareto grid (rows=domain, cols=family), bigger panels +
    side legend + annotated winners + per-pair connecting lines."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.lines as mlines
    except ImportError:
        print("[stage3_pareto] matplotlib unavailable; skipping plots",
              file=sys.stderr)
        return

    # Reserve right strip for legend so panels stay big.
    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.42], hspace=0.30,
                          wspace=0.20, left=0.07, right=0.99,
                          top=0.92, bottom=0.07)
    families = ("yolov8", "yolo26")
    domains = ("glass", "porcelain")
    domain_winner = {"glass": "combined_hyst", "porcelain": "entropy_only"}

    for r_i, dom in enumerate(domains):
        for c_i, fam in enumerate(families):
            ax = fig.add_subplot(gs[r_i, c_i])
            cell = [r for r in rows
                    if r["_key"].domain == dom
                    and r["_key"].family == fam
                    and r["_key"].device == device_filter]
            if not cell:
                ax.set_title(f"{fam} / {dom} (no data)", fontsize=11)
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, color="0.5", fontsize=10)
                continue

            # Random-baseline reference line per pair.
            for pair in ("n_s", "n_l"):
                pair_rows = [r for r in cell
                             if f"{r['_key'].fast}_{r['_key'].acc}" == pair]
                rb = _random_baseline_line(pair_rows, x_col)
                if rb:
                    xs, ys = rb
                    ax.plot(xs, ys, lw=1.0, ls=":", color="0.55",
                            alpha=0.7, zorder=1)

            for r in cell:
                k = r["_key"]
                pol = r["policy"]
                x = r.get(x_col, float("nan"))
                y = r.get("iou_match_rate", float("nan"))
                marker = PAIR_MARKERS.get(f"{k.fast}_{k.acc}", "o")
                color = POLICY_COLORS.get(pol, "0.4")
                is_winner = annotate_winners and pol == domain_winner.get(dom)
                is_baseline = pol in ("n_only", "s_only")
                size = 220 if is_winner else (140 if is_baseline else 120)
                edge = ("black" if (is_winner or is_baseline) else "0.35")
                lw = 1.2 if is_winner else 0.7
                ax.scatter(x, y, s=size, marker=marker,
                           color=color, edgecolor=edge,
                           linewidth=lw, zorder=3)
                # Annotate winner per pair.
                if is_winner and not _isnan(x) and not _isnan(y):
                    ax.annotate(
                        f"{pol}\n{f'{k.fast}_{k.acc}'}",
                        xy=(x, y), xytext=(8, -6),
                        textcoords="offset points",
                        fontsize=8, color="black", fontweight="bold",
                    )

            ax.set_title(f"{fam.upper()}  /  {dom}", fontsize=12,
                         fontweight="bold")
            ax.set_xlabel(x_label, fontsize=10)
            ax.set_ylabel("iou_match", fontsize=10)
            ax.grid(True, alpha=0.3, zorder=0)
            ax.set_axisbelow(True)
            ax.set_ylim(-0.02, 1.04)

    # Side legend column on the right (spans both rows).
    leg_ax = fig.add_subplot(gs[:, 2])
    leg_ax.axis("off")
    handles: List = []
    # Headline winners first.
    handles.append(mlines.Line2D([], [], color=POLICY_COLORS["combined_hyst"],
                                  marker="s", linestyle="None", markersize=11,
                                  markeredgecolor="black", markeredgewidth=1.0,
                                  label="combined_hyst (glass winner)"))
    handles.append(mlines.Line2D([], [], color=POLICY_COLORS["entropy_only"],
                                  marker="o", linestyle="None", markersize=11,
                                  markeredgecolor="black", markeredgewidth=1.0,
                                  label="entropy_only (porcelain winner)"))
    handles.append(mlines.Line2D([], [], color="white", label=" "))
    # Other policies (one shape, n_s example).
    for pol, col in POLICY_COLORS.items():
        if pol in ("combined_hyst", "entropy_only"):
            continue
        edge = "black" if pol in ("n_only", "s_only") else "0.35"
        handles.append(mlines.Line2D([], [], color=col, marker="o",
                                      linestyle="None", markersize=9,
                                      markeredgecolor=edge,
                                      markeredgewidth=0.7, label=pol))
    handles.append(mlines.Line2D([], [], color="white", label=" "))
    handles.append(mlines.Line2D([], [], color="0.4", marker="o",
                                  linestyle="None", markersize=9,
                                  label="circle = n_s pair"))
    handles.append(mlines.Line2D([], [], color="0.4", marker="s",
                                  linestyle="None", markersize=9,
                                  label="square = n_l pair"))
    handles.append(mlines.Line2D([], [], color="white", label=" "))
    handles.append(mlines.Line2D([], [], color="0.55", lw=1.0, ls=":",
                                  label="random-switch reference"))
    leg_ax.legend(handles=handles, loc="center left", fontsize=10,
                  frameon=False)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[stage3_pareto] wrote {out_path}")


def _plot_random_gain(rows: List[Dict], out_path: Path) -> None:
    """Bar chart of informed_gain_over_random per policy.

    Stage 3 only (excludes ablation_* runs from the average so the
    bar represents the deployed-default behaviour, not the tuned/
    swept variants).

    On RTX 5090 vs CPU iou_match is identical (deterministic policies
    on identical frames), so the GAIN values are identical too. We
    show only the per-domain average across the 4 pair configs and
    drop the redundant device dimension. Two panels: glass + porcelain.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if not rows or "informed_gain_over_random" not in rows[0]:
        print("[stage3_pareto] no informed_gain column; skipping",
              file=sys.stderr)
        return

    # Filter to stage3 (non-ablation) rows so the gain shown is the
    # deployed-default value, not the swept ones.
    stage3_rows = [r for r in rows if r["run_id"].startswith("stage3_")]
    if not stage3_rows:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    domains = ("glass", "porcelain")
    pols_to_plot = ["n_only", "multi_proxy", "entropy_only", "combined",
                    "combined_hyst", "conf_ema"]

    # Find the y-axis range across both panels so bars are comparable.
    all_gains: List[float] = []
    panel_data = {}
    for dom in domains:
        cell = [r for r in stage3_rows if r["_key"].domain == dom]
        gains = {p: [] for p in pols_to_plot}
        for r in cell:
            if r["policy"] in gains:
                g = r.get("informed_gain_over_random")
                if isinstance(g, float) and not _isnan(g):
                    gains[r["policy"]].append(g)
        means = {p: (sum(gs) / len(gs) if gs else 0.0)
                 for p, gs in gains.items()}
        panel_data[dom] = means
        all_gains.extend(means.values())

    y_max = max(all_gains + [0.0]) * 1.4
    y_min = min(all_gains + [0.0]) * 1.4 if min(all_gains, default=0.0) < 0 else -0.005

    for ax, dom in zip(axes, domains):
        means = panel_data[dom]
        xs = list(range(len(pols_to_plot)))
        ys = [means[p] for p in pols_to_plot]
        cols = [POLICY_COLORS.get(p, "0.5") for p in pols_to_plot]
        bars = ax.bar(xs, ys, color=cols, edgecolor="black", linewidth=0.7,
                      width=0.7)
        ax.axhline(0, color="black", lw=1.0, zorder=2)

        # Value labels on each bar (above for positive, below for negative).
        for bar, val in zip(bars, ys):
            if val == 0.0:
                continue
            x = bar.get_x() + bar.get_width() / 2
            offset = 0.0015 if val >= 0 else -0.0015
            va = "bottom" if val >= 0 else "top"
            ax.text(x, val + offset, f"{val:+.3f}",
                    ha="center", va=va, fontsize=10, fontweight="bold")

        ax.set_xticks(xs)
        ax.set_xticklabels(pols_to_plot, rotation=20, ha="right",
                           fontsize=10)
        ax.set_title(f"{dom}", fontsize=13, fontweight="bold")
        if ax is axes[0]:
            ax.set_ylabel("informed_gain_over_random\n"
                          "(higher = trigger beats coin-flip at same s%)",
                          fontsize=11)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, axis="y", alpha=0.3, zorder=0)
        ax.set_axisbelow(True)
        # Light shading for "below random" zone.
        ax.axhspan(y_min, 0, alpha=0.08, color="red", zorder=0)

    fig.suptitle(
        "Stage 3 — informed gain vs random-switch baseline\n"
        "(positive = trigger is informative; negative = below coin-flip)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[stage3_pareto] wrote {out_path}")


def _plot_p95_tail(rows: List[Dict], device_filter: str,
                   out_path: Path) -> None:
    """p95 T_total vs iou — same layout as Pareto but on tail latency."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if not rows or "t_total_p95_ms" not in rows[0]:
        return
    _plot_pareto_grid(
        rows, device_filter=device_filter,
        x_col="t_total_p95_ms",
        x_label="T_total p95 tail (ms)",
        title=f"Stage 3 {device_filter.upper()} p95 tail-latency frontier",
        out_path=out_path,
    )


def _plot_platform_inversion(rows: List[Dict], out_path: Path) -> None:
    """Domain-winner CPU vs CUDA on log-x latency.

    For clarity: only the n_l pair (where the latency story has the
    most teeth). One panel per domain. CPU points solid red, CUDA
    points hollow blue; lines connect policy points by latency.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.lines as mlines
    except ImportError:
        return

    x_col = ("t_total_mean_ms"
             if rows and "t_total_mean_ms" in rows[0]
             else "lat_mean_ms")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    panels = (
        ("glass",     "combined_hyst",  axes[0]),
        ("porcelain", "entropy_only",   axes[1]),
    )

    # Stage 3 only (no ablations) and only n_l pair (clean story).
    for dom, winner, ax in panels:
        for device, dev_color, dev_ls in (
            ("cpu", "tab:red", "-"),
            ("cuda", "tab:blue", "--"),
        ):
            cell = [r for r in rows
                    if r["run_id"].startswith("stage3_")
                    and r["_key"].domain == dom
                    and r["_key"].device == device
                    and r["_key"].acc == "l"]
            if not cell:
                continue
            cell.sort(key=lambda r: r.get(x_col) or 0.0)
            xs = [r.get(x_col) for r in cell]
            ys = [r.get("iou_match_rate") for r in cell]
            ax.plot(xs, ys, ls=dev_ls, lw=1.6, color=dev_color, alpha=0.55,
                    zorder=2, label=f"{device} (n_l pair)")
            # Random-baseline ref.
            rb = _random_baseline_line(cell, x_col)
            if rb:
                rxs, rys = rb
                ax.plot(rxs, rys, lw=0.9, ls=":", color="0.5",
                        alpha=0.7, zorder=1)
            for r in cell:
                pol = r["policy"]
                x = r.get(x_col)
                y = r.get("iou_match_rate")
                is_winner = (pol == winner)
                is_baseline = pol in ("n_only", "s_only")
                size = 220 if is_winner else (140 if is_baseline else 110)
                ec = ("black" if (is_winner or is_baseline) else "0.35")
                lw = 1.2 if is_winner else 0.7
                marker = "o" if device == "cpu" else "D"
                ax.scatter(x, y, s=size, marker=marker,
                           color=POLICY_COLORS.get(pol, "0.4"),
                           edgecolor=ec, linewidth=lw, zorder=3)
                if pol == winner and device == "cpu":
                    ax.annotate(
                        f"{pol}\nT_total = {x:.0f} ms\niou = {y:.3f}",
                        xy=(x, y),
                        xytext=(12, -10),
                        textcoords="offset points",
                        fontsize=9, fontweight="bold",
                        color="black",
                        bbox=dict(boxstyle="round,pad=0.3",
                                  facecolor="#fffec8", alpha=0.9,
                                  edgecolor="0.5"))

        ax.set_title(f"{dom}  /  winner-by-iou: {winner}",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("T_total mean (ms, log scale)", fontsize=11)
        ax.set_ylabel("iou_match", fontsize=11)
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3, which="both", zorder=0)
        ax.set_axisbelow(True)
        # Per-axis legend (small) about device + winner marker meaning.
        custom = [
            mlines.Line2D([], [], color="tab:red", lw=1.6, label="CPU"),
            mlines.Line2D([], [], color="tab:blue", lw=1.6, ls="--",
                          label="CUDA (RTX 5090)"),
            mlines.Line2D([], [], color="0.5", ls=":", lw=0.9,
                          label="random-switch reference"),
        ]
        ax.legend(handles=custom, loc="lower right", fontsize=9,
                  frameon=True)

    fig.suptitle(
        "Stage 3 — platform inversion (n→l pair only)\n"
        "Switching saves real latency on CPU; nearly free quality lift on saturated GPU",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[stage3_pareto] wrote {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build Stage 3 summary table + Pareto figures.",
    )
    p.add_argument("--audit-table", type=Path,
                   default=Path(
                       "analysis/figures/stage3/master_table_stage3_audit.csv"
                   ),
                   help="Preferred input: stage3_audit output with T_total + "
                        "random baseline + trigger F1.")
    p.add_argument("--master-table", type=Path,
                   default=Path("results/master_table.csv"),
                   help="Fallback input if --audit-table is missing.")
    p.add_argument("--out-dir", type=Path,
                   default=Path("analysis/figures/stage3"))
    p.add_argument("--no-plots", action="store_true",
                   help="Write summary CSV only.")
    return p.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    src: Optional[Path] = None
    if args.audit_table.exists():
        src = args.audit_table
        print(f"[stage3_pareto] using audit table: {src}")
    elif args.master_table.exists():
        src = args.master_table
        print(f"[stage3_pareto] WARNING: audit table not found; "
              f"falling back to {src} (selected-model latency, no T_total)")
    else:
        print(f"[stage3_pareto] no input table found", file=sys.stderr)
        return 2

    rows = read_stage3(src)
    if not rows:
        print("[stage3_pareto] no stage3_ rows in input table",
              file=sys.stderr)
        return 2
    is_audit = has_audit_columns(rows)

    # Long-format clean CSV (one row per run-policy, stage3 only).
    clean_path = args.out_dir / "master_table_stage3_clean.csv"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    skip = {"_key"}
    cols = [c for c in rows[0].keys() if c not in skip]
    with clean_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    print(f"[stage3_pareto] wrote {clean_path} ({len(rows)} rows)")

    # Pivoted summary.
    summary_path = args.out_dir / "stage3_summary.csv"
    write_summary_csv(rows, summary_path)
    n_runs = len({(r['_key'].family, r['_key'].fast, r['_key'].acc,
                   r['_key'].domain, r['_key'].device) for r in rows})
    print(f"[stage3_pareto] wrote {summary_path} ({n_runs} runs)")

    if args.no_plots:
        return 0

    x_col = "t_total_mean_ms" if is_audit else "lat_mean_ms"
    x_label = ("T_total mean (ms)  =  T_scene + T_ctrl + T_infer"
               if is_audit else "selected-model latency (ms)")
    suffix = " — honest T_total" if is_audit else " — selected-model latency"

    # GPU Pareto.
    _plot_pareto_grid(
        rows, device_filter="cuda",
        x_col=x_col, x_label=x_label,
        title=f"Stage 3 GPU Pareto — RTX 5090 saturated platform{suffix}",
        out_path=args.out_dir / "fig_gpu_pareto.png",
    )
    # CPU Pareto.
    _plot_pareto_grid(
        rows, device_filter="cpu",
        x_col=x_col, x_label=x_label,
        title=f"Stage 3 CPU Pareto — real latency-quality frontier{suffix}",
        out_path=args.out_dir / "fig_cpu_pareto.png",
    )
    # Platform inversion.
    _plot_platform_inversion(
        rows, out_path=args.out_dir / "fig_platform_inversion.png",
    )

    if is_audit:
        # Random-baseline gain figure.
        _plot_random_gain(rows,
                          out_path=args.out_dir / "fig_random_baseline.png")
        # Tail-latency figures.
        _plot_p95_tail(rows, device_filter="cpu",
                       out_path=args.out_dir / "fig_cpu_p95_tail.png")
        _plot_p95_tail(rows, device_filter="cuda",
                       out_path=args.out_dir / "fig_gpu_p95_tail.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
