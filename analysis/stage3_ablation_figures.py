"""analysis/stage3_ablation_figures.py — frame-skip + proxy-resolution figures.

Reads the audit table (which now includes ablation rows tagged with
``ablation_kind`` ∈ {'skip', 'proxy', ''} and ``ablation_value``) and
emits two ablation figures plus a summary CSV per ablation.

Output (under --out-dir, default ``analysis/figures/stage3``)
-------------------------------------------------------------
* ``fig_ablation_frame_skip.png``    K=1..10 video-subsampling frontier
                                     per (policy, pair). Two panels
                                     (one per pair config).
* ``fig_ablation_proxy_size.png``    proxy_size in {32, 80, 160, 320}
                                     vs T_total + iou_match per policy,
                                     two panels.
* ``ablation_frame_skip_summary.csv`` long table for thesis tables.
* ``ablation_proxy_size_summary.csv``
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional


POLICIES_TO_PLOT = ("n_only", "s_only", "entropy_only", "combined",
                    "combined_hyst", "conf_ema", "multi_proxy")
POLICY_COLORS = {
    "n_only":         "#888888",
    "s_only":         "#222222",
    "entropy_only":   "#2ca02c",
    "combined":       "#1f77b4",
    "combined_hyst":  "#0033cc",
    "conf_ema":       "#d62728",
    "multi_proxy":    "#9467bd",
}


def _f(x) -> float:
    if x is None or x == "":
        return float("nan")
    try:
        return float(x)
    except ValueError:
        return float("nan")


def read_audit(audit_csv: Path) -> List[Dict]:
    rows = []
    with audit_csv.open("r", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def filter_ablation(rows: List[Dict], kind: str) -> List[Dict]:
    """Keep ablation_kind == kind plus K=1 / S=160 baseline (from stage3)."""
    out: List[Dict] = []
    for r in rows:
        if r.get("ablation_kind") == kind:
            out.append(r)
    return out


def write_summary_csv(rows: List[Dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["run_id", "ablation_kind", "ablation_value", "policy",
            "iou_match_rate", "det_coverage_rate", "s_choice_rate",
            "informed_gain_over_random", "trigger_f1",
            "t_scene_mean_ms", "t_total_mean_ms", "t_total_p95_ms"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def plot_skip_ablation(rows: List[Dict], out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if not rows:
        print("[stage3_ablation_figures] no skip rows; skipping plot",
              file=sys.stderr)
        return

    # Group by (run pair signature, policy) -> list of (K, T_total, iou).
    by_panel: Dict = defaultdict(lambda: defaultdict(list))
    for r in rows:
        pol = r["policy"]
        if pol not in POLICIES_TO_PLOT:
            continue
        K = int(_f(r["ablation_value"])) if r["ablation_value"] != "" else 1
        # Use the pair signature from run_id (after the ablation prefix).
        run_id = r["run_id"]
        # ablation_skip_K{K}_<rest>
        rest = run_id.split(f"_K{K}_", 1)[-1] if f"_K{K}_" in run_id else run_id
        panel_key = rest  # e.g. "yolov8_n_l_glass_cpu"
        by_panel[panel_key][pol].append(
            (K, _f(r["t_total_mean_ms"]), _f(r["iou_match_rate"])))

    panel_keys = sorted(by_panel.keys())
    n_panels = len(panel_keys)
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5),
                             sharey=True)
    if n_panels == 1:
        axes = [axes]

    for ax, pkey in zip(axes, panel_keys):
        for pol, points in by_panel[pkey].items():
            points.sort()
            xs = [p[0] for p in points]
            ys = [p[2] for p in points]  # iou_match
            ts = [p[1] for p in points]  # T_total
            color = POLICY_COLORS.get(pol, "0.4")
            ax.plot(xs, ys, "-o", color=color, label=pol, lw=1.4,
                    markersize=7, markeredgecolor="black",
                    markeredgewidth=0.6)
        ax.set_title(f"video-subsampling K  /  {pkey}")
        ax.set_xlabel("frame-skip K (every K-th raw frame processed)")
        ax.set_ylabel("iou_match")
        ax.set_xscale("log")
        ax.set_xticks([1, 3, 5, 10])
        ax.set_xticklabels(["1", "3", "5", "10"])
        ax.set_ylim(-0.02, 1.04)
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(loc="lower left", fontsize=8)

    fig.suptitle("Stage 3 ablation — video subsampling frame-skip K "
                 "(NOT proxy-stride)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[stage3_ablation_figures] wrote {out_path}")


def plot_proxy_size_ablation(rows: List[Dict], out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if not rows:
        print("[stage3_ablation_figures] no proxy rows; skipping plot",
              file=sys.stderr)
        return

    by_panel: Dict = defaultdict(lambda: defaultdict(list))
    for r in rows:
        pol = r["policy"]
        if pol not in POLICIES_TO_PLOT:
            continue
        S = int(_f(r["ablation_value"])) if r["ablation_value"] != "" else 160
        run_id = r["run_id"]
        rest = run_id.split(f"_S{S}_", 1)[-1] if f"_S{S}_" in run_id else run_id
        panel_key = rest
        by_panel[panel_key][pol].append(
            (S, _f(r["t_scene_mean_ms"]),
             _f(r["t_total_mean_ms"]), _f(r["iou_match_rate"])))

    panel_keys = sorted(by_panel.keys())
    n_panels = len(panel_keys)
    fig, axes = plt.subplots(2, n_panels, figsize=(7 * n_panels, 9),
                             sharex=True)
    if n_panels == 1:
        axes = axes.reshape(2, 1)

    for col, pkey in enumerate(panel_keys):
        ax_top = axes[0][col]
        ax_bot = axes[1][col]
        for pol, points in by_panel[pkey].items():
            points.sort()
            xs = [p[0] for p in points]
            scenes = [p[1] for p in points]
            iou = [p[3] for p in points]
            color = POLICY_COLORS.get(pol, "0.4")
            ax_top.plot(xs, iou, "-o", color=color, label=pol, lw=1.3,
                        markersize=7, markeredgecolor="black",
                        markeredgewidth=0.6)
            ax_bot.plot(xs, scenes, "-o", color=color, lw=1.0,
                        markersize=6, markeredgecolor="black",
                        markeredgewidth=0.5)
        ax_top.set_title(f"proxy-size sweep  /  {pkey}")
        ax_top.set_ylabel("iou_match")
        ax_top.set_ylim(-0.02, 1.04)
        ax_top.grid(True, alpha=0.3, which="both")
        ax_top.legend(loc="lower right", fontsize=8)
        ax_bot.set_xlabel("proxy_size (pixels, square)")
        ax_bot.set_ylabel("T_scene mean (ms)")
        ax_bot.grid(True, alpha=0.3, which="both")
        ax_bot.set_xticks([32, 80, 160, 320])
        ax_bot.set_xticklabels(["32", "80", "160", "320"])

    fig.suptitle("Stage 3 ablation — proxy-resolution sweep "
                 "(T_scene cost vs trigger fidelity)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[stage3_ablation_figures] wrote {out_path}")


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Frame-skip + proxy-resolution ablation figures.",
    )
    p.add_argument("--audit-csv", type=Path,
                   default=Path("analysis/figures/stage3/master_table_stage3_audit.csv"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("analysis/figures/stage3"))
    return p.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    if not args.audit_csv.exists():
        print(f"[stage3_ablation_figures] missing {args.audit_csv}",
              file=sys.stderr)
        return 2
    rows = read_audit(args.audit_csv)
    print(f"[stage3_ablation_figures] read {len(rows)} audit rows")

    skip_rows = filter_ablation(rows, "skip")
    proxy_rows = filter_ablation(rows, "proxy")

    print(f"  ablation_skip rows:  {len(skip_rows)}")
    print(f"  ablation_proxy rows: {len(proxy_rows)}")

    write_summary_csv(skip_rows,
                      args.out_dir / "ablation_frame_skip_summary.csv")
    write_summary_csv(proxy_rows,
                      args.out_dir / "ablation_proxy_size_summary.csv")

    plot_skip_ablation(skip_rows,
                       args.out_dir / "fig_ablation_frame_skip.png")
    plot_proxy_size_ablation(proxy_rows,
                             args.out_dir / "fig_ablation_proxy_size.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
