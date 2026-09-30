"""analysis/training_stats.py — Stage 1 per-model training table.

Reads each Path-B trained model's ``results.csv`` (Ultralytics output)
and extracts the best-epoch row per model. Produces:

  * ``stage1_training_stats.csv``   one row per model with:
      family, size, domain, epochs_total, best_epoch,
      best_precision, best_recall, best_f1,
      best_mAP50, best_mAP50_95
  * ``fig_training_curves.png``     PR + F1 + mAP curves per model
                                    (4-panel: family x size grid).

Failed runs (``.FAILED_*``, ``.OOM_*``) are excluded from analysis but
still listed at the end of the CSV for transparency.

Usage
-----
    python -m analysis.training_stats \
        --runs-dir results/remote_pull/stage3_ablation/extracted_training \
        --out-dir analysis/figures/stage3
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _f(x) -> float:
    if x is None or x == "":
        return float("nan")
    try:
        return float(x)
    except ValueError:
        return float("nan")


def parse_dirname(name: str) -> Tuple[Optional[Tuple[str, str, str]], bool]:
    """Return ((family, size, domain), is_failed) or (None, ?) for unknown."""
    is_failed = (".FAILED" in name or ".OOM" in name)
    base = name.split(".")[0]  # strip .FAILED_* / .OOM_*
    m = re.match(r"^(yolov8|yolo26)_(n|s|l)_(glass|porcelain)$", base)
    if not m:
        return None, is_failed
    return tuple(m.groups()), is_failed  # type: ignore


def best_epoch_row(results_csv: Path) -> Optional[Dict]:
    """Return the row with highest mAP@[.5:.95]; None if file empty."""
    rows = []
    with results_csv.open("r", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    if not rows:
        return None
    best = max(rows, key=lambda r: _f(r.get("metrics/mAP50-95(B)", "0")))
    return best


def f1_from(row: Dict) -> float:
    p = _f(row.get("metrics/precision(B)", 0.0))
    r = _f(row.get("metrics/recall(B)", 0.0))
    if (p + r) <= 0 or _f("nan") in (p, r):
        return 0.0
    return 2.0 * p * r / (p + r)


def write_summary(rows: List[Dict], path: Path) -> None:
    if not rows:
        return
    cols = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def plot_training_curves(rows: List[Tuple[str, Path]], out_path: Path) -> None:
    """For each clean model, plot mAP50, mAP50-95 vs epoch.

    rows: list of (model_name, results_csv_path).
    Layout: 2x3 grid (family x size).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey=True)
    # Map (family, size) -> ax row, col.
    family_row = {"yolov8": 0, "yolo26": 1}
    size_col = {"n": 0, "s": 1, "l": 2}
    domain_color = {"glass": "tab:blue", "porcelain": "tab:orange"}
    for name, path in rows:
        parsed, failed = parse_dirname(name)
        if parsed is None or failed:
            continue
        fam, sz, dom = parsed
        ax = axes[family_row[fam]][size_col[sz]]
        epochs = []
        map5 = []
        map5_95 = []
        with path.open("r", newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                epochs.append(int(_f(r["epoch"])))
                map5.append(_f(r["metrics/mAP50(B)"]))
                map5_95.append(_f(r["metrics/mAP50-95(B)"]))
        col = domain_color[dom]
        ax.plot(epochs, map5, "-", color=col, lw=1.4,
                label=f"{dom} mAP@0.5")
        ax.plot(epochs, map5_95, "--", color=col, lw=1.0, alpha=0.7,
                label=f"{dom} mAP@[.5:.95]")
        ax.set_title(f"{fam} / {sz}")
        ax.set_xlabel("epoch")
        ax.set_ylabel("mAP")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("Stage 1 — per-model training curves "
                 "(YOLOv8 + YOLO26 trained on glass + porcelain)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[training_stats] wrote {out_path}")


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract Stage 1 per-model training stats.",
    )
    p.add_argument("--runs-dir", type=Path, required=True,
                   help="Directory with one subdir per model "
                        "(e.g., extracted training_runs_results.tgz).")
    p.add_argument("--out-dir", type=Path,
                   default=Path("analysis/figures/stage3"))
    return p.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    if not args.runs_dir.exists():
        print(f"[training_stats] missing {args.runs_dir}", file=sys.stderr)
        return 2

    out_rows: List[Dict] = []
    csv_paths: List[Tuple[str, Path]] = []
    for d in sorted(args.runs_dir.iterdir()):
        if not d.is_dir():
            continue
        rcsv = d / "results.csv"
        if not rcsv.exists():
            continue
        csv_paths.append((d.name, rcsv))

        parsed, failed = parse_dirname(d.name)
        best = best_epoch_row(rcsv)
        if best is None:
            continue
        n_epochs = 0
        with rcsv.open("r", newline="", encoding="utf-8") as f:
            n_epochs = sum(1 for _ in csv.DictReader(f))

        row = dict(
            run_dir=d.name,
            failed_run=int(failed),
            family=parsed[0] if parsed else "",
            size=parsed[1] if parsed else "",
            domain=parsed[2] if parsed else "",
            epochs_total=n_epochs,
            best_epoch=int(_f(best.get("epoch", "0"))),
            best_precision=round(_f(best.get("metrics/precision(B)", 0)), 4),
            best_recall=round(_f(best.get("metrics/recall(B)", 0)), 4),
            best_f1=round(f1_from(best), 4),
            best_mAP50=round(_f(best.get("metrics/mAP50(B)", 0)), 4),
            best_mAP50_95=round(_f(best.get("metrics/mAP50-95(B)", 0)), 4),
        )
        out_rows.append(row)

    # Sort: clean runs first, by family/size/domain; failed last.
    out_rows.sort(key=lambda r: (r["failed_run"], r["family"], r["size"],
                                 r["domain"]))

    summary_path = args.out_dir / "stage1_training_stats.csv"
    write_summary(out_rows, summary_path)
    print(f"[training_stats] wrote {summary_path} ({len(out_rows)} models)")

    # Print clean models as a thesis-ready table.
    print()
    print(f"{'family':7s} {'size':4s} {'domain':10s}  "
          f"{'epochs':>6s} {'best_ep':>7s}  "
          f"{'P':>6s} {'R':>6s} {'F1':>6s}  "
          f"{'mAP@0.5':>8s} {'mAP@.5:.95':>10s}")
    for r in out_rows:
        marker = " (FAILED)" if r["failed_run"] else ""
        print(f"{r['family']:7s} {r['size']:4s} {r['domain']:10s}  "
              f"{r['epochs_total']:>6d} {r['best_epoch']:>7d}  "
              f"{r['best_precision']:>6.3f} "
              f"{r['best_recall']:>6.3f} "
              f"{r['best_f1']:>6.3f}  "
              f"{r['best_mAP50']:>8.4f} "
              f"{r['best_mAP50_95']:>10.4f}{marker}")

    plot_training_curves(csv_paths,
                         args.out_dir / "fig_training_curves.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
