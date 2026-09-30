"""experiments/build_master_table.py — aggregate sweep summaries.

Aggregate every sweep_summary.json under a results root into a single
``master_table.csv`` (one row per (run_id, policy)) for the analysis
layer to consume.

Why one wide table per (run_id, policy)
---------------------------------------
- Pareto plotting is one scatter point per (policy, video, model_pair,
  device): exactly the shape of a master_table row.
- Comparing platforms (CPU vs CUDA vs Jetson TRT) is a filter on the
  ``device`` column, no JSON re-traversal required.
- The thesis chapter and the GUI dashboard both read this CSV; one
  source of truth, no per-tool aggregation drift.

Schema
------
The flat CSV has these columns (missing values written as empty):
    run_id              from the summary JSON 'run_id'
    host                machine that ran the sweep
    device              cpu / cuda / tensorrt
    video               source video path
    fast_weights        fast model weights path
    accurate_weights    accurate model weights path
    imgsz               inference image size
    proxy_size          proxy resize length
    preserve_aspect     bool
    iou_threshold       IoU threshold used for matching
    conf_floor          confidence floor applied before matching
    n_frames            frames processed
    policy              policy name
    iou_match_rate      strict 1:1 rate
    det_coverage_rate   relaxed binary recall rate (recall == 1.0)
    det_recall_mean     relaxed continuous recall (mean over frames)
    count_agree_rate    cheap count-only secondary
    benefit_rate        locked benefit-positive rate
    s_choice_rate       fraction of frames the policy picked s
    lat_mean_ms         mean per-frame latency (chosen-model, ms)
    lat_p50_ms          p50
    lat_p95_ms          p95
    lat_p99_ms          p99
    fps_steady          1000 / lat_mean_ms

CLI
---
python -m experiments.build_master_table \
    --results-root results \
    --out results/master_table.csv

By default, recursively scans for any ``*summary.json`` containing a
``per_policy`` block (i.e., produced by run_sweep). ``run_reference``
summaries are skipped (they are pair-of-baselines, not a sweep row).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable, List, Optional


# ===========================================================================
# Output schema
# ===========================================================================
MASTER_FIELDS = [
    "run_id",
    "host",
    "device",
    "video",
    "fast_weights",
    "accurate_weights",
    "imgsz",
    "proxy_size",
    "preserve_aspect",
    "iou_threshold",
    "conf_floor",
    "n_frames",
    "policy",
    "iou_match_rate",
    "det_coverage_rate",
    "det_recall_mean",
    "count_agree_rate",
    "benefit_rate",
    "s_choice_rate",
    "lat_mean_ms",
    "lat_p50_ms",
    "lat_p95_ms",
    "lat_p99_ms",
    "fps_steady",
]


# ===========================================================================
# Discovery
# ===========================================================================
def find_sweep_summaries(root: Path) -> List[Path]:
    """Return all *summary.json files under root that look like sweep
    outputs (have a top-level 'per_policy' key).
    """
    out = []
    for p in root.rglob("*summary.json"):
        try:
            with p.open("r", encoding="utf-8") as f:
                doc = json.load(f)
            if isinstance(doc, dict) and "per_policy" in doc:
                out.append(p)
        except Exception:
            # corrupt or partial JSON — skip rather than abort the build
            continue
    return out


# ===========================================================================
# Row building
# ===========================================================================
def _row(summary: dict, policy: str, p_summary: dict) -> dict:
    cfg = summary.get("config", {})
    lat = p_summary.get("latency", {}) or {}
    return dict(
        run_id=summary.get("run_id", ""),
        host=summary.get("host", ""),
        device=cfg.get("device", ""),
        video=cfg.get("video", ""),
        fast_weights=cfg.get("fast_weights", ""),
        accurate_weights=cfg.get("accurate_weights", ""),
        imgsz=cfg.get("imgsz", ""),
        proxy_size=cfg.get("proxy_size", ""),
        preserve_aspect=cfg.get("preserve_aspect", ""),
        iou_threshold=cfg.get("iou_threshold", ""),
        conf_floor=cfg.get("conf_floor", ""),
        n_frames=p_summary.get("n_frames", ""),
        policy=policy,
        iou_match_rate=p_summary.get("iou_match_rate", ""),
        det_coverage_rate=p_summary.get("det_coverage_rate", ""),
        det_recall_mean=p_summary.get("det_recall_mean", ""),
        count_agree_rate=p_summary.get("count_agree_rate", ""),
        benefit_rate=p_summary.get("benefit_rate", ""),
        s_choice_rate=p_summary.get("s_choice_rate", ""),
        lat_mean_ms=lat.get("mean", ""),
        lat_p50_ms=lat.get("p50", ""),
        lat_p95_ms=lat.get("p95", ""),
        lat_p99_ms=lat.get("p99", ""),
        fps_steady=lat.get("fps", ""),
    )


def build_rows(summaries: Iterable[Path]) -> List[dict]:
    rows: List[dict] = []
    for path in summaries:
        try:
            with path.open("r", encoding="utf-8") as f:
                doc = json.load(f)
        except Exception as e:
            print(f"[build_master_table] WARN: cannot parse {path}: {e}",
                  file=sys.stderr)
            continue
        per_policy = doc.get("per_policy", {})
        if not per_policy:
            continue
        for pname, p_summary in per_policy.items():
            if not isinstance(p_summary, dict):
                continue
            if p_summary.get("n_frames", 0) == 0:
                continue
            rows.append(_row(doc, pname, p_summary))
    return rows


# ===========================================================================
# CLI
# ===========================================================================
def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aggregate sweep summary JSONs into one master_table.csv.",
    )
    p.add_argument("--results-root", type=Path,
                   default=Path(__file__).resolve().parent.parent / "results",
                   help="Directory to search recursively for summary JSONs.")
    p.add_argument("--out", type=Path,
                   default=Path(__file__).resolve().parent.parent
                   / "results" / "master_table.csv",
                   help="Output CSV path.")
    return p.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    if not args.results_root.exists():
        print(f"[build_master_table] ERROR: results root does not exist: "
              f"{args.results_root}", file=sys.stderr)
        return 2

    summaries = find_sweep_summaries(args.results_root)
    print(f"[build_master_table] found {len(summaries)} sweep summary "
          f"file(s) under {args.results_root}")
    if not summaries:
        print("[build_master_table] nothing to aggregate.")
        return 0

    rows = build_rows(summaries)
    print(f"[build_master_table] aggregated {len(rows)} (run, policy) rows")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MASTER_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[build_master_table] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
