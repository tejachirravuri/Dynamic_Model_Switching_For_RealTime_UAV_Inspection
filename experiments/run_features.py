"""experiments/run_features.py — per-frame scene proxies.

Compute scene proxies per frame for one video, with NO inference and
NO model switching. Isolates proxy computation from everything else.

Why this comes after run_reference and before run_sweep
-------------------------------------------------------
- run_reference verified inference + matching + metrics + I/O
- run_features verifies the proxy layer in isolation
- run_sweep then composes the two: proxies feed the policy, the
  policy chooses, and inference runs on whichever model the policy
  selects. Debugging that big composition is much easier when its
  pieces are individually known good.

Usage
-----
python -m experiments.run_features \
    --video <path> \
    --proxy-size 160 \
    --max-frames 100 \
    --features L H color_entropy tenengrad edge_density local_contrast \
    --run-id features_glass_p160

Output
------
results/<stage>/<run_id>/
    per_frame.csv   one row per frame: frame_idx, time_ms, <each feature>
    summary.json    config snapshot + per-feature mean/std/min/max
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Optional

from dms.proxies import ProxyConfig, compute_proxies


# ===========================================================================
# CLI
# ===========================================================================
DEFAULT_FEATURES = (
    "L", "H", "color_entropy",
    "tenengrad", "edge_density", "local_contrast",
)


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute per-frame scene proxies on one video. "
                    "No model inference.",
    )
    p.add_argument("--video", type=Path, required=True,
                   help="Path to input video.")
    p.add_argument("--proxy-size", type=int, default=160,
                   help="Resize length for proxy computation. "
                        "0 = native resolution (no resize).")
    p.add_argument("--preserve-aspect", action="store_true",
                   help="Letterbox (aspect-preserving) instead of warp-resize.")
    p.add_argument("--features", type=str, nargs="+",
                   default=list(DEFAULT_FEATURES),
                   help=f"Which features to compute. "
                        f"Default: {' '.join(DEFAULT_FEATURES)}")
    p.add_argument("--max-frames", type=int, default=0,
                   help="Cap frames processed (0 = no cap).")
    p.add_argument("--results-root", type=Path,
                   default=Path(__file__).resolve().parent.parent
                   / "results" / "stage_features",
                   help="Root directory for run outputs.")
    p.add_argument("--run-id", type=str, default=None,
                   help="Run subdirectory name. Default: video stem + ts.")
    return p.parse_args(argv)


# ===========================================================================
# Per-feature summary
# ===========================================================================
def _summarise_series(values: list) -> dict:
    if not values:
        return dict(n=0, mean=0.0, std=0.0, min=0.0, max=0.0)
    return dict(
        n=len(values),
        mean=float(statistics.fmean(values)),
        std=float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        min=float(min(values)),
        max=float(max(values)),
    )


# ===========================================================================
# Main
# ===========================================================================
def main(argv: Optional[list] = None) -> int:
    import cv2  # local import

    args = parse_args(argv)

    if args.run_id:
        run_id = args.run_id
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        run_id = f"{args.video.stem}_p{args.proxy_size}_{ts}"
    out_dir = args.results_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "per_frame.csv"
    summary_path = out_dir / "summary.json"

    print(f"[run_features] writing to {out_dir}")
    print(f"[run_features] proxy_size={args.proxy_size} "
          f"preserve_aspect={args.preserve_aspect} "
          f"features={args.features}")

    cfg = ProxyConfig(
        size=int(args.proxy_size),
        preserve_aspect=bool(args.preserve_aspect),
        features=tuple(args.features),
    )

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"[run_features] ERROR: cannot open video {args.video}",
              file=sys.stderr)
        return 2

    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or -1
    fps_video = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[run_features] video: {n_total} frames, {fps_video:.2f} fps, "
          f"{width}x{height}")

    cap_n = args.max_frames if args.max_frames > 0 else n_total

    csv_fields = ["frame_idx", "time_ms"] + list(args.features)
    csv_file = csv_path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    writer.writeheader()
    csv_file.flush()

    series = {f: [] for f in args.features}
    times_ms = []

    frame_idx = 0
    t_run0 = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if cap_n > 0 and frame_idx >= cap_n:
                break

            out = compute_proxies(frame, cfg)
            row = dict(frame_idx=frame_idx, time_ms=float(out["time_ms"]))
            for f in args.features:
                v = float(out[f])
                row[f] = v
                series[f].append(v)
            times_ms.append(float(out["time_ms"]))
            writer.writerow(row)
            csv_file.flush()

            if (frame_idx + 1) % 50 == 0 or frame_idx == 0:
                elapsed = time.perf_counter() - t_run0
                preview = "  ".join(
                    f"{f}={row[f]:.3f}" for f in args.features[:3]
                )
                print(f"  frame {frame_idx + 1:>5d}  "
                      f"t={out['time_ms']:6.2f}ms  {preview}  "
                      f"[wall {elapsed:.1f}s]")

            frame_idx += 1
    finally:
        csv_file.close()
        cap.release()

    t_run = time.perf_counter() - t_run0
    print(f"[run_features] processed {frame_idx} frames in {t_run:.1f}s")

    # ----- Summary --------------------------------------------------------
    summary = dict(
        run_id=run_id,
        wall_seconds=float(t_run),
        n_frames=frame_idx,
        host=platform.node(),
        platform=platform.platform(),
        config=dict(
            video=str(args.video),
            proxy_size=int(args.proxy_size),
            preserve_aspect=bool(args.preserve_aspect),
            features=list(args.features),
            max_frames=int(args.max_frames),
            video_meta=dict(
                total_frames=n_total, fps=fps_video,
                width=width, height=height,
            ),
        ),
        time_ms=_summarise_series(times_ms),
        feature_stats={f: _summarise_series(series[f]) for f in args.features},
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n[run_features] summary:")
    if frame_idx > 0:
        print(f"  proxy time mean={summary['time_ms']['mean']:.2f}ms "
              f"p95={summary['time_ms'].get('max', 0):.2f}ms")
        for f in args.features:
            s = summary["feature_stats"][f]
            print(f"  {f:>16s}: mean={s['mean']:.4f}  std={s['std']:.4f}  "
                  f"[{s['min']:.4f}, {s['max']:.4f}]")
    print(f"  per-frame CSV: {csv_path}")
    print(f"  summary JSON:  {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
