"""experiments/run_reference.py — inference + timing reference.

Run **n_only and s_only only** on a single video. No policies, no
proxies, no switching logic. The point is to isolate the inference +
timing + output format from everything else.

Why this comes first
--------------------
The rule of thumb: "One video, one model pair, CPU first, 100-frame
limit first." If this runs to a clean per-frame CSV + summary JSON
with believable numbers, then the inference / matching / metrics /
file-output paths are all known good — and the policy sweep just
plugs in on top.

What it writes
--------------
results/<stage>/<run_id>/
    per_frame.csv      one row per frame with both n_only and s_only
                       traces side-by-side (one frame, both models)
    summary.json       aggregated run metrics + run config snapshot

Each per-frame row carries:
    frame_idx               int
    n_time_ms, s_time_ms    float           per-model wall-clock
    n_count,   s_count      int             post-conf-floor det counts
    iou_match               bool            strict (n vs s)
    det_coverage            bool            relaxed binary recall
    det_recall              float in [0,1]  relaxed continuous recall
    count_agree             bool            cheap count-only secondary
    benefit_positive        bool            locked benefit-positive label
                                            (n=fast, s=accurate)

CLI
---
python -m experiments.run_reference \
    --video <path> \
    --fast-weights <pt> \
    --accurate-weights <pt> \
    --device cpu \
    --max-frames 100 \
    --conf-floor 0.25 \
    --iou-threshold 0.5 \
    --run-id reference_glass_cpu100

Notes
-----
- Lazy-imports cv2 and ultralytics so the module can be inspected on
  hosts without those packages.
- Writes incrementally (CSV row-by-row) so a Ctrl-C leaves a partial
  but well-formed CSV that the analysis layer can still read.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from dms.inference import Detections, UltralyticsYOLOBackend, time_inference
from dms.matching import filter_by_score
from dms.metrics import (
    benefit_positive_frame,
    count_agree_frame,
    det_coverage_frame,
    det_recall_frame,
    iou_match_frame,
    summarise_run,
)


# ===========================================================================
# Per-frame record (matches CSV schema exactly)
# ===========================================================================
@dataclass
class FrameRow:
    frame_idx: int
    n_time_ms: float
    s_time_ms: float
    n_count: int
    s_count: int
    iou_match: int          # 0/1 for clean CSV ints
    det_coverage: int       # 0/1
    det_recall: float       # in [0, 1]
    count_agree: int        # 0/1
    benefit_positive: int   # 0/1


CSV_FIELDS = list(FrameRow.__dataclass_fields__.keys())


# ===========================================================================
# CLI
# ===========================================================================
def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run n_only + s_only reference traces on one video.",
    )
    p.add_argument("--video", type=Path, required=True,
                   help="Path to input video (e.g., .mp4).")
    p.add_argument("--fast-weights", type=Path, required=True,
                   help="Path to fast-model weights (.pt / .onnx / .engine).")
    p.add_argument("--accurate-weights", type=Path, required=True,
                   help="Path to accurate-model weights.")
    p.add_argument("--device", type=str, default="cpu",
                   choices=("cpu", "cuda", "0", "1"),
                   help="Inference device.")
    p.add_argument("--imgsz", type=int, default=640,
                   help="Inference image size (square).")
    p.add_argument("--max-frames", type=int, default=100,
                   help="Cap frames processed (0 = no cap).")
    p.add_argument("--start-frame", type=int, default=0,
                   help="Seek to this frame index after warmup, before "
                        "the timed loop.")
    p.add_argument("--warmup-frames", type=int, default=3,
                   help="Per-backend warmup iterations on a discarded "
                        "frame before timing. Default 3 absorbs CUDA "
                        "kernel-compile cost.")
    p.add_argument("--conf-floor", type=float, default=0.25,
                   help="Confidence floor applied before matching.")
    p.add_argument("--iou-threshold", type=float, default=0.5,
                   help="IoU threshold for matching n vs s detections.")
    p.add_argument("--results-root", type=Path,
                   default=Path(__file__).resolve().parent.parent
                   / "results" / "stage_reference",
                   help="Root directory for run outputs.")
    p.add_argument("--run-id", type=str, default=None,
                   help="Run subdirectory name. Default: video stem + ts.")
    p.add_argument("--use-classes-in-match", action="store_true",
                   help="If set, require class agreement when matching "
                        "n vs s detections (default: off, class-agnostic).")
    return p.parse_args(argv)


# ===========================================================================
# Main
# ===========================================================================
def main(argv: Optional[list] = None) -> int:
    import cv2  # local import: keeps module importable without OpenCV

    args = parse_args(argv)

    # ----- Resolve output dir ----------------------------------------------
    if args.run_id:
        run_id = args.run_id
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        run_id = f"{args.video.stem}_{args.device}_{ts}"
    out_dir = args.results_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "per_frame.csv"
    summary_path = out_dir / "summary.json"

    print(f"[run_reference] writing to {out_dir}")

    # ----- Load both backends ----------------------------------------------
    print(f"[run_reference] loading fast weights:     {args.fast_weights}")
    fast_backend = UltralyticsYOLOBackend(
        weights_path=str(args.fast_weights),
        device=args.device, imgsz=args.imgsz, conf=0.0, iou_nms=0.7,
    )
    print(f"[run_reference] loading accurate weights: {args.accurate_weights}")
    acc_backend = UltralyticsYOLOBackend(
        weights_path=str(args.accurate_weights),
        device=args.device, imgsz=args.imgsz, conf=0.0, iou_nms=0.7,
    )

    # ----- Open video -------------------------------------------------------
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"[run_reference] ERROR: cannot open video {args.video}",
              file=sys.stderr)
        return 2

    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or -1
    fps_video = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[run_reference] video: {n_total} frames, {fps_video:.2f} fps, "
          f"{width}x{height}")

    cap_n = args.max_frames if args.max_frames > 0 else n_total
    if cap_n > 0 and cap_n != n_total:
        print(f"[run_reference] frame cap: {cap_n}")

    # ----- GPU warmup ------------------------------------------------------
    # Absorb cuDNN kernel-compile cost on first CUDA call so the timed
    # loop sees steady-state latency from frame 0.
    if args.warmup_frames > 0:
        ok, warmup_frame = cap.read()
        if ok and warmup_frame is not None:
            print(f"[run_reference] warming up backends "
                  f"({args.warmup_frames} iters each)...")
            for _ in range(args.warmup_frames):
                _ = fast_backend.predict(warmup_frame)
                _ = acc_backend.predict(warmup_frame)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if args.start_frame > 0:
        print(f"[run_reference] seeking to frame {args.start_frame}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.start_frame))

    # ----- Per-frame loop ---------------------------------------------------
    n_only_records = []
    s_only_records = []
    rows: list[FrameRow] = []

    csv_file = csv_path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    writer.writeheader()
    csv_file.flush()

    frame_idx = 0
    t_run0 = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if cap_n > 0 and frame_idx >= cap_n:
                break

            # Run both models on the SAME frame (back-to-back, not parallel).
            n_dets, n_ms = time_inference(fast_backend, frame)
            s_dets, s_ms = time_inference(acc_backend, frame)

            # Apply confidence floor before matching.
            n_dets_f = n_dets.filter_score(args.conf_floor)
            s_dets_f = s_dets.filter_score(args.conf_floor)

            kw_match = {}
            if args.use_classes_in_match:
                kw_match["policy_classes"] = n_dets_f.classes
                kw_match["ref_classes"] = s_dets_f.classes

            iou_m = iou_match_frame(
                n_dets_f.boxes, s_dets_f.boxes,
                iou_threshold=args.iou_threshold,
                **kw_match,
            )
            det_cov = det_coverage_frame(
                n_dets_f.boxes, s_dets_f.boxes,
                iou_threshold=args.iou_threshold,
                **kw_match,
            )
            det_rec = det_recall_frame(
                n_dets_f.boxes, s_dets_f.boxes,
                iou_threshold=args.iou_threshold,
                **kw_match,
            )
            cnt_a = count_agree_frame(n_dets_f.boxes, s_dets_f.boxes)

            kw_benefit = {}
            if args.use_classes_in_match:
                kw_benefit["fast_classes"] = n_dets_f.classes
                kw_benefit["accurate_classes"] = s_dets_f.classes
            ben = benefit_positive_frame(
                n_dets_f.boxes, s_dets_f.boxes,
                iou_threshold=args.iou_threshold,
                **kw_benefit,
            )

            row = FrameRow(
                frame_idx=frame_idx,
                n_time_ms=float(n_ms),
                s_time_ms=float(s_ms),
                n_count=len(n_dets_f),
                s_count=len(s_dets_f),
                iou_match=int(bool(iou_m)),
                det_coverage=int(bool(det_cov)),
                det_recall=float(det_rec),
                count_agree=int(bool(cnt_a)),
                benefit_positive=int(bool(ben)),
            )
            rows.append(row)
            writer.writerow(asdict(row))
            csv_file.flush()  # so a Ctrl-C leaves a clean partial CSV

            # Build the per-frame records for run summarisation. The
            # n_only trace pretends every frame chose 'n', so its
            # iou_match / benefit fields equal the n-vs-s comparison.
            n_only_records.append(dict(
                iou_match=row.iou_match,
                det_coverage=row.det_coverage,
                det_recall=row.det_recall,
                count_agree=row.count_agree,
                benefit=row.benefit_positive,
                choice="n",
                time_ms=row.n_time_ms,
            ))
            # The s_only trace is the reference, so vs itself it's a
            # perfect match by construction; we still log it for
            # latency / fps.
            s_only_records.append(dict(
                iou_match=1, det_coverage=1, det_recall=1.0, count_agree=1,
                benefit=0, choice="s", time_ms=row.s_time_ms,
            ))

            if (frame_idx + 1) % 25 == 0 or frame_idx == 0:
                elapsed = time.perf_counter() - t_run0
                print(f"  frame {frame_idx + 1:>4d}  "
                      f"n={n_ms:6.1f}ms  s={s_ms:6.1f}ms  "
                      f"|n|={len(n_dets_f)}  |s|={len(s_dets_f)}  "
                      f"iou_match={int(iou_m)}  "
                      f"recall={det_rec:.2f}  benefit={int(ben)}  "
                      f"[wall {elapsed:.1f}s]")

            frame_idx += 1
    finally:
        csv_file.close()
        cap.release()

    t_run = time.perf_counter() - t_run0
    print(f"[run_reference] processed {frame_idx} frames in {t_run:.1f}s")

    # ----- Aggregate + write summary ---------------------------------------
    n_summary = summarise_run(n_only_records)
    s_summary = summarise_run(s_only_records)

    summary = dict(
        run_id=run_id,
        wall_seconds=float(t_run),
        n_frames=frame_idx,
        host=platform.node(),
        platform=platform.platform(),
        config=dict(
            video=str(args.video),
            fast_weights=str(args.fast_weights),
            accurate_weights=str(args.accurate_weights),
            device=args.device,
            imgsz=args.imgsz,
            max_frames=args.max_frames,
            start_frame=int(args.start_frame),
            warmup_frames=int(args.warmup_frames),
            conf_floor=args.conf_floor,
            iou_threshold=args.iou_threshold,
            use_classes_in_match=bool(args.use_classes_in_match),
            video_meta=dict(
                total_frames=n_total, fps=fps_video,
                width=width, height=height,
            ),
        ),
        n_only=n_summary,
        s_only=s_summary,
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # ----- Console report --------------------------------------------------
    print("\n[run_reference] summary:")
    if frame_idx > 0:
        nm = n_summary["latency"]["mean"]
        sm = s_summary["latency"]["mean"]
        print(f"  n_only:  mean={nm:.2f} ms  fps={n_summary['latency']['fps']:.2f}  "
              f"p95={n_summary['latency']['p95']:.2f}")
        print(f"  s_only:  mean={sm:.2f} ms  fps={s_summary['latency']['fps']:.2f}  "
              f"p95={s_summary['latency']['p95']:.2f}")
        print(f"  speedup (n_only / s_only):  {sm / max(nm, 1e-9):.2f}x")
        print(f"  iou_match rate (n vs s):    {n_summary['iou_match_rate']:.3f}")
        print(f"  det_coverage rate (n vs s): {n_summary['det_coverage_rate']:.3f}")
        print(f"  det_recall mean (n vs s):   {n_summary['det_recall_mean']:.3f}")
        print(f"  benefit-positive rate:      {n_summary['benefit_rate']:.3f}")
    print(f"  per-frame CSV: {csv_path}")
    print(f"  summary JSON:  {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
