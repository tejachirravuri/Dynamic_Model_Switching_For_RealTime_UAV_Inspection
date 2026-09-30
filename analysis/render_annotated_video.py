"""analysis/render_annotated_video.py — annotated demo video.

Per pathB_plan.md Stage 6 spec:
  * Boxes 4-5 px thick. GREEN = fast (n) active. RED = accurate (s) active.
  * Frame border: 12 px solid ring in active-model color.
  * Transition flash: 150 ms yellow fade on border at switch frames
    (= ~4 frames at 30 fps).
  * HUD top-left: policy | active_model | T_infer ms | n/s confidence.
  * Encode: H.264 / mp4v via cv2.VideoWriter (no ffmpeg dep).
  * Same resolution as input (no upscaling).

Runs locally on CPU. For each frame:
  1. compute_proxies on a 160-px proxy
  2. instantiate the chosen policy class with default PolicyConfig
     (override conf_ema c_high via --conf-ema-c-high)
  3. policy.decide(features) -> 'n' or 's'
  4. run only the chosen model and time it (mirrors deployment)
  5. render boxes + border + HUD on the original frame
  6. write to MP4

Input
-----
  --video         path to video file
  --fast-weights  path to fast .pt
  --accurate-weights path to accurate .pt
  --policy        one of {n_only, s_only, entropy_only, combined,
                          combined_hyst, conf_ema, multi_proxy}
  --out           output MP4 path
  --max-frames    cap (0 = full video)
  --conf-ema-c-high  threshold override (only meaningful for conf_ema)

Usage
-----
    python -m analysis.render_annotated_video \\
        --video    G:/.../20190916-722.mp4 \\
        --fast-weights    results/.../yolov8_n_glass/weights/best.pt \\
        --accurate-weights results/.../yolov8_s_glass/weights/best.pt \\
        --policy   conf_ema \\
        --out      analysis/figures/stage3/demo/glass_conf_ema.mp4 \\
        --max-frames 600
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

from dms.policies import FrameFeatures, PolicyConfig, make_policy
from dms.proxies import ProxyConfig, compute_proxies


# Locked color convention (BGR for cv2):
COLOR_FAST_BGR     = (30, 180, 30)    # green
COLOR_ACCURATE_BGR = (30, 30, 220)    # red
COLOR_FLASH_BGR    = (40, 230, 240)   # yellow (BGR for cv2)
COLOR_HUD_BG_BGR   = (35, 35, 35)
COLOR_HUD_FG_BGR   = (245, 245, 245)


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render an annotated demo MP4.")
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--fast-weights", type=Path, required=True)
    p.add_argument("--accurate-weights", type=Path, required=True)
    p.add_argument("--policy", type=str, default="conf_ema",
                   choices=("n_only", "s_only", "entropy_only", "combined",
                            "combined_hyst", "conf_ema", "multi_proxy"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--proxy-size", type=int, default=160)
    p.add_argument("--conf-floor", type=float, default=0.25)
    p.add_argument("--conf-ema-c-high", type=float, default=None)
    p.add_argument("--conf-ema-c-low", type=float, default=None)
    p.add_argument("--codec", type=str, default="mp4v",
                   help="FourCC code for cv2.VideoWriter. mp4v = MPEG-4, "
                        "avc1 = H.264 (needs ffmpeg backend).")
    # ----- Optional presentation-grade overlay tweaks ----------------------
    # All three default to the locked v1 values so omitting them reproduces
    # the previous behaviour exactly.
    p.add_argument("--box-thickness", type=int, default=4,
                   help="Bounding-box stroke thickness in pixels. "
                        "Default 4 reproduces v1 demos.")
    p.add_argument("--hud-font-scale", type=float, default=0.9,
                   help="OpenCV font scale for the HUD. The HUD line "
                        "height auto-scales with this value. "
                        "Default 0.9 reproduces v1 demos.")
    p.add_argument("--switch-fade-ms", type=int, default=150,
                   help="Yellow border-flash duration on a switch event, "
                        "in milliseconds. Default 150 ms reproduces v1 "
                        "demos.")
    return p.parse_args(argv)


def draw_hud(frame, text_lines, color_bg, color_fg, font_scale=0.9):
    """Render the top-left HUD.

    ``font_scale`` defaults to 0.9 to preserve the v1 demo appearance.
    The text-row height and the per-row vertical baseline auto-scale
    with ``font_scale`` so the box stays proportional at any size.
    """
    import cv2
    pad = 12
    # Scale the row height and baseline offset linearly with font_scale
    # so that doubling the font roughly doubles the row spacing.
    line_h = int(round(36 * (font_scale / 0.9)))
    baseline_offset = int(round(22 * (font_scale / 0.9)))
    n = len(text_lines)
    box_w = 0
    for t in text_lines:
        size = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
        box_w = max(box_w, size[0])
    box_w += pad * 2
    box_h = line_h * n + pad
    cv2.rectangle(frame, (10, 10), (10 + box_w, 10 + box_h),
                  color_bg, thickness=-1)
    for i, t in enumerate(text_lines):
        y = 10 + pad + line_h * i + baseline_offset
        cv2.putText(frame, t, (10 + pad, y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_fg, 2)


def draw_border(frame, color_bgr, thickness=12):
    import cv2
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color_bgr, thickness)


def main(argv: Optional[list] = None) -> int:
    import cv2
    import numpy as np
    from ultralytics import YOLO

    args = parse_args(argv)
    if not args.video.exists():
        print(f"[render] missing video: {args.video}", file=sys.stderr)
        return 2
    if not args.fast_weights.exists() or not args.accurate_weights.exists():
        print(f"[render] missing weights", file=sys.stderr)
        return 2

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"[render] cannot open {args.video}", file=sys.stderr)
        return 2
    fps_in = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or -1
    cap_n = args.max_frames if args.max_frames > 0 else n_total
    print(f"[render] video: {n_total} frames, {fps_in:.1f} fps, {w}x{h}")
    print(f"[render] policy: {args.policy}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    writer = cv2.VideoWriter(str(args.out), fourcc, fps_in, (w, h))
    if not writer.isOpened():
        # Fallback to MJPG (always available, larger files).
        print(f"[render] WARN: codec {args.codec} unavailable; "
              f"falling back to MJPG (larger files).", file=sys.stderr)
        args.out = args.out.with_suffix(".avi")
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(str(args.out), fourcc, fps_in, (w, h))
        if not writer.isOpened():
            print(f"[render] ERROR: cannot open output writer",
                  file=sys.stderr)
            return 2

    fast_model = YOLO(str(args.fast_weights))
    acc_model = YOLO(str(args.accurate_weights))
    cfg_kwargs = {}
    if args.conf_ema_c_high is not None:
        cfg_kwargs["conf_ema_c_high"] = float(args.conf_ema_c_high)
    if args.conf_ema_c_low is not None:
        cfg_kwargs["conf_ema_c_low"] = float(args.conf_ema_c_low)
    cfg = PolicyConfig(**cfg_kwargs)
    policy = make_policy(args.policy, cfg)
    proxy_cfg = ProxyConfig(size=args.proxy_size)

    if args.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    last_choice = "n"
    flash_frames_left = 0   # ~4 frames at 30 fps = 150ms
    # Convert switch-fade-ms (CLI flag, default 150 ms) to a frame count.
    # Default (150 ms) matches the v1 demos; larger values make switches
    # more visible from a projector at the cost of brief border-colour
    # ambiguity around switch frames.
    flash_window_total = max(
        2, int(round((args.switch_fade_ms / 1000.0) * fps_in))
    )
    last_fast_mean_conf = 0.0
    frame_idx = 0
    t0 = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if cap_n > 0 and frame_idx >= cap_n:
            break

        # Compute proxies + build features.
        prox = compute_proxies(frame, proxy_cfg)
        feat = FrameFeatures(
            frame_idx=frame_idx,
            L=prox.get("L"), H=prox.get("H"),
            color_entropy=prox.get("color_entropy"),
            tenengrad=prox.get("tenengrad"),
            edge_density=prox.get("edge_density"),
            local_contrast=prox.get("local_contrast"),
            bright_fraction=prox.get("bright_fraction"),
            hue_std=prox.get("hue_std"),
            last_mean_conf=last_fast_mean_conf,
        )
        choice = policy.decide(feat)

        # Run the chosen model AND the fast model (the latter is needed
        # every frame for conf_ema's next-frame trigger; matches the
        # always-watchdog deployment we report in the audit).
        t_inf0 = time.perf_counter()
        if choice == "s":
            # Always run fast first as watchdog.
            fres = fast_model.predict(frame, conf=args.conf_floor,
                                      iou=0.7, verbose=False)
            sres = acc_model.predict(frame, conf=args.conf_floor,
                                     iou=0.7, verbose=False)
        else:
            fres = fast_model.predict(frame, conf=args.conf_floor,
                                      iou=0.7, verbose=False)
            sres = None
        t_inf_ms = (time.perf_counter() - t_inf0) * 1000.0

        # Update fast_mean_conf for next frame.
        n_boxes, n_scores = ([], [])
        if fres and fres[0].boxes is not None:
            n_boxes = fres[0].boxes.xyxy.cpu().numpy()
            n_scores = fres[0].boxes.conf.cpu().numpy()
        if len(n_scores) > 0:
            last_fast_mean_conf = float(n_scores.mean())
        else:
            last_fast_mean_conf = 0.0

        # Pick boxes to draw.
        if choice == "s" and sres is not None and sres[0].boxes is not None:
            boxes = sres[0].boxes.xyxy.cpu().numpy()
            scores = sres[0].boxes.conf.cpu().numpy()
            color = COLOR_ACCURATE_BGR
            active_model = "accurate (s)"
        else:
            boxes = n_boxes
            scores = n_scores
            color = COLOR_FAST_BGR
            active_model = "fast (n)"

        # Draw boxes.
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(frame, (x1, y1), (x2, y2),
                          color, args.box_thickness)
            cv2.putText(frame, f"{score:.2f}", (x1, max(28, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Border.
        if choice != last_choice:
            flash_frames_left = flash_window_total
        if flash_frames_left > 0:
            draw_border(frame, COLOR_FLASH_BGR, thickness=12)
            flash_frames_left -= 1
        else:
            draw_border(frame, color, thickness=12)
        last_choice = choice

        # HUD.
        mean_conf = (float(scores.mean()) if len(scores) > 0 else 0.0)
        hud = [
            f"policy        : {args.policy}",
            f"active model  : {active_model}",
            f"T_infer       : {t_inf_ms:6.1f} ms",
            f"#detections   : {len(boxes)}",
            f"mean conf     : {mean_conf:.3f}",
            f"frame         : {frame_idx}",
        ]
        draw_hud(frame, hud, COLOR_HUD_BG_BGR, COLOR_HUD_FG_BGR,
                 font_scale=args.hud_font_scale)

        writer.write(frame)
        frame_idx += 1
        if (frame_idx % 100) == 0:
            elapsed = time.perf_counter() - t0
            print(f"  frame {frame_idx} / {cap_n if cap_n>0 else '?'}  "
                  f"choice={choice}  T_infer={t_inf_ms:.1f}ms  "
                  f"[wall {elapsed:.1f}s]")

    cap.release()
    writer.release()
    elapsed = time.perf_counter() - t0
    print(f"\n[render] wrote {args.out} ({frame_idx} frames in {elapsed:.1f}s, "
          f"{frame_idx / max(elapsed, 1e-9):.1f} fps avg)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
