"""analysis/viz_examples.py — render representative frames with
n / s detections side by side + per-frame metadata overlay.

Answers the "give visualised example" question by rendering 2
frames per domain in two evidence-based categories (no subjective
"easy / hard" terminology, which would invite challenges about how
those categories are defined). The categories are:

  * agreement frame:        benefit_positive = 0 AND n_count == s_count > 0
  * benefit-positive frame: benefit_positive = 1 (locked dms.metrics
                            definition: accurate detection unmatched
                            by any fast detection at IoU≥0.5)

with:

  Left : original frame.
  Mid  : same frame + FAST-model bounding boxes (red), with
         n_count + fast_mean_conf in the corner.
  Right: same frame + ACCURATE-model bounding boxes (green), with
         s_count + benefit_positive label.

Plus a metadata footer showing the per-frame proxy values
(L, H, color_entropy) + per-policy decisions on this frame.

Inputs
------
  --weights-dir   ``results/remote_pull/.../extracted_weights``
                  (after extracting weights_for_viz.tgz)
  --videos        space-separated list of (label, path) entries

The script picks frames automatically by reading the matching
sweep_per_frame.csv from --sweep-root and selecting:
  * an "easy" frame: benefit_positive=0, n_count == s_count > 0
  * a "hard" frame: benefit_positive=1, s_count > n_count

Output: one PNG per (video, frame_idx) under ``analysis/figures/stage3/viz/``.

Usage (Windows path-friendly)
-----------------------------
    python -m analysis.viz_examples \\
        --weights-dir results/remote_pull/stage3_ablation/extracted_weights \\
        --features-dir results/remote_pull/stage3_ablation \\
        --sweep-root results/remote_pull/stage3_combined \\
        --out-dir analysis/figures/stage3/viz \\
        --pairs glass=yolov8_n_glass:yolov8_s_glass:G:/Teja_Master_Thesis/MT_Chirravuri/input_videos/glass_insulator_videos/20190916-722.mp4 \\
                porcelain=yolo26_n_porcelain:yolo26_l_porcelain:G:/Teja_Master_Thesis/MT_Chirravuri/input_videos/porcelain_insulator_videos/YUN_0037.MP4
"""
from __future__ import annotations

import argparse
import csv
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


def find_representative_frames(
    sweep_csv: Path,
    skip_fraction: float = 0.25,
) -> Dict[str, Optional[Dict]]:
    """Return four evidence-based frame categories.

    Selection prefers frames where (1) the drone has reached the scene
    (skip first ``skip_fraction`` of the video to avoid transit /
    backlit-silhouette early shots) and (2) there are MULTIPLE
    detections (so the viewer can actually see what's happening).
    Within each category, picks the frame with the most dramatic
    visual difference between fast and accurate (largest
    s_count - n_count for switch categories; highest detection count
    for agreement) so the figure communicates the story at a glance.

    Categories use the LOCKED thesis terminology:

      * 'agreement'      : benefit_positive == 0 AND n_count == s_count > 0
      * 'correct_switch' : benefit_positive == 1 AND conf_ema chose 's'
      * 'missed_switch'  : benefit_positive == 1 AND conf_ema chose 'n'
      * 'false_switch'   : benefit_positive == 0 AND conf_ema chose 's'
    """
    rows = list(csv.DictReader(sweep_csv.open("r", newline="",
                                              encoding="utf-8")))
    by_frame: Dict[int, Dict[str, Dict]] = {}
    for r in rows:
        fi = int(r["frame_idx"])
        by_frame.setdefault(fi, {})[r["policy"]] = r

    if not by_frame:
        return dict(agreement=None, correct_switch=None,
                    missed_switch=None, false_switch=None)
    max_idx = max(by_frame.keys())
    min_keep_idx = int(max_idx * skip_fraction)

    def candidates():
        """Yield (frame_idx, n_only_row, decisions) for every frame past
        the transit cutoff."""
        for fi in sorted(by_frame.keys()):
            if fi < min_keep_idx:
                continue
            polrows = by_frame[fi]
            if "n_only" not in polrows:
                continue
            n_row = polrows["n_only"]
            decisions = {pn: polrows[pn]["choice"] for pn in polrows}
            yield fi, n_row, decisions

    def best_for(category: str):
        """Score every candidate for a given category, return the best.

        Score function favours frames with non-trivial detection counts
        and (for switch categories) large s_count - n_count gaps.
        """
        best = None
        best_score = -1.0
        for fi, r, d in candidates():
            n_count = int(r["n_count"])
            s_count = int(r["s_count"])
            ben = int(r["benefit_positive"])
            ce = d.get("conf_ema")

            if category == "agreement":
                if ben != 0 or n_count <= 0 or n_count != s_count:
                    continue
                # Prefer frames with more detections; visually richer.
                score = float(n_count) + 0.001 * fi
            elif category == "correct_switch":
                if ben != 1 or ce != "s" or s_count <= n_count:
                    continue
                # Prefer big detection gap + decent total count.
                score = float(s_count - n_count) * 10 + float(s_count)
            elif category == "missed_switch":
                if ben != 1 or ce != "n" or s_count <= n_count:
                    continue
                score = float(s_count - n_count) * 10 + float(s_count)
            elif category == "false_switch":
                if ben != 0 or ce != "s":
                    continue
                # Prefer frames with at least some detections so the
                # "unnecessary switch" visualisation shows context.
                score = float(s_count) + 0.001
            else:
                continue
            if score > best_score:
                best_score = score
                best = {"n_only_row": r, "decisions": d}
        return best

    return dict(
        agreement=best_for("agreement"),
        correct_switch=best_for("correct_switch"),
        missed_switch=best_for("missed_switch"),
        false_switch=best_for("false_switch"),
    )


def features_for_frame(
    features_csv: Path, frame_idx: int,
) -> Optional[Dict]:
    rows = csv.DictReader(features_csv.open("r", newline="", encoding="utf-8"))
    for r in rows:
        if int(r["frame_idx"]) == frame_idx:
            return r
    return None


def render_one_panel(
    video_path: Path,
    frame_idx: int,
    fast_weights: Path,
    acc_weights: Path,
    out_path: Path,
    title: str,
    subtitle: str,
    panel_label: str,
    meta_pairs: List[Tuple[str, str]],
) -> bool:
    """Render a thesis-quality 1x3 panel + structured metadata table.

    Layout:
      Top row of 3 image axes (original | FAST | ACCURATE)
      Bottom: a key:value metadata table (NOT a monospace blob)
      One main title + one subtitle (the takeaway in plain English)
      Panel label (a/b/c/d) for thesis cross-references
    """
    import cv2
    import numpy as np
    from ultralytics import YOLO
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        print(f"[viz] failed to read frame {frame_idx} from {video_path}",
              file=sys.stderr)
        return False
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    fast_model = YOLO(str(fast_weights))
    acc_model = YOLO(str(acc_weights))

    def predict(model, img):
        res = model.predict(img, conf=0.25, iou=0.7, verbose=False)
        if not res or res[0].boxes is None:
            return [], []
        b = res[0].boxes
        return (b.xyxy.cpu().numpy(), b.conf.cpu().numpy())

    n_boxes, n_scores = predict(fast_model, frame)
    s_boxes, s_scores = predict(acc_model, frame)

    # Locked convention (pathB_plan.md):
    #   GREEN = fast model active (n)
    #   RED   = accurate model active (s)
    FAST_COLOR_RGB = (30, 180, 30)   # green
    ACC_COLOR_RGB = (220, 30, 30)    # red
    img_n = rgb.copy()
    for box, score in zip(n_boxes, n_scores):
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(img_n, (x1, y1), (x2, y2), FAST_COLOR_RGB, 5)
        cv2.putText(img_n, f"{score:.2f}", (x1, max(28, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, FAST_COLOR_RGB, 3)
    img_s = rgb.copy()
    for box, score in zip(s_boxes, s_scores):
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(img_s, (x1, y1), (x2, y2), ACC_COLOR_RGB, 5)
        cv2.putText(img_s, f"{score:.2f}", (x1, max(28, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, ACC_COLOR_RGB, 3)

    # ---- figure layout: 3 image axes top, 1 metadata axis bottom ----
    fig = plt.figure(figsize=(20, 9), constrained_layout=False)
    gs = GridSpec(2, 3, figure=fig,
                  height_ratios=[3.6, 1.0],
                  left=0.04, right=0.99, top=0.88, bottom=0.04,
                  hspace=0.18, wspace=0.04)

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    ax_meta = fig.add_subplot(gs[1, :])

    ax0.imshow(rgb)
    ax0.set_title("(a)  Original frame", fontsize=14, fontweight="bold",
                  loc="left")
    ax1.imshow(img_n)
    ax1.set_title(
        f"(b)  FAST model — {fast_weights.parent.parent.name}\n"
        f"      {len(n_boxes)} detection(s) at conf≥0.25 (green boxes)",
        fontsize=14, fontweight="bold", color="#157515", loc="left",
    )
    ax2.imshow(img_s)
    ax2.set_title(
        f"(c)  ACCURATE model — {acc_weights.parent.parent.name}\n"
        f"      {len(s_boxes)} detection(s) at conf≥0.25 (red boxes)",
        fontsize=14, fontweight="bold", color="#aa1818", loc="left",
    )
    for ax in (ax0, ax1, ax2):
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # ---- metadata as a structured table (NOT monospace blob) ----
    ax_meta.axis("off")
    n_cols = 3
    rows_per_col = (len(meta_pairs) + n_cols - 1) // n_cols
    col_w = 1.0 / n_cols
    for i, (k, v) in enumerate(meta_pairs):
        col = i // rows_per_col
        row = i % rows_per_col
        x_left = col * col_w + 0.02
        y = 0.85 - row * 0.20
        ax_meta.text(x_left, y, f"{k}",
                     transform=ax_meta.transAxes,
                     fontsize=12, color="#444444", ha="left", va="top")
        ax_meta.text(x_left + 0.16, y, f"{v}",
                     transform=ax_meta.transAxes,
                     fontsize=13, color="black", ha="left", va="top",
                     fontweight="bold")
    # Frame the metadata block.
    ax_meta.add_patch(plt.Rectangle((0.0, 0.0), 1.0, 1.0,
                                    transform=ax_meta.transAxes,
                                    fill=True, facecolor="#fffdf2",
                                    edgecolor="#999999", lw=0.8,
                                    zorder=-1))

    # Suptitle (panel label + headline) + subtitle (takeaway).
    fig.text(0.04, 0.965, panel_label, fontsize=18, fontweight="bold",
             ha="left", va="top")
    fig.text(0.5, 0.965, title, fontsize=15, fontweight="bold",
             ha="center", va="top")
    fig.text(0.5, 0.925, subtitle, fontsize=11, ha="center", va="top",
             color="#444444", style="italic")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"[viz] wrote {out_path}")
    return True


def parse_pair(pair_arg: str) -> Tuple[str, str, str, str]:
    """Format: domain=fast_dir:acc_dir:video_path"""
    domain, rest = pair_arg.split("=", 1)
    fast_dir, acc_dir, video = rest.split(":", 2)
    return domain, fast_dir, acc_dir, video


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render visualised example frames.")
    p.add_argument("--weights-dir", type=Path, required=True,
                   help="Dir with extracted yolov8_*/weights/best.pt etc.")
    p.add_argument("--features-dir", type=Path, required=True,
                   help="Dir with features_<video> subdirs.")
    p.add_argument("--sweep-root", type=Path, required=True,
                   help="Dir with stage3_<pair>_cpu subdirs.")
    p.add_argument("--out-dir", type=Path,
                   default=Path("analysis/figures/stage3/viz"))
    p.add_argument("--pairs", type=str, nargs="+", required=True,
                   help="Format domain=fast:acc:video_path per pair.")
    return p.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)

    for pair_arg in args.pairs:
        domain, fast_dir, acc_dir, video_path_s = parse_pair(pair_arg)
        video_path = Path(video_path_s)
        if not video_path.exists():
            print(f"[viz] missing video: {video_path}", file=sys.stderr)
            continue
        fast_w = args.weights_dir / fast_dir / "weights" / "best.pt"
        acc_w = args.weights_dir / acc_dir / "weights" / "best.pt"
        if not fast_w.exists() or not acc_w.exists():
            print(f"[viz] missing weights: {fast_w} / {acc_w}",
                  file=sys.stderr)
            continue

        # Locate the matching sweep dir + features dir for this domain.
        if domain == "glass":
            sweep_dir = args.sweep_root / "stage3_yolov8_n_s_glass_cpu"
            features_dir = args.features_dir / "features_20190916-722"
        elif domain == "porcelain":
            sweep_dir = args.sweep_root / "stage3_yolo26_n_l_porcelain_cpu"
            features_dir = args.features_dir / "features_yun_0037.mp4"
        else:
            print(f"[viz] unknown domain {domain}", file=sys.stderr)
            continue

        sweep_csv = sweep_dir / "sweep_per_frame.csv"
        feat_csv = features_dir / "per_frame.csv"
        if not (sweep_csv.exists() and feat_csv.exists()):
            print(f"[viz] missing csv inputs for {domain}", file=sys.stderr)
            continue

        frames = find_representative_frames(sweep_csv)
        # Stable panel labels per (domain × category).
        panel_letters = {
            ("glass", "agreement"):       "(a)",
            ("glass", "correct_switch"):  "(b)",
            ("glass", "missed_switch"):   "(c)",
            ("glass", "false_switch"):    "(d)",
            ("porcelain", "agreement"):       "(e)",
            ("porcelain", "correct_switch"):  "(f)",
            ("porcelain", "missed_switch"):   "(g)",
            ("porcelain", "false_switch"):    "(h)",
        }

        for kind, payload in frames.items():
            if payload is None:
                print(f"[viz] no {kind} frame found for {domain}",
                      file=sys.stderr)
                continue
            row = payload["n_only_row"]
            decisions = payload["decisions"]
            fi = int(row["frame_idx"])
            feats = features_for_frame(feat_csv, fi) or {}

            n_count = int(row["n_count"])
            s_count = int(row["s_count"])
            ben = int(row["benefit_positive"])
            fast_conf = _f(row["fast_mean_conf"])

            # Render per-policy decisions as one comma-separated string.
            decision_order = ["n_only", "s_only", "entropy_only",
                              "combined", "combined_hyst", "conf_ema",
                              "multi_proxy"]
            dec_short = "  ".join(
                f"{p[:9]}={decisions.get(p, '?')}"
                for p in decision_order
                if p not in ("n_only", "s_only")
            )

            meta_pairs = [
                ("frame index",       f"{fi}"),
                ("benefit_positive",  "1" if ben else "0"),
                ("fast detections",   f"{n_count}"),
                ("accurate detections", f"{s_count}"),
                ("fast mean conf",    f"{fast_conf:.3f}"),
                ("Laplacian var (L)", f"{_f(feats.get('L', 0)):.2f}"),
                ("entropy (H)",       f"{_f(feats.get('H', 0)):.3f}  bits"),
                ("color entropy",     f"{_f(feats.get('color_entropy', 0)):.3f}  bits"),
                ("tenengrad",         f"{_f(feats.get('tenengrad', 0)):.1f}"),
                ("edge density",      f"{_f(feats.get('edge_density', 0)):.4f}"),
                ("policies → choice", dec_short),
                ("video",             video_path.name),
            ]

            CATEGORY_INFO = {
                "agreement": (
                    "agreement frame (benefit_positive = 0, fast/accurate match)",
                    "Fast and accurate models produce the same detections and "
                    "every accurate detection matches a fast detection at IoU "
                    "≥ 0.5. Switching is unnecessary; ideal policy choice is 'n'."
                ),
                "correct_switch": (
                    "correctly-triggered switch (benefit_positive = 1, conf_ema → s)",
                    "Frame is benefit-positive (accurate has a detection the "
                    "fast model misses at IoU ≥ 0.5) AND conf_ema's trigger "
                    "fired. The policy made the right call here."
                ),
                "missed_switch": (
                    "missed switch (benefit_positive = 1, conf_ema → n)",
                    "Frame is benefit-positive but conf_ema's trigger did NOT "
                    "fire. The fast model is used and the missed detections "
                    "are not recovered. Counts toward the policy's false-"
                    "negative rate on the trigger."
                ),
                "false_switch": (
                    "false-positive switch (benefit_positive = 0, conf_ema → s)",
                    "Frame is NOT benefit-positive yet conf_ema's trigger "
                    "fired anyway. The accurate model runs unnecessarily; "
                    "counts toward false-positive rate. This is what causes "
                    "the trigger F1 to fall below 1.0."
                ),
            }
            cat_title, cat_subtitle = CATEGORY_INFO[kind]
            title = f"{domain.upper()}  —  {cat_title}"
            subtitle = cat_subtitle
            panel = panel_letters.get((domain, kind), "")

            out_path = args.out_dir / f"viz_{domain}_{kind}_frame{fi:04d}.png"
            render_one_panel(
                video_path, fi, fast_w, acc_w, out_path,
                title=title, subtitle=subtitle,
                panel_label=panel, meta_pairs=meta_pairs,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
