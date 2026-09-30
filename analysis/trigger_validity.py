"""analysis/trigger_validity.py — predictive (NOT causal) value of per-frame
image statistics for identifying frames where the accurate model helps.

Methodological note (locked)
----------------------------
This script measures **statistical association**, not causation. A
high AUROC/AUPRC for an image feature against the locked
benefit_positive label means the feature has **predictive value** for
identifying benefit-positive frames in this dataset; it does not
prove that the feature causally drives benefit. The thesis text MUST
use "predictive value" / "statistically associated", never "causally
informative".

Question this script answers
----------------------------
For each scene-feature image statistic (L, H, color_entropy, tenengrad,
edge_density, local_contrast, bright_fraction, hue_std), and for each
of several alternative target labels, how well does the feature's raw
value predict the target per frame?

Targets supported (multi-target validity):
  * benefit_positive       (locked label, primary target)
  * n_iou_match_failure    (fast model failed strict 1:1 vs s_only)
  * n_det_coverage_failure (fast model missed at least one ref det)
  * count_disagreement     (n_count != s_count)

Per (domain, target, feature) we report
---------------------------------------
* AUROC — area under ROC. 0.5 = no signal.
* AUPRC — area under precision-recall curve. Compare to base rate.
* Random baseline AUPRC = base rate of the positive label.
* Spearman ρ + Mann-Whitney U two-tailed p-value (effect significance).
* Cohen's d on the feature distribution (effect size).
* Top-quartile lift = P(positive | feature in top quartile) / base.
* KS statistic = max distance between feature CDFs on +/- frames.

Inputs
------
A pair of CSVs per domain:
  * features:  per_frame.csv from experiments.run_features
  * sweep:     sweep_per_frame.csv from experiments.run_sweep
              (any policy row is fine; benefit_positive is broadcast)

The two CSVs must come from the same source video.

Output
------
* trigger_validity.csv      one row per (domain, feature) tuple
* fig_trigger_validity.png  AUC bar chart + ROC curves
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _f(x) -> float:
    if x is None or x == "":
        return float("nan")
    try:
        return float(x)
    except ValueError:
        return float("nan")


def read_features(p: Path) -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str, float]] = {}
    with p.open("r", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("policy") not in (None, "", "n_only"):
                continue
            fi = int(r["frame_idx"])
            out[fi] = {k: _f(v) for k, v in r.items() if k != "frame_idx"}
    return out


def read_targets(p: Path) -> Dict[int, Dict[str, int]]:
    """Pull all per-frame target labels from sweep_per_frame.csv.

    Targets per frame:
      benefit_positive       (locked label, broadcast across policies)
      n_iou_match_failure    1 iff n_only's iou_match == 0 on this frame
      n_det_coverage_failure 1 iff n_only's det_coverage == 0
      count_disagreement     1 iff n_count != s_count
    """
    by_frame: Dict[int, Dict[str, Dict]] = defaultdict(dict)
    with p.open("r", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            fi = int(r["frame_idx"])
            by_frame[fi][r["policy"]] = r
    out: Dict[int, Dict[str, int]] = {}
    for fi, polrows in by_frame.items():
        if "n_only" not in polrows:
            continue
        n_row = polrows["n_only"]
        out[fi] = dict(
            benefit_positive=int(n_row["benefit_positive"]),
            n_iou_match_failure=1 - int(n_row["iou_match"]),
            n_det_coverage_failure=1 - int(n_row["det_coverage"]),
            count_disagreement=int(int(n_row["n_count"]) !=
                                    int(n_row["s_count"])),
        )
    return out


def _feature_csv_for_dir(run_dir: Path) -> Path:
    """Prefer de-duplicated frame features; fall back to sweep rows."""
    for name in ("frame_features.csv", "per_frame.csv", "sweep_per_frame.csv"):
        p = run_dir / name
        if p.exists():
            return p
    return run_dir / "frame_features.csv"


# ---------------------------------------------------------------------------
# AUC + correlation
# ---------------------------------------------------------------------------
def auc_score(scores: List[float], labels: List[int]) -> float:
    """Mann-Whitney U formulation of AUC (no sklearn dependency)."""
    pairs = sorted(zip(scores, labels))
    rank_sum_pos = 0.0
    n_pos = 0
    n_neg = 0
    n = len(pairs)
    i = 0
    while i < n:
        j = i
        while j < n and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = 0.5 * ((i + 1) + j)
        for k in range(i, j):
            if pairs[k][1] == 1:
                rank_sum_pos += avg_rank
                n_pos += 1
            else:
                n_neg += 1
        i = j
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return auc


def spearman_rho(xs: List[float], ys: List[float]) -> float:
    """Spearman = Pearson on ranks. No external deps."""
    if len(xs) < 2:
        return float("nan")

    def rank(arr: List[float]) -> List[float]:
        order = sorted(range(len(arr)), key=lambda i: arr[i])
        ranks = [0.0] * len(arr)
        i = 0
        while i < len(arr):
            j = i
            while j < len(arr) and arr[order[j]] == arr[order[i]]:
                j += 1
            avg_rank = 0.5 * ((i + 1) + j)
            for k in range(i, j):
                ranks[order[k]] = avg_rank
            i = j
        return ranks

    rx = rank(xs)
    ry = rank(ys)
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(len(rx)))
    dx = math.sqrt(sum((r - mx) ** 2 for r in rx))
    dy = math.sqrt(sum((r - my) ** 2 for r in ry))
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def quartile_lift(scores: List[float], labels: List[int]) -> float:
    """P(benefit | top quartile of feature) / base rate.

    >1.0 means the top quartile of this feature is enriched in
    benefit-positive frames. =1.0 means the feature is uninformative.
    """
    n = len(scores)
    if n == 0:
        return float("nan")
    base = sum(labels) / n
    if base == 0.0:
        return float("nan")
    pairs = sorted(zip(scores, labels), reverse=True)
    top = pairs[: max(1, n // 4)]
    top_pos_rate = sum(l for _, l in top) / len(top)
    return top_pos_rate / base


def ks_statistic(scores: List[float], labels: List[int]) -> float:
    """Two-sample KS between feature distributions on +/- frames.

    O(n) two-pointer sweep over sorted positive and negative samples.
    (The naive O(n^2) implementation made multi-group runs unusable
    on the 8190-frame glass video.)
    """
    pos = sorted([s for s, l in zip(scores, labels) if l == 1])
    neg = sorted([s for s, l in zip(scores, labels) if l == 0])
    n_p = len(pos)
    n_n = len(neg)
    if n_p == 0 or n_n == 0:
        return float("nan")
    i_p = 0
    i_n = 0
    max_d = 0.0
    while i_p < n_p or i_n < n_n:
        v_p = pos[i_p] if i_p < n_p else float("inf")
        v_n = neg[i_n] if i_n < n_n else float("inf")
        v = v_p if v_p <= v_n else v_n
        while i_p < n_p and pos[i_p] <= v:
            i_p += 1
        while i_n < n_n and neg[i_n] <= v:
            i_n += 1
        d = abs(i_p / n_p - i_n / n_n)
        if d > max_d:
            max_d = d
    return max_d


def auprc_score(scores: List[float], labels: List[int]) -> float:
    """Average-precision approximation of AUPRC.

    Sort by descending score; sweep through and compute precision at
    each rank where a positive is encountered; average those
    precisions. Equivalent to sklearn.metrics.average_precision_score.
    """
    n = len(scores)
    if n == 0:
        return float("nan")
    n_pos = sum(labels)
    if n_pos == 0:
        return float("nan")
    pairs = sorted(zip(scores, labels), reverse=True)
    tp = 0
    fp = 0
    last_recall = 0.0
    ap = 0.0
    for s, l in pairs:
        if l == 1:
            tp += 1
        else:
            fp += 1
        precision = tp / max(tp + fp, 1)
        recall = tp / n_pos
        if l == 1:
            ap += precision * (recall - last_recall)
            last_recall = recall
    return ap


def mann_whitney_p(
    scores_pos: List[float], scores_neg: List[float],
) -> float:
    """Two-tailed Mann-Whitney U test, normal approximation.

    Implements the standard rank-sum formulation with continuity
    correction. Returns the two-tailed p-value. No external deps.
    Reasonably accurate for n_pos + n_neg >= 20.
    """
    n_pos = len(scores_pos)
    n_neg = len(scores_neg)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    combined = sorted([(s, "p") for s in scores_pos] +
                      [(s, "n") for s in scores_neg])
    # Average ranks for ties.
    ranks: List[float] = [0.0] * len(combined)
    i = 0
    while i < len(combined):
        j = i
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        avg = 0.5 * ((i + 1) + j)
        for k in range(i, j):
            ranks[k] = avg
        i = j
    rank_sum_pos = sum(ranks[k] for k in range(len(combined))
                       if combined[k][1] == "p")
    u_pos = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    u_neg = n_pos * n_neg - u_pos
    u = min(u_pos, u_neg)
    mu = n_pos * n_neg / 2.0
    sigma = math.sqrt(n_pos * n_neg * (n_pos + n_neg + 1) / 12.0)
    if sigma == 0:
        return float("nan")
    z = (u - mu + 0.5) / sigma  # continuity correction
    # Two-tailed p from standard normal.
    return 2.0 * (1.0 - _phi(abs(z)))


def _phi(x: float) -> float:
    """Standard normal CDF using error-function approximation."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def cohens_d(scores_pos: List[float], scores_neg: List[float]) -> float:
    """Cohen's d effect size on the feature distribution.

    d = (mean_pos - mean_neg) / pooled_std.
    Conventional thresholds: |d|<0.2 negligible, ~0.5 medium, >0.8 large.
    """
    if not scores_pos or not scores_neg:
        return float("nan")
    n_p, n_n = len(scores_pos), len(scores_neg)
    m_p = sum(scores_pos) / n_p
    m_n = sum(scores_neg) / n_n
    var_p = sum((x - m_p) ** 2 for x in scores_pos) / max(n_p - 1, 1)
    var_n = sum((x - m_n) ** 2 for x in scores_neg) / max(n_n - 1, 1)
    pooled = math.sqrt(((n_p - 1) * var_p + (n_n - 1) * var_n)
                       / max(n_p + n_n - 2, 1))
    if pooled == 0:
        return float("nan")
    return (m_p - m_n) / pooled


def cliffs_delta(scores_pos: List[float], scores_neg: List[float]) -> float:
    """Cliff's delta: P(pos>neg) - P(pos<neg), in [-1, 1]."""
    if not scores_pos or not scores_neg:
        return float("nan")
    pos = sorted(scores_pos)
    neg = sorted(scores_neg)
    n_pos = len(pos)
    n_neg = len(neg)
    gt = 0
    lt = 0
    j_lt = 0
    j_le = 0
    for x in pos:
        while j_lt < n_neg and neg[j_lt] < x:
            j_lt += 1
        while j_le < n_neg and neg[j_le] <= x:
            j_le += 1
        gt += j_lt
        lt += n_neg - j_le
    return float((gt - lt) / (n_pos * n_neg))


def roc_curve(
    scores: List[float], labels: List[int], n_thresholds: int = 50,
) -> Tuple[List[float], List[float]]:
    """Sweep thresholds; return (fpr, tpr) lists."""
    if not scores or not labels:
        return [], []
    s_min, s_max = min(scores), max(scores)
    if s_min == s_max:
        return [0.0, 1.0], [0.0, 1.0]
    thresholds = [s_min + (s_max - s_min) * i / n_thresholds
                  for i in range(n_thresholds + 1)]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return [], []
    fprs, tprs = [], []
    for t in thresholds:
        tp = sum(1 for s, l in zip(scores, labels) if s >= t and l == 1)
        fp = sum(1 for s, l in zip(scores, labels) if s >= t and l == 0)
        tprs.append(tp / n_pos)
        fprs.append(fp / n_neg)
    fprs.append(0.0); tprs.append(0.0)
    return fprs, tprs


# ---------------------------------------------------------------------------
# Per-domain analysis
# ---------------------------------------------------------------------------
SCENE_FEATURES = ("L", "H", "color_entropy", "tenengrad",
                  "edge_density", "local_contrast",
                  "bright_fraction", "hue_std")


TARGET_NAMES = ("benefit_positive", "n_iou_match_failure",
                "n_det_coverage_failure", "count_disagreement")


def analyse_one(
    group_label: str,
    features_csv: Path,
    sweep_csv: Path,
) -> Tuple[List[Dict], Dict[Tuple[str, str], Tuple[List[float], List[float]]]]:
    """Per-(target, feature) trigger-validity row + ROC curves.

    Returns:
      rows: list of dicts (one per (target, feature) tuple).
            Each row has the group_label baked in so all rows can be
            concatenated cross-group.
      rocs: dict mapping (target, feature) -> (fpr_list, tpr_list).
    """
    feats = read_features(features_csv)
    targets = read_targets(sweep_csv)

    common = sorted(set(feats.keys()) & set(targets.keys()))
    if not common:
        return [], {}
    print(f"  {group_label}: {len(common)} aligned frames "
          f"(features {len(feats)} vs sweep {len(targets)})")

    rows: List[Dict] = []
    rocs: Dict[Tuple[str, str], Tuple[List[float], List[float]]] = {}

    for target_name in TARGET_NAMES:
        labels_all = [targets[fi][target_name] for fi in common]
        n_pos = sum(labels_all)
        n_neg = len(labels_all) - n_pos
        if n_pos == 0 or n_neg == 0:
            print(f"  {group_label} / target={target_name}: degenerate "
                  f"({n_pos} positives, {n_neg} negatives) — skipping")
            continue
        base_rate = n_pos / len(labels_all)

        for feat_name in SCENE_FEATURES:
            if feat_name not in feats[common[0]]:
                continue
            scores: List[float] = []
            labels: List[int] = []
            for fi in common:
                v = feats[fi].get(feat_name)
                if v is None or math.isnan(v):
                    continue
                scores.append(float(v))
                labels.append(int(targets[fi][target_name]))
            if len(scores) < 10 or sum(labels) == 0 \
                    or sum(labels) == len(labels):
                continue
            auc = auc_score(scores, labels)
            auprc = auprc_score(scores, labels)
            rho = spearman_rho(scores, [float(l) for l in labels])
            lift = quartile_lift(scores, labels)
            ks = ks_statistic(scores, labels)
            pos_scores = [s for s, l in zip(scores, labels) if l == 1]
            neg_scores = [s for s, l in zip(scores, labels) if l == 0]
            mw_p = mann_whitney_p(pos_scores, neg_scores)
            d = cohens_d(pos_scores, neg_scores)
            delta = cliffs_delta(pos_scores, neg_scores)

            # Direction-aware AUROC: max(auc, 1-auc). A feature with
            # auc < 0.5 is informative in the opposite direction.
            auroc_dir = max(auc, 1.0 - auc) if not math.isnan(auc) else float("nan")
            direction = ""
            if not math.isnan(auc):
                if auc > 0.5:
                    direction = "positive"   # high feature -> positive label
                elif auc < 0.5:
                    direction = "negative"   # low feature -> positive label
                else:
                    direction = "none"

            row = dict(
                group=group_label,
                target=target_name,
                feature=feat_name,
                n_frames=len(scores),
                base_rate=round(base_rate, 4),
                auroc_raw=round(auc, 4),
                auroc_directional=round(auroc_dir, 4),
                direction=direction,
                auprc=round(auprc, 4),
                random_auprc=round(base_rate, 4),
                auprc_lift_over_random=round(auprc - base_rate, 4),
                spearman=round(rho, 4),
                spearman_rho=round(rho, 4),
                mann_whitney_p=("<1e-12" if (mw_p is not None
                                              and not math.isnan(mw_p)
                                              and mw_p < 1e-12)
                                else (round(mw_p, 6)
                                      if not math.isnan(mw_p) else "")),
                cohens_d=round(d, 4) if not math.isnan(d) else "",
                cliffs_delta=(round(delta, 4)
                               if not math.isnan(delta) else ""),
                top_quartile_lift=(round(lift, 3)
                                    if not math.isnan(lift) else ""),
                ks_statistic=round(ks, 4),
            )
            rows.append(row)
            if not math.isnan(auc):
                rocs[(target_name, feat_name)] = roc_curve(scores, labels)
    return rows, rocs


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_validity(
    rows: List[Dict],
    rocs_by_group: Dict[str, Dict[Tuple[str, str], Tuple[List[float], List[float]]]],
    out_path: Path,
    target: str = "benefit_positive",
) -> None:
    """AUROC_directional bars + ROC overlays per group, for ONE target.

    Bars use direction-aware AUROC (max(raw, 1-raw)) so a feature is
    rated by HOW informative it is, not by which direction. Bar
    edge color encodes the direction (green = positive, orange =
    negative).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    rows_t = [r for r in rows if r["target"] == target]
    if not rows_t:
        print(f"[trigger_validity] no rows for target={target}",
              file=sys.stderr)
        return

    groups = sorted(set(r["group"] for r in rows_t))
    feature_order = SCENE_FEATURES

    n_panels = len(groups)
    fig = plt.figure(figsize=(16, 4.2 * n_panels))
    for g_i, grp in enumerate(groups):
        ax_l = fig.add_subplot(n_panels, 2, 2 * g_i + 1)
        feats_for_g = [r for r in rows_t if r["group"] == grp]
        feats_for_g.sort(key=lambda r: feature_order.index(r["feature"])
                         if r["feature"] in feature_order else 999)
        xs = list(range(len(feats_for_g)))
        ys = [r["auroc_directional"] for r in feats_for_g]
        names = [r["feature"] for r in feats_for_g]
        # Color by directional AUROC magnitude; edge encodes direction.
        face_colors = []
        edge_colors = []
        for r, y in zip(feats_for_g, ys):
            if y >= 0.65:
                face_colors.append("#2ca02c")
            elif y <= 0.55:
                face_colors.append("#aaaaaa")
            else:
                face_colors.append("#ffaa44")
            edge_colors.append("#2ca02c" if r["direction"] == "positive"
                                else ("#cc4444" if r["direction"] == "negative"
                                      else "black"))
        bars = ax_l.bar(xs, ys, color=face_colors, edgecolor=edge_colors,
                        linewidth=2.2, width=0.7)
        for bar, y, r in zip(bars, ys, feats_for_g):
            sign = "+" if r["direction"] == "positive" else \
                   ("−" if r["direction"] == "negative" else "")
            ax_l.text(bar.get_x() + bar.get_width() / 2, y + 0.01,
                      f"{y:.3f}{sign}", ha="center", va="bottom",
                      fontsize=10, fontweight="bold")
        ax_l.axhline(0.5, color="black", lw=1.0, ls="--", alpha=0.6)
        ax_l.text(len(xs) - 0.5, 0.51, "random (AUROC=0.5)",
                  ha="right", va="bottom", fontsize=9, color="0.3")
        ax_l.set_xticks(xs)
        ax_l.set_xticklabels(names, rotation=20, ha="right", fontsize=10)
        ax_l.set_ylim(0.3, 1.02)
        ax_l.set_ylabel("AUROC_directional")
        ax_l.set_title(f"{grp}", fontsize=11, fontweight="bold")
        ax_l.grid(True, axis="y", alpha=0.3)
        ax_l.set_axisbelow(True)

        ax_r = fig.add_subplot(n_panels, 2, 2 * g_i + 2)
        ax_r.plot([0, 1], [0, 1], "--", color="0.5", lw=0.8, alpha=0.6,
                  label="random")
        rocs = rocs_by_group.get(grp, {})
        for feat in feature_order:
            key = (target, feat)
            if key not in rocs:
                continue
            fpr, tpr = rocs[key]
            if not fpr:
                continue
            ax_r.plot(fpr, tpr, lw=1.3, label=feat)
        ax_r.set_xlabel("False Positive Rate")
        ax_r.set_ylabel("True Positive Rate")
        ax_r.set_title(f"{grp}: ROC per feature (raw direction)",
                       fontsize=11, fontweight="bold")
        ax_r.set_xlim(-0.01, 1.01)
        ax_r.set_ylim(-0.01, 1.01)
        ax_r.grid(True, alpha=0.3)
        ax_r.legend(loc="lower right", fontsize=7)

    fig.suptitle(
        f"Trigger validity — predictive (NOT causal) value of image "
        f"features\n"
        f"target = {target}    "
        f"(green edge = high feature predicts label; "
        f"red edge = low feature predicts label)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[trigger_validity] wrote {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Per-feature trigger validity (direction-aware) "
                    "across (domain x model_family x pair_type)."
    )
    p.add_argument("--pairs", type=str, nargs="+", default=None,
                   help=("Format: <key>=<features_dir>:<sweep_dir>. "
                         "key is a free-form group label, e.g. "
                         "'glass_yolov8_n_l' or 'porcelain_yolo26_n_s'. "
                         "Each pair contributes one logical 'group' to "
                         "the analysis. Domain / family / pair are "
                         "parsed from the key when possible."))
    p.add_argument("--stage3-root", type=Path, action="append",
                   default=None,
                   help=("Root containing stage3_* run dirs. If --pairs is "
                         "omitted, runs are auto-discovered here. May be "
                         "passed multiple times. Defaults to "
                         "results/remote_pull/stage3_combined and "
                         "results/remote_pull/stage3_cuda."))
    p.add_argument("--prefer-device", type=str, default="cpu",
                   help="When CPU/CUDA duplicate a group, keep this device.")
    p.add_argument("--out-dir", type=Path,
                   default=Path("analysis/figures/stage3"))
    return p.parse_args(argv)


def parse_group_key(key: str) -> Dict[str, str]:
    """Parse '<domain>_<family>_<fast>_<acc>' into a grouping dict.

    Falls back to {'group': key} on unrecognised keys.
    """
    import re
    m = re.match(
        r"^(glass|porcelain)_(yolov8|yolo26)_(n)_(s|l)$", key
    )
    if not m:
        return {"group": key, "domain": key, "model_family": "",
                "pair_type": ""}
    dom, fam, fast, acc = m.groups()
    return {"group": key, "domain": dom, "model_family": fam,
            "pair_type": f"{fast}_{acc}"}


def discover_stage3_pairs(
    roots: List[Path],
    prefer_device: str,
) -> List[Tuple[str, Path, Path]]:
    """Return de-duplicated (key, feature_csv, sweep_csv) inputs.

    Deduplication key is (domain, model_family, pair_type); this avoids
    double-counting identical frame labels from CPU and CUDA runs.
    """
    import re
    candidates: Dict[Tuple[str, str, str], Tuple[int, str, Path, Path]] = {}
    pat = re.compile(
        r"^stage3_(yolov8|yolo26)_n_(s|l)_(glass|porcelain)_(cpu|cuda)$"
    )
    for root in roots:
        if not root.exists():
            continue
        for sweep_csv in root.glob("stage3_*/sweep_per_frame.csv"):
            run_dir = sweep_csv.parent
            m = pat.match(run_dir.name)
            if not m:
                continue
            fam, acc, dom, dev = m.groups()
            pair_type = f"n_{acc}"
            group_key = f"{dom}_{fam}_{pair_type}"
            feature_csv = _feature_csv_for_dir(run_dir)
            if (not feature_csv.exists()) or feature_csv.name == "sweep_per_frame.csv":
                # Older Stage 3 sweep CSVs did not duplicate image features.
                # Reuse the matching per-frame feature trace generated for
                # the Stage 3 ablations; labels still come from each run's
                # sweep CSV, so model-family/pair differences are preserved.
                ablation_root = root.parent / "stage3_ablation"
                fallback = {
                    "glass": ablation_root / "features_20190916-722"
                             / "per_frame.csv",
                    "porcelain": ablation_root / "features_yun_0037.mp4"
                                 / "per_frame.csv",
                }.get(dom)
                feature_csv = fallback if fallback and fallback.exists() else sweep_csv
            score = 0 if dev == prefer_device else 1
            key = (dom, fam, pair_type)
            old = candidates.get(key)
            if old is None or score < old[0]:
                candidates[key] = (score, group_key, feature_csv, sweep_csv)
    return [(v[1], v[2], v[3]) for v in sorted(candidates.values(),
                                               key=lambda x: x[1])]


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    all_rows: List[Dict] = []
    all_rocs: Dict[str, Dict] = {}
    input_pairs: List[Tuple[str, Path, Path]] = []
    if args.pairs:
        for arg in args.pairs:
            key, rest = arg.split("=", 1)
            feat_dir, sweep_dir = rest.split(":", 1)
            feat_path = Path(feat_dir)
            sweep_path = Path(sweep_dir)
            feat_csv = feat_path if feat_path.is_file() else _feature_csv_for_dir(feat_path)
            sweep_csv = (sweep_path if sweep_path.is_file()
                         else sweep_path / "sweep_per_frame.csv")
            input_pairs.append((key, feat_csv, sweep_csv))
    else:
        roots = args.stage3_root or [
            Path("results/remote_pull/stage3_combined"),
            Path("results/remote_pull/stage3_cuda"),
        ]
        input_pairs = discover_stage3_pairs(roots, args.prefer_device)
        print(f"[trigger_validity] auto-discovered {len(input_pairs)} "
              f"deduplicated stage3 groups")

    for key, feat_csv, sweep_csv in input_pairs:
        if not feat_csv.exists() or not sweep_csv.exists():
            print(f"[trigger_validity] missing input for {key}",
                  file=sys.stderr)
            continue
        meta = parse_group_key(key)
        rows, rocs = analyse_one(key, feat_csv, sweep_csv)
        # Splice group metadata into every row.
        for r in rows:
            for k, v in meta.items():
                r[k] = v
        all_rows.extend(rows)
        all_rocs[key] = rocs

    if not all_rows:
        print("[trigger_validity] no analysable rows", file=sys.stderr)
        return 2

    # Print table grouped by target, then by group.
    for target in TARGET_NAMES:
        rows_t = [r for r in all_rows if r["target"] == target]
        if not rows_t:
            continue
        print()
        print(f"=== TARGET: {target} ===")
        print(f"{'group':28s} {'feature':16s} {'n':>5s} "
              f"{'base':>6s} {'AUROC_raw':>9s} {'AUROC_dir':>9s} "
              f"{'dir':>9s} {'AUPRC':>6s} {'+rand':>6s} {'rho':>6s} "
                  f"{'mw_p':>10s} {'d':>6s} {'delta':>7s}")
        for r in rows_t:
            mw = r["mann_whitney_p"]
            mw_s = mw if isinstance(mw, str) else \
                   ("<1e-12" if mw < 1e-12 else f"{mw:.2e}")
            d = r["cohens_d"]
            d_s = (f"{d:>6.3f}" if isinstance(d, float) else f"{d:>6}")
            delta = r["cliffs_delta"]
            delta_s = (f"{delta:>7.3f}" if isinstance(delta, float)
                       else f"{delta:>7}")
            print(f"{r['group']:28s} {r['feature']:16s} "
                  f"{r['n_frames']:>5d} "
                  f"{r['base_rate']:>6.3f} "
                  f"{r['auroc_raw']:>9.3f} "
                  f"{r['auroc_directional']:>9.3f} "
                  f"{r['direction']:>9s} "
                  f"{r['auprc']:>6.3f} "
                  f"{r['auprc_lift_over_random']:>+6.3f} "
                  f"{r['spearman_rho']:>+6.3f} "
                  f"{mw_s:>10s} "
                  f"{d_s} "
                  f"{delta_s}")

    print()
    print("=== TOP 3 FEATURES: benefit_positive ===")
    group_keys = sorted({
        (r.get("domain", ""), r.get("model_family", ""), r.get("pair_type", ""))
        for r in all_rows if r["target"] == "benefit_positive"
    })
    for dom, fam, pair_type in group_keys:
        rows_g = [
            r for r in all_rows
            if r["target"] == "benefit_positive"
            and r.get("domain", "") == dom
            and r.get("model_family", "") == fam
            and r.get("pair_type", "") == pair_type
        ]
        rows_g.sort(key=lambda r: r["auroc_directional"], reverse=True)
        print(f"{dom or '?'} / {fam or '?'} / {pair_type or '?'}")
        for r in rows_g[:3]:
            print(f"  {r['feature']}: AUROC_dir={r['auroc_directional']:.3f}, "
                  f"dir={r['direction']}, AUPRC={r['auprc']:.3f}, "
                  f"lift={r['auprc_lift_over_random']:+.3f}")

    out_csv = args.out_dir / "trigger_validity_all_stage3.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        required = [
            "domain", "model_family", "pair_type", "target", "feature",
            "base_rate", "auroc_raw", "auroc_directional", "direction",
            "auprc", "auprc_lift_over_random", "spearman",
            "mann_whitney_p", "cohens_d", "cliffs_delta",
        ]
        cols = required + [c for c in all_rows[0].keys() if c not in required]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    print(f"\n[trigger_validity] wrote {out_csv}")

    # One figure per target. Plot uses 'group' as the panel splitter
    # so 8 groups get 8 panel rows; we cap to focus on benefit_positive.
    for target in TARGET_NAMES:
        out_png = args.out_dir / f"fig_trigger_validity_{target}.png"
        plot_validity(all_rows, all_rocs, out_png, target=target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
