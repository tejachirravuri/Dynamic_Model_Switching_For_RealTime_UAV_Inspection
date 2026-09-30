"""Evaluate Dynamic Model Selection on Anti-UAV YOLO-format data.

This script is intentionally separate from ``experiments/run_sweep.py``.
The insulator sweeps use accurate-model detections as the reference;
Anti-UAV has frame-level human ground-truth boxes, so this evaluator
reports GT metrics directly while reusing the same DMS policy and proxy
implementations.

Outputs, under ``--out-dir/--run-id``:
    * ``anti_uav_per_image.csv``: one row per image/policy
    * ``anti_uav_policy_summary.csv``: thesis-facing policy summary
    * ``anti_uav_feature_validity.csv``: feature association with
      ``benefit_positive_gt``
    * ``anti_uav_run_summary.json``: compact metadata and paths
    * optional PNG figures

Example:
    python -m analysis.anti_uav.evaluate_dms \
      --images-dir /home/gchi/datasets/anti_uav_yolo/visible_stride10/images/val \
      --labels-dir /home/gchi/datasets/anti_uav_yolo/visible_stride10/labels/val \
      --fast-weights /home/gchi/training_runs_anti_uav/yolov8n_visible_stride10_e50/weights/best.pt \
      --accurate-weights /home/gchi/training_runs_anti_uav/yolov8l_visible_stride10_e50/weights/best.pt \
      --model-family yolov8 --pair-type n_l --device cuda --max-images 1000
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from dms.inference import Detections, UltralyticsYOLOBackend, time_inference
from dms.matching import greedy_match
from dms.policies import FrameFeatures, PolicyConfig, make_policy


DEFAULT_POLICIES = (
    "n_only",
    "s_only",
    "entropy_only",
    "combined",
    "combined_hyst",
    "conf_ema",
    "multi_proxy",
    "local_contrast_hyst",
)

READER_POLICY_LABEL = {
    "n_only": "fast_only",
    "s_only": "accurate_only",
}

PROXY_FEATURES = (
    "L",
    "H",
    "color_entropy",
    "tenengrad",
    "edge_density",
    "local_contrast",
    "bright_fraction",
    "hue_std",
    "brightness_mean",
)

FEATURE_LABELS = {
    "L": "laplacian",
    "H": "entropy",
    "local_contrast": "local_contrast",
    "edge_density": "edge_density",
    "tenengrad": "tenengrad",
    "color_entropy": "color_entropy",
    "bright_fraction": "bright_fraction",
    "hue_std": "hue_std",
    "brightness_mean": "brightness_mean",
    "conf_drop": "conf_drop",
}

SCENE_POLICY_PROXY_FEATURES = {
    "entropy_only": ("H",),
    "combined": ("L", "H"),
    "combined_hyst": ("L", "H"),
    "local_contrast_hyst": ("local_contrast",),
    "multi_proxy": ("L", "H", "tenengrad", "color_entropy"),
}


@dataclass(frozen=True)
class ImageItem:
    image_path: Path
    label_path: Path


@dataclass
class GTMetrics:
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    iou_match_gt: float
    det_coverage_gt: float
    count_agree_gt: float
    matched_iou_mean: float


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate DMS policies on Anti-UAV YOLO-format validation data."
    )
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--labels-dir", required=True, type=Path)
    parser.add_argument("--fast-weights", required=True, type=Path)
    parser.add_argument("--accurate-weights", required=True, type=Path)
    parser.add_argument("--model-family", default="yolov8")
    parser.add_argument("--pair-type", default="n_l")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--imgsz", default=640, type=int)
    parser.add_argument("--conf-floor", default=0.25, type=float)
    parser.add_argument("--iou-threshold", default=0.5, type=float)
    parser.add_argument("--iou-nms", default=0.7, type=float)
    parser.add_argument("--proxy-size", default=160, type=int)
    parser.add_argument("--preserve-aspect", action="store_true")
    parser.add_argument("--max-images", default=0, type=int)
    parser.add_argument("--sample-seed", default=0, type=int)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--out-dir", default=Path("results/anti_uav_dms"), type=Path)
    parser.add_argument("--policies", nargs="+", default=list(DEFAULT_POLICIES))
    parser.add_argument("--warmup-images", default=3, type=int)
    parser.add_argument("--conf-ema-c-low", default=0.007, type=float)
    parser.add_argument("--conf-ema-c-high", default=0.02, type=float)
    parser.add_argument("--local-contrast-low", default=0.45, type=float)
    parser.add_argument("--local-contrast-high", default=0.65, type=float)
    parser.add_argument("--no-figures", action="store_true")
    return parser.parse_args(argv)


def read_yolo_labels(label_path: Path, width: int, height: int) -> Detections:
    boxes: List[List[float]] = []
    classes: List[int] = []
    if not label_path.exists():
        return Detections.empty()
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        xc, yc, bw, bh = map(float, parts[1:5])
        x1 = (xc - bw / 2.0) * width
        y1 = (yc - bh / 2.0) * height
        x2 = (xc + bw / 2.0) * width
        y2 = (yc + bh / 2.0) * height
        boxes.append([
            max(0.0, min(float(width), x1)),
            max(0.0, min(float(height), y1)),
            max(0.0, min(float(width), x2)),
            max(0.0, min(float(height), y2)),
        ])
        classes.append(cls)
    if not boxes:
        return Detections.empty()
    return Detections(
        boxes=np.asarray(boxes, dtype=float),
        scores=np.ones((len(boxes),), dtype=float),
        classes=np.asarray(classes, dtype=int),
    )


def discover_items(images_dir: Path, labels_dir: Path, max_images: int, seed: int) -> List[ImageItem]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    images = sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in exts)
    if not images:
        raise FileNotFoundError(f"no images found under {images_dir}")
    items = [ImageItem(p, labels_dir / f"{p.stem}.txt") for p in images]
    if max_images and max_images > 0 and max_images < len(items):
        rng = random.Random(seed)
        items = sorted(rng.sample(items, max_images), key=lambda it: str(it.image_path))
    return items


def mean_conf(dets: Detections) -> float:
    return float(dets.scores.mean()) if len(dets) else 0.0


def gt_metrics(dets: Detections, gt: Detections, iou_threshold: float) -> GTMetrics:
    match = greedy_match(
        dets.boxes,
        gt.boxes,
        iou_threshold=iou_threshold,
        classes_a=dets.classes,
        classes_b=gt.classes,
    )
    tp = len(match.pairs)
    fp = len(match.unmatched_a)
    fn = len(match.unmatched_b)
    precision = tp / (tp + fp) if (tp + fp) else (1.0 if len(gt) == 0 else 0.0)
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    iou_match_gt = 1.0 if fp == 0 and fn == 0 else 0.0
    det_coverage_gt = tp / len(gt) if len(gt) else (1.0 if len(dets) == 0 else 0.0)
    count_agree_gt = 1.0 if len(dets) == len(gt) else 0.0
    matched_iou_mean = float(np.mean([m.iou for m in match.pairs])) if match.pairs else 0.0
    return GTMetrics(
        tp=tp,
        fp=fp,
        fn=fn,
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        iou_match_gt=float(iou_match_gt),
        det_coverage_gt=float(det_coverage_gt),
        count_agree_gt=float(count_agree_gt),
        matched_iou_mean=float(matched_iou_mean),
    )


def build_features(frame_idx: int, proxies: Dict[str, float], last_fast_mean_conf: float) -> FrameFeatures:
    return FrameFeatures(
        frame_idx=frame_idx,
        L=proxies.get("L"),
        H=proxies.get("H"),
        color_entropy=proxies.get("color_entropy"),
        tenengrad=proxies.get("tenengrad"),
        edge_density=proxies.get("edge_density"),
        local_contrast=proxies.get("local_contrast"),
        bright_fraction=proxies.get("bright_fraction"),
        hue_std=proxies.get("hue_std"),
        last_mean_conf=float(last_fast_mean_conf),
    )


def policy_relevant_proxy_ms(policy_name: str, proxies: Dict[str, float]) -> float:
    feats = SCENE_POLICY_PROXY_FEATURES.get(policy_name)
    if not feats:
        return 0.0
    vals = [float(proxies.get("time_overhead_ms", math.nan))]
    vals.extend(float(proxies.get(f"time_{f}_ms", math.nan)) for f in feats)
    if any(math.isnan(v) for v in vals):
        return math.nan
    return float(sum(vals))


def deployed_runtime_ms(
    policy_name: str,
    choice: str,
    n_ms: float,
    s_ms: float,
    proxy_rel_ms: float,
    decision_ms: float,
) -> Tuple[float, float]:
    if policy_name == "n_only":
        return 0.0, float(n_ms)
    if policy_name == "s_only":
        return 0.0, float(s_ms)
    if policy_name == "conf_ema":
        infer = float(n_ms + (s_ms if choice == "s" else 0.0))
        return infer, float(infer + decision_ms)
    infer = float(n_ms if choice == "n" else s_ms)
    return infer, float(proxy_rel_ms + decision_ms + infer)


def safe_mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if not math.isnan(float(v))]
    return float(sum(vals) / len(vals)) if vals else math.nan


def safe_quantile(values: Iterable[float], q: float) -> float:
    vals = sorted(float(v) for v in values if not math.isnan(float(v)))
    if not vals:
        return math.nan
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return float(vals[lo] * (hi - pos) + vals[hi] * (pos - lo))


def trigger_scores(rows: List[Dict[str, object]], policy: str) -> Tuple[float, float, float]:
    subset = [r for r in rows if r["policy"] == policy]
    tp = sum(1 for r in subset if int(r["s_choice"]) == 1 and int(r["benefit_positive_gt"]) == 1)
    fp = sum(1 for r in subset if int(r["s_choice"]) == 1 and int(r["benefit_positive_gt"]) == 0)
    fn = sum(1 for r in subset if int(r["s_choice"]) == 0 and int(r["benefit_positive_gt"]) == 1)
    precision = tp / (tp + fp) if (tp + fp) else math.nan
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision == precision and (precision + recall) else math.nan
    return float(precision), float(recall), float(f1)


def ap_from_pr(recall: np.ndarray, precision: np.ndarray) -> float:
    if recall.size == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def average_precision(
    preds: List[Dict[str, object]],
    gt_by_image: Dict[int, Detections],
    iou_threshold: float,
) -> float:
    total_gt = int(sum(len(gt) for gt in gt_by_image.values()))
    if total_gt == 0:
        return math.nan
    preds_sorted = sorted(preds, key=lambda p: float(p["score"]), reverse=True)
    matched: Dict[int, set] = {idx: set() for idx in gt_by_image}
    tp = np.zeros((len(preds_sorted),), dtype=float)
    fp = np.zeros((len(preds_sorted),), dtype=float)
    for i, pred in enumerate(preds_sorted):
        image_idx = int(pred["image_idx"])
        gt = gt_by_image.get(image_idx, Detections.empty())
        if len(gt) == 0:
            fp[i] = 1.0
            continue
        box = np.asarray(pred["box"], dtype=float).reshape(1, 4)
        cls = int(pred["class"])
        match = greedy_match(
            box,
            gt.boxes,
            iou_threshold=iou_threshold,
            classes_a=np.asarray([cls], dtype=int),
            classes_b=gt.classes,
        )
        best = None
        for pair in match.pairs:
            if pair.b_idx not in matched[image_idx]:
                best = pair
                break
        if best is None:
            fp[i] = 1.0
        else:
            tp[i] = 1.0
            matched[image_idx].add(best.b_idx)
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recall = cum_tp / max(total_gt, 1)
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)
    return ap_from_pr(recall, precision)


def map_metrics(preds_by_policy: Dict[str, List[Dict[str, object]]], gt_by_image: Dict[int, Detections]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    thresholds = [0.50 + 0.05 * i for i in range(10)]
    for policy, preds in preds_by_policy.items():
        aps = [average_precision(preds, gt_by_image, thr) for thr in thresholds]
        out[policy] = {
            "mAP50": float(aps[0]),
            "mAP50_95": float(safe_mean(aps)),
        }
    return out


def roc_auc_manual(y: np.ndarray, x: np.ndarray) -> float:
    pos = x[y == 1]
    neg = x[y == 0]
    if pos.size == 0 or neg.size == 0:
        return math.nan
    vals = np.concatenate([pos, neg])
    order = np.argsort(vals)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, vals.size + 1)
    # Average tied ranks.
    uniq, inv, counts = np.unique(vals, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        for k, count in enumerate(counts):
            if count > 1:
                ranks[inv == k] = ranks[inv == k].mean()
    rank_sum_pos = ranks[:pos.size].sum()
    auc = (rank_sum_pos - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size)
    return float(auc)


def auprc_manual(y: np.ndarray, x: np.ndarray) -> float:
    if y.sum() == 0:
        return math.nan
    order = np.argsort(-x)
    ys = y[order]
    tp = np.cumsum(ys == 1)
    fp = np.cumsum(ys == 0)
    recall = tp / max(int(y.sum()), 1)
    precision = tp / np.maximum(tp + fp, 1e-12)
    return ap_from_pr(recall.astype(float), precision.astype(float))


def spearman_manual(y: np.ndarray, x: np.ndarray) -> float:
    if x.size < 2 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return math.nan
    xr = np.argsort(np.argsort(x)).astype(float)
    yr = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(xr, yr)[0, 1])


def cohens_d(y: np.ndarray, x: np.ndarray) -> float:
    pos = x[y == 1]
    neg = x[y == 0]
    if pos.size < 2 or neg.size < 2:
        return math.nan
    pooled = math.sqrt(((pos.size - 1) * pos.var(ddof=1) + (neg.size - 1) * neg.var(ddof=1)) / (pos.size + neg.size - 2))
    return float((pos.mean() - neg.mean()) / pooled) if pooled > 0 else math.nan


def cliffs_delta(y: np.ndarray, x: np.ndarray) -> float:
    pos = x[y == 1]
    neg = x[y == 0]
    if pos.size == 0 or neg.size == 0:
        return math.nan
    gt = sum(float(p > n) for p in pos for n in neg)
    lt = sum(float(p < n) for p in pos for n in neg)
    return float((gt - lt) / (pos.size * neg.size))


def feature_validity(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    # One feature row per image is enough; policy rows repeat the same image features.
    by_image: Dict[int, Dict[str, object]] = {}
    conf_drop_by_image: Dict[int, float] = {}
    for row in rows:
        idx = int(row["image_idx"])
        if idx not in by_image:
            by_image[idx] = dict(row)
        if row["policy"] == "conf_ema":
            conf_drop_by_image[idx] = float(row.get("conf_drop", math.nan))
    image_rows = []
    for idx in sorted(by_image):
        merged = dict(by_image[idx])
        merged["conf_drop"] = conf_drop_by_image.get(idx, math.nan)
        image_rows.append(merged)
    if not image_rows:
        return []
    y = np.asarray([int(r["benefit_positive_gt"]) for r in image_rows], dtype=int)
    base_rate = float(y.mean()) if y.size else math.nan
    out: List[Dict[str, object]] = []
    for feature in list(PROXY_FEATURES) + ["conf_drop"]:
        if feature not in image_rows[0]:
            continue
        x = np.asarray([float(r.get(feature, math.nan)) for r in image_rows], dtype=float)
        keep = np.isfinite(x)
        if keep.sum() < 20 or len(np.unique(y[keep])) < 2 or len(np.unique(x[keep])) < 2:
            continue
        yy = y[keep]
        xx = x[keep]
        auc = roc_auc_manual(yy, xx)
        auprc = auprc_manual(yy, xx)
        out.append({
            "feature": feature,
            "reader_feature_label": FEATURE_LABELS.get(feature, feature),
            "target": "benefit_positive_gt",
            "base_rate": base_rate,
            "auroc_raw": auc,
            "auroc_directional": max(auc, 1.0 - auc) if auc == auc else math.nan,
            "direction": "positive" if auc >= 0.5 else "negative",
            "auprc": auprc,
            "auprc_lift_over_base_rate": auprc - base_rate if auprc == auprc else math.nan,
            "spearman_rho": spearman_manual(yy, xx),
            "cohens_d": cohens_d(yy, xx),
            "cliffs_delta": cliffs_delta(yy, xx),
            "n_images": int(keep.sum()),
            "positive_images": int(yy.sum()),
        })
    out.sort(key=lambda r: (
        -float(r["auprc_lift_over_base_rate"]) if r["auprc_lift_over_base_rate"] == r["auprc_lift_over_base_rate"] else 0.0,
        -float(r["auroc_directional"]) if r["auroc_directional"] == r["auroc_directional"] else 0.0,
    ))
    return out


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarise_policies(rows: List[Dict[str, object]], preds_by_policy: Dict[str, List[Dict[str, object]]], gt_by_image: Dict[int, Detections]) -> List[Dict[str, object]]:
    map_by_policy = map_metrics(preds_by_policy, gt_by_image)
    fast_rows = [r for r in rows if r["policy"] == "n_only"]
    acc_rows = [r for r in rows if r["policy"] == "s_only"]
    fast_f1 = np.asarray([float(r["gt_f1"]) for r in fast_rows])
    acc_f1 = np.asarray([float(r["gt_f1"]) for r in acc_rows])
    fast_iou = np.asarray([float(r["iou_match_gt"]) for r in fast_rows])
    acc_iou = np.asarray([float(r["iou_match_gt"]) for r in acc_rows])

    out: List[Dict[str, object]] = []
    policies = list(dict.fromkeys(str(r["policy"]) for r in rows))
    for policy in policies:
        subset = [r for r in rows if r["policy"] == policy]
        p = safe_mean(float(r["s_choice"]) for r in subset)
        precision_tr, recall_tr, f1_tr = trigger_scores(rows, policy)
        tp = sum(int(r["gt_tp"]) for r in subset)
        fp = sum(int(r["gt_fp"]) for r in subset)
        fn = sum(int(r["gt_fn"]) for r in subset)
        micro_p = tp / (tp + fp) if (tp + fp) else math.nan
        micro_r = tp / (tp + fn) if (tp + fn) else math.nan
        micro_f1 = 2.0 * micro_p * micro_r / (micro_p + micro_r) if micro_p == micro_p and micro_r == micro_r and (micro_p + micro_r) else math.nan
        expected_f1 = float(np.mean((1.0 - p) * fast_f1 + p * acc_f1))
        expected_iou = float(np.mean((1.0 - p) * fast_iou + p * acc_iou))
        t_total = [float(r["t_total_policy_relevant_ms"]) for r in subset]
        out.append({
            "policy": policy,
            "reader_policy_label": READER_POLICY_LABEL.get(policy, policy),
            "n_images": len(subset),
            "accurate_usage_rate": p,
            "benefit_rate_gt": safe_mean(float(r["benefit_positive_gt"]) for r in subset),
            "gt_precision_macro": safe_mean(float(r["gt_precision"]) for r in subset),
            "gt_recall_macro": safe_mean(float(r["gt_recall"]) for r in subset),
            "gt_f1_macro": safe_mean(float(r["gt_f1"]) for r in subset),
            "gt_precision_micro": micro_p,
            "gt_recall_micro": micro_r,
            "gt_f1_micro": micro_f1,
            "mAP50": map_by_policy.get(policy, {}).get("mAP50", math.nan),
            "mAP50_95": map_by_policy.get(policy, {}).get("mAP50_95", math.nan),
            "iou_match_gt_rate": safe_mean(float(r["iou_match_gt"]) for r in subset),
            "det_coverage_gt_rate": safe_mean(float(r["det_coverage_gt"]) for r in subset),
            "count_agree_gt_rate": safe_mean(float(r["count_agree_gt"]) for r in subset),
            "trigger_precision": precision_tr,
            "trigger_recall": recall_tr,
            "trigger_f1": f1_tr,
            "expected_random_gt_f1_at_same_usage": expected_f1,
            "informed_gt_f1_gain_over_random": safe_mean(float(r["gt_f1"]) for r in subset) - expected_f1,
            "expected_random_iou_match_gt_at_same_usage": expected_iou,
            "informed_iou_match_gt_gain_over_random": safe_mean(float(r["iou_match_gt"]) for r in subset) - expected_iou,
            "t_scene_policy_relevant_mean_ms": safe_mean(float(r["t_scene_policy_relevant_ms"]) for r in subset),
            "t_ctrl_mean_ms": safe_mean(float(r["decision_ms"]) for r in subset),
            "t_inference_deployed_mean_ms": safe_mean(float(r["t_inference_deployed_ms"]) for r in subset),
            "t_total_policy_relevant_mean_ms": safe_mean(t_total),
            "t_total_policy_relevant_p50_ms": safe_quantile(t_total, 0.50),
            "t_total_policy_relevant_p95_ms": safe_quantile(t_total, 0.95),
            "fps_policy_relevant": 1000.0 / safe_mean(t_total) if safe_mean(t_total) > 0 else math.nan,
            "fast_latency_mean_ms": safe_mean(float(r["n_latency_ms"]) for r in subset),
            "accurate_latency_mean_ms": safe_mean(float(r["s_latency_ms"]) for r in subset),
        })
    return out


def add_prediction_rows(
    pred_store: List[Dict[str, object]],
    image_idx: int,
    dets: Detections,
) -> None:
    for box, score, cls in zip(dets.boxes, dets.scores, dets.classes):
        pred_store.append({
            "image_idx": int(image_idx),
            "box": [float(v) for v in box],
            "score": float(score),
            "class": int(cls),
        })


def plot_outputs(out_dir: Path, summary_rows: List[Dict[str, object]], feature_rows: List[Dict[str, object]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"[anti_uav_dms] skipping figures: matplotlib unavailable ({exc})", file=sys.stderr)
        return

    if summary_rows:
        labels = [str(r["reader_policy_label"]) for r in summary_rows]
        x = [float(r["t_total_policy_relevant_mean_ms"]) for r in summary_rows]
        y = [float(r["gt_f1_macro"]) for r in summary_rows]
        plt.figure(figsize=(9, 5.5))
        plt.scatter(x, y, s=72)
        for label, xx, yy in zip(labels, x, y):
            plt.annotate(label, (xx, yy), xytext=(4, 4), textcoords="offset points", fontsize=8)
        plt.xlabel("Policy-relevant deployed runtime, T_total (ms/image)")
        plt.ylabel("Macro F1 against Anti-UAV human GT")
        plt.title("Anti-UAV DMS Runtime-Quality Tradeoff")
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(out_dir / "fig_anti_uav_runtime_quality.png", dpi=180)
        plt.close()

        dyn = [r for r in summary_rows if r["policy"] not in ("n_only", "s_only")]
        plt.figure(figsize=(9, 4.8))
        plt.bar([str(r["reader_policy_label"]) for r in dyn], [float(r["trigger_f1"]) for r in dyn])
        plt.ylabel("Trigger F1 vs benefit_positive_gt")
        plt.title("Anti-UAV Trigger Precision/Recall Balance")
        plt.xticks(rotation=25, ha="right")
        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(out_dir / "fig_anti_uav_trigger_f1.png", dpi=180)
        plt.close()

    if feature_rows:
        top = feature_rows[:8]
        plt.figure(figsize=(9, 4.8))
        plt.bar([str(r["reader_feature_label"]) for r in top], [float(r["auprc_lift_over_base_rate"]) for r in top])
        plt.ylabel("AUPRC lift over benefit base rate")
        plt.title("Image Feature Association with Accurate-Model Benefit")
        plt.xticks(rotation=25, ha="right")
        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(out_dir / "fig_anti_uav_feature_validity.png", dpi=180)
        plt.close()


def run(args: argparse.Namespace) -> Path:
    import cv2
    from dms.proxies import ProxyConfig, compute_proxies

    run_id = args.run_id or f"anti_uav_{args.model_family}_{args.pair_type}_{args.device}"
    out_dir = args.out_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    items = discover_items(args.images_dir, args.labels_dir, args.max_images, args.sample_seed)
    print(f"[anti_uav_dms] run_id={run_id}")
    print(f"[anti_uav_dms] images={len(items)}")
    print(f"[anti_uav_dms] loading fast:     {args.fast_weights}")
    print(f"[anti_uav_dms] loading accurate: {args.accurate_weights}")

    fast_backend = UltralyticsYOLOBackend(
        str(args.fast_weights), device=args.device, imgsz=args.imgsz,
        conf=args.conf_floor, iou_nms=args.iou_nms,
    )
    accurate_backend = UltralyticsYOLOBackend(
        str(args.accurate_weights), device=args.device, imgsz=args.imgsz,
        conf=args.conf_floor, iou_nms=args.iou_nms,
    )
    proxy_cfg = ProxyConfig(
        size=args.proxy_size,
        preserve_aspect=bool(args.preserve_aspect),
        features=PROXY_FEATURES,
    )
    policy_cfg = PolicyConfig(
        conf_ema_c_low=args.conf_ema_c_low,
        conf_ema_c_high=args.conf_ema_c_high,
        local_contrast_low=args.local_contrast_low,
        local_contrast_high=args.local_contrast_high,
    )
    policies = {name: make_policy(name, policy_cfg) for name in args.policies}

    # Light warmup on the first image avoids counting first-call CUDA setup in the run.
    first_img = cv2.imread(str(items[0].image_path), cv2.IMREAD_COLOR)
    if first_img is None:
        raise RuntimeError(f"failed to read first image: {items[0].image_path}")
    for _ in range(max(0, args.warmup_images)):
        fast_backend.predict(first_img)
        accurate_backend.predict(first_img)

    per_rows: List[Dict[str, object]] = []
    gt_by_image: Dict[int, Detections] = {}
    preds_by_policy: Dict[str, List[Dict[str, object]]] = {name: [] for name in args.policies}
    last_fast_mean_conf = 0.0

    t_run0 = time.perf_counter()
    for image_idx, item in enumerate(items, start=1):
        frame = cv2.imread(str(item.image_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"failed to read image: {item.image_path}")
        h, w = frame.shape[:2]
        gt = read_yolo_labels(item.label_path, w, h)
        gt_by_image[image_idx] = gt

        proxies = compute_proxies(frame, proxy_cfg)
        fast_dets_raw, n_ms = time_inference(fast_backend, frame)
        accurate_dets_raw, s_ms = time_inference(accurate_backend, frame)
        fast_dets = fast_dets_raw.filter_score(args.conf_floor)
        accurate_dets = accurate_dets_raw.filter_score(args.conf_floor)

        fast_m = gt_metrics(fast_dets, gt, args.iou_threshold)
        accurate_m = gt_metrics(accurate_dets, gt, args.iou_threshold)
        benefit_positive_gt = int(accurate_m.f1 > fast_m.f1 + 1e-9)
        feat = build_features(image_idx, proxies, last_fast_mean_conf)

        for policy_name, policy in policies.items():
            if policy_name == "n_only":
                choice = "n"
                decision_ms = 0.0
            elif policy_name == "s_only":
                choice = "s"
                decision_ms = 0.0
            else:
                t0 = time.perf_counter()
                choice = policy.decide(feat)
                decision_ms = float((time.perf_counter() - t0) * 1000.0)
            chosen = fast_dets if choice == "n" else accurate_dets
            chosen_metrics = gt_metrics(chosen, gt, args.iou_threshold)
            proxy_rel = policy_relevant_proxy_ms(policy_name, proxies)
            infer_deployed, total_deployed = deployed_runtime_ms(
                policy_name, choice, float(n_ms), float(s_ms),
                float(proxy_rel), float(decision_ms),
            )
            diag = dict(getattr(policy, "last_diag", {}))
            add_prediction_rows(preds_by_policy[policy_name], image_idx, chosen)
            row: Dict[str, object] = {
                "run_id": run_id,
                "dataset": "anti_uav_visible_stride10",
                "model_family": args.model_family,
                "pair_type": args.pair_type,
                "device": args.device,
                "image_idx": image_idx,
                "image_file": str(item.image_path.name),
                "label_file": str(item.label_path.name),
                "width": int(w),
                "height": int(h),
                "policy": policy_name,
                "reader_policy_label": READER_POLICY_LABEL.get(policy_name, policy_name),
                "choice": choice,
                "s_choice": int(choice == "s"),
                "n_count": len(fast_dets),
                "s_count": len(accurate_dets),
                "gt_count": len(gt),
                "benefit_positive_gt": benefit_positive_gt,
                "n_gt_f1": fast_m.f1,
                "s_gt_f1": accurate_m.f1,
                "gt_tp": chosen_metrics.tp,
                "gt_fp": chosen_metrics.fp,
                "gt_fn": chosen_metrics.fn,
                "gt_precision": chosen_metrics.precision,
                "gt_recall": chosen_metrics.recall,
                "gt_f1": chosen_metrics.f1,
                "iou_match_gt": chosen_metrics.iou_match_gt,
                "det_coverage_gt": chosen_metrics.det_coverage_gt,
                "count_agree_gt": chosen_metrics.count_agree_gt,
                "matched_iou_mean_gt": chosen_metrics.matched_iou_mean,
                "n_latency_ms": float(n_ms),
                "s_latency_ms": float(s_ms),
                "decision_ms": decision_ms,
                "t_scene_policy_relevant_ms": float(proxy_rel),
                "t_inference_deployed_ms": infer_deployed,
                "t_total_policy_relevant_ms": total_deployed,
                "proxy_ms_total": float(proxies.get("time_ms", math.nan)),
                "proxy_ms_overhead": float(proxies.get("time_overhead_ms", math.nan)),
                "policy_score": float(diag.get("policy_score", math.nan)),
                "policy_thresh_low": float(diag.get("policy_thresh_low", math.nan)),
                "policy_thresh_high": float(diag.get("policy_thresh_high", math.nan)),
                "policy_thresh_mid": float(diag.get("policy_thresh_mid", math.nan)),
                "fast_ema": float(diag.get("fast_ema", math.nan)),
                "slow_ema": float(diag.get("slow_ema", math.nan)),
                "conf_drop": float(diag.get("conf_drop", math.nan)),
            }
            for f in PROXY_FEATURES:
                row[f] = float(proxies.get(f, math.nan))
                row[f"proxy_ms_{f}"] = float(proxies.get(f"time_{f}_ms", math.nan))
            per_rows.append(row)

        last_fast_mean_conf = mean_conf(fast_dets)
        if image_idx == 1 or image_idx % 100 == 0 or image_idx == len(items):
            elapsed = time.perf_counter() - t_run0
            print(
                f"  image {image_idx:5d}/{len(items)} "
                f"n={n_ms:6.2f}ms s={s_ms:6.2f}ms "
                f"|n|={len(fast_dets)} |s|={len(accurate_dets)} "
                f"gt={len(gt)} benefit_gt={benefit_positive_gt} "
                f"[wall {elapsed:.1f}s]"
            )

    policy_summary = summarise_policies(per_rows, preds_by_policy, gt_by_image)
    feature_rows = feature_validity(per_rows)

    per_path = out_dir / "anti_uav_per_image.csv"
    summary_path = out_dir / "anti_uav_policy_summary.csv"
    feature_path = out_dir / "anti_uav_feature_validity.csv"
    write_csv(per_path, per_rows)
    write_csv(summary_path, policy_summary)
    write_csv(feature_path, feature_rows)

    if not args.no_figures:
        plot_outputs(out_dir, policy_summary, feature_rows)

    run_summary = {
        "run_id": run_id,
        "dataset": "anti_uav_visible_stride10",
        "images_dir": str(args.images_dir),
        "labels_dir": str(args.labels_dir),
        "fast_weights": str(args.fast_weights),
        "accurate_weights": str(args.accurate_weights),
        "model_family": args.model_family,
        "pair_type": args.pair_type,
        "device": args.device,
        "n_images": len(items),
        "policies": list(args.policies),
        "conf_floor": args.conf_floor,
        "iou_threshold": args.iou_threshold,
        "runtime_semantics": {
            "fast_only": "T_fast",
            "accurate_only": "T_accurate",
            "scene_policies": "T_scene(policy-required proxy features) + T_ctrl + T_selected",
            "conf_ema": "T_fast + T_ctrl + I[accurate] * T_accurate",
        },
        "outputs": {
            "per_image": str(per_path),
            "policy_summary": str(summary_path),
            "feature_validity": str(feature_path),
        },
    }
    (out_dir / "anti_uav_run_summary.json").write_text(json.dumps(run_summary, indent=2))

    print("[anti_uav_dms] done")
    print(f"  per-image CSV:      {per_path}")
    print(f"  policy summary CSV: {summary_path}")
    print(f"  feature validity:   {feature_path}")
    return out_dir


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
