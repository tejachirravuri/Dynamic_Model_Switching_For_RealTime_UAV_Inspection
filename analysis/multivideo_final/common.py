"""Shared utilities for final multi-video DMS analysis.

These helpers operate on frozen sweep outputs only. They do not import or
modify experiment/inference code.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


EXPECTED_POLICIES = [
    "n_only",
    "s_only",
    "entropy_only",
    "combined",
    "combined_hyst",
    "conf_ema",
    "multi_proxy",
    "local_contrast_hyst",
]

POLICY_LABELS = {
    "n_only": "fast_only",
    "s_only": "accurate_only",
    "entropy_only": "entropy_only",
    "combined": "combined",
    "combined_hyst": "combined_hyst",
    "conf_ema": "conf_ema",
    "multi_proxy": "multi_proxy",
    "local_contrast_hyst": "local_contrast_hyst",
}

RUNS_PER_VIDEO = [
    ("yolov8", "n_s", "cuda"),
    ("yolov8", "n_l", "cuda"),
    ("yolov8", "n_l", "cpu"),
    ("yolo26", "n_s", "cuda"),
    ("yolo26", "n_l", "cuda"),
    ("yolo26", "n_l", "cpu"),
]

OUTPUT_COLUMNS = [
    "domain",
    "video_slug",
    "model_family",
    "pair_type",
    "device",
    "policy",
]

CRITICAL_TIMING_COLS = [
    "t_total_deployed_ms",
    "n_latency_ms",
    "s_latency_ms",
]

FEATURE_ALIASES = {
    "laplacian": ["laplacian", "L"],
    "entropy": ["entropy", "H"],
    "local_contrast": ["local_contrast", "contrast", "brightness_std"],
}

FEATURES = [
    "local_contrast",
    "edge_density",
    "tenengrad",
    "laplacian",
    "entropy",
    "color_entropy",
    "bright_fraction",
    "hue_std",
    "brightness_mean",
    "conf_drop",
    "fast_ema",
    "slow_ema",
]

TARGETS = [
    "benefit_positive",
    "fast_iou_match_failure",
    "fast_det_coverage_failure",
    "count_disagreement",
]


@dataclass(frozen=True)
class RunInfo:
    run_id: str
    domain: str
    video_slug: str
    model_family: str
    pair_type: str
    device: str
    run_dir: Path


def output_root() -> Path:
    return Path("analysis/figures/stage3/multivideo_final")


def final_results_root() -> Path:
    return Path("results/remote_pull/multi_video_full_gpu_cpu_nl_final")


def parse_run_id(run_id: str, run_dir: Path | None = None) -> RunInfo:
    parts = run_id.split("_")
    if len(parts) < 7 or parts[0] != "multi":
        raise ValueError(f"Unexpected run_id format: {run_id}")
    domain = parts[1]
    model_idx = next((i for i, p in enumerate(parts) if p in {"yolov8", "yolo26"}), None)
    if model_idx is None or model_idx + 3 >= len(parts):
        raise ValueError(f"Could not parse model/pair/device from {run_id}")
    video_slug = "_".join(parts[2:model_idx])
    model_family = parts[model_idx]
    pair_type = "_".join(parts[model_idx + 1 : model_idx + 3])
    device = parts[model_idx + 3]
    return RunInfo(run_id, domain, video_slug, model_family, pair_type, device, run_dir or Path())


def discover_runs(root: Path) -> list[RunInfo]:
    runs: list[RunInfo] = []
    for summary in sorted(root.rglob("sweep_summary.json")):
        run_dir = summary.parent
        run_id = run_dir.name
        info = parse_run_id(run_id, run_dir)
        runs.append(info)
    return runs


def read_summary(run: RunInfo) -> dict:
    return json.loads((run.run_dir / "sweep_summary.json").read_text(encoding="utf-8"))


def read_sweep(run: RunInfo, usecols: Iterable[str] | None = None) -> pd.DataFrame:
    return pd.read_csv(run.run_dir / "sweep_per_frame.csv", usecols=usecols)


def read_frame_features(run: RunInfo, usecols: Iterable[str] | None = None) -> pd.DataFrame:
    return pd.read_csv(run.run_dir / "frame_features.csv", usecols=usecols)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def reader_policy_label(policy: str) -> str:
    return POLICY_LABELS.get(policy, policy)


def safe_rate(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce")
    return float(values.mean()) if len(values) else math.nan


def f1_score(precision: float, recall: float) -> float:
    if not np.isfinite(precision) or not np.isfinite(recall):
        return math.nan
    denom = precision + recall
    return float(2 * precision * recall / denom) if denom else 0.0


def policy_trigger_metrics(df: pd.DataFrame, policy: str) -> dict[str, float]:
    rows = df[df["policy"] == policy]
    if rows.empty or "benefit_positive" not in rows or "s_choice" not in rows:
        return {"trigger_precision": math.nan, "trigger_recall": math.nan, "trigger_f1": math.nan}
    s = pd.to_numeric(rows["s_choice"], errors="coerce").fillna(0).astype(int)
    benefit = pd.to_numeric(rows["benefit_positive"], errors="coerce").fillna(0).astype(int)
    tp = int(((s == 1) & (benefit == 1)).sum())
    fp = int(((s == 1) & (benefit == 0)).sum())
    fn = int(((s == 0) & (benefit == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else math.nan
    recall = tp / (tp + fn) if (tp + fn) else math.nan
    return {
        "trigger_precision": precision,
        "trigger_recall": recall,
        "trigger_f1": f1_score(precision, recall),
    }


def expected_random_metrics(df: pd.DataFrame, policy: str) -> dict[str, float]:
    metrics = ["iou_match", "det_coverage", "det_recall", "count_agree"]
    out: dict[str, float] = {}
    if "policy" not in df or "s_choice" not in df:
        return {f"expected_random_{m}": math.nan for m in metrics}
    fast = df[df["policy"] == "n_only"].sort_values("frame_idx")
    accurate = df[df["policy"] == "s_only"].sort_values("frame_idx")
    rows = df[df["policy"] == policy]
    p = safe_rate(rows["s_choice"]) if "s_choice" in rows else math.nan
    for metric in metrics:
        if metric in fast and metric in accurate and np.isfinite(p):
            f = pd.to_numeric(fast[metric], errors="coerce").to_numpy()
            a = pd.to_numeric(accurate[metric], errors="coerce").to_numpy()
            n = min(len(f), len(a))
            out[f"expected_random_{metric}"] = float(np.nanmean((1 - p) * f[:n] + p * a[:n])) if n else math.nan
        else:
            out[f"expected_random_{metric}"] = math.nan
    return out


def add_fast_failure_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["fast_iou_match_failure"] = (pd.to_numeric(out.get("iou_match"), errors="coerce") < 1).astype(int)
    out["fast_det_coverage_failure"] = (pd.to_numeric(out.get("det_coverage"), errors="coerce") < 1).astype(int)
    out["count_disagreement"] = (pd.to_numeric(out.get("count_agree"), errors="coerce") < 1).astype(int)
    return out


def summarize_group(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    numeric = [
        "accurate_usage_rate",
        "iou_match",
        "det_coverage",
        "det_recall",
        "count_agree",
        "benefit_rate",
        "trigger_precision",
        "trigger_recall",
        "trigger_f1",
        "expected_random_iou_match",
        "informed_gain_over_random",
        "t_scene_policy_relevant_mean_ms",
        "t_ctrl_mean_ms",
        "t_selected_inference_mean_ms",
        "t_inference_deployed_mean_ms",
        "t_total_deployed_mean_ms",
        "t_total_deployed_p50_ms",
        "t_total_deployed_p95_ms",
        "t_total_deployed_p99_ms",
        "t_total_policy_relevant_mean_ms",
        "t_total_policy_relevant_p50_ms",
        "t_total_policy_relevant_p95_ms",
        "t_total_policy_relevant_p99_ms",
        "t_total_shared_proxy_mean_ms",
        "t_total_shared_proxy_p50_ms",
        "t_total_shared_proxy_p95_ms",
        "t_total_shared_proxy_p99_ms",
        "t_total_logged_deployed_mean_ms",
        "t_total_logged_deployed_p50_ms",
        "t_total_logged_deployed_p95_ms",
        "t_total_logged_deployed_p99_ms",
        "fps_deployed",
        "proxy_overhead_mean_ms",
        "proxy_policy_relevant_mean_ms",
        "proxy_shared_mean_ms",
        "fast_latency_mean_ms",
        "accurate_latency_mean_ms",
        "n_frames",
    ]
    agg = {c: "mean" for c in numeric if c in df.columns}
    return df.groupby(group_cols, dropna=False).agg(agg).reset_index()


def pvalue_mannwhitney(x0: np.ndarray, x1: np.ndarray) -> tuple[float, float]:
    try:
        from scipy.stats import mannwhitneyu

        res = mannwhitneyu(x1, x0, alternative="two-sided")
        return float(res.statistic), float(res.pvalue)
    except Exception:
        return math.nan, math.nan


def ks_statistic(x0: np.ndarray, x1: np.ndarray) -> float:
    try:
        from scipy.stats import ks_2samp

        return float(ks_2samp(x0, x1).statistic)
    except Exception:
        return math.nan


def auroc_score(y: np.ndarray, x: np.ndarray) -> float:
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y, x))
    except Exception:
        pos = x[y == 1]
        neg = x[y == 0]
        if len(pos) == 0 or len(neg) == 0:
            return math.nan
        u, _ = pvalue_mannwhitney(neg, pos)
        return float(u / (len(pos) * len(neg))) if np.isfinite(u) else math.nan


def auprc_score(y: np.ndarray, x: np.ndarray) -> float:
    try:
        from sklearn.metrics import average_precision_score

        return float(average_precision_score(y, x))
    except Exception:
        return math.nan


def spearman_score(y: np.ndarray, x: np.ndarray) -> float:
    try:
        from scipy.stats import spearmanr

        return float(spearmanr(x, y, nan_policy="omit").statistic)
    except Exception:
        return math.nan


def cliffs_delta(x0: np.ndarray, x1: np.ndarray) -> float:
    if len(x0) == 0 or len(x1) == 0:
        return math.nan
    u, _ = pvalue_mannwhitney(x0, x1)
    return float((2 * u / (len(x0) * len(x1))) - 1) if np.isfinite(u) else math.nan


def cohens_d(x0: np.ndarray, x1: np.ndarray) -> float:
    if len(x0) < 2 or len(x1) < 2:
        return math.nan
    v0 = np.nanvar(x0, ddof=1)
    v1 = np.nanvar(x1, ddof=1)
    pooled = math.sqrt(((len(x0) - 1) * v0 + (len(x1) - 1) * v1) / (len(x0) + len(x1) - 2))
    return float((np.nanmean(x1) - np.nanmean(x0)) / pooled) if pooled else math.nan


def validity_for_group(df: pd.DataFrame, group: dict[str, object], target: str, feature: str) -> dict[str, object] | None:
    if target not in df or feature not in df:
        return None
    sub = df[[target, feature]].copy()
    sub[target] = pd.to_numeric(sub[target], errors="coerce")
    sub[feature] = pd.to_numeric(sub[feature], errors="coerce")
    sub = sub.dropna()
    if sub.empty:
        return None
    y = sub[target].astype(int).to_numpy()
    x = sub[feature].to_numpy(dtype=float)
    base_rate = float(np.mean(y)) if len(y) else math.nan
    if len(np.unique(y)) < 2:
        raw_auc = math.nan
        direction = "undefined"
        auprc = math.nan
    else:
        raw_auc = auroc_score(y, x)
        direction = "positive" if raw_auc >= 0.5 else "negative"
        auprc = auprc_score(y, x if direction == "positive" else -x)
    x0 = x[y == 0]
    x1 = x[y == 1]
    return {
        **group,
        "target": target,
        "feature": feature,
        "base_rate": base_rate,
        "auroc_raw": raw_auc,
        "auroc_directional": max(raw_auc, 1 - raw_auc) if np.isfinite(raw_auc) else math.nan,
        "direction": direction,
        "auprc": auprc,
        "auprc_lift_over_random": auprc - base_rate if np.isfinite(auprc) else math.nan,
        "spearman": spearman_score(y, x) if len(np.unique(y)) >= 2 else math.nan,
        "cohens_d": cohens_d(x0, x1),
        "cliffs_delta": cliffs_delta(x0, x1),
        "mannwhitney_p": pvalue_mannwhitney(x0, x1)[1],
        "ks_statistic": ks_statistic(x0, x1),
        "n_frames": int(len(y)),
        "n_videos": int(df["video_slug"].nunique()) if "video_slug" in df else math.nan,
    }


def clean_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def canonicalize_feature_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """Collapse known duplicate feature aliases into one thesis-facing feature.

    The sweep logs both historical shorthand names (`L`, `H`) and descriptive
    names (`laplacian`, `entropy`). It also logs `contrast` and
    `brightness_std`, which equal `local_contrast` in the current feature
    implementation. Keeping all aliases in final validity plots would double
    count the same signal, so final analysis uses one canonical name.
    """
    out = df.copy()
    audit: list[dict[str, str]] = []
    for canonical, aliases in FEATURE_ALIASES.items():
        present = [c for c in aliases if c in out.columns]
        if not present:
            continue
        source = present[0]
        if canonical not in out.columns:
            out[canonical] = out[source]
        for alias in present:
            if alias == canonical:
                continue
            audit.append({"canonical_feature": canonical, "excluded_alias": alias, "reason": "exact_duplicate_or_legacy_alias"})
    return out, audit


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Render a small/medium DataFrame as a GitHub-style Markdown table.

    Avoids pandas' optional tabulate dependency so scripts run in the thesis
    virtual environment without installing anything.
    """
    if df.empty:
        return "None."
    cols = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for col in df.columns:
            val = row[col]
            if pd.isna(val):
                text = ""
            else:
                text = str(val)
            text = text.replace("|", "\\|").replace("\n", " ")
            vals.append(text)
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)
