"""Selected-video deep-dive package for thesis presentation.

This script consumes frozen final sweep outputs and creates a compact,
presentation-oriented evidence package for two glass and two porcelain videos.
It does not modify experiment or inference code.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import discover_runs, ensure_dir, final_results_root, output_root, reader_policy_label


SELECTED_VIDEOS = {
    "glass": ["glass_ins", "161_YUN_0001_96"],
    "porcelain": ["UAV_porcelain", "porcelain_maybe"],
}

MODEL_FAMILIES = ["yolov8", "yolo26"]
DEVICE = "cpu"
PAIR_TYPE = "n_l"

POLICY_ORDER = [
    "n_only",
    "s_only",
    "conf_ema",
    "entropy_only",
    "combined",
    "combined_hyst",
    "local_contrast_hyst",
    "multi_proxy",
]

POLICY_LABEL = {
    "n_only": "Fast only",
    "s_only": "Accurate only",
    "entropy_only": "Entropy",
    "combined": "Combined",
    "combined_hyst": "Combined + hyst.",
    "conf_ema": "Confidence EMA",
    "multi_proxy": "Naive multi-proxy",
    "local_contrast_hyst": "Local contrast hyst.",
}

COLORS = {
    "n_only": "#31688E",
    "s_only": "#F28E2B",
    "entropy_only": "#59A14F",
    "combined": "#7B5EA7",
    "combined_hyst": "#8C6D31",
    "conf_ema": "#D62728",
    "multi_proxy": "#76B7B2",
    "local_contrast_hyst": "#D6A100",
}

FEATURES = [
    "local_contrast",
    "edge_density",
    "laplacian",
    "tenengrad",
    "entropy",
    "color_entropy",
    "bright_fraction",
    "hue_std",
    "brightness_mean",
]

STRUCTURAL_FEATURES = ["local_contrast", "edge_density", "laplacian", "tenengrad"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=final_results_root())
    parser.add_argument("--output-root", type=Path, default=output_root() / "selected_video_deep_dive")
    return parser.parse_args()


def safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def f1(precision: float, recall: float) -> float:
    if not np.isfinite(precision) or not np.isfinite(recall):
        return math.nan
    denom = precision + recall
    return float(2 * precision * recall / denom) if denom else 0.0


def trigger_metrics(rows: pd.DataFrame) -> tuple[float, float, float]:
    s = safe_num(rows["s_choice"]).fillna(0).astype(int)
    b = safe_num(rows["benefit_positive"]).fillna(0).astype(int)
    tp = int(((s == 1) & (b == 1)).sum())
    fp = int(((s == 1) & (b == 0)).sum())
    fn = int(((s == 0) & (b == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else math.nan
    recall = tp / (tp + fn) if (tp + fn) else math.nan
    return precision, recall, f1(precision, recall)


def expected_random_iou(df: pd.DataFrame, policy: str) -> float:
    rows = df[df["policy"] == policy]
    p = float(safe_num(rows["s_choice"]).mean()) if len(rows) else math.nan
    fast = df[df["policy"] == "n_only"].sort_values("frame_idx")
    acc = df[df["policy"] == "s_only"].sort_values("frame_idx")
    if not np.isfinite(p) or fast.empty or acc.empty:
        return math.nan
    f = safe_num(fast["iou_match"]).to_numpy()
    a = safe_num(acc["iou_match"]).to_numpy()
    n = min(len(f), len(a))
    return float(np.nanmean((1 - p) * f[:n] + p * a[:n])) if n else math.nan


def deployed_inference(rows: pd.DataFrame, policy: str) -> pd.Series:
    if policy == "conf_ema":
        return safe_num(rows["n_latency_ms"]) + safe_num(rows["s_choice"]).fillna(0) * safe_num(rows["s_latency_ms"])
    return safe_num(rows["chosen_latency_ms"])


def summarize_policy(run, df: pd.DataFrame) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for policy in POLICY_ORDER:
        rows = df[df["policy"] == policy]
        if rows.empty:
            continue
        precision, recall, trig_f1 = trigger_metrics(rows)
        rand = expected_random_iou(df, policy)
        t_total = safe_num(rows["t_total_policy_relevant_ms"])
        t_scene = safe_num(rows["proxy_ms_policy_relevant"])
        t_ctrl = safe_num(rows["decision_ms"])
        t_selected = safe_num(rows["chosen_latency_ms"])
        t_infer = deployed_inference(rows, policy)
        row = {
            "domain": run.domain,
            "video_slug": run.video_slug,
            "model_family": run.model_family,
            "pair_type": run.pair_type,
            "device": run.device,
            "run_id": run.run_id,
            "policy": policy,
            "reader_policy_label": reader_policy_label(policy),
            "presentation_label": POLICY_LABEL[policy],
            "n_frames": int(rows["frame_idx"].nunique()),
            "accurate_usage_rate": float(safe_num(rows["s_choice"]).mean()),
            "benefit_rate": float(safe_num(rows["benefit_positive"]).mean()),
            "iou_match": float(safe_num(rows["iou_match"]).mean()),
            "det_coverage": float(safe_num(rows["det_coverage"]).mean()),
            "det_recall": float(safe_num(rows["det_recall"]).mean()),
            "count_agree": float(safe_num(rows["count_agree"]).mean()),
            "trigger_precision": precision,
            "trigger_recall": recall,
            "trigger_f1": trig_f1,
            "expected_random_iou_match": rand,
            "informed_gain_over_random": float(safe_num(rows["iou_match"]).mean() - rand) if np.isfinite(rand) else math.nan,
            "t_scene_policy_relevant_mean_ms": float(t_scene.mean()),
            "t_ctrl_mean_ms": float(t_ctrl.mean()),
            "t_selected_inference_mean_ms": float(t_selected.mean()),
            "t_inference_deployed_mean_ms": float(t_infer.mean()),
            "t_total_policy_relevant_mean_ms": float(t_total.mean()),
            "t_total_policy_relevant_p95_ms": float(t_total.quantile(0.95)),
            "fps_policy_relevant": float(1000.0 / t_total.mean()) if t_total.mean() else math.nan,
            "t_total_shared_proxy_mean_ms": float(safe_num(rows["t_total_shared_proxy_ms"]).mean()),
            "proxy_shared_mean_ms": float(safe_num(rows["proxy_ms_total"]).mean()),
        }
        out.append(row)
    return out


def auc_ap_metrics(y: np.ndarray, x: np.ndarray) -> dict[str, float | str]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask].astype(float)
    y = y[mask].astype(int)
    out: dict[str, float | str] = {
        "base_rate": float(np.mean(y)) if len(y) else math.nan,
        "auroc_raw": math.nan,
        "auroc_directional": math.nan,
        "direction": "",
        "auprc": math.nan,
        "auprc_lift_over_random": math.nan,
        "spearman": math.nan,
        "cohens_d": math.nan,
        "cliffs_delta": math.nan,
        "n_frames": int(len(y)),
    }
    if len(np.unique(y)) < 2:
        return out
    try:
        from scipy.stats import mannwhitneyu, spearmanr
        from sklearn.metrics import average_precision_score, roc_auc_score

        auc = float(roc_auc_score(y, x))
        direction = "positive" if auc >= 0.5 else "negative"
        score = x if auc >= 0.5 else -x
        auprc = float(average_precision_score(y, score))
        base = float(np.mean(y))
        pos = x[y == 1]
        neg = x[y == 0]
        pooled = math.sqrt(((len(pos) - 1) * np.var(pos, ddof=1) + (len(neg) - 1) * np.var(neg, ddof=1)) / max(len(pos) + len(neg) - 2, 1))
        d = float((np.mean(pos) - np.mean(neg)) / pooled) if pooled else math.nan
        u = float(mannwhitneyu(pos, neg, alternative="two-sided").statistic)
        cliffs = float((2 * u) / (len(pos) * len(neg)) - 1) if len(pos) and len(neg) else math.nan
        rho = float(spearmanr(x, y).statistic)
        out.update(
            {
                "auroc_raw": auc,
                "auroc_directional": max(auc, 1 - auc),
                "direction": direction,
                "auprc": auprc,
                "auprc_lift_over_random": auprc - base,
                "spearman": rho,
                "cohens_d": d,
                "cliffs_delta": cliffs,
            }
        )
    except Exception:
        pass
    return out


def feature_validity(run, df: pd.DataFrame) -> list[dict[str, object]]:
    # Use one row per frame; all policies broadcast the same features/benefit.
    base = df[df["policy"] == "n_only"].sort_values("frame_idx")
    y = safe_num(base["benefit_positive"]).fillna(0).to_numpy()
    rows: list[dict[str, object]] = []
    for feature in FEATURES:
        if feature not in base.columns:
            continue
        metrics = auc_ap_metrics(y, safe_num(base[feature]).to_numpy())
        rows.append(
            {
                "domain": run.domain,
                "video_slug": run.video_slug,
                "model_family": run.model_family,
                "pair_type": run.pair_type,
                "device": run.device,
                "run_id": run.run_id,
                "target": "benefit_positive",
                "feature": feature,
                **metrics,
            }
        )
    return rows


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=230)
    plt.close()


def run_tag(run) -> str:
    return f"{run.domain}_{run.video_slug}_{run.model_family}_{run.pair_type}_{run.device}"


def plot_pareto(summary: pd.DataFrame, run, out: Path) -> None:
    rows = summary[summary["run_id"] == run.run_id].copy()
    plt.figure(figsize=(8.6, 5.2))
    for policy in POLICY_ORDER:
        r = rows[rows["policy"] == policy]
        if r.empty:
            continue
        x = float(r["t_total_policy_relevant_mean_ms"].iloc[0])
        y = float(r["iou_match"].iloc[0])
        plt.scatter(x, y, color=COLORS[policy], s=70)
        plt.text(x, y, POLICY_LABEL[policy], fontsize=8, va="bottom", ha="left")
    plt.xlabel("Policy-relevant T_total mean (ms)")
    plt.ylabel("Accurate-reference IoU agreement")
    plt.title(f"{run.domain}/{run.video_slug} | {run.model_family} CPU n_l")
    plt.grid(True, alpha=0.25)
    plt.ylim(0, 1.05)
    plt.xlim(left=0)
    savefig(out / f"{run_tag(run)}_runtime_quality_pareto.png")


def plot_runtime_components(summary: pd.DataFrame, run, out: Path) -> None:
    rows = summary[summary["run_id"] == run.run_id].set_index("policy").loc[POLICY_ORDER].reset_index()
    labels = [POLICY_LABEL[p] for p in rows["policy"]]
    scene = rows["t_scene_policy_relevant_mean_ms"].to_numpy(float)
    ctrl = rows["t_ctrl_mean_ms"].to_numpy(float)
    infer = rows["t_inference_deployed_mean_ms"].to_numpy(float)
    x = np.arange(len(rows))
    plt.figure(figsize=(11.5, 5.2))
    plt.bar(x, infer, label="Detector inference", color="#4E79A7")
    plt.bar(x, scene, bottom=infer, label="T_scene: policy-required proxy", color="#F28E2B")
    plt.bar(x, ctrl, bottom=infer + scene, label="T_ctrl: policy decision", color="#59A14F")
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Mean time per frame (ms)")
    plt.title(f"T_total components | {run.domain}/{run.video_slug} | {run.model_family} CPU n_l")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    savefig(out / f"{run_tag(run)}_runtime_components.png")


def plot_trigger_scores(summary: pd.DataFrame, run, out: Path) -> None:
    policies = [p for p in POLICY_ORDER if p not in {"n_only", "s_only"}]
    rows = summary[(summary["run_id"] == run.run_id) & (summary["policy"].isin(policies))].set_index("policy").loc[policies].reset_index()
    x = np.arange(len(rows))
    width = 0.25
    plt.figure(figsize=(10.5, 5))
    plt.bar(x - width, rows["trigger_precision"], width, label="Precision", color="#76B7B2")
    plt.bar(x, rows["trigger_recall"], width, label="Recall", color="#59A14F")
    plt.bar(x + width, rows["trigger_f1"], width, label="F1", color="#E15759")
    plt.xticks(x, [POLICY_LABEL[p] for p in rows["policy"]], rotation=30, ha="right")
    plt.ylabel("Score vs benefit-positive frame label")
    plt.ylim(0, 1)
    plt.title(f"Trigger alignment | {run.domain}/{run.video_slug} | {run.model_family} CPU n_l")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    savefig(out / f"{run_tag(run)}_trigger_precision_recall_f1.png")


def plot_benefit_timeline(df: pd.DataFrame, run, out: Path) -> None:
    base = df[df["policy"] == "n_only"].sort_values("frame_idx")
    x = safe_num(base["frame_idx"])
    benefit = safe_num(base["benefit_positive"]).rolling(100, min_periods=1).mean()
    plt.figure(figsize=(11.5, 4.8))
    plt.plot(x, benefit, color="black", linewidth=1.8, label="benefit_positive rolling mean")
    for policy in ["conf_ema", "combined_hyst", "local_contrast_hyst"]:
        rows = df[df["policy"] == policy].sort_values("frame_idx")
        if rows.empty:
            continue
        usage = safe_num(rows["s_choice"]).rolling(100, min_periods=1).mean()
        plt.plot(safe_num(rows["frame_idx"]), usage, linewidth=1.2, label=f"{POLICY_LABEL[policy]} accurate usage")
    plt.xlabel("Frame index")
    plt.ylabel("Rolling rate, window=100 frames")
    plt.ylim(-0.02, 1.02)
    plt.title(f"Where the accurate model helps vs policy switching | {run.domain}/{run.video_slug} | {run.model_family}")
    plt.legend(fontsize=8, ncol=2)
    plt.grid(True, alpha=0.25)
    savefig(out / f"{run_tag(run)}_benefit_and_switching_timeline.png")


def plot_feature_heatmap(validity: pd.DataFrame, run, out: Path) -> None:
    rows = validity[validity["run_id"] == run.run_id].copy()
    rows = rows.set_index("feature").reindex(FEATURES).reset_index()
    vals = rows[["auroc_directional", "auprc_lift_over_random", "spearman", "cohens_d"]].to_numpy(float)
    plt.figure(figsize=(8.2, 5.8))
    vmax = np.nanmax(np.abs(vals)) if np.isfinite(vals).any() else 1
    im = plt.imshow(vals, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    plt.colorbar(im, label="Metric value")
    plt.xticks(np.arange(4), ["AUROC dir.", "AUPRC lift", "Spearman", "Cohen's d"], rotation=20, ha="right")
    plt.yticks(np.arange(len(rows)), rows["feature"])
    plt.title(f"Feature association with benefit_positive | {run.domain}/{run.video_slug} | {run.model_family}")
    savefig(out / f"{run_tag(run)}_feature_benefit_heatmap.png")


def plot_pr_curves(df: pd.DataFrame, validity: pd.DataFrame, run, out: Path) -> None:
    try:
        from sklearn.metrics import average_precision_score, precision_recall_curve
    except Exception:
        return
    base = df[df["policy"] == "n_only"].sort_values("frame_idx")
    y = safe_num(base["benefit_positive"]).fillna(0).astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        return
    valid = validity[validity["run_id"] == run.run_id].set_index("feature")
    plt.figure(figsize=(7.5, 5.6))
    for feature in STRUCTURAL_FEATURES:
        if feature not in base.columns or feature not in valid.index:
            continue
        x = safe_num(base[feature]).to_numpy(float)
        direction = valid.loc[feature, "direction"]
        score = x if direction == "positive" else -x
        mask = np.isfinite(score)
        if len(np.unique(y[mask])) < 2:
            continue
        precision, recall, _ = precision_recall_curve(y[mask], score[mask])
        ap = average_precision_score(y[mask], score[mask])
        plt.plot(recall, precision, linewidth=1.8, label=f"{feature} AP={ap:.3f}")
    plt.xlabel("Recall for benefit-positive frames")
    plt.ylabel("Precision")
    plt.title(f"Feature PR curves for benefit_positive | {run.domain}/{run.video_slug} | {run.model_family}")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.25)
    savefig(out / f"{run_tag(run)}_feature_pr_curves.png")


def cross_video_figures(summary: pd.DataFrame, validity: pd.DataFrame, out: Path) -> None:
    fig_dir = out / "figures_cross_selected"
    ensure_dir(fig_dir)
    core = summary[summary["policy"].isin(["n_only", "s_only", "conf_ema", "combined_hyst", "local_contrast_hyst"])]
    plt.figure(figsize=(9, 5.2))
    for policy in ["n_only", "s_only", "conf_ema", "combined_hyst", "local_contrast_hyst"]:
        rows = core[core["policy"] == policy]
        plt.scatter(rows["t_total_policy_relevant_mean_ms"], rows["iou_match"], label=POLICY_LABEL[policy], s=55, alpha=0.78, color=COLORS[policy])
    plt.xlabel("Policy-relevant T_total mean (ms)")
    plt.ylabel("Accurate-reference IoU agreement")
    plt.title("Selected videos: runtime-quality tradeoff across CPU n_l runs")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    savefig(fig_dir / "selected_videos_runtime_quality_overview.png")

    dyn = summary[~summary["policy"].isin(["n_only", "s_only"])]
    piv = dyn.pivot_table(index=["domain", "video_slug", "model_family"], columns="policy", values="informed_gain_over_random")
    cols = [p for p in ["conf_ema", "combined_hyst", "local_contrast_hyst", "multi_proxy"] if p in piv.columns]
    piv = piv[cols]
    plt.figure(figsize=(8.5, max(4.5, len(piv) * 0.4)))
    im = plt.imshow(piv.to_numpy(float), aspect="auto", cmap="RdBu_r", vmin=-0.08, vmax=0.08)
    plt.colorbar(im, label="IoU gain over same-usage random")
    plt.xticks(np.arange(len(cols)), [POLICY_LABEL[c] for c in cols], rotation=25, ha="right")
    plt.yticks(np.arange(len(piv)), ["/".join(map(str, idx)) for idx in piv.index])
    plt.title("Trigger intelligence varies by video and model family")
    savefig(fig_dir / "selected_videos_informed_gain_heatmap.png")

    top = (
        validity[validity["feature"].isin(STRUCTURAL_FEATURES)]
        .sort_values("auprc_lift_over_random", ascending=False)
        .groupby(["domain", "video_slug", "model_family", "run_id"])
        .head(1)
    )
    counts = top.groupby(["domain", "feature"]).size().reset_index(name="top_count")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True)
    for ax, domain in zip(axes, ["glass", "porcelain"]):
        dom = counts[counts["domain"] == domain].set_index("feature").reindex(STRUCTURAL_FEATURES).fillna(0)
        ax.bar(dom.index, dom["top_count"], color=["#D6A100", "#76B7B2", "#4E79A7", "#D62728"])
        ax.set_title(domain.capitalize())
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Times ranked top")
    fig.suptitle("Selected videos: best structural feature is not universal")
    savefig(fig_dir / "selected_videos_top_structural_features.png")


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    d = df.copy()
    for col in d.select_dtypes(include=[float]).columns:
        d[col] = d[col].map(lambda x: f"{x:.4f}" if np.isfinite(x) else "")
    headers = [str(c) for c in d.columns]
    rows = [[str(v) for v in row] for row in d.to_numpy()]
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows)) if rows else len(headers[i])
        for i in range(len(headers))
    ]
    header_line = "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |"
    sep_line = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    body = [
        "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header_line, sep_line, *body])


def write_narrative(out: Path, summary: pd.DataFrame, validity: pd.DataFrame) -> None:
    core_policies = ["n_only", "s_only", "conf_ema", "combined_hyst", "local_contrast_hyst"]
    core = summary[summary["policy"].isin(core_policies)].copy()
    overview = core.groupby(["domain", "policy"], as_index=False).agg(
        iou=("iou_match", "mean"),
        t_total=("t_total_policy_relevant_mean_ms", "mean"),
        fps=("fps_policy_relevant", "mean"),
        gain=("informed_gain_over_random", "mean"),
        usage=("accurate_usage_rate", "mean"),
        f1=("trigger_f1", "mean"),
    )
    overview["policy"] = overview["policy"].map(POLICY_LABEL)

    top_features = (
        validity[validity["feature"].isin(STRUCTURAL_FEATURES)]
        .sort_values("auprc_lift_over_random", ascending=False)
        .groupby(["domain", "video_slug", "model_family"])
        .head(1)[
            [
                "domain",
                "video_slug",
                "model_family",
                "feature",
                "auroc_directional",
                "direction",
                "auprc_lift_over_random",
                "spearman",
            ]
        ]
    )

    lines = [
        "# Selected-Video Deep Dive: Presentation Narrative",
        "",
        "This package explains DMS behavior on four representative videos using CPU `n_l` runs only. CPU is used because the fast/accurate latency gap is large enough for DMS to be meaningful and easy to explain visually.",
        "",
        "Selected videos:",
        "",
        "- Glass: `glass_ins` and `161_YUN_0001_96`.",
        "- Porcelain: `UAV_porcelain` and `porcelain_maybe`.",
        "- For each video, both `yolov8 n_l` and `yolo26 n_l` CPU pairs are included.",
        "",
        "## Runtime Accounting Used Here",
        "",
        "`T_total` uses policy-relevant deployed accounting:",
        "",
        "`T_total = T_scene(policy-required) + T_ctrl + T_selected_detector`.",
        "",
        "For `conf_ema`, the deployed formula is `T_fast + T_ctrl + I[accurate] * T_accurate`, because the fast detector is the watchdog that produces the confidence signal.",
        "",
        "`T_scene` is not the cost of all logged image features. It is the estimated cost of the feature(s) required by each policy: entropy for `entropy_only`, L/H for `combined`, local contrast for `local_contrast_hyst`, and the naive multi-feature set for `multi_proxy`.",
        "",
        "## Step-by-Step Presentation Order",
        "",
        "1. Start with `figures_cross_selected/selected_videos_runtime_quality_overview.png`. Explain that DMS creates points between Fast only and Accurate only, trading latency for agreement.",
        "2. Show one per-video Pareto plot. Explain the x-axis as policy-relevant deployed latency and the y-axis as agreement with the accurate detector.",
        "3. Show the matching runtime-component plot. Explain `T_scene`, `T_ctrl`, and detector inference separately; this is where the corrected benchmarking is defended.",
        "4. Show the trigger precision/recall/F1 plot. Explain that a useful trigger should select accurate on benefit-positive frames, not merely use accurate often.",
        "5. Show the benefit-and-switching timeline. Explain whether policy switches occur near regions where `benefit_positive` increases.",
        "6. Show the feature heatmap and PR curves. Explain that structural-texture features predict benefit-positive frames statistically, not causally.",
        "7. Finish with `figures_cross_selected/selected_videos_informed_gain_heatmap.png` and the full 84-run generalization figures listed below.",
        "",
        "## Compact Selected-Video Result Table",
        "",
        markdown_table(overview.round(4)),
        "",
        "## Top Structural Feature Per Selected Run",
        "",
        markdown_table(top_features.round(4)),
        "",
        "## Notes For Explaining To A Normal Technical Audience",
        "",
        "- `benefit_positive` is the frame-level opportunity signal: the accurate model sees useful evidence that the fast model does not capture as well.",
        "- `iou_match` is agreement with the accurate detector reference, not human-labelled video mAP.",
        "- `trigger F1` measures whether a policy sends the right frames to the accurate model.",
        "- `informed_gain_over_random` prevents a misleading conclusion where a policy looks good only because it uses the accurate model often.",
        "- The image-feature analysis supports a structural-texture family claim. It does not prove that one feature universally causes difficulty.",
        "",
        "## Full 84-Run Generalization Figures To Add After The Deep Dive",
        "",
        "- `../figures_core/fig_core_cpu_tradeoff.png`: main CPU tradeoff summary.",
        "- `../figures_core/fig_core_cuda_tradeoff.png`: CUDA/proxy-overhead caution.",
        "- `../figures_core/fig_core_informed_gain.png`: random-baseline correction.",
        "- `../figures_core/fig_core_latency_fps.png`: latency and FPS summary.",
        "- `../figures_core/fig_core_trigger_precision_recall_f1.png`: trigger alignment summary.",
        "- `../figures_core/fig_core_structural_feature_counts.png`: structural-texture generalization summary.",
        "- `../figures_core/CORE_FIGURE_EXPLANATION_GUIDE.md`: concise explanation guide for all six core figures.",
        "",
        "## Thesis-Safe Conclusion",
        "",
        "On these selected videos, DMS behavior can be explained as a controlled runtime-quality tradeoff. The strongest visual argument is not that DMS always wins, but that policies occupy interpretable operating points between Fast only and Accurate only. The feature analysis suggests that structural-texture complexity is predictively associated with frames where the accurate detector helps, while the full 84-run analysis is needed for generalization claims.",
    ]
    (out / "SELECTED_VIDEO_PRESENTATION_NARRATIVE.md").write_text("\n".join(lines), encoding="utf-8")


def write_readme(out: Path) -> None:
    lines = [
        "# Selected Video Deep Dive",
        "",
        "This folder contains a compact thesis-presentation package built from the completed 84-run final sweep. It focuses on four representative videos and CPU `n_l` model pairs.",
        "",
        "## Folder Structure",
        "",
        "- `tables/selected_run_inventory.csv`: selected run list.",
        "- `tables/selected_policy_summary.csv`: policy metrics and policy-relevant runtime accounting.",
        "- `tables/selected_runtime_components.csv`: explicit `T_scene`, `T_ctrl`, inference, `T_total`, and FPS.",
        "- `tables/selected_feature_validity.csv`: feature association with `benefit_positive`.",
        "- `figures_per_run/`: per-video/model figures.",
        "- `figures_cross_selected/`: summary figures across the selected videos.",
        "- `SELECTED_VIDEO_PRESENTATION_NARRATIVE.md`: step-by-step speaking notes.",
        "",
        "All results use existing local evidence only. No remote jobs or experiment code changes are involved.",
    ]
    (out / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = args.output_root
    tables = out / "tables"
    per_run_figs = out / "figures_per_run"
    ensure_dir(tables)
    ensure_dir(per_run_figs)

    runs = discover_runs(args.results_root)
    wanted = []
    for run in runs:
        if (
            run.domain in SELECTED_VIDEOS
            and run.video_slug in SELECTED_VIDEOS[run.domain]
            and run.model_family in MODEL_FAMILIES
            and run.device == DEVICE
            and run.pair_type == PAIR_TYPE
        ):
            wanted.append(run)
    wanted = sorted(wanted, key=lambda r: (r.domain, r.video_slug, r.model_family))
    if len(wanted) != 8:
        raise SystemExit(f"Expected 8 selected runs, found {len(wanted)}")

    inventory = pd.DataFrame(
        [
            {
                "domain": r.domain,
                "video_slug": r.video_slug,
                "model_family": r.model_family,
                "pair_type": r.pair_type,
                "device": r.device,
                "run_id": r.run_id,
                "run_dir": str(r.run_dir),
            }
            for r in wanted
        ]
    )
    inventory.to_csv(tables / "selected_run_inventory.csv", index=False)

    summary_rows: list[dict[str, object]] = []
    validity_rows: list[dict[str, object]] = []
    data_by_run = {}
    for run in wanted:
        df = pd.read_csv(run.run_dir / "sweep_per_frame.csv")
        data_by_run[run.run_id] = df
        summary_rows.extend(summarize_policy(run, df))
        validity_rows.extend(feature_validity(run, df))

    summary = pd.DataFrame(summary_rows)
    validity = pd.DataFrame(validity_rows)
    summary.to_csv(tables / "selected_policy_summary.csv", index=False)
    summary[
        [
            "domain",
            "video_slug",
            "model_family",
            "policy",
            "presentation_label",
            "t_scene_policy_relevant_mean_ms",
            "t_ctrl_mean_ms",
            "t_selected_inference_mean_ms",
            "t_inference_deployed_mean_ms",
            "t_total_policy_relevant_mean_ms",
            "t_total_policy_relevant_p95_ms",
            "fps_policy_relevant",
            "t_total_shared_proxy_mean_ms",
            "proxy_shared_mean_ms",
        ]
    ].to_csv(tables / "selected_runtime_components.csv", index=False)
    summary[
        [
            "domain",
            "video_slug",
            "model_family",
            "policy",
            "presentation_label",
            "accurate_usage_rate",
            "benefit_rate",
            "trigger_precision",
            "trigger_recall",
            "trigger_f1",
            "iou_match",
            "expected_random_iou_match",
            "informed_gain_over_random",
        ]
    ].to_csv(tables / "selected_trigger_and_quality_metrics.csv", index=False)
    validity.to_csv(tables / "selected_feature_validity.csv", index=False)

    for run in wanted:
        df = data_by_run[run.run_id]
        plot_pareto(summary, run, per_run_figs)
        plot_runtime_components(summary, run, per_run_figs)
        plot_trigger_scores(summary, run, per_run_figs)
        plot_benefit_timeline(df, run, per_run_figs)
        plot_feature_heatmap(validity, run, per_run_figs)
        plot_pr_curves(df, validity, run, per_run_figs)

    cross_video_figures(summary, validity, out)
    write_narrative(out, summary, validity)
    write_readme(out)
    print(f"wrote selected-video deep dive to {out}")
    print(f"selected runs={len(wanted)}")
    print(f"policy summary rows={len(summary)}")
    print(f"feature validity rows={len(validity)}")


if __name__ == "__main__":
    main()
