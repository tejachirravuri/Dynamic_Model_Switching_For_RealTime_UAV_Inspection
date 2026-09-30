"""Polished thesis-facing figures for final multi-video results.

This script consumes derived final CSVs only. It does not read raw sweep files
or modify experiment/inference code.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import dataframe_to_markdown, ensure_dir, output_root


POLICY_ORDER = [
    "fast_only",
    "accurate_only",
    "conf_ema",
    "combined",
    "combined_hyst",
    "entropy_only",
    "local_contrast_hyst",
    "multi_proxy",
]

POLICY_SHORT = {
    "fast_only": "Fast only",
    "accurate_only": "Accurate only",
    "conf_ema": "Conf-EMA",
    "combined": "Combined",
    "combined_hyst": "Combined hyst.",
    "entropy_only": "Entropy",
    "local_contrast_hyst": "Local contrast hyst.",
    "multi_proxy": "Naive multi-proxy",
}

COLORS = {
    "fast_only": "#3B6EA8",
    "accurate_only": "#E17C05",
    "conf_ema": "#C7362F",
    "combined": "#7B5EA7",
    "combined_hyst": "#8C6D31",
    "entropy_only": "#4C9A51",
    "local_contrast_hyst": "#D6A100",
    "multi_proxy": "#5FA8A8",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=output_root())
    return parser.parse_args()


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def labels_in(df: pd.DataFrame) -> list[str]:
    present = set(df["reader_policy_label"])
    return [p for p in POLICY_ORDER if p in present]


def policy_name(policy: str) -> str:
    return POLICY_SHORT.get(policy, policy)


def common_pair_pareto(master: pd.DataFrame, out: Path, device: str) -> None:
    sub = master[(master["pair_type"] == "n_l") & (master["device"] == device)]
    agg = sub.groupby(["domain", "reader_policy_label"], as_index=False).agg(
        runtime_mean=("t_total_deployed_mean_ms", "mean"),
        runtime_sem=("t_total_deployed_mean_ms", lambda s: s.std(ddof=1) / np.sqrt(len(s))),
        iou_mean=("iou_match", "mean"),
        iou_sem=("iou_match", lambda s: s.std(ddof=1) / np.sqrt(len(s))),
        usage=("accurate_usage_rate", "mean"),
        gain=("informed_gain_over_random", "mean"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), sharey=True)
    for ax, domain in zip(axes, ["glass", "porcelain"]):
        dom = agg[agg["domain"] == domain]
        for _, row in dom.iterrows():
            p = row["reader_policy_label"]
            ax.errorbar(
                row["runtime_mean"],
                row["iou_mean"],
                xerr=row["runtime_sem"],
                yerr=row["iou_sem"],
                fmt="o",
                markersize=7,
                color=COLORS.get(p, "#777777"),
                ecolor=COLORS.get(p, "#777777"),
                alpha=0.9,
                capsize=2,
                label=policy_name(p),
            )
            ax.text(row["runtime_mean"], row["iou_mean"], policy_name(p), fontsize=7, ha="left", va="bottom")
        ax.set_title(domain.capitalize())
        ax.set_xlabel("Mean deployed T_total (ms)")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Mean accurate-reference IoU agreement")
    fig.suptitle(f"{device.upper()} common-pair Pareto tradeoff (n_l only, unweighted per-run means)")
    savefig(out / f"fig_thesis_{device}_common_pair_pareto.png")


def common_pair_runtime_quality_bars(master: pd.DataFrame, out: Path) -> None:
    common = master[master["pair_type"] == "n_l"]
    agg = common.groupby(["device", "reader_policy_label"], as_index=False).agg(
        runtime=("t_total_deployed_mean_ms", "mean"),
        iou=("iou_match", "mean"),
        gain=("informed_gain_over_random", "mean"),
    )
    policies = labels_in(agg)
    x = np.arange(len(policies))
    width = 0.38
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for ax, metric, ylabel in [
        (axes[0], "runtime", "Mean deployed T_total (ms)"),
        (axes[1], "iou", "Mean IoU agreement"),
    ]:
        for j, device in enumerate(["cpu", "cuda"]):
            vals = []
            for p in policies:
                row = agg[(agg["device"] == device) & (agg["reader_policy_label"] == p)]
                vals.append(float(row[metric].iloc[0]) if not row.empty else np.nan)
            ax.bar(x + (j - 0.5) * width, vals, width, label=device.upper(), color="#5B8AC6" if device == "cpu" else "#F0A33A")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([policy_name(p) for p in policies], rotation=30, ha="right")
    fig.suptitle("CPU vs CUDA on common n_l pair coverage")
    savefig(out / "fig_thesis_common_pair_platform_bars.png")


def informed_gain_delta_bars(master: pd.DataFrame, out: Path) -> None:
    dyn = master[~master["reader_policy_label"].isin(["fast_only", "accurate_only"])]
    agg = dyn.groupby(["domain", "device", "reader_policy_label"], as_index=False).agg(
        gain=("informed_gain_over_random", "mean"),
        gain_sem=("informed_gain_over_random", lambda s: s.std(ddof=1) / np.sqrt(len(s))),
    )
    policies = [p for p in POLICY_ORDER if p in set(agg["reader_policy_label"]) and p not in {"fast_only", "accurate_only"}]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=True)
    for ax, (domain, device) in zip(axes.flat, [("glass", "cpu"), ("glass", "cuda"), ("porcelain", "cpu"), ("porcelain", "cuda")]):
        sub = agg[(agg["domain"] == domain) & (agg["device"] == device)]
        vals = []
        errs = []
        labels = []
        colors = []
        for p in policies:
            row = sub[sub["reader_policy_label"] == p]
            if row.empty:
                continue
            vals.append(float(row["gain"].iloc[0]))
            errs.append(float(row["gain_sem"].iloc[0]))
            labels.append(policy_name(p))
            colors.append(COLORS.get(p, "#777777"))
        ax.bar(np.arange(len(vals)), vals, yerr=errs, capsize=2, color=colors)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_title(f"{domain.capitalize()} / {device.upper()}")
        ax.set_xticks(np.arange(len(vals)))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Informed gain over same-usage random switching")
    axes[0, 0].set_ylabel("Mean IoU gain over random")
    axes[1, 0].set_ylabel("Mean IoU gain over random")
    savefig(out / "fig_thesis_informed_gain_delta_bars.png")


def per_video_policy_heatmap(master: pd.DataFrame, out: Path, metric: str, filename: str, title: str) -> None:
    dyn = master.copy()
    dyn["video_domain"] = dyn["domain"] + "/" + dyn["video_slug"]
    piv = dyn.pivot_table(index="video_domain", columns="reader_policy_label", values=metric, aggfunc="mean")
    cols = [p for p in POLICY_ORDER if p in piv.columns]
    piv = piv[cols].sort_index()
    plt.figure(figsize=(11, max(5.5, len(piv) * 0.35)))
    data = piv.to_numpy(dtype=float)
    im = plt.imshow(data, aspect="auto", cmap="viridis")
    plt.colorbar(im, label=metric)
    plt.xticks(np.arange(len(cols)), [policy_name(c) for c in cols], rotation=35, ha="right")
    plt.yticks(np.arange(len(piv.index)), piv.index)
    plt.title(title)
    savefig(out / filename)


def feature_video_heatmap(by_video: pd.DataFrame, out: Path) -> None:
    structural = ["local_contrast", "edge_density", "laplacian", "tenengrad"]
    sub = by_video[(by_video["target"] == "benefit_positive") & (by_video["feature"].isin(structural))].copy()
    sub["video_domain"] = sub["domain"] + "/" + sub["video_slug"]
    piv = sub.pivot_table(index="video_domain", columns="feature", values="auprc_lift_over_random", aggfunc="mean")
    piv = piv[[c for c in structural if c in piv.columns]].sort_index()
    plt.figure(figsize=(7.5, max(5.5, len(piv) * 0.35)))
    data = piv.to_numpy(dtype=float)
    vmax = np.nanmax(np.abs(data)) if np.isfinite(data).any() else 1
    im = plt.imshow(data, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    plt.colorbar(im, label="AUPRC lift over base rate")
    plt.xticks(np.arange(len(piv.columns)), piv.columns, rotation=25, ha="right")
    plt.yticks(np.arange(len(piv.index)), piv.index)
    plt.title("Per-video structural-texture association with benefit-positive frames")
    savefig(out / "fig_thesis_feature_video_lighting_variability_heatmap.png")


def top_feature_counts(by_video: pd.DataFrame, out: Path) -> None:
    structural = ["local_contrast", "edge_density", "laplacian", "tenengrad"]
    sub = by_video[(by_video["target"] == "benefit_positive") & (by_video["feature"].isin(structural))]
    top = (
        sub.sort_values("auprc_lift_over_random", ascending=False)
        .groupby(["domain", "video_slug", "model_family", "pair_type", "device"])
        .head(1)
    )
    counts = top.groupby(["domain", "feature"]).size().reset_index(name="top_count")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), sharey=True)
    for ax, domain in zip(axes, ["glass", "porcelain"]):
        dom = counts[counts["domain"] == domain].set_index("feature").reindex(structural).fillna(0)
        ax.bar(dom.index, dom["top_count"], color=["#D6A100", "#5FA8A8", "#3B6EA8", "#C7362F"])
        ax.set_title(domain.capitalize())
        ax.set_ylabel("Top-feature count")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Which structural feature ranks highest per video/configuration?")
    savefig(out / "fig_thesis_top_structural_feature_counts.png")


def table_policy_summary(master: pd.DataFrame, out: Path) -> None:
    common = master[master["pair_type"] == "n_l"]
    table = common.groupby(["device", "reader_policy_label"], as_index=False).agg(
        iou_match=("iou_match", "mean"),
        t_total_ms=("t_total_deployed_mean_ms", "mean"),
        accurate_usage=("accurate_usage_rate", "mean"),
        informed_gain=("informed_gain_over_random", "mean"),
        trigger_precision=("trigger_precision", "mean"),
        trigger_recall=("trigger_recall", "mean"),
    )
    table["policy"] = table["reader_policy_label"].map(policy_name)
    table = table[["device", "policy", "iou_match", "t_total_ms", "accurate_usage", "informed_gain", "trigger_precision", "trigger_recall"]]
    table.round(4).to_csv(out / "table_thesis_common_pair_policy_summary.csv", index=False)


def latency_fps_tail_figures(master: pd.DataFrame, out: Path) -> None:
    common = master[master["pair_type"] == "n_l"].copy()
    common["fps"] = 1000.0 / common["t_total_deployed_mean_ms"]
    policies = labels_in(common)
    x = np.arange(len(policies))
    width = 0.38

    for metric, ylabel, filename, title in [
        ("t_total_deployed_mean_ms", "Mean deployed latency (ms)", "fig_thesis_latency_mean_common_pair.png", "Mean deployed latency on common n_l pair coverage"),
        ("t_total_deployed_p95_ms", "p95 deployed latency (ms)", "fig_thesis_latency_p95_common_pair.png", "p95 deployed latency on common n_l pair coverage"),
        ("fps", "Mean deployed FPS", "fig_thesis_fps_common_pair.png", "Mean deployed FPS on common n_l pair coverage"),
    ]:
        agg = common.groupby(["device", "reader_policy_label"], as_index=False)[metric].mean()
        plt.figure(figsize=(11.5, 5))
        for j, device in enumerate(["cpu", "cuda"]):
            vals = []
            for p in policies:
                row = agg[(agg["device"] == device) & (agg["reader_policy_label"] == p)]
                vals.append(float(row[metric].iloc[0]) if not row.empty else np.nan)
            plt.bar(x + (j - 0.5) * width, vals, width, label=device.upper(), color="#5B8AC6" if device == "cpu" else "#F0A33A")
        plt.xticks(x, [policy_name(p) for p in policies], rotation=30, ha="right")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(axis="y", alpha=0.25)
        plt.legend()
        savefig(out / filename)


def trigger_f1_figures(master: pd.DataFrame, out: Path) -> None:
    dyn = master[~master["reader_policy_label"].isin(["fast_only", "accurate_only"])].copy()
    agg = dyn.groupby(["domain", "device", "reader_policy_label"], as_index=False).agg(
        precision=("trigger_precision", "mean"),
        recall=("trigger_recall", "mean"),
        f1=("trigger_f1", "mean"),
        usage=("accurate_usage_rate", "mean"),
        gain=("informed_gain_over_random", "mean"),
    )
    policies = [p for p in POLICY_ORDER if p in set(agg["reader_policy_label"]) and p not in {"fast_only", "accurate_only"}]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=True)
    for ax, (domain, device) in zip(axes.flat, [("glass", "cpu"), ("glass", "cuda"), ("porcelain", "cpu"), ("porcelain", "cuda")]):
        sub = agg[(agg["domain"] == domain) & (agg["device"] == device)]
        labels = []
        vals = []
        colors = []
        for p in policies:
            row = sub[sub["reader_policy_label"] == p]
            if row.empty:
                continue
            labels.append(policy_name(p))
            vals.append(float(row["f1"].iloc[0]))
            colors.append(COLORS.get(p, "#777777"))
        ax.bar(np.arange(len(vals)), vals, color=colors)
        ax.set_title(f"{domain.capitalize()} / {device.upper()}")
        ax.set_xticks(np.arange(len(vals)))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.25)
    axes[0, 0].set_ylabel("Trigger F1 vs benefit-positive")
    axes[1, 0].set_ylabel("Trigger F1 vs benefit-positive")
    fig.suptitle("Policy trigger F1 for selecting benefit-positive frames")
    savefig(out / "fig_thesis_trigger_f1_by_domain_device.png")

    # Precision-recall scatter, useful to explain whether a policy is conservative or broad.
    plt.figure(figsize=(8, 6))
    for policy, rows in dyn.groupby("reader_policy_label"):
        plt.scatter(
            rows["trigger_recall"],
            rows["trigger_precision"],
            s=45,
            alpha=0.65,
            color=COLORS.get(policy, "#777777"),
            label=policy_name(policy),
        )
    plt.xlabel("Trigger recall: benefit-positive frames selected")
    plt.ylabel("Trigger precision: selected frames that are benefit-positive")
    plt.title("Trigger precision/recall across final runs")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8, ncol=2)
    savefig(out / "fig_thesis_trigger_precision_recall_scatter.png")

    agg.round(4).to_csv(out / "table_thesis_trigger_f1_summary.csv", index=False)


def latency_fps_tables(master: pd.DataFrame, out: Path) -> None:
    common = master[master["pair_type"] == "n_l"].copy()
    common["fps_mean"] = 1000.0 / common["t_total_deployed_mean_ms"]
    table = common.groupby(["device", "reader_policy_label"], as_index=False).agg(
        latency_mean_ms=("t_total_deployed_mean_ms", "mean"),
        latency_p50_ms=("t_total_deployed_p50_ms", "mean"),
        latency_p95_ms=("t_total_deployed_p95_ms", "mean"),
        latency_p99_ms=("t_total_deployed_p99_ms", "mean"),
        fps_mean=("fps_mean", "mean"),
        iou_match=("iou_match", "mean"),
        informed_gain=("informed_gain_over_random", "mean"),
        accurate_usage=("accurate_usage_rate", "mean"),
    )
    table["policy"] = table["reader_policy_label"].map(policy_name)
    table = table[
        [
            "device",
            "policy",
            "latency_mean_ms",
            "latency_p50_ms",
            "latency_p95_ms",
            "latency_p99_ms",
            "fps_mean",
            "iou_match",
            "informed_gain",
            "accurate_usage",
        ]
    ]
    table.round(4).to_csv(out / "table_thesis_latency_fps_common_pair.csv", index=False)


def write_captions(out: Path) -> None:
    lines = [
        "# Thesis-Ready Figure Captions: Final Multi-Video Evaluation",
        "",
        "These figures use canonical feature names and avoid duplicate aliases (`L/laplacian`, `H/entropy`, `contrast/local_contrast/brightness_std`).",
        "",
        "## fig_thesis_cpu_common_pair_pareto.png",
        "CPU Pareto plot using common `n_l` pair coverage only. It shows the deployed runtime/accurate-reference agreement tradeoff without mixing CUDA-only `n_s` runs.",
        "",
        "## fig_thesis_cuda_common_pair_pareto.png",
        "CUDA Pareto plot using common `n_l` pair coverage only. It supports the platform-dependence finding that proxy overhead can dominate when detector latency gaps are small.",
        "",
        "## fig_thesis_common_pair_platform_bars.png",
        "Bar comparison of CPU and CUDA runtime/agreement on common `n_l` pair coverage. This is the correct figure for strict CPU-vs-CUDA claims.",
        "",
        "## fig_thesis_latency_mean_common_pair.png",
        "Mean deployed latency by policy on common `n_l` pair coverage. Use this for runtime benchmarking claims.",
        "",
        "## fig_thesis_latency_p95_common_pair.png",
        "p95 deployed latency by policy on common `n_l` pair coverage. Use this for tail-latency and real-time stability discussion.",
        "",
        "## fig_thesis_fps_common_pair.png",
        "Mean deployed FPS by policy on common `n_l` pair coverage. Use this for reader-friendly real-time performance comparison.",
        "",
        "## fig_thesis_informed_gain_delta_bars.png",
        "Mean gain over same-usage random switching by policy, domain, and device. This is the main trigger-intelligence figure because it corrects for accurate-usage rate.",
        "",
        "## fig_thesis_trigger_f1_by_domain_device.png",
        "Trigger F1 for selecting benefit-positive frames. This measures how well policy switching aligns with frames where the accurate detector adds evidence.",
        "",
        "## fig_thesis_trigger_precision_recall_scatter.png",
        "Trigger precision/recall scatter across runs. It shows whether a policy is conservative, broad, or poorly aligned with benefit-positive frames.",
        "",
        "## fig_thesis_per_video_iou_heatmap.png",
        "Per-video policy agreement heatmap. It shows whether aggregate policy conclusions are stable across videos or driven by a few clips.",
        "",
        "## fig_thesis_per_video_informed_gain_heatmap.png",
        "Per-video informed-gain heatmap. It exposes video-level variability and protects against overgeneralizing from pooled means.",
        "",
        "## fig_thesis_feature_video_lighting_variability_heatmap.png",
        "Per-video structural-texture feature association with benefit-positive frames. It is the main figure for discussing lighting/viewpoint variability.",
        "",
        "## fig_thesis_top_structural_feature_counts.png",
        "Counts how often each structural feature ranks first per video/configuration. It supports the feature-family conclusion rather than a single universal trigger claim.",
    ]
    (out / "THESIS_FIGURE_CAPTIONS_REFINED.md").write_text("\n".join(lines), encoding="utf-8")


def write_interpretation_note(out: Path) -> None:
    lines = [
        "# How To Read The Refined Final Figures",
        "",
        "## What changed",
        "",
        "The refined figures remove duplicate feature aliases and use common-pair-only data for strict CPU/CUDA comparison. They also expose per-video variability so lighting and viewpoint differences are visible rather than hidden inside pooled averages.",
        "",
        "## How conclusions are derived",
        "",
        "Policy conclusions come from unweighted run-level summaries in `master_table_multivideo_final.csv`. Trigger-intelligence conclusions use `informed_gain_over_random`, which compares a policy to same-usage random switching.",
        "",
        "Feature conclusions come from device-specific and per-video feature-validity tables. A feature is considered useful only as a predictive signal if it has positive AUPRC lift over the base rate and behaves consistently enough across videos to support the claim.",
        "",
        "## Lighting caveat",
        "",
        "The videos were recorded under different lighting, viewpoint, and background conditions. This helps test robustness but does not prove lighting invariance. The thesis should therefore claim cross-video predictive association under the evaluated dataset, not universal feature validity.",
        "",
        "## Thesis-safe feature claim",
        "",
        "Structural-texture complexity is predictively associated with accurate-model benefit. The exact best scalar feature varies across videos, domains, model pairs, and devices.",
        "",
        "## Thesis-safe policy claim",
        "",
        "Dynamic model selection is a conditional runtime-quality tradeoff. It is useful when the policy's informed gain and runtime savings justify the additional controller/proxy overhead.",
    ]
    (out / "THESIS_FIGURE_INTERPRETATION_GUIDE.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = args.output_root / "figures_refined"
    ensure_dir(out)
    root = args.output_root
    master = pd.read_csv(root / "master_table_multivideo_final.csv")
    by_video = pd.read_csv(root / "feature_validity_by_video_final.csv")

    common_pair_pareto(master, out, "cpu")
    common_pair_pareto(master, out, "cuda")
    common_pair_runtime_quality_bars(master, out)
    latency_fps_tail_figures(master, out)
    latency_fps_tables(master, out)
    informed_gain_delta_bars(master, out)
    trigger_f1_figures(master, out)
    per_video_policy_heatmap(master, out, "iou_match", "fig_thesis_per_video_iou_heatmap.png", "Per-video accurate-reference agreement")
    per_video_policy_heatmap(master, out, "informed_gain_over_random", "fig_thesis_per_video_informed_gain_heatmap.png", "Per-video informed gain over random")
    feature_video_heatmap(by_video, out)
    top_feature_counts(by_video, out)
    table_policy_summary(master, out)
    write_captions(out)
    write_interpretation_note(out)
    print(f"wrote refined thesis figures to {out}")


if __name__ == "__main__":
    main()
