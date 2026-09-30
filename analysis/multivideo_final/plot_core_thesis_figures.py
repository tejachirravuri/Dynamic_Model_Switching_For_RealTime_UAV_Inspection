"""Minimal, thesis-facing core figures for explaining final DMS results.

These figures are intentionally simpler than the diagnostic plots. Each figure
is designed to support one defensible claim in the thesis/viva.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import dataframe_to_markdown, ensure_dir, output_root


CORE_POLICIES = [
    "fast_only",
    "accurate_only",
    "conf_ema",
    "combined_hyst",
    "local_contrast_hyst",
]

POLICY_LABEL = {
    "fast_only": "Fast only",
    "accurate_only": "Accurate only",
    "conf_ema": "Conf-EMA",
    "combined_hyst": "Combined hyst.",
    "local_contrast_hyst": "Local contrast hyst.",
    "multi_proxy": "Naive multi-proxy",
}

COLORS = {
    "fast_only": "#31688E",
    "accurate_only": "#F28E2B",
    "conf_ema": "#D62728",
    "combined_hyst": "#7B5EA7",
    "local_contrast_hyst": "#D6A100",
    "multi_proxy": "#6BAED6",
}


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=240)
    plt.close()


def load_tables(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    master = pd.read_csv(root / "master_table_multivideo_final.csv")
    feature_by_video = pd.read_csv(root / "feature_validity_by_video_final.csv")
    feature_validity = pd.read_csv(root / "feature_vs_benefit_multivideo_final.csv")
    return master, feature_by_video, feature_validity


def mean_sem(series: pd.Series) -> tuple[float, float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return np.nan, np.nan
    return float(s.mean()), float(s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 1 else 0.0


def core_cpu_tradeoff(master: pd.DataFrame, out: Path) -> None:
    """CPU common-pair tradeoff: the strongest positive DMS story."""
    sub = master[(master["pair_type"] == "n_l") & (master["device"] == "cpu") & (master["reader_policy_label"].isin(CORE_POLICIES))]
    agg_rows = []
    for policy, rows in sub.groupby("reader_policy_label"):
        rt, rt_sem = mean_sem(rows["t_total_deployed_mean_ms"])
        iou, iou_sem = mean_sem(rows["iou_match"])
        gain, _ = mean_sem(rows["informed_gain_over_random"])
        usage, _ = mean_sem(rows["accurate_usage_rate"])
        agg_rows.append({"policy": policy, "runtime": rt, "runtime_sem": rt_sem, "iou": iou, "iou_sem": iou_sem, "gain": gain, "usage": usage})
    agg = pd.DataFrame(agg_rows).set_index("policy").loc[CORE_POLICIES].reset_index()

    plt.figure(figsize=(8.5, 5.4))
    for _, row in agg.iterrows():
        p = row["policy"]
        plt.errorbar(row["runtime"], row["iou"], xerr=row["runtime_sem"], yerr=row["iou_sem"], fmt="o", color=COLORS[p], capsize=2, markersize=8)
        plt.text(row["runtime"] + 3, row["iou"], POLICY_LABEL[p], fontsize=9, va="center")
    plt.xlabel("Mean deployed latency, T_total (ms)")
    plt.ylabel("Accurate-reference IoU agreement")
    plt.title("CPU: DMS forms a runtime-quality middle ground")
    plt.grid(True, alpha=0.25)
    plt.xlim(left=0)
    plt.ylim(0.55, 1.03)
    savefig(out / "fig_core_cpu_tradeoff.png")


def core_cuda_tradeoff(master: pd.DataFrame, out: Path) -> None:
    """CUDA common-pair tradeoff: proxy overhead caution."""
    sub = master[(master["pair_type"] == "n_l") & (master["device"] == "cuda") & (master["reader_policy_label"].isin(CORE_POLICIES))]
    agg_rows = []
    for policy, rows in sub.groupby("reader_policy_label"):
        rt, rt_sem = mean_sem(rows["t_total_deployed_mean_ms"])
        iou, iou_sem = mean_sem(rows["iou_match"])
        agg_rows.append({"policy": policy, "runtime": rt, "runtime_sem": rt_sem, "iou": iou, "iou_sem": iou_sem})
    agg = pd.DataFrame(agg_rows).set_index("policy").loc[CORE_POLICIES].reset_index()

    plt.figure(figsize=(8.5, 5.4))
    for _, row in agg.iterrows():
        p = row["policy"]
        plt.errorbar(row["runtime"], row["iou"], xerr=row["runtime_sem"], yerr=row["iou_sem"], fmt="o", color=COLORS[p], capsize=2, markersize=8)
        plt.text(row["runtime"] + 0.35, row["iou"], POLICY_LABEL[p], fontsize=9, va="center")
    plt.xlabel("Mean deployed latency, T_total (ms)")
    plt.ylabel("Accurate-reference IoU agreement")
    plt.title("CUDA: proxy overhead can outweigh detector savings")
    plt.grid(True, alpha=0.25)
    plt.xlim(left=0)
    plt.ylim(0.55, 1.03)
    savefig(out / "fig_core_cuda_tradeoff.png")


def core_informed_gain(master: pd.DataFrame, out: Path) -> None:
    """Trigger intelligence after random correction."""
    dyn = master[master["reader_policy_label"].isin(["conf_ema", "combined_hyst", "local_contrast_hyst", "multi_proxy"])]
    agg = dyn.groupby("reader_policy_label", as_index=False).agg(
        gain=("informed_gain_over_random", "mean"),
        gain_sem=("informed_gain_over_random", lambda s: s.std(ddof=1) / np.sqrt(len(s))),
        usage=("accurate_usage_rate", "mean"),
    )
    order = ["local_contrast_hyst", "conf_ema", "combined_hyst", "multi_proxy"]
    agg = agg.set_index("reader_policy_label").loc[order].reset_index()
    x = np.arange(len(agg))
    plt.figure(figsize=(8.5, 4.8))
    plt.bar(x, agg["gain"], yerr=agg["gain_sem"], capsize=3, color=[COLORS[p] for p in agg["reader_policy_label"]])
    plt.axhline(0, color="black", linewidth=1)
    plt.xticks(x, [POLICY_LABEL[p] for p in agg["reader_policy_label"]], rotation=20, ha="right")
    plt.ylabel("Mean IoU gain over same-usage random")
    plt.title("Trigger intelligence requires beating same-usage random switching")
    plt.grid(axis="y", alpha=0.25)
    savefig(out / "fig_core_informed_gain.png")


def core_latency_fps(master: pd.DataFrame, out: Path) -> None:
    """Simple latency/FPS table-like figure."""
    sub = master[(master["pair_type"] == "n_l") & (master["reader_policy_label"].isin(CORE_POLICIES))].copy()
    sub["fps"] = 1000.0 / sub["t_total_deployed_mean_ms"]
    agg = sub.groupby(["device", "reader_policy_label"], as_index=False).agg(
        latency=("t_total_deployed_mean_ms", "mean"),
        fps=("fps", "mean"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    x = np.arange(len(CORE_POLICIES))
    width = 0.38
    for ax, metric, ylabel in [(axes[0], "latency", "Latency T_total (ms)"), (axes[1], "fps", "FPS")]:
        for j, device in enumerate(["cpu", "cuda"]):
            vals = []
            for p in CORE_POLICIES:
                row = agg[(agg["device"] == device) & (agg["reader_policy_label"] == p)]
                vals.append(float(row[metric].iloc[0]) if not row.empty else np.nan)
            ax.bar(x + (j - 0.5) * width, vals, width, label=device.upper(), color="#4E79A7" if device == "cpu" else "#F28E2B")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        ax.set_xticks(x)
        ax.set_xticklabels([POLICY_LABEL[p] for p in CORE_POLICIES], rotation=25, ha="right")
        ax.legend()
    fig.suptitle("Runtime benchmark on common n_l pair coverage")
    savefig(out / "fig_core_latency_fps.png")


def core_trigger_f1(master: pd.DataFrame, out: Path) -> None:
    """Simple trigger-F1 figure."""
    dyn_order = ["local_contrast_hyst", "conf_ema", "combined_hyst", "multi_proxy"]
    dyn = master[master["reader_policy_label"].isin(dyn_order)]
    agg = dyn.groupby("reader_policy_label", as_index=False).agg(
        precision=("trigger_precision", "mean"),
        recall=("trigger_recall", "mean"),
        f1=("trigger_f1", "mean"),
    ).set_index("reader_policy_label").loc[dyn_order].reset_index()
    x = np.arange(len(agg))
    width = 0.25
    plt.figure(figsize=(9, 4.8))
    plt.bar(x - width, agg["precision"], width, label="Precision", color="#76B7B2")
    plt.bar(x, agg["recall"], width, label="Recall", color="#59A14F")
    plt.bar(x + width, agg["f1"], width, label="F1", color="#E15759")
    plt.xticks(x, [POLICY_LABEL[p] for p in agg["reader_policy_label"]], rotation=20, ha="right")
    plt.ylabel("Score vs benefit-positive frame label")
    plt.title("Trigger alignment with frames where the accurate model helps")
    plt.ylim(0, 0.6)
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    savefig(out / "fig_core_trigger_precision_recall_f1.png")


def core_feature_counts(feature_by_video: pd.DataFrame, out: Path) -> None:
    """Feature family figure that is robust to lighting variability."""
    structural = ["local_contrast", "edge_density", "laplacian", "tenengrad"]
    sub = feature_by_video[(feature_by_video["target"] == "benefit_positive") & (feature_by_video["feature"].isin(structural))]
    top = (
        sub.sort_values("auprc_lift_over_random", ascending=False)
        .groupby(["domain", "video_slug", "model_family", "pair_type", "device"])
        .head(1)
    )
    counts = top.groupby(["domain", "feature"]).size().reset_index(name="top_count")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.5), sharey=True)
    for ax, domain in zip(axes, ["glass", "porcelain"]):
        dom = counts[counts["domain"] == domain].set_index("feature").reindex(structural).fillna(0)
        ax.bar(dom.index, dom["top_count"], color=[COLORS["local_contrast_hyst"], "#6BAED6", "#4E79A7", "#D62728"])
        ax.set_title(domain.capitalize())
        ax.set_ylabel("Times ranked top per video/config")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Best structural-texture feature varies across videos")
    savefig(out / "fig_core_structural_feature_counts.png")


def write_core_guide(out: Path, master: pd.DataFrame) -> None:
    common = master[(master["pair_type"] == "n_l") & (master["reader_policy_label"].isin(CORE_POLICIES))].copy()
    common["fps"] = 1000.0 / common["t_total_deployed_mean_ms"]
    table = common.groupby(["device", "reader_policy_label"], as_index=False).agg(
        iou=("iou_match", "mean"),
        latency_ms=("t_total_deployed_mean_ms", "mean"),
        fps=("fps", "mean"),
        gain=("informed_gain_over_random", "mean"),
        usage=("accurate_usage_rate", "mean"),
    )
    table["policy"] = table["reader_policy_label"].map(POLICY_LABEL)
    table = table[["device", "policy", "iou", "latency_ms", "fps", "gain", "usage"]].round(4)
    table.to_csv(out / "table_core_policy_numbers.csv", index=False)

    lines = [
        "# Core Figure Explanation Guide",
        "",
        "Use these figures instead of the diagnostic plots for thesis writing and viva. Each figure has one message.",
        "",
        "## Figure Order",
        "",
        "1. `fig_core_cpu_tradeoff.png`: CPU is the positive DMS story. Dynamic policies sit between fast-only and accurate-only.",
        "2. `fig_core_cuda_tradeoff.png`: CUDA is the cautionary story. Accurate-only is already very fast, so proxy overhead can remove speed benefits.",
        "3. `fig_core_informed_gain.png`: Raw agreement is not enough; policies must beat same-usage random switching.",
        "4. `fig_core_latency_fps.png`: Converts latency into FPS for easy real-time interpretation.",
        "5. `fig_core_trigger_precision_recall_f1.png`: Shows how well policy switches align with benefit-positive frames.",
        "6. `fig_core_structural_feature_counts.png`: Explains feature validity without overclaiming one universal feature.",
        "",
        "## Exact Explanation",
        "",
        "The final DMS result is not a universal speedup claim. It is a runtime-quality tradeoff. On CPU, the accurate model is much slower than the fast model, so dynamic policies can recover some accurate-model agreement while saving runtime. On CUDA, the accurate model is already close to the fast model in latency, so scene-proxy policies add overhead and can become slower than accurate-only.",
        "",
        "The trigger-intelligence figure uses informed gain over same-usage random switching. This is necessary because a policy can get high agreement simply by choosing the accurate model often. Positive informed gain means the policy selects accurate frames better than random at the same accurate usage.",
        "",
        "Trigger F1 is modest because benefit-positive frames are difficult and sometimes sparse. This is not a failure; it shows why raw switching rate is not enough and why the thesis reports precision, recall, F1, informed gain, and runtime together.",
        "",
        "For features, do not say one scalar feature is universally best. The evidence supports the structural-texture feature family. Lighting and viewpoint differences change which feature ranks highest in a given video/configuration.",
        "",
        "## Core Numbers",
        "",
        dataframe_to_markdown(table),
    ]
    (out / "CORE_FIGURE_EXPLANATION_GUIDE.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    root = output_root()
    out = root / "figures_core"
    ensure_dir(out)
    master = pd.read_csv(root / "master_table_multivideo_final.csv")
    feature_by_video = pd.read_csv(root / "feature_validity_by_video_final.csv")
    core_cpu_tradeoff(master, out)
    core_cuda_tradeoff(master, out)
    core_informed_gain(master, out)
    core_latency_fps(master, out)
    core_trigger_f1(master, out)
    core_feature_counts(feature_by_video, out)
    write_core_guide(out, master)
    print(f"wrote core thesis figures to {out}")


if __name__ == "__main__":
    main()
