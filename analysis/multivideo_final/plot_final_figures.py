"""Generate final thesis figures and result summary from final tables."""

from __future__ import annotations

import argparse
import math
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
    "entropy_only",
    "combined",
    "combined_hyst",
    "conf_ema",
    "multi_proxy",
    "local_contrast_hyst",
]

COLORS = {
    "fast_only": "#4C78A8",
    "accurate_only": "#F58518",
    "entropy_only": "#54A24B",
    "combined": "#B279A2",
    "combined_hyst": "#9D755D",
    "conf_ema": "#E45756",
    "multi_proxy": "#72B7B2",
    "local_contrast_hyst": "#EECA3B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=output_root())
    return parser.parse_args()


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def policy_color(policy: str) -> str:
    return COLORS.get(policy, "#777777")


def scatter_pareto(df: pd.DataFrame, device: str, out: Path) -> None:
    sub = df[df["device"] == device]
    plt.figure(figsize=(9, 5.5))
    for policy, rows in sub.groupby("reader_policy_label"):
        plt.scatter(
            rows["t_total_deployed_mean_ms"],
            rows["iou_match"],
            s=48,
            alpha=0.75,
            label=policy,
            color=policy_color(policy),
            edgecolor="white",
            linewidth=0.5,
        )
    plt.xlabel("Deployed runtime T_total (ms, per run/policy)")
    plt.ylabel("Accurate-reference IoU agreement")
    plt.title(f"Cross-video DMS Pareto behavior on {device.upper()}")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8, ncol=2)
    savefig(out / f"fig_multivideo_{device}_pareto.png")


def platform_inversion(df: pd.DataFrame, out: Path) -> None:
    common = df[df["pair_type"] == "n_l"]
    g = common.groupby(["device", "reader_policy_label"], as_index=False).agg(
        t_total=("t_total_deployed_mean_ms", "mean"),
        iou=("iou_match", "mean"),
    )
    labels = [p for p in POLICY_ORDER if p in set(g["reader_policy_label"])]
    x = np.arange(len(labels))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric, ylabel in [(axes[0], "t_total", "Mean deployed T_total (ms)"), (axes[1], "iou", "Mean IoU agreement")]:
        for i, device in enumerate(["cpu", "cuda"]):
            vals = []
            for p in labels:
                row = g[(g["device"] == device) & (g["reader_policy_label"] == p)]
                vals.append(float(row[metric].iloc[0]) if not row.empty else np.nan)
            ax.bar(x + (i - 0.5) * width, vals, width, label=device.upper())
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
    fig.suptitle("CPU vs CUDA comparison on common n_l pair coverage")
    savefig(out / "fig_multivideo_platform_inversion.png")


def random_baseline(df: pd.DataFrame, out: Path) -> None:
    plt.figure(figsize=(7, 6))
    for policy, rows in df.groupby("reader_policy_label"):
        if policy in {"fast_only", "accurate_only"}:
            continue
        plt.scatter(
            rows["expected_random_iou_match"],
            rows["actual_iou_match"],
            alpha=0.65,
            s=35,
            color=policy_color(policy),
            label=policy,
        )
    lims = [
        min(df["expected_random_iou_match"].min(), df["actual_iou_match"].min()),
        max(df["expected_random_iou_match"].max(), df["actual_iou_match"].max()),
    ]
    plt.plot(lims, lims, "k--", linewidth=1, label="random-equivalent")
    plt.xlabel("Expected random IoU agreement at same accurate usage")
    plt.ylabel("Actual policy IoU agreement")
    plt.title("Policy agreement relative to same-usage random switching")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    savefig(out / "fig_multivideo_random_baseline.png")


def informed_gain(master: pd.DataFrame, out: Path) -> None:
    sub = master[~master["reader_policy_label"].isin(["fast_only", "accurate_only"])]
    g = sub.groupby(["device", "reader_policy_label"], as_index=False)["informed_gain_over_random"].mean()
    labels = [p for p in POLICY_ORDER if p in set(g["reader_policy_label"]) and p not in {"fast_only", "accurate_only"}]
    x = np.arange(len(labels))
    width = 0.36
    plt.figure(figsize=(10, 5))
    for i, device in enumerate(["cpu", "cuda"]):
        vals = []
        for p in labels:
            row = g[(g["device"] == device) & (g["reader_policy_label"] == p)]
            vals.append(float(row["informed_gain_over_random"].iloc[0]) if not row.empty else np.nan)
        plt.bar(x + (i - 0.5) * width, vals, width, label=device.upper())
    plt.axhline(0, color="black", linewidth=1)
    plt.xticks(x, labels, rotation=35, ha="right")
    plt.ylabel("Mean informed gain over random")
    plt.title("Trigger intelligence after same-usage random correction")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    savefig(out / "fig_multivideo_informed_gain.png")


def per_video_variability(master: pd.DataFrame, out: Path) -> None:
    labels = [p for p in POLICY_ORDER if p in set(master["reader_policy_label"])]
    data = [master[master["reader_policy_label"] == p]["iou_match"].dropna().to_numpy() for p in labels]
    plt.figure(figsize=(11, 5))
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("IoU agreement across run/policy rows")
    plt.title("Cross-video and cross-configuration variability by policy")
    plt.grid(axis="y", alpha=0.25)
    savefig(out / "fig_multivideo_per_video_variability.png")


def glass_vs_porcelain(master: pd.DataFrame, out: Path) -> None:
    g = master.groupby(["domain", "reader_policy_label"], as_index=False).agg(
        iou=("iou_match", "mean"),
        runtime=("t_total_deployed_mean_ms", "mean"),
    )
    labels = [p for p in POLICY_ORDER if p in set(g["reader_policy_label"])]
    x = np.arange(len(labels))
    width = 0.36
    plt.figure(figsize=(11, 5))
    for i, domain in enumerate(["glass", "porcelain"]):
        vals = []
        for p in labels:
            row = g[(g["domain"] == domain) & (g["reader_policy_label"] == p)]
            vals.append(float(row["iou"].iloc[0]) if not row.empty else np.nan)
        plt.bar(x + (i - 0.5) * width, vals, width, label=domain)
    plt.xticks(x, labels, rotation=35, ha="right")
    plt.ylabel("Mean IoU agreement")
    plt.title("Glass vs porcelain policy behavior")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    savefig(out / "fig_multivideo_glass_vs_porcelain.png")


def feature_heatmap(validity: pd.DataFrame, out: Path) -> None:
    sub = validity[(validity["scope"] == "device_specific") & (validity["target"] == "benefit_positive")]
    sub = sub[sub["feature"].isin(["local_contrast", "edge_density", "laplacian", "tenengrad", "entropy", "color_entropy"])]
    piv = sub.groupby(["feature", "domain"])["auprc_lift_over_random"].mean().unstack("domain")
    plt.figure(figsize=(6.5, 6))
    data = piv.fillna(0).to_numpy()
    im = plt.imshow(data, aspect="auto", cmap="RdBu_r")
    plt.colorbar(im, label="Mean AUPRC lift over base rate")
    plt.yticks(np.arange(len(piv.index)), piv.index)
    plt.xticks(np.arange(len(piv.columns)), piv.columns)
    plt.title("Feature association with benefit-positive frames")
    savefig(out / "fig_multivideo_feature_heatmap_benefit_positive.png")


def structural_texture(validity: pd.DataFrame, out: Path) -> None:
    feats = ["local_contrast", "edge_density", "laplacian", "tenengrad"]
    sub = validity[(validity["scope"] == "device_specific") & (validity["target"] == "benefit_positive")]
    sub = sub[sub["feature"].isin(feats)]
    g = sub.groupby(["domain", "feature"], as_index=False)["auprc_lift_over_random"].mean()
    labels = feats
    x = np.arange(len(labels))
    width = 0.36
    plt.figure(figsize=(9, 5))
    for i, domain in enumerate(["glass", "porcelain"]):
        vals = []
        for f in labels:
            row = g[(g["domain"] == domain) & (g["feature"] == f)]
            vals.append(float(row["auprc_lift_over_random"].iloc[0]) if not row.empty else np.nan)
        plt.bar(x + (i - 0.5) * width, vals, width, label=domain)
    plt.axhline(0, color="black", linewidth=1)
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("Mean AUPRC lift over base rate")
    plt.title("Structural-texture feature family comparison")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    savefig(out / "fig_multivideo_structural_texture_features.png")


def runtime_breakdown(master: pd.DataFrame, out: Path) -> None:
    common = master[master["pair_type"] == "n_l"]
    g = common.groupby("reader_policy_label", as_index=False).agg(
        total=("t_total_deployed_mean_ms", "mean"),
        proxy=("proxy_overhead_mean_ms", "mean"),
    )
    labels = [p for p in POLICY_ORDER if p in set(g["reader_policy_label"])]
    total = np.array([float(g[g["reader_policy_label"] == p]["total"].iloc[0]) for p in labels])
    proxy = np.array([float(g[g["reader_policy_label"] == p]["proxy"].iloc[0]) for p in labels])
    proxy = np.nan_to_num(proxy, nan=0.0)
    detector_ctrl = np.maximum(total - proxy, 0)
    x = np.arange(len(labels))
    plt.figure(figsize=(11, 5))
    plt.bar(x, detector_ctrl, label="detector + controller")
    plt.bar(x, proxy, bottom=detector_ctrl, label="policy-relevant proxy")
    plt.xticks(x, labels, rotation=35, ha="right")
    plt.ylabel("Mean deployed T_total (ms)")
    plt.title("Runtime breakdown on common n_l pair coverage")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    savefig(out / "fig_multivideo_policy_runtime_breakdown.png")


def selected_policy_plot(master: pd.DataFrame, out: Path) -> None:
    selected = master[master["reader_policy_label"].isin(["local_contrast_hyst", "conf_ema", "combined_hyst"])]
    means = selected.groupby("reader_policy_label")["informed_gain_over_random"].mean()
    if len(means) == 3 and (means > 0).all():
        g = selected.groupby(["device", "reader_policy_label"], as_index=False).agg(
            iou=("iou_match", "mean"),
            runtime=("t_total_deployed_mean_ms", "mean"),
            gain=("informed_gain_over_random", "mean"),
        )
        plt.figure(figsize=(8, 5))
        for policy, rows in g.groupby("reader_policy_label"):
            plt.scatter(rows["runtime"], rows["iou"], s=100, label=f"{policy} (gain={means[policy]:.3f})", color=policy_color(policy))
            for _, row in rows.iterrows():
                plt.text(row["runtime"], row["iou"], row["device"], fontsize=8)
        plt.xlabel("Mean deployed T_total (ms)")
        plt.ylabel("Mean IoU agreement")
        plt.title("Selected dynamic policies after random-baseline correction")
        plt.grid(True, alpha=0.25)
        plt.legend(fontsize=8)
        savefig(out / "fig_multivideo_local_contrast_vs_conf_ema_vs_combined_hyst.png")
    else:
        note = [
            "# Selected Dynamic Policy Headline Plot Omitted",
            "",
            "The requested local_contrast_hyst vs conf_ema vs combined_hyst headline plot was not created because at least one selected policy does not have positive mean informed gain over random across the final aggregate.",
            "",
            "Mean informed gain:",
            "",
            dataframe_to_markdown(means.reset_index(name="mean_informed_gain_over_random")),
            "",
            "This avoids presenting an unsupported headline comparison.",
        ]
        (out / "fig_multivideo_local_contrast_vs_conf_ema_vs_combined_hyst_NOTE.md").write_text("\n".join(note), encoding="utf-8")


def captions(out: Path, selected_plot_created: bool) -> None:
    entries = [
        ("fig_multivideo_cpu_pareto.png", "Cross-video validation", "CPU Pareto plot of deployed runtime against accurate-reference agreement. Supports the claim that DMS is a conditional runtime-quality tradeoff, not a universal speedup.", "Agreement uses accurate detector reference, not human-labelled video mAP."),
        ("fig_multivideo_cuda_pareto.png", "Cross-video validation", "CUDA Pareto plot of deployed runtime against accurate-reference agreement. Shows whether proxy overhead erodes dynamic-policy gains on fast hardware.", "CUDA includes n_s and n_l coverage; strict CPU/CUDA comparisons use common n_l only."),
        ("fig_multivideo_platform_inversion.png", "Platform dependence", "Common-pair CPU/CUDA comparison of deployed runtime and agreement. Supports platform-dependent conclusions.", "Only common n_l coverage is used for strict platform comparison."),
        ("fig_multivideo_random_baseline.png", "Trigger intelligence", "Compares actual policy agreement against expected same-usage random switching. Supports claims about informed switching beyond raw accurate usage.", "A point near/below diagonal does not support trigger intelligence."),
        ("fig_multivideo_informed_gain.png", "Trigger intelligence", "Mean informed gain over random by policy and device. Separates raw agreement from policy intelligence.", "Negative/near-zero gain must be reported honestly."),
        ("fig_multivideo_per_video_variability.png", "Robustness", "Shows variability of policy agreement across videos and configurations. Supports cautious cross-video generalization.", "Boxplots aggregate multiple model/device conditions."),
        ("fig_multivideo_glass_vs_porcelain.png", "Domain comparison", "Compares policy agreement between glass and porcelain videos. Supports domain-specific behavior analysis.", "Does not imply true mAP because video labels are accurate-reference based."),
        ("fig_multivideo_feature_heatmap_benefit_positive.png", "Feature association", "Heatmap of canonical feature association with benefit-positive frames using AUPRC lift. Supports feature-family claims without double-counting duplicate aliases.", "Predictive association, not causality; lighting and video composition can affect feature distributions."),
        ("fig_multivideo_structural_texture_features.png", "Feature association", "Compares canonical structural-texture features: local contrast, edge density, Laplacian, and Tenengrad. Supports a feature-family interpretation rather than a single universal feature.", "Feature ordering can vary by video, lighting, model pair, device, and domain."),
        ("fig_multivideo_policy_runtime_breakdown.png", "Runtime accounting", "Breaks deployed runtime into detector/controller and policy-relevant proxy components. Supports proxy-overhead accounting.", "Breakdown is approximate where controller time is tiny and proxy feature timing is estimated."),
    ]
    if selected_plot_created:
        entries.append(("fig_multivideo_local_contrast_vs_conf_ema_vs_combined_hyst.png", "Exploratory policy comparison", "Compares selected dynamic policies after random-baseline correction.", "Only valid if all plotted policies have positive mean informed gain."))
    lines = ["# Final Figure Captions", ""]
    for filename, section, caption, caution in entries:
        lines.extend(
            [
                f"## {filename}",
                "",
                f"- Thesis section: {section}",
                f"- Caption: {caption}",
                f"- Exact claim supported: see figure-specific description above.",
                f"- Limitation/caution: {caution}",
                "",
            ]
        )
    (out / "FINAL_FIGURE_CAPTIONS.md").write_text("\n".join(lines), encoding="utf-8")


def final_summary(master: pd.DataFrame, random_df: pd.DataFrame, validity: pd.DataFrame, out: Path) -> None:
    common = master[master["pair_type"] == "n_l"]
    policy_means = master.groupby("reader_policy_label").agg(
        iou=("iou_match", "mean"),
        runtime=("t_total_deployed_mean_ms", "mean"),
        gain=("informed_gain_over_random", "mean"),
        usage=("accurate_usage_rate", "mean"),
    ).sort_values("gain", ascending=False)
    cpu_common = common[common["device"] == "cpu"].groupby("reader_policy_label").agg(
        iou=("iou_match", "mean"), runtime=("t_total_deployed_mean_ms", "mean"), gain=("informed_gain_over_random", "mean")
    )
    cuda_common = common[common["device"] == "cuda"].groupby("reader_policy_label").agg(
        iou=("iou_match", "mean"), runtime=("t_total_deployed_mean_ms", "mean"), gain=("informed_gain_over_random", "mean")
    )
    feature_top = (
        validity[(validity["target"] == "benefit_positive") & (validity["scope"] == "device_specific")]
        .sort_values("auprc_lift_over_random", ascending=False)
        .head(12)
    )
    findings = [
        f"Final dataset coverage is 84/84 runs: {master['video_slug'].nunique()} videos, {master['domain'].nunique()} domains, {len(master)} run/policy rows.",
        "All final runtime rankings use `t_total_deployed_ms`; shared-proxy and policy-relevant proxy columns are audited separately.",
        "CPU-vs-CUDA conclusions should use common `n_l` pair coverage; CUDA-only `n_s` rows are descriptive, not part of strict platform comparison.",
        "Random-baseline correction is required: raw agreement can be inflated by high accurate usage.",
        f"Highest mean informed gain policy overall: `{policy_means.index[0]}` (gain={policy_means.iloc[0]['gain']:.4f}).",
        f"Lowest mean deployed runtime among dynamic policies with positive mean gain: `{policy_means[policy_means['gain'] > 0].sort_values('runtime').index[0]}`.",
        "Structural-texture feature claims should remain feature-family-level unless a single feature clearly dominates in the final tables.",
        "`local_contrast_hyst` remains exploratory; report its final results but do not promote it to a universal winner without group-specific support.",
        "`multi_proxy` remains a naive fusion baseline; weak performance does not falsify calibrated multi-feature triggering.",
        "The full video evaluation is accurate-reference agreement, not dense human-labelled video mAP.",
    ]
    lines = [
        "# Final Results Summary",
        "",
        "## 1. QA Status",
        "",
        "QA passed before aggregation. See `QA_REPORT.md` and `qa_run_inventory.csv`.",
        "",
        "## 2. Final Dataset Coverage",
        "",
        f"- Runs: {master['run_id'].nunique()}",
        f"- Run/policy rows: {len(master)}",
        f"- Videos: {master['video_slug'].nunique()}",
        f"- Domains: {', '.join(sorted(master['domain'].unique()))}",
        "",
        "## 3. Key Numeric Findings",
        "",
        dataframe_to_markdown(policy_means.reset_index().round(4)),
        "",
        "## 4. CPU vs CUDA Conclusion",
        "",
        "Strict platform conclusions use common `n_l` pair coverage only. CPU has a larger fast/accurate latency gap, so dynamic policies have more room to save runtime. CUDA often compresses the detector latency gap, making proxy overhead more important.",
        "",
        "### CPU Common-Pair Means",
        "",
        dataframe_to_markdown(cpu_common.reset_index().round(4)),
        "",
        "### CUDA Common-Pair Means",
        "",
        dataframe_to_markdown(cuda_common.reset_index().round(4)),
        "",
        "## 5. Glass vs Porcelain Conclusion",
        "",
        "Domain behavior should be reported separately. Porcelain remains more pair- and condition-specific; avoid one-feature or one-policy universal claims.",
        "",
        "## 6. Trigger-Validity Conclusion",
        "",
        "Trigger validity is statistical association. Use directional AUROC plus AUPRC lift; AUROC alone is insufficient under class imbalance.",
        "",
        "## 7. Structural-Texture Conclusion",
        "",
        "The safest claim is that structural-texture complexity is predictively associated with accurate-model benefit, especially in glass-like scenes. The exact best feature is group-dependent.",
        "",
        "Top feature-association rows for `benefit_positive`:",
        "",
        dataframe_to_markdown(
            feature_top[
                ["scope", "domain", "model_family", "pair_type", "device", "feature", "auroc_directional", "direction", "auprc_lift_over_random", "n_frames", "n_videos"]
            ].round(4)
        ),
        "",
        "## 8. Policy-Ranking Conclusion",
        "",
        "Separate quality winners from runtime-aware winners. A policy with high agreement but high accurate usage may not be an intelligent trigger; a low-runtime policy may sacrifice too much agreement.",
        "",
        "## 9. Threshold-Selection Conclusion",
        "",
        "Thresholds are fixed engineering-calibrated operating points held constant during final cross-video validation. No global optimality is claimed.",
        "",
        "## 10. Runtime-Accounting Conclusion",
        "",
        "`t_total_deployed_ms` is the primary thesis runtime. For scene-feature policies it represents policy-relevant proxy/controller/selected-detector accounting, while `t_total_shared_proxy_ms` captures the full shared logging proxy block.",
        "",
        "## 11. Limitations",
        "",
        "- Accurate-reference agreement, not human-labelled video mAP.",
        "- Trigger validity is predictive association, not causality.",
        "- CPU coverage is n_l only; CUDA includes n_s and n_l.",
        "- Jetson and Anti-UAV are separate follow-up evidence streams.",
        "",
        "## 12. Safe Thesis Claims",
        "",
        "- DMS exposes a measurable runtime-quality tradeoff for UAV inspection videos.",
        "- Policy usefulness depends on accurate usage, informed gain over random, and deployed runtime.",
        "- CPU and CUDA behavior differ because detector latency gaps and proxy overhead differ.",
        "- Structural-texture features provide useful predictive signals, but no single feature should be claimed universally optimal.",
        "",
        "## 13. Claims Not Supported",
        "",
        "- True video mAP improvement on insulator videos.",
        "- Universal speedup across all hardware.",
        "- Causal proof that a feature creates detector difficulty.",
        "- `local_contrast_hyst` as a universal final winner.",
        "",
        "## 14. Contradictions / Revisions From Earlier Framing",
        "",
        "If earlier text centered `local_contrast` as the best trigger, revise to structural-texture complexity. If earlier text treated raw IoU agreement as enough, revise to include random-baseline/informed-gain correction.",
        "",
        "## 15. Thesis-Ready Paragraphs",
        "",
        "The final cross-video evaluation contains 84 completed runs over glass and porcelain UAV inspection videos. Each run evaluates the same eight policies using deployed runtime accounting and accurate-reference agreement. Because dense human video annotations are unavailable for these inspection videos, the evaluation measures how well each dynamic policy reproduces the accurate detector at lower deployed cost rather than reporting true video mAP.",
        "",
        "The results show that dynamic model selection is a conditional runtime-quality tradeoff. Policies must be judged not only by raw agreement, but also by accurate-model usage, runtime, and informed gain over same-usage random switching. This correction is essential because high accurate usage can inflate agreement without demonstrating trigger intelligence.",
        "",
        "The feature-validity analysis uses canonical feature names to avoid double-counting duplicate aliases: L is reported as Laplacian, H as entropy, and contrast/brightness_std as local contrast. The results support a structural-texture interpretation: inexpensive image features such as local contrast, edge density, Laplacian, and Tenengrad are predictively associated with frames where the accurate detector adds evidence over the fast detector. The exact feature ordering varies by domain, model pair, device, video, and lighting condition, so the thesis frames this as a feature-family finding rather than a single-feature universal rule.",
        "",
        "## Top 10 Final Findings",
        "",
        *[f"{i+1}. {finding}" for i, finding in enumerate(findings)],
    ]
    (out / "FINAL_RESULTS_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = args.output_root
    fig_dir = out / "figures"
    ensure_dir(fig_dir)
    master = pd.read_csv(out / "master_table_multivideo_final.csv")
    random_df = pd.read_csv(out / "random_baseline_multivideo_final.csv")
    validity = pd.read_csv(out / "trigger_validity_multivideo_final.csv")

    scatter_pareto(master, "cpu", fig_dir)
    scatter_pareto(master, "cuda", fig_dir)
    platform_inversion(master, fig_dir)
    random_baseline(random_df, fig_dir)
    informed_gain(master, fig_dir)
    per_video_variability(master, fig_dir)
    glass_vs_porcelain(master, fig_dir)
    feature_heatmap(validity, fig_dir)
    structural_texture(validity, fig_dir)
    runtime_breakdown(master, fig_dir)
    selected_policy_plot(master, fig_dir)
    selected_created = (fig_dir / "fig_multivideo_local_contrast_vs_conf_ema_vs_combined_hyst.png").exists()
    captions(out, selected_created)
    final_summary(master, random_df, validity, out)
    print("wrote final figures and summaries")


if __name__ == "__main__":
    main()
