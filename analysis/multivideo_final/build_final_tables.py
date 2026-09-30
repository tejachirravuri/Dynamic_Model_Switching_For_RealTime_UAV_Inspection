"""Build final multi-video aggregate tables and policy rankings."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    EXPECTED_POLICIES,
    dataframe_to_markdown,
    discover_runs,
    ensure_dir,
    expected_random_metrics,
    final_results_root,
    output_root,
    policy_trigger_metrics,
    read_summary,
    read_sweep,
    reader_policy_label,
    summarize_group,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=final_results_root())
    parser.add_argument("--output-root", type=Path, default=output_root())
    return parser.parse_args()


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def quantile(series: pd.Series, q: float) -> float:
    s = numeric(series).dropna()
    return float(s.quantile(q)) if len(s) else math.nan


def mean_or_nan(series: pd.Series) -> float:
    s = numeric(series).dropna()
    return float(s.mean()) if len(s) else math.nan


def runtime_series(rows: pd.DataFrame, preferred: str, fallback: str) -> pd.Series:
    """Return the requested timing series, falling back only if necessary."""
    if preferred in rows:
        return numeric(rows[preferred])
    if fallback in rows:
        return numeric(rows[fallback])
    return pd.Series(dtype=float)


def deployed_inference_ms(rows: pd.DataFrame, policy: str) -> pd.Series:
    """Detector inference component under deployed policy semantics.

    Scene policies pay only the selected detector. `conf_ema` is different:
    it must always run the fast detector to obtain the confidence signal and
    additionally runs the accurate detector when it selects accurate.
    """
    if rows.empty:
        return pd.Series(dtype=float)
    if policy == "conf_ema" and {"n_latency_ms", "s_latency_ms", "s_choice"}.issubset(rows.columns):
        n = numeric(rows["n_latency_ms"])
        s = numeric(rows["s_latency_ms"])
        choice = numeric(rows["s_choice"]).fillna(0.0)
        return n + choice * s
    if "chosen_latency_ms" in rows:
        return numeric(rows["chosen_latency_ms"])
    return pd.Series(dtype=float)


def summarize_run_policy(run, summary: dict, df: pd.DataFrame, policy: str) -> dict[str, object]:
    rows = df[df["policy"] == policy]
    per_policy = summary.get("per_policy", {}).get(policy, {})
    random = expected_random_metrics(df, policy)
    trigger = policy_trigger_metrics(df, policy)
    fast_rows = df[df["policy"] == "n_only"]
    base_rows = fast_rows if not fast_rows.empty else rows
    # Primary thesis runtime: policy-relevant deployed accounting.
    # This avoids charging a policy for all features in the shared logging
    # proxy block when it only requires a subset, e.g. local_contrast_hyst.
    t = runtime_series(rows, "t_total_policy_relevant_ms", "t_total_deployed_ms")
    t_logged = runtime_series(rows, "t_total_deployed_ms", "t_total_policy_relevant_ms")
    t_shared = runtime_series(rows, "t_total_shared_proxy_ms", "t_total_deployed_ms")
    t_scene = runtime_series(rows, "proxy_ms_policy_relevant", "proxy_overhead_ms")
    t_ctrl = numeric(rows["decision_ms"]) if "decision_ms" in rows else pd.Series(dtype=float)
    t_selected = numeric(rows["chosen_latency_ms"]) if "chosen_latency_ms" in rows else pd.Series(dtype=float)
    t_infer_deployed = deployed_inference_ms(rows, policy)
    proxy_col = "proxy_ms_policy_relevant" if "proxy_ms_policy_relevant" in rows else "proxy_overhead_ms"
    row = {
        "run_id": run.run_id,
        "domain": run.domain,
        "video_slug": run.video_slug,
        "model_family": run.model_family,
        "pair_type": run.pair_type,
        "device": run.device,
        "policy": policy,
        "reader_policy_label": reader_policy_label(policy),
        "accurate_usage_rate": float(per_policy.get("s_choice_rate", numeric(rows["s_choice"]).mean() if "s_choice" in rows else math.nan)),
        "iou_match": float(per_policy.get("iou_match_rate", numeric(rows["iou_match"]).mean() if "iou_match" in rows else math.nan)),
        "det_coverage": float(per_policy.get("det_coverage_rate", numeric(rows["det_coverage"]).mean() if "det_coverage" in rows else math.nan)),
        "det_recall": float(per_policy.get("det_recall_mean", numeric(rows["det_recall"]).mean() if "det_recall" in rows else math.nan)),
        "count_agree": float(per_policy.get("count_agree_rate", numeric(rows["count_agree"]).mean() if "count_agree" in rows else math.nan)),
        "benefit_rate": float(per_policy.get("benefit_rate", numeric(rows["benefit_positive"]).mean() if "benefit_positive" in rows else math.nan)),
        "trigger_precision": trigger["trigger_precision"],
        "trigger_recall": trigger["trigger_recall"],
        "trigger_f1": trigger["trigger_f1"],
        "expected_random_iou_match": random["expected_random_iou_match"],
        "expected_random_det_coverage": random["expected_random_det_coverage"],
        "expected_random_det_recall": random["expected_random_det_recall"],
        "expected_random_count_agree": random["expected_random_count_agree"],
        "runtime_accounting_basis": "policy_relevant_proxy",
        "t_scene_policy_relevant_mean_ms": mean_or_nan(t_scene),
        "t_ctrl_mean_ms": mean_or_nan(t_ctrl),
        "t_selected_inference_mean_ms": mean_or_nan(t_selected),
        "t_inference_deployed_mean_ms": mean_or_nan(t_infer_deployed),
        "t_total_deployed_mean_ms": float(t.mean()) if len(t) else math.nan,
        "t_total_deployed_p50_ms": quantile(t, 0.50),
        "t_total_deployed_p95_ms": quantile(t, 0.95),
        "t_total_deployed_p99_ms": quantile(t, 0.99),
        "t_total_policy_relevant_mean_ms": mean_or_nan(t),
        "t_total_policy_relevant_p50_ms": quantile(t, 0.50),
        "t_total_policy_relevant_p95_ms": quantile(t, 0.95),
        "t_total_policy_relevant_p99_ms": quantile(t, 0.99),
        "t_total_shared_proxy_mean_ms": mean_or_nan(t_shared),
        "t_total_shared_proxy_p50_ms": quantile(t_shared, 0.50),
        "t_total_shared_proxy_p95_ms": quantile(t_shared, 0.95),
        "t_total_shared_proxy_p99_ms": quantile(t_shared, 0.99),
        "t_total_logged_deployed_mean_ms": mean_or_nan(t_logged),
        "t_total_logged_deployed_p50_ms": quantile(t_logged, 0.50),
        "t_total_logged_deployed_p95_ms": quantile(t_logged, 0.95),
        "t_total_logged_deployed_p99_ms": quantile(t_logged, 0.99),
        "fps_deployed": float(1000.0 / t.mean()) if len(t) and t.mean() else math.nan,
        "proxy_overhead_mean_ms": float(numeric(rows[proxy_col]).mean()) if proxy_col in rows else math.nan,
        "proxy_policy_relevant_mean_ms": float(numeric(rows[proxy_col]).mean()) if proxy_col in rows else math.nan,
        "proxy_shared_mean_ms": mean_or_nan(numeric(rows["proxy_ms_total"])) if "proxy_ms_total" in rows else math.nan,
        "fast_latency_mean_ms": float(numeric(base_rows["n_latency_ms"]).mean()) if "n_latency_ms" in base_rows else math.nan,
        "accurate_latency_mean_ms": float(numeric(base_rows["s_latency_ms"]).mean()) if "s_latency_ms" in base_rows else math.nan,
        "n_frames": int(rows["frame_idx"].nunique()) if "frame_idx" in rows else int(per_policy.get("n_frames", 0)),
    }
    row["informed_gain_over_random"] = row["iou_match"] - row["expected_random_iou_match"]
    return row


def threshold_table(master: pd.DataFrame) -> pd.DataFrame:
    specs = [
        {
            "policy": "n_only",
            "threshold_parameters_used": "none",
            "selection_rationale": "fast detector baseline",
            "status": "fixed baseline",
            "sensitivity_evidence": "not applicable",
            "final_interpretation": "lower-cost baseline, not a trigger",
        },
        {
            "policy": "s_only",
            "threshold_parameters_used": "none",
            "selection_rationale": "accurate detector reference",
            "status": "fixed baseline",
            "sensitivity_evidence": "not applicable",
            "final_interpretation": "upper agreement reference, highest detector cost",
        },
        {
            "policy": "entropy_only",
            "threshold_parameters_used": "entropy threshold from locked policy defaults",
            "selection_rationale": "single cheap scene-complexity operating point",
            "status": "fixed/default",
            "sensitivity_evidence": "controlled evaluation and cross-video validation",
            "final_interpretation": "simple scene trigger; interpret through random-baseline correction",
        },
        {
            "policy": "combined",
            "threshold_parameters_used": "combined L/H score thresholds from locked policy defaults",
            "selection_rationale": "engineering-calibrated scene-complexity operating point",
            "status": "fixed/default",
            "sensitivity_evidence": "controlled threshold/proxy ablations",
            "final_interpretation": "quality-oriented proxy policy, overhead-sensitive",
        },
        {
            "policy": "combined_hyst",
            "threshold_parameters_used": "low/high hysteresis thresholds from locked policy defaults",
            "selection_rationale": "reduce switching oscillation around scene-complexity boundary",
            "status": "fixed/default with hysteresis",
            "sensitivity_evidence": "controlled threshold/proxy ablations",
            "final_interpretation": "hysteretic scene policy; not globally optimal",
        },
        {
            "policy": "conf_ema",
            "threshold_parameters_used": "c_low=0.007, c_high=0.02, fast_beta=0.3, slow_beta=0.02, warmup=5",
            "selection_rationale": "runtime-aware watchdog using fast detector confidence history",
            "status": "fixed/default override in sweep",
            "sensitivity_evidence": "confidence-threshold sweep and cross-video validation",
            "final_interpretation": "proxy-overhead-sensitive runtime-aware candidate, not necessarily highest agreement",
        },
        {
            "policy": "multi_proxy",
            "threshold_parameters_used": "naive multi-feature fusion defaults",
            "selection_rationale": "baseline for uncalibrated multi-feature fusion",
            "status": "naive baseline",
            "sensitivity_evidence": "cross-video validation",
            "final_interpretation": "weak performance does not disprove calibrated multi-feature triggers",
        },
        {
            "policy": "local_contrast_hyst",
            "threshold_parameters_used": "rolling-normalised local_contrast_low=0.45, high=0.65",
            "selection_rationale": "trigger-validity-informed exploratory glass structural-texture policy",
            "status": "exploratory, not canonical",
            "sensitivity_evidence": "feature association analysis and cross-video validation",
            "final_interpretation": "report cautiously; not a universal winner unless supported by final data",
        },
    ]
    stats = master.groupby("policy").agg(
        accurate_usage_min=("accurate_usage_rate", "min"),
        accurate_usage_mean=("accurate_usage_rate", "mean"),
        accurate_usage_max=("accurate_usage_rate", "max"),
        informed_gain_mean=("informed_gain_over_random", "mean"),
    )
    rows = []
    for spec in specs:
        st = stats.loc[spec["policy"]] if spec["policy"] in stats.index else pd.Series(dtype=float)
        rows.append(
            {
                **spec,
                "reader_policy_label": reader_policy_label(spec["policy"]),
                "accurate_usage_rate_min": st.get("accurate_usage_min", math.nan),
                "accurate_usage_rate_mean": st.get("accurate_usage_mean", math.nan),
                "accurate_usage_rate_max": st.get("accurate_usage_max", math.nan),
                "informed_gain_over_random_mean": st.get("informed_gain_mean", math.nan),
            }
        )
    return pd.DataFrame(rows)


def ranking_outputs(master: pd.DataFrame, out: Path) -> None:
    dynamic = master[~master["policy"].isin(["n_only", "s_only"])].copy()
    common = master[master["pair_type"] == "n_l"].copy()
    rows = []

    def add_rank(category: str, df: pd.DataFrame, metric: str, ascending: bool = False, note: str = "") -> None:
        if df.empty:
            return
        grouped = df.groupby("policy", dropna=False).agg(
            metric_value=(metric, "mean"),
            iou_match=("iou_match", "mean"),
            t_total_deployed_mean_ms=("t_total_deployed_mean_ms", "mean"),
            accurate_usage_rate=("accurate_usage_rate", "mean"),
            informed_gain_over_random=("informed_gain_over_random", "mean"),
        )
        grouped = grouped.sort_values("metric_value", ascending=ascending)
        for rank, (policy, vals) in enumerate(grouped.iterrows(), 1):
            rows.append(
                {
                    "category": category,
                    "rank": rank,
                    "policy": policy,
                    "reader_policy_label": reader_policy_label(policy),
                    "ranking_metric": metric,
                    **vals.to_dict(),
                    "note": note,
                }
            )

    add_rank("best_agreement_dynamic", dynamic, "iou_match", note="Dynamic policies only; agreement is not trigger intelligence by itself.")
    add_rank("best_runtime_aware_dynamic", dynamic[dynamic["informed_gain_over_random"] > 0], "t_total_deployed_mean_ms", True, "Positive informed gain required; lower runtime wins.")
    add_rank("best_cpu_common_pair_dynamic", common[(common["device"] == "cpu") & (~common["policy"].isin(["n_only", "s_only"]))], "informed_gain_over_random")
    add_rank("best_cuda_common_pair_dynamic", common[(common["device"] == "cuda") & (~common["policy"].isin(["n_only", "s_only"]))], "informed_gain_over_random")
    add_rank("best_glass_dynamic", dynamic[dynamic["domain"] == "glass"], "informed_gain_over_random")
    add_rank("best_porcelain_dynamic", dynamic[dynamic["domain"] == "porcelain"], "informed_gain_over_random")
    add_rank("best_robust_dynamic", dynamic, "informed_gain_over_random")
    add_rank("best_exploratory_policy", master[master["policy"] == "local_contrast_hyst"], "informed_gain_over_random")

    ranking = pd.DataFrame(rows)
    dominated = dynamic.groupby("policy").agg(
        mean_informed_gain=("informed_gain_over_random", "mean"),
        mean_iou=("iou_match", "mean"),
        mean_runtime=("t_total_deployed_mean_ms", "mean"),
    ).reset_index()
    dominated["dominated_or_concern"] = np.where(
        dominated["mean_informed_gain"] <= 0,
        "worse_than_or_equal_random_on_average",
        "",
    )
    ranking.to_csv(out / "final_policy_ranking.csv", index=False)
    dominated.to_csv(out / "final_policy_dominated_baseline_audit.csv", index=False)

    top_lines = [
        "# Final Policy Ranking",
        "",
        "Rankings are computed after random-baseline correction where relevant. `informed_gain_over_random` is the primary trigger-intelligence signal.",
        "",
        "Important caution: quality winners and runtime winners are different concepts; CPU/CUDA conclusions use common `n_l` pair coverage where platform comparisons are strict.",
        "",
        "## Ranking Table",
        "",
        dataframe_to_markdown(ranking.head(80)),
        "",
        "## Dominated / Concern Audit",
        "",
        dataframe_to_markdown(dominated),
    ]
    (out / "final_policy_ranking.md").write_text("\n".join(top_lines), encoding="utf-8")


def threshold_markdown(out: Path) -> None:
    lines = [
        "# Threshold Selection Rationale",
        "",
        "Detector thresholds are fixed uniformly across runs: `conf_floor=0.25` and `iou_threshold=0.5`. These thresholds define detection filtering and matching, not policy tuning.",
        "",
        "Policy thresholds are engineering-calibrated operating points selected before final cross-video validation. The final 84-run evaluation holds them fixed; it does not tune thresholds per video.",
        "",
        "Hysteresis thresholds are used to reduce oscillation near a decision boundary. They introduce temporal stability, but they are not claimed to be globally optimal.",
        "",
        "Proxy-size and proxy-update-stride/k ablations test sensitivity to proxy overhead and temporal reuse. Threshold sweeps provide evidence that conclusions are not based only on a single unexplained setting.",
        "",
        "The thesis should state that thresholds are defensible fixed operating points, not theoretically optimal values.",
    ]
    (out / "threshold_selection_rationale.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = args.output_root
    ensure_dir(out)
    runs = discover_runs(args.results_root)
    if len(runs) != 84:
        raise SystemExit(f"Expected 84 runs before aggregation, found {len(runs)}")

    master_rows = []
    random_rows = []
    for idx, run in enumerate(runs, 1):
        summary = read_summary(run)
        df = read_sweep(run)
        for policy in EXPECTED_POLICIES:
            row = summarize_run_policy(run, summary, df, policy)
            master_rows.append(row)
            random_rows.append(
                {
                    "run_id": row["run_id"],
                    "domain": row["domain"],
                    "video_slug": row["video_slug"],
                    "model_family": row["model_family"],
                    "pair_type": row["pair_type"],
                    "device": row["device"],
                    "policy": row["policy"],
                    "reader_policy_label": row["reader_policy_label"],
                    "actual_iou_match": row["iou_match"],
                    "accurate_usage_rate": row["accurate_usage_rate"],
                    "expected_random_iou_match": row["expected_random_iou_match"],
                    "informed_gain_over_random": row["informed_gain_over_random"],
                    "expected_random_det_coverage": row["expected_random_det_coverage"],
                    "expected_random_det_recall": row["expected_random_det_recall"],
                    "expected_random_count_agree": row["expected_random_count_agree"],
                }
            )
        if idx % 10 == 0:
            print(f"processed {idx}/{len(runs)} runs")

    master = pd.DataFrame(master_rows)
    random_df = pd.DataFrame(random_rows)
    master.to_csv(out / "master_table_multivideo_final.csv", index=False)
    master.to_csv(out / "per_video_policy_summary_final.csv", index=False)
    summarize_group(master, ["domain", "model_family", "pair_type", "device", "policy", "reader_policy_label"]).to_csv(
        out / "policy_summary_multivideo_final.csv", index=False
    )
    summarize_group(master, ["domain", "device", "policy", "reader_policy_label"]).to_csv(
        out / "domain_policy_summary_final.csv", index=False
    )
    common = master[master["pair_type"] == "n_l"]
    platform = summarize_group(common, ["device", "policy", "reader_policy_label"])
    platform.insert(0, "comparison_scope", "common_pair_only_n_l")
    platform.to_csv(out / "platform_policy_summary_final.csv", index=False)
    unbalanced = summarize_group(master, ["device", "policy", "reader_policy_label"])
    unbalanced.insert(0, "comparison_scope", "descriptive_unbalanced_all_available_pairs")
    unbalanced.to_csv(out / "platform_policy_summary_unbalanced_descriptive.csv", index=False)
    random_df.to_csv(out / "random_baseline_multivideo_final.csv", index=False)

    thresholds = threshold_table(master)
    thresholds.to_csv(out / "threshold_policy_justification_table.csv", index=False)
    threshold_markdown(out)
    ranking_outputs(master, out)

    print("wrote final aggregate tables")
    print(f"master rows={len(master)}")


if __name__ == "__main__":
    main()
