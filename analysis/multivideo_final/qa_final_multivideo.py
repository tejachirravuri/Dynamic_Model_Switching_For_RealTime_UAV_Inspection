"""QA checks for the final 84-run multi-video sweep pull."""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from common import (
    CRITICAL_TIMING_COLS,
    EXPECTED_POLICIES,
    RUNS_PER_VIDEO,
    dataframe_to_markdown,
    discover_runs,
    ensure_dir,
    final_results_root,
    output_root,
    read_sweep,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=final_results_root())
    parser.add_argument("--output-root", type=Path, default=output_root())
    return parser.parse_args()


def log_findings(log_path: Path) -> tuple[bool, list[str], list[str]]:
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    failures = re.findall(r"\[FAIL\]\s+([^\n]+)", text)
    suspicious = []
    if "Traceback" in text or "ERROR" in text or "Exception" in text:
        suspicious.append("error_or_exception_text")
    if "mean(s_count) = 0.000" in text:
        suspicious.append("zero_accurate_detections_validation_warning")
    passed = "all gates passed" in text and not failures
    return passed, failures, suspicious


def main() -> None:
    args = parse_args()
    out = args.output_root
    ensure_dir(out)

    runs = discover_runs(args.results_root)
    critical_errors: list[str] = []
    warnings: list[str] = []

    if len(runs) != 84:
        critical_errors.append(f"Expected 84 runs, found {len(runs)}")

    run_ids = [r.run_id for r in runs]
    dupes = [rid for rid, n in Counter(run_ids).items() if n > 1]
    if dupes:
        critical_errors.append(f"Duplicate run IDs: {dupes}")

    file_counts = {
        "sweep_summary.json": len(list(args.results_root.rglob("sweep_summary.json"))),
        "sweep_per_frame.csv": len(list(args.results_root.rglob("sweep_per_frame.csv"))),
        "frame_features.csv": len(list(args.results_root.rglob("frame_features.csv"))),
        "run.log": len(list(args.results_root.rglob("run.log"))),
    }
    for name, count in file_counts.items():
        if count != 84:
            critical_errors.append(f"Expected 84 {name} files, found {count}")

    videos_by_domain = defaultdict(set)
    for run in runs:
        videos_by_domain[run.domain].add(run.video_slug)
    if len(videos_by_domain["glass"]) != 9:
        critical_errors.append(f"Expected 9 glass videos, found {len(videos_by_domain['glass'])}")
    if len(videos_by_domain["porcelain"]) != 5:
        critical_errors.append(f"Expected 5 porcelain videos, found {len(videos_by_domain['porcelain'])}")

    expected = set()
    for domain, videos in videos_by_domain.items():
        for video in videos:
            for model_family, pair_type, device in RUNS_PER_VIDEO:
                expected.add((domain, video, model_family, pair_type, device))
    observed = {(r.domain, r.video_slug, r.model_family, r.pair_type, r.device) for r in runs}
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing:
        critical_errors.append(f"Missing expected tuples: {missing[:20]}{'...' if len(missing) > 20 else ''}")
    if extra:
        warnings.append(f"Unexpected tuples: {extra[:20]}{'...' if len(extra) > 20 else ''}")

    inventory_rows = []
    all_columns: dict[str, set[str]] = {}
    for i, run in enumerate(runs, 1):
        run_status = "PASS"
        run_errors: list[str] = []
        run_warnings: list[str] = []
        required = {
            "sweep_summary.json": run.run_dir / "sweep_summary.json",
            "sweep_per_frame.csv": run.run_dir / "sweep_per_frame.csv",
            "frame_features.csv": run.run_dir / "frame_features.csv",
            "run.log": run.run_dir / "run.log",
        }
        for label, path in required.items():
            if not path.exists():
                run_status = "FAIL"
                run_errors.append(f"missing {label}")
        if (run.run_dir / "sweep_per_frame.csv").exists():
            df = read_sweep(run)
            all_columns[run.run_id] = set(df.columns)
            policies = sorted(df["policy"].dropna().unique().tolist()) if "policy" in df else []
            missing_policies = sorted(set(EXPECTED_POLICIES) - set(policies))
            if missing_policies:
                run_status = "FAIL"
                run_errors.append(f"missing policies {missing_policies}")
            if "t_total_deployed_ms" not in df:
                run_status = "FAIL"
                run_errors.append("missing t_total_deployed_ms")
            elif pd.to_numeric(df["t_total_deployed_ms"], errors="coerce").isna().any():
                run_status = "FAIL"
                run_errors.append("NaN/empty t_total_deployed_ms")
            for col in CRITICAL_TIMING_COLS:
                if col not in df.columns:
                    run_warnings.append(f"missing timing column {col}")
            for col in ["t_total_selected_ms", "t_total_shared_proxy_ms", "t_total_policy_relevant_ms", "proxy_ms_policy_relevant", "proxy_ms_total"]:
                if col not in df.columns:
                    run_warnings.append(f"missing optional runtime-accounting column {col}")
            lch = df[df["policy"] == "local_contrast_hyst"] if "policy" in df else pd.DataFrame()
            for col in ["policy_score", "policy_thresh_low", "policy_thresh_high", "t_total_deployed_ms"]:
                if col not in lch:
                    run_status = "FAIL"
                    run_errors.append(f"local_contrast_hyst missing {col}")
                elif pd.to_numeric(lch[col], errors="coerce").isna().all():
                    run_status = "FAIL"
                    run_errors.append(f"local_contrast_hyst empty {col}")
            if "policy_thresh_mid" in lch and pd.to_numeric(lch["policy_thresh_mid"], errors="coerce").isna().all():
                run_warnings.append("local_contrast_hyst policy_thresh_mid empty")
            if "s_count" in df:
                s_mean = pd.to_numeric(df["s_count"], errors="coerce").mean()
                if s_mean < 0.5:
                    run_warnings.append(f"suspicious low mean(s_count)={s_mean:.3f}")
        log_pass, log_failures, log_suspicious = log_findings(run.run_dir / "run.log")
        if log_failures:
            run_warnings.append("validation failures in log: " + "; ".join(log_failures))
        run_warnings.extend(log_suspicious)
        inventory_rows.append(
            {
                "run_id": run.run_id,
                "domain": run.domain,
                "video_slug": run.video_slug,
                "model_family": run.model_family,
                "pair_type": run.pair_type,
                "device": run.device,
                "status": run_status,
                "errors": " | ".join(run_errors),
                "warnings": " | ".join(run_warnings),
                "log_all_gates_passed": log_pass,
                "run_dir": str(run.run_dir),
            }
        )
        if run_errors:
            critical_errors.append(f"{run.run_id}: {'; '.join(run_errors)}")

    inv = pd.DataFrame(inventory_rows)
    inv.to_csv(out / "qa_run_inventory.csv", index=False)

    all_seen_cols = sorted(set().union(*all_columns.values())) if all_columns else []
    col_rows = []
    for col in all_seen_cols:
        present = sum(1 for cols in all_columns.values() if col in cols)
        col_rows.append({"column": col, "present_in_runs": present, "missing_in_runs": len(all_columns) - present})
    col_df = pd.DataFrame(col_rows)

    column_map_lines = [
        "# Column Map Audit",
        "",
        f"Runs inspected: {len(runs)}",
        "",
        "Reader-facing label map:",
        "",
        "- `n_only` -> `fast_only`",
        "- `s_only` -> `accurate_only`",
        "- `s_choice_rate` -> `accurate_usage_rate`",
        "- `n_latency_ms` -> `fast_latency_ms`",
        "- `s_latency_ms` -> `accurate_latency_ms`",
        "",
        "Column presence:",
        "",
        dataframe_to_markdown(col_df) if not col_df.empty else "No columns found.",
    ]
    (out / "column_map_audit.md").write_text("\n".join(column_map_lines), encoding="utf-8")

    runtime_lines = [
        "# Runtime Accounting Audit",
        "",
        "Primary ranking runtime column: `t_total_deployed_ms`.",
        "",
        "Runtime semantics used in final tables:",
        "",
        "- `n_only`: fast detector runtime (`T_fast`).",
        "- `s_only`: accurate detector runtime (`T_accurate`).",
        "- Scene-feature policies: policy-required proxy features + controller + selected detector.",
        "- `conf_ema`: fast watchdog + controller + accurate detector only when selected.",
        "",
        "Important available columns:",
        "",
        "- `t_total_shared_proxy_ms`: full shared proxy/logging block cost.",
        "- `t_total_policy_relevant_ms`: estimated cost of only policy-required proxy features.",
        "- `t_total_deployed_ms`: final deployed runtime used for ranking.",
        "",
        "Presence audit:",
        "",
        dataframe_to_markdown(col_df[col_df["column"].str.contains("t_total|proxy_ms|decision_ms|latency", regex=True)]),
    ]
    (out / "runtime_accounting_audit.md").write_text("\n".join(runtime_lines), encoding="utf-8")

    status = "PASS" if not critical_errors else "FAIL"
    critical_lines = [f"- {e}" for e in critical_errors] if critical_errors else ["- None"]
    warning_lines = [f"- {w}" for w in warnings] if warnings else ["- None"]
    warned_inv = inv[(inv["errors"] != "") | (inv["warnings"] != "")]
    warned_table = (
        dataframe_to_markdown(warned_inv[["run_id", "status", "errors", "warnings"]])
        if not warned_inv.empty
        else "None."
    )

    qa_lines = [
        "# Final Multi-Video QA Report",
        "",
        f"QA status: **{status}**",
        "",
        f"Results root: `{args.results_root}`",
        "",
        "File counts:",
        "",
        *[f"- `{k}`: {v}" for k, v in file_counts.items()],
        "",
        f"Run folders discovered from summaries: {len(runs)}",
        f"Unique run IDs: {len(set(run_ids))}",
        f"Glass videos: {len(videos_by_domain['glass'])} ({', '.join(sorted(videos_by_domain['glass']))})",
        f"Porcelain videos: {len(videos_by_domain['porcelain'])} ({', '.join(sorted(videos_by_domain['porcelain']))})",
        "",
        "Critical errors:",
        "",
        *critical_lines,
        "",
        "Warnings:",
        "",
        *warning_lines,
        "",
        "Per-run status summary:",
        "",
        dataframe_to_markdown(inv.groupby("status").size().reset_index(name="n")),
        "",
        "Runs with warnings or errors:",
        "",
        warned_table,
    ]
    (out / "QA_REPORT.md").write_text("\n".join(qa_lines), encoding="utf-8")

    if critical_errors:
        raise SystemExit("QA failed. See analysis/figures/stage3/multivideo_final/QA_REPORT.md")
    print("QA PASS")
    print(f"runs={len(runs)} files={file_counts}")


if __name__ == "__main__":
    main()
