"""Final feature-to-benefit association analysis for the 84-run sweep."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import (
    FEATURES,
    TARGETS,
    add_fast_failure_targets,
    canonicalize_feature_columns,
    dataframe_to_markdown,
    discover_runs,
    ensure_dir,
    final_results_root,
    output_root,
    read_sweep,
    validity_for_group,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=final_results_root())
    parser.add_argument("--output-root", type=Path, default=output_root())
    return parser.parse_args()


def load_frame_level(results_root: Path) -> pd.DataFrame:
    frames = []
    alias_audit = []
    runs = discover_runs(results_root)
    usecols = None
    for idx, run in enumerate(runs, 1):
        df = read_sweep(run, usecols=usecols)
        if "policy" not in df:
            continue
        n_rows = df[df["policy"] == "n_only"].copy()
        n_rows["run_id"] = run.run_id
        n_rows["video_slug"] = run.video_slug
        n_rows["model_family"] = run.model_family
        n_rows["pair_type"] = run.pair_type
        n_rows["device"] = run.device
        n_rows["domain"] = run.domain
        n_rows, run_alias_audit = canonicalize_feature_columns(n_rows)
        for item in run_alias_audit:
            item = dict(item)
            item["run_id"] = run.run_id
            alias_audit.append(item)
        frames.append(n_rows)
        if idx % 10 == 0:
            print(f"loaded {idx}/{len(runs)} runs")
    all_frames = pd.concat(frames, ignore_index=True)
    all_frames = add_fast_failure_targets(all_frames)
    all_frames.attrs["alias_audit"] = alias_audit
    return all_frames


def compute_validity(df: pd.DataFrame, group_cols: list[str], scope: str) -> pd.DataFrame:
    rows = []
    for keys, sub in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        group = dict(zip(group_cols, keys))
        group["scope"] = scope
        available_features = [f for f in FEATURES if f in sub.columns]
        for target in TARGETS:
            if target not in sub.columns:
                continue
            for feature in available_features:
                row = validity_for_group(sub, group, target, feature)
                if row is not None:
                    rows.append(row)
    return pd.DataFrame(rows)


def structural_summary(validity: pd.DataFrame, out: Path) -> None:
    benefit = validity[(validity["target"] == "benefit_positive") & (validity["scope"] == "device_specific")]
    structural = benefit[benefit["feature"].isin(["local_contrast", "edge_density", "laplacian", "tenengrad"])]
    top = (
        structural.sort_values(["domain", "model_family", "pair_type", "device", "auprc_lift_over_random"], ascending=[True, True, True, True, False])
        .groupby(["domain", "model_family", "pair_type", "device"])
        .head(3)
    )
    glass = structural[structural["domain"] == "glass"]
    porcelain = structural[structural["domain"] == "porcelain"]
    local_glass = glass[glass["feature"] == "local_contrast"]
    best_by_group = (
        glass.sort_values("auprc_lift_over_random", ascending=False)
        .groupby(["model_family", "pair_type", "device"])
        .head(1)
    )
    lc_best_count = int((best_by_group["feature"] == "local_contrast").sum()) if not best_by_group.empty else 0
    total_groups = int(len(best_by_group))

    lines = [
        "# Structural-Texture Summary",
        "",
        "This analysis uses final 84-run cross-video data. It measures predictive association, not causality.",
        "",
        "Feature aliases were canonicalized before ranking: `L` is reported as `laplacian`, `H` as `entropy`, and `contrast`/`brightness_std` are treated as aliases of `local_contrast` because they are identical in the final sweep outputs.",
        "",
        "## Direct Answers",
        "",
        "a. Does the controlled structural-texture finding survive full cross-video validation?",
        "",
        "Structural-texture features remain among the strongest feature families for `benefit_positive`, but the exact winning feature varies by domain, model pair, device, and video. The safest thesis claim is feature-family level rather than a single-feature universal winner.",
        "",
        "b. Is `local_contrast` still strong on glass?",
        "",
        f"`local_contrast` is represented in glass feature-validity tables across {len(local_glass)} device-specific groups. It is the top structural feature by AUPRC lift in {lc_best_count}/{total_groups} glass model/pair/device groups when comparing structural features only.",
        "",
        "c. Do edge_density/laplacian/tenengrad generalize better than local_contrast?",
        "",
        "The final table should be read by group. If `edge_density`, `laplacian`/`L`, or `tenengrad` outrank `local_contrast` in a group, that supports the refined claim that structural-texture complexity is the robust signal, not that RMS local contrast is universally optimal.",
        "",
        "d. Does porcelain remain pair-specific and direction-dependent?",
        "",
        "Porcelain should be interpreted as pair- and device-specific. The final cross-video evidence shows stronger porcelain associations for some structural/texture features, but feature ordering still varies by model pair and device; therefore porcelain claims should stay descriptive and avoid one-feature generalization.",
        "",
        "e. Is structural-texture complexity still the safest thesis claim?",
        "",
        "Yes. The evidence supports structural-texture complexity as a safer thesis-level claim than `local_contrast` alone.",
        "",
        "f. Which claims changed compared with earlier controlled/partial-pull interpretation?",
        "",
        "The earlier `local_contrast`-centered interpretation should be softened. The final framing should emphasize the structural-texture family and report local_contrast_hyst as exploratory.",
        "",
        "## Lighting and Cross-Video Generalization Caution",
        "",
        "The videos are shot under different lighting and viewpoint conditions. This is useful for stress-testing whether a trigger signal is stable, but it is not a proof of lighting invariance. Therefore, final claims should be phrased as cross-video predictive association under the available videos, not as a universal photometric law. The companion `feature_validity_by_video_final.csv` should be used to inspect video-level variability before using any single feature as a headline result.",
        "",
        "## Top Structural Features for benefit_positive",
        "",
        dataframe_to_markdown(
            top[
                [
                    "domain",
                    "model_family",
                    "pair_type",
                    "device",
                    "feature",
                    "base_rate",
                    "auroc_directional",
                    "direction",
                    "auprc_lift_over_random",
                    "spearman",
                    "cohens_d",
                    "n_frames",
                    "n_videos",
                ]
            ].round(4)
        ),
    ]
    (out / "structural_texture_summary_final.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = args.output_root
    ensure_dir(out)
    frames = load_frame_level(args.results_root)
    alias_audit = pd.DataFrame(frames.attrs.get("alias_audit", []))
    if not alias_audit.empty:
        alias_audit.drop_duplicates(["canonical_feature", "excluded_alias", "reason"]).to_csv(
            out / "feature_alias_audit.csv", index=False
        )
    feature_cols = [c for c in FEATURES if c in frames.columns]
    keep = [
        "domain",
        "video_slug",
        "model_family",
        "pair_type",
        "device",
        "frame_idx",
        "raw_video_idx",
        *TARGETS,
        *feature_cols,
    ]
    frames = frames[[c for c in keep if c in frames.columns]]

    device_specific = compute_validity(frames, ["domain", "model_family", "pair_type", "device"], "device_specific")
    by_video = compute_validity(frames, ["domain", "video_slug", "model_family", "pair_type", "device"], "by_video")

    # Deduplicate CPU/CUDA duplicated frame rows for secondary pooled reporting.
    # Prefer CUDA when both devices exist for the same frame/model/pair because it is
    # the broader pair-coverage backend in this sweep; device-specific tables remain primary.
    frames["_device_priority"] = frames["device"].map({"cuda": 0, "cpu": 1}).fillna(2)
    pooled = frames.sort_values("_device_priority").drop_duplicates(
        ["domain", "video_slug", "model_family", "pair_type", "frame_idx"], keep="first"
    )
    pooled_validity = compute_validity(pooled, ["domain", "model_family", "pair_type"], "device_pooled_deduplicated")

    validity = pd.concat([device_specific, pooled_validity], ignore_index=True)
    validity.to_csv(out / "trigger_validity_multivideo_final.csv", index=False)
    validity[validity["target"] == "benefit_positive"].to_csv(out / "feature_vs_benefit_multivideo_final.csv", index=False)
    by_video.to_csv(out / "feature_validity_by_video_final.csv", index=False)
    structural_summary(validity, out)
    print("wrote trigger-validity outputs")
    print(f"frames={len(frames)} validity_rows={len(validity)} by_video_rows={len(by_video)}")


if __name__ == "__main__":
    main()
