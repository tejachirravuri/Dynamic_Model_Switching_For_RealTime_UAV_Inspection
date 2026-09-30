"""Build compact feature-vs-benefit and policy-vs-benefit tables.

Inputs are existing Stage 3 evidence CSVs:
* analysis/figures/stage3/trigger_validity_all_stage3.csv
* analysis/figures/stage3/master_table_stage3_audit.csv

Outputs:
* analysis/figures/stage3/feature_vs_benefit_stage3.csv
* analysis/figures/stage3/policy_vs_benefit_stage3.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

from analysis.stage3_audit import parse_run_id


DEFAULT_OUT_DIR = Path("analysis/figures/stage3")


def _run_meta(run_id: str) -> Optional[dict]:
    key = parse_run_id(run_id)
    if key is None:
        return None
    return {
        "domain": key.domain,
        "model_family": key.family,
        "pair_type": key.pair,
        "device": key.device,
    }


def build_feature_table(out_dir: Path) -> Path:
    src = out_dir / "trigger_validity_all_stage3.csv"
    if not src.exists():
        raise FileNotFoundError(src)
    df = pd.read_csv(src)
    cols = [
        "domain",
        "model_family",
        "pair_type",
        "target",
        "feature",
        "auroc_directional",
        "direction",
        "auprc_lift_over_random",
        "spearman",
        "cohens_d",
    ]
    out = df[cols].rename(columns={"auprc_lift_over_random": "auprc_lift"})
    out_path = out_dir / "feature_vs_benefit_stage3.csv"
    out.to_csv(out_path, index=False)
    return out_path


def build_policy_table(out_dir: Path, prefer_device: str) -> Path:
    src = out_dir / "master_table_stage3_audit.csv"
    if not src.exists():
        raise FileNotFoundError(src)
    df = pd.read_csv(src)
    df = df[df["run_id"].astype(str).str.startswith("stage3_")].copy()

    meta_rows = []
    for run_id in df["run_id"]:
        meta_rows.append(_run_meta(str(run_id)))
    meta = pd.DataFrame(meta_rows)
    keep = meta.notna().all(axis=1)
    df = df[keep].reset_index(drop=True)
    meta = meta[keep].reset_index(drop=True)
    df = pd.concat([meta, df], axis=1)

    # Keep one device to avoid duplicate frame labels / trigger rows.
    df = df[df["device"] == prefer_device].copy()
    cols = [
        "domain",
        "model_family",
        "pair_type",
        "policy",
        "s_choice_rate",
        "trigger_precision",
        "trigger_recall",
        "trigger_f1",
        "informed_gain_over_random",
        "iou_match_rate",
        "t_total_mean_ms",
    ]
    out = df[cols].rename(columns={
        "trigger_f1": "trigger_F1",
        "iou_match_rate": "iou_match",
        "t_total_mean_ms": "t_total_deployed_ms",
    })
    out = out.sort_values(["domain", "model_family", "pair_type", "policy"])
    out_path = out_dir / "policy_vs_benefit_stage3.csv"
    out.to_csv(out_path, index=False)
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build compact Stage 3 feature/policy benefit tables.",
    )
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--prefer-device", type=str, default="cpu")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    feature_path = build_feature_table(args.out_dir)
    policy_path = build_policy_table(args.out_dir, args.prefer_device)
    print(f"[benefit_tables] wrote {feature_path}")
    print(f"[benefit_tables] wrote {policy_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
