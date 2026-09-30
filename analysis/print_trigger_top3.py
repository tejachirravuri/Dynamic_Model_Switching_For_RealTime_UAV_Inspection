r"""Print top trigger-validity features for benefit_positive.

Run after analysis/figures/stage3/trigger_validity_all_stage3.csv exists:
    python analysis\print_trigger_top3.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


CSV_PATH = Path("analysis/figures/stage3/trigger_validity_all_stage3.csv")


def main() -> int:
    if not CSV_PATH.exists():
        raise SystemExit(f"missing CSV: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    df = df[df["target"] == "benefit_positive"].copy()
    df = df.sort_values(
        ["domain", "model_family", "pair_type", "auroc_directional"],
        ascending=[True, True, True, False],
    )

    cols = [
        "feature",
        "base_rate",
        "auroc_raw",
        "auroc_directional",
        "direction",
        "auprc",
        "auprc_lift_over_random",
        "spearman",
        "cohens_d",
        "cliffs_delta",
    ]

    for key, group in df.groupby(["domain", "model_family", "pair_type"]):
        domain, family, pair_type = key
        print(f"\n{domain} / {family} / {pair_type}")
        print(group[cols].head(3).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
