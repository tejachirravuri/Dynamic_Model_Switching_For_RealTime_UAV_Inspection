"""Check that a sweep CSV contains an expected policy and diagnostics.

This is intentionally lightweight: it validates the recorded schema after a
smoke/sweep run, but it does not launch inference or modify results.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_REQUIRED = (
    "policy_score",
    "policy_thresh_low",
    "policy_thresh_high",
    "t_total_deployed_ms",
)


def _is_bad(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def check_policy_schema(
    sweep_csv: Path,
    policy: str,
    required_columns: Iterable[str],
) -> None:
    df = pd.read_csv(sweep_csv)
    missing = [col for col in ("policy", *required_columns) if col not in df.columns]
    if missing:
        raise SystemExit(f"{sweep_csv}: missing columns {missing}")

    policy_rows = df[df["policy"] == policy]
    if policy_rows.empty:
        policies = sorted(str(p) for p in df["policy"].dropna().unique())
        raise SystemExit(
            f"{sweep_csv}: policy {policy!r} not found; available={policies}"
        )

    bad = {}
    for col in required_columns:
        n_bad = int(policy_rows[col].map(_is_bad).sum())
        if n_bad:
            bad[col] = n_bad
    if bad:
        raise SystemExit(f"{sweep_csv}: bad values for {policy!r}: {bad}")

    print(
        f"[policy_schema] OK {sweep_csv}: policy={policy} "
        f"rows={len(policy_rows)} checked={','.join(required_columns)}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify policy diagnostics in sweep_per_frame.csv."
    )
    parser.add_argument("sweep_csv", type=Path)
    parser.add_argument("--policy", default="local_contrast_hyst")
    parser.add_argument(
        "--required-columns",
        nargs="+",
        default=list(DEFAULT_REQUIRED),
    )
    args = parser.parse_args()

    check_policy_schema(args.sweep_csv, args.policy, args.required_columns)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
