"""Single source of truth for filesystem paths.

All paths used by the library, experiments, and analysis modules are
defined here. External roots (datasets, model weights, legacy results)
can be overridden via environment variables for portability between
the Windows workstation, the remote Linux box, and the Jetson Nano.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo root — this file's parent
# ---------------------------------------------------------------------------
REPO_ROOT: Path = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# External thesis roots (override via env vars)
# ---------------------------------------------------------------------------
THESIS_ROOT: Path = Path(
    os.environ.get("DMS_THESIS_ROOT", "G:/Teja_Master_Thesis")
)
DELIVERABLE_ROOT: Path = THESIS_ROOT / "thesis_deliverable"
LEGACY_PATHB_ROOT: Path = THESIS_ROOT / "pathB"
LEGACY_PATHA_ROOT: Path = THESIS_ROOT / "MT_Chirravuri"

# ---------------------------------------------------------------------------
# Trained models (kept from previous work, not retrained)
# ---------------------------------------------------------------------------
PATHB_MODELS: Path = LEGACY_PATHB_ROOT / "models"
JETSON_ENGINES_BACKUP: Path = THESIS_ROOT / "jetson_engines_backup"

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
APOLI_DATASET_ROOT: Path = THESIS_ROOT / "APOLI_dataset"
APOLI_INS_S_TGZ: Path = THESIS_ROOT / "APOLI_ins_s.tgz"

# ---------------------------------------------------------------------------
# New repo outputs (this codebase writes here only)
# ---------------------------------------------------------------------------
RESULTS_ROOT: Path = REPO_ROOT / "results"
STAGE1_DIR: Path = RESULTS_ROOT / "stage1"
STAGE2_DIR: Path = RESULTS_ROOT / "stage2"
STAGE3_DIR: Path = RESULTS_ROOT / "stage3"
STAGE4_DIR: Path = RESULTS_ROOT / "stage4"
STAGE5_DIR: Path = RESULTS_ROOT / "stage5"

# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------
CONFIGS_DIR: Path = REPO_ROOT / "experiments" / "configs"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def ensure(path: Path) -> Path:
    """Create the directory if it does not exist; return the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def describe() -> str:
    """Human-readable summary used by experiment runners on startup."""
    lines = [
        f"REPO_ROOT          = {REPO_ROOT}",
        f"THESIS_ROOT        = {THESIS_ROOT}",
        f"DELIVERABLE_ROOT   = {DELIVERABLE_ROOT}",
        f"PATHB_MODELS       = {PATHB_MODELS}",
        f"APOLI_DATASET_ROOT = {APOLI_DATASET_ROOT}",
        f"RESULTS_ROOT       = {RESULTS_ROOT}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
