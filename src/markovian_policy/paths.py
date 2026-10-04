"""Default locations of downloaded assets and built datasets, relative to the working directory (git-ignored)."""

from pathlib import Path
from typing import Final

DATA_DIR: Final = Path("data")
TRACE_WEIGHTS: Final = DATA_DIR / "weights" / "trace_anything.pt"  # pretrained TraceAnything weights
HUMAN_LOGS_ZIP: Final = DATA_DIR / "raw" / "kitchen_demos_multitask.zip"  # human teleoperation logs
EXPERT_DATASET: Final = DATA_DIR / "kitchen_demos_expert.zarr"
HUMAN_DATASET: Final = DATA_DIR / "kitchen_demos_human.zarr"
