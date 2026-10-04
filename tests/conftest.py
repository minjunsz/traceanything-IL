from pathlib import Path

import numpy as np
import pytest
import torch
import zarr
from accelerate.state import AcceleratorState, GradientState

torch.set_num_threads(2)  # torch would otherwise spawn one thread per node core, thrashing a 4-CPU cgroup

N_EPISODES, LENGTH = 6, 40


@pytest.fixture(autouse=True)
def fresh_accelerator_state():
    AcceleratorState._reset_state(True)
    GradientState._reset_state()


@pytest.fixture
def demo_zarr(tmp_path: Path) -> Path:
    """Small synthetic demonstration set in the recorder's layout (tiny 8x8 camera frames)."""
    rng = np.random.default_rng(0)
    total = N_EPISODES * LENGTH
    root = zarr.open_group(str(tmp_path / "demos.zarr"), mode="w")
    data = root.create_group("data")
    arrays = {
        "action": rng.normal(size=(total, 9)),
        "state": rng.normal(size=(total, 60)),
        "subtask_sequence": np.eye(28)[rng.integers(0, 28, total)],
        "scene": rng.integers(0, 255, (total, 8, 8, 3), dtype=np.uint8),
        "wrist": rng.integers(0, 255, (total, 8, 8, 3), dtype=np.uint8),
    }
    for key, value in arrays.items():
        data.create_array(key, data=value)
    root.create_group("meta").create_array("episode_ends", data=np.arange(1, N_EPISODES + 1) * LENGTH)
    return tmp_path / "demos.zarr"
