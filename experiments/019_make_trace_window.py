"""Cut a small (B, T, H, W, 3) uint8 frame window from a demo zarr for the TraceAnything equivalence test."""

from pathlib import Path

import numpy as np
import zarr

root = zarr.open_group("output/kitchen_demos_markovian_scripted_expert.zarr", mode="r")
frames = np.asarray(root["data/scene"][100 : 100 + 9])  # (9, 240, 320, 3)
window = np.stack([frames[:8], frames[1:9]])  # two overlapping windows, T=8
Path("output/trace_reference").mkdir(parents=True, exist_ok=True)
np.savez("output/trace_reference/window.npz", window=window)
print(window.shape, window.dtype)
