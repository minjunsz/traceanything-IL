"""Stage: export the camera frames of a dataset as uncompressed `.npy` files for memory-mapped training.

Loading a compressed zarr decompresses every frame into each training process's RAM (28 GiB per camera for the human
data); the exported files are memory-mapped by `DataConfig.mmap_dir`, so all processes on a node share one page-cache copy.
"""

from dataclasses import dataclass
from pathlib import Path

from markovian_policy.data.replay_buffer import export_npy


@dataclass
class Config:
    zarr_path: Path
    out_dir: Path | None = None  # default: next to the zarr, `<name>.raw`
    keys: tuple[str, ...] = ("scene", "wrist")


def run(config: Config) -> list[Path]:
    out_dir = config.out_dir or config.zarr_path.with_suffix(".raw")
    paths = export_npy(str(config.zarr_path), out_dir, list(config.keys))
    for path in paths:
        print(f"{path} ({path.stat().st_size / 2**30:.1f} GiB)")
    return paths
