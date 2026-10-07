"""In-memory view of a recorded demonstration set: flat (T_total, ...) arrays per key + episode_ends.

Large image arrays can be served from uncompressed `.npy` files instead (`export_npy`, `from_zarr(mmap_dir=...)`):
they are memory-mapped, so every training process (and dataloader worker) on a node shares one page-cache copy
instead of each decompressing its own multi-GB copy into RAM, and the pages are reclaimable.
"""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import zarr

EXPORT_STEP = 2048  # frames decompressed at a time by `export_npy`


class ReplayBuffer:
    def __init__(self, data: dict[str, np.ndarray], episode_ends: np.ndarray) -> None:
        self.data, self.episode_ends = data, episode_ends

    @classmethod
    def from_zarr(cls, path: str, keys: list[str] | None = None, mmap_dir: Path | None = None) -> "ReplayBuffer":
        """Load `data/<key>` (all keys by default) and `meta/episode_ends` from a zarr store.

        Keys with an exported `<mmap_dir>/<key>.npy` are memory-mapped from there; the others are read into RAM.
        """
        root = zarr.open_group(path, mode="r")
        data = root["data"]
        assert isinstance(data, zarr.Group)
        keys = keys if keys is not None else list(data)

        def load(key: str) -> np.ndarray:
            exported = mmap_dir / f"{key}.npy" if mmap_dir is not None else None
            return np.load(exported, mmap_mode="r") if exported is not None and exported.exists() else np.asarray(data[key])

        return cls({key: load(key) for key in keys}, np.asarray(root["meta/episode_ends"], dtype=np.int64))

    def keys(self) -> list[str]:
        return list(self.data)

    def __iter__(self) -> Iterator[str]:
        return iter(self.data)

    def __getitem__(self, key: str) -> np.ndarray:
        return self.data[key]

    @property
    def n_episodes(self) -> int:
        return len(self.episode_ends)


def export_npy(zarr_path: str, out_dir: Path, keys: list[str]) -> list[Path]:
    """Write `data/<key>` of a zarr store as `<out_dir>/<key>.npy`, streamed so memory stays small.

    Resumable: a finished file is kept; files are written under a temporary name and renamed when complete.
    """
    root = zarr.open_group(zarr_path, mode="r")
    data = root["data"]
    assert isinstance(data, zarr.Group)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for key in keys:
        source = data[key]
        assert isinstance(source, zarr.Array)
        target = out_dir / f"{key}.npy"
        paths.append(target)
        if target.exists():
            continue
        tmp = out_dir / f"{key}.npy.tmp"
        with open(tmp, "wb") as handle:
            sink = np.lib.format.open_memmap(handle.name, mode="w+", dtype=source.dtype, shape=source.shape)
            for start in range(0, source.shape[0], EXPORT_STEP):
                sink[start : start + EXPORT_STEP] = source[start : start + EXPORT_STEP]
            sink.flush()
            del sink
        tmp.rename(target)
    return paths
