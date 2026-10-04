"""In-memory view of a recorded demonstration set: flat (T_total, ...) arrays per key + episode_ends."""

from collections.abc import Iterator

import numpy as np
import zarr


class ReplayBuffer:
    def __init__(self, data: dict[str, np.ndarray], episode_ends: np.ndarray) -> None:
        self.data, self.episode_ends = data, episode_ends

    @classmethod
    def from_zarr(cls, path: str, keys: list[str] | None = None) -> "ReplayBuffer":
        """Load `data/<key>` (all keys by default) and `meta/episode_ends` from a zarr store."""
        root = zarr.open_group(path, mode="r")
        data = root["data"]
        assert isinstance(data, zarr.Group)
        keys = keys if keys is not None else list(data)
        return cls({key: np.asarray(data[key]) for key in keys}, np.asarray(root["meta/episode_ends"], dtype=np.int64))

    def keys(self) -> list[str]:
        return list(self.data)

    def __iter__(self) -> Iterator[str]:
        return iter(self.data)

    def __getitem__(self, key: str) -> np.ndarray:
        return self.data[key]

    @property
    def n_episodes(self) -> int:
        return len(self.episode_ends)
