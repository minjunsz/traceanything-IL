"""Episodes as arrays, and their on-disk dataset layout: data/<key> (T, ...) plus meta/episode_ends (E,)."""

from pathlib import Path

import numpy as np
import zarr
from zarr.codecs import BloscCodec

from markovian_policy.arrays import Array

type EpisodeArrays = dict[str, Array]  # key -> (T, ...) array; a step stores (o_t, a_t), the obs before the action


class EpisodeWriter:
    """Appends episodes straight to the on-disk arrays (camera streams never sit in RAM). `episode_ends` is
    rewritten after every episode, so the store stays valid if the run is killed. With `resume`, continues an
    existing store instead of overwriting it."""

    def __init__(self, path: Path, resume: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.root = zarr.open_group(str(path), mode="a" if resume else "w")
        self.data = self.root.require_group("data")
        self.meta = self.root.require_group("meta")
        self.episode_ends: list[int] = np.asarray(self.meta["episode_ends"]).tolist() if "episode_ends" in self.meta else []

    @property
    def n_episodes(self) -> int:
        return len(self.episode_ends)

    @property
    def n_steps(self) -> int:
        return self.episode_ends[-1] if self.episode_ends else 0

    def add(self, episode: EpisodeArrays) -> None:
        for key, value in episode.items():
            if key not in self.data:
                chunks = (128, *value.shape[1:]) if value.ndim == 4 else (1024, *value.shape[1:])
                self.data.create_array(
                    key, shape=(0, *value.shape[1:]), chunks=chunks, dtype=value.dtype,
                    compressors=BloscCodec(cname="lz4", clevel=5, shuffle="shuffle"),
                )  # fmt: skip
            self.data[key].append(value)  # pyright: ignore[reportAttributeAccessIssue]
        self.episode_ends.append(self.n_steps + len(episode["action"]))
        self.meta.create_array("episode_ends", data=np.array(self.episode_ends, dtype=np.int64), overwrite=True)
