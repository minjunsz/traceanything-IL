"""Franka Kitchen demonstration dataset, backed by one zarr store (see `data.generation` for the layout).

    data/action (T, 9) | data/state (T, 60) = [qpos(9), obj_qpos(21), goal(30)] | data/<camera> (T, H, W, 3) uint8
    data/<lowdim key> (T, d) | meta/episode_ends (E,)

Windowing/padding lives in `data.sampler.SequenceSampler`. With `trace_cache_dirs` (built by `TokenCacheBuilder`, one
cache per camera) the dataset serves the frozen TraceAnything tokens of each window under
`trace_cache.tokens_obs_key(camera, n_cameras)` and never loads the rgb frames.
"""

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from markovian_policy.arrays import Array
from markovian_policy.data import trace_cache
from markovian_policy.data.replay_buffer import ReplayBuffer
from markovian_policy.data.sampler import SequenceSampler, get_val_mask
from markovian_policy.nn.normalizer import LinearNormalizer
from markovian_policy.paths import EXPERT_DATASET
from markovian_policy.utils import dict_apply

# Only the first PROPRIO_DIM dims of "state" (7 arm joints + 2 gripper) exist at execution time; the rest
# (object qpos, goal) is privileged simulator state available only in recorded demonstrations.
PROPRIO_DIM: Final = 9
STATE_KEY: Final = "state"
AGENT_POS_KEY: Final = "agent_pos"


@dataclass
class DataConfig:
    zarr_path: Path = EXPERT_DATASET
    horizon: int = 22
    n_obs_steps: int = 8
    lowdim_keys: tuple[str, ...] = ("agent_pos", "subtask_sequence")
    cameras: tuple[str, ...] = ("scene", "wrist")
    val_ratio: float = 0.03
    seed: int = 41
    trace_cache_dirs: dict[str, Path] | None = None  # camera -> TraceAnything token cache
    mmap_dir: Path | None = None  # exported frames (stage export_frames): memory-mapped, shared by all processes of a node


class FrankaKitchenDataset(Dataset[dict[str, Any]]):
    def __init__(self, config: DataConfig) -> None:
        self.config = config
        self.key_map = {STATE_KEY: AGENT_POS_KEY} if AGENT_POS_KEY in config.lowdim_keys else {}
        buffer_keys = ["action"] + [next((b for b, o in self.key_map.items() if o == k), k) for k in config.lowdim_keys]
        if not config.trace_cache_dirs:
            buffer_keys += list(config.cameras)
        self.replay_buffer = ReplayBuffer.from_zarr(
            str(config.zarr_path), keys=buffer_keys, mmap_dir=config.mmap_dir
        )  # only what is consumed

        self._tokens: dict[str, np.memmap] | None = None  # opened lazily so each dataloader worker gets its own
        if config.trace_cache_dirs:
            assert set(config.trace_cache_dirs) == set(config.cameras), "need one trace cache per camera"
            for cache_dir in config.trace_cache_dirs.values():
                self._check_cache(cache_dir)

        val_mask = get_val_mask(self.replay_buffer.n_episodes, config.val_ratio, config.seed)
        self.train_mask, self.val_mask = ~val_mask, val_mask
        self.sampler, self.cache_rows = self._build_sampler(self.train_mask)

    def _check_cache(self, cache_dir: Path) -> None:
        assert (cache_dir / trace_cache.COMPLETE_FILE).exists(), f"{cache_dir} is not a complete cache"
        meta = trace_cache.load_meta(str(cache_dir))
        assert meta["n_obs_steps"] == self.config.n_obs_steps, "cache was built for another n_obs_steps"
        assert meta["episode_ends"] == self.replay_buffer.episode_ends.tolist(), "cache was built for another dataset"

    def _build_sampler(self, episode_mask: Array) -> tuple[SequenceSampler, Array | None]:
        c = self.config
        sampler = SequenceSampler(
            self.replay_buffer, c.horizon, (*c.lowdim_keys, *c.cameras), c.cameras,
            pad_before=c.n_obs_steps - 1, pad_after=c.horizon - 1, episode_mask=episode_mask, key_map=self.key_map,
            key_first_k={k: c.n_obs_steps for k in self.replay_buffer if k != "action"},
        )  # fmt: skip
        rows = None
        if c.trace_cache_dirs:
            rows = trace_cache.sampler_window_rows(sampler.indices, self.replay_buffer.episode_ends, c.n_obs_steps)
        return sampler, rows

    def get_validation_dataset(self) -> "FrankaKitchenDataset":
        val_set = copy.copy(self)
        val_set.sampler, val_set.cache_rows = self._build_sampler(self.val_mask)
        return val_set

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, index: int) -> dict[str, Any]:
        data = self.sampler.sample_data(index)
        if AGENT_POS_KEY in data["obs"]:
            data["obs"][AGENT_POS_KEY] = data["obs"][AGENT_POS_KEY][..., :PROPRIO_DIM]
        cache_dirs = self.config.trace_cache_dirs
        if self.cache_rows is not None and cache_dirs:
            if self._tokens is None:
                self._tokens = {camera: trace_cache.open_tokens(str(d)) for camera, d in cache_dirs.items()}
            for camera, tokens in self._tokens.items():
                key = trace_cache.tokens_obs_key(camera, len(self._tokens))
                data["obs"][key] = tokens[self.cache_rows[index]].astype(np.float32)
        return dict_apply(data, torch.from_numpy)

    def get_normalizer(self, mode: Literal["limits", "gaussian"] = "limits") -> LinearNormalizer:
        """Fit scale/offset from the low-dim obs and action ranges of the whole dataset."""
        data = {"action": self.replay_buffer["action"]}
        for key in self.config.lowdim_keys:
            array = self.replay_buffer[STATE_KEY if key == AGENT_POS_KEY else key]
            data[key] = array[..., :PROPRIO_DIM] if key == AGENT_POS_KEY else array
        normalizer = LinearNormalizer()
        normalizer.fit(data, last_n_dims=1, mode=mode)
        return normalizer
