"""TokenCacheBuilder with a stub encoder: layout against the sampler, sharding, resume and the dataset round trip."""

from pathlib import Path

import numpy as np
import torch
import zarr

from markovian_policy.data import trace_cache
from markovian_policy.data.dataset import DataConfig, FrankaKitchenDataset
from markovian_policy.data.token_cache_builder import TokenCacheBuilder, TokenCacheConfig

CPU = torch.device("cpu")
N_OBS = 2


class MeanColorEncoder:
    """Deterministic stand-in for the frozen encoder: tokens are functions of the window's mean colours."""

    token_dim, tokens = 4, 3

    def num_tokens(self, height: int, width: int) -> int:
        return self.tokens

    def __call__(self, window: torch.Tensor) -> torch.Tensor:
        colour = window.float().mean(dim=(2, 3))  # (B, T, 3): mean colour of every frame
        mixed = torch.stack([colour.mean(1), colour[:, -1], colour[:, 0], colour.sum(1)], dim=-1)  # (B, 3, 4)
        return mixed.half().float()


def _builder(zarr_path: Path, cache_dir: Path, **kwargs: int) -> TokenCacheBuilder:
    config = TokenCacheConfig(zarr_path, cache_dir, camera="scene", n_obs_steps=N_OBS, horizon=6, batch_size=16, n_verify=8, **kwargs)
    return TokenCacheBuilder(config, lambda: MeanColorEncoder(), CPU)


def test_cache_build_verify_and_dataset_round_trip(demo_zarr: Path, tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    for rank in (0, 1):  # two shards, as if two GPUs
        builder = _builder(demo_zarr, cache, rank=rank, world=2)
        if rank == 0:
            builder.init()
        builder.encode()
    verifier = _builder(demo_zarr, cache)
    assert verifier.verify() < 1e-3
    assert (cache / trace_cache.COMPLETE_FILE).exists()

    dataset = FrankaKitchenDataset(
        DataConfig(
            demo_zarr,
            horizon=6,
            n_obs_steps=N_OBS,
            lowdim_keys=("agent_pos",),
            cameras=("scene",),
            val_ratio=0.0,
            trace_cache_dirs={"scene": cache},
        )
    )
    frames = np.asarray(zarr.open_group(str(demo_zarr), mode="r")["data/scene"])
    for index in (0, 5, len(dataset) - 1):
        item = dataset[index]
        row = int(dataset.cache_rows[index])  # type: ignore[index]
        episode = int(np.searchsorted(verifier.offsets, row, side="right") - 1)
        window = trace_cache.episode_window_frames(verifier.episode_ends, N_OBS, episode)[row - verifier.offsets[episode]]
        expected = MeanColorEncoder()(torch.from_numpy(frames[window])[None]).half().float()[0]
        torch.testing.assert_close(item["obs"]["trace_tokens"], expected)


def test_encode_resumes_without_redoing_finished_episodes(demo_zarr: Path, tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    first = _builder(demo_zarr, cache, rank=0, world=2)
    first.init()
    first.encode()  # episodes 0, 2, 4
    assert sorted(first._finished_episodes()) == [0, 2, 4]
    resumed = _builder(demo_zarr, cache, rank=0, world=1)  # a rerun with another world size takes what is left
    resumed.encode()
    assert sorted(resumed._finished_episodes()) == [0, 1, 2, 3, 4, 5]
    resumed.verify()
    assert (cache / trace_cache.COMPLETE_FILE).exists()
