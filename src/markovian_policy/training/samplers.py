"""Epoch samplers: resumable mid-epoch, and with data echoing."""

from typing import Any

import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler


class ResumableDistributedSampler(DistributedSampler[int]):
    """DistributedSampler that can drop the first `skip_samples` indices of its (epoch-seeded, hence reproducible)
    shard, so a run resumed from a mid-epoch checkpoint continues the interrupted epoch. The trainer resets
    `skip_samples` to 0 at the end of each epoch."""

    def __init__(self, dataset: Dataset[Any], num_replicas: int, rank: int, shuffle: bool = True, seed: int = 0) -> None:
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed)
        self.skip_samples = 0

    def _epoch_indices(self) -> list[int]:
        return list(super().__iter__())

    def _epoch_len(self) -> int:
        return self.num_samples

    def __iter__(self):
        return iter(self._epoch_indices()[self.skip_samples :])

    def __len__(self) -> int:
        return max(0, self._epoch_len() - self.skip_samples)


class EchoDistributedSampler(ResumableDistributedSampler):
    """Data echoing (Choi et al. 2019): every index of this rank's epoch shard is yielded `echo` times, for when
    reading samples (the token cache) is slower than the GPU step. Repeats stay useful since each draws a fresh
    diffusion timestep and noise.

    The shard is cut into blocks of `block` indices; output segment k is a shuffle of block k (first use) plus
    `echo - 1` copies of block k - 1, so repeats follow their first read closely (page cache stays warm) and first
    reads are spread evenly. An "epoch" is still one pass over the unique data with `echo` times as many steps.
    """

    def __init__(
        self, dataset: Dataset[Any], num_replicas: int, rank: int, echo: int, block: int, shuffle: bool = True, seed: int = 0
    ) -> None:
        super().__init__(dataset, num_replicas, rank, shuffle, seed)
        assert echo >= 1 and block >= 1
        self.echo, self.block = echo, block

    def _epoch_indices(self) -> list[int]:
        base = list(DistributedSampler.__iter__(self))
        rng = np.random.default_rng([self.seed, self.epoch, self.rank])
        blocks = [base[i : i + self.block] for i in range(0, len(base), self.block)]
        out: list[int] = []
        prev: list[int] = []
        for block in [*blocks, []]:
            segment = np.array(block + prev * (self.echo - 1), dtype=np.int64)
            rng.shuffle(segment)
            out.extend(segment.tolist())
            prev = block
        return out

    def _epoch_len(self) -> int:
        return self.num_samples * self.echo


def build_train_sampler(
    dataset: Dataset[Any], num_replicas: int, rank: int, seed: int, echo_factor: int = 1, echo_block: int = 5000
) -> ResumableDistributedSampler:
    """Strategy choice: plain resumable sharding, or data echoing when `echo_factor > 1`."""
    if echo_factor > 1:
        return EchoDistributedSampler(dataset, num_replicas, rank, echo_factor, echo_block, seed=seed)
    return ResumableDistributedSampler(dataset, num_replicas, rank, seed=seed)
