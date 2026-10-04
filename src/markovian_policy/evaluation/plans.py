"""Plan sources and seeds: everything depends on (seed, trial number) only, so a resumed run is reproducible."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from markovian_policy.sim.tasks import sample_plan


@dataclass(frozen=True)
class RandomPlans:
    """A fresh random plan of `chain_len` distinct subtasks per trial."""

    seed: int = 0
    chain_len: int = 4

    def plan(self, trial: int) -> list[str]:
        return sample_plan(np.random.default_rng([self.seed, trial]), self.chain_len)


@dataclass(frozen=True)
class FixedPlan:
    """The same plan for every trial."""

    sequence: Sequence[str]

    def plan(self, trial: int) -> list[str]:
        return list(self.sequence)


def trial_seed(seed: int, trial: int) -> int:
    """Seed of the environment's noise stream for a trial."""
    return int(np.random.default_rng([seed, trial, 1]).integers(2**31))
