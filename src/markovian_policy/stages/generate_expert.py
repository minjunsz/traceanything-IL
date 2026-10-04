"""Stage: generate demonstrations with the scripted Markovian expert."""

from dataclasses import dataclass, field
from pathlib import Path

from markovian_policy.data.generation import BuildSummary, DatasetBuilder, ExpertDemoSource, SuccessOnly
from markovian_policy.paths import EXPERT_DATASET
from markovian_policy.sim.env import KitchenEnvConfig


@dataclass
class Config:
    out: Path = EXPERT_DATASET
    n_episodes: int = 581  # successful episodes to store
    n_workers: int = 4  # replay processes, each with its own simulator
    chain_len: int = 4  # subtasks per random plan
    seed: int = 0
    max_steps_per_subtask: int = 200
    env: KitchenEnvConfig = field(default_factory=KitchenEnvConfig)


def run(config: Config) -> BuildSummary:
    source = ExpertDemoSource(config.env, config.chain_len, config.seed, config.max_steps_per_subtask)
    return DatasetBuilder(source, SuccessOnly(), config.out, config.n_workers, config.n_episodes).build()
