"""Stage: build a dataset by replaying human teleoperation logs with physics."""

from dataclasses import dataclass, field
from pathlib import Path

from markovian_policy.data.generation import BuildSummary, DatasetBuilder, HumanReplaySource, KeepAll, SuccessOnly
from markovian_policy.paths import HUMAN_DATASET, HUMAN_LOGS_ZIP
from markovian_policy.sim.env import KitchenEnvConfig


@dataclass
class Config:
    zip_path: Path = HUMAN_LOGS_ZIP
    out: Path = HUMAN_DATASET
    n_workers: int = 4  # replay processes (each with its own GL context); 0: replay in this process
    keep_incomplete: bool = False  # also keep replays that miss planned subtasks
    max_logs: int | None = None  # smoke test: replay only this many logs, evenly spread over the usable ones
    env: KitchenEnvConfig = field(default_factory=KitchenEnvConfig)


def run(config: Config) -> BuildSummary:
    source = HumanReplaySource(config.zip_path, config.env, config.max_logs)
    acceptance = KeepAll() if config.keep_incomplete else SuccessOnly()
    return DatasetBuilder(source, acceptance, config.out, config.n_workers).build()
