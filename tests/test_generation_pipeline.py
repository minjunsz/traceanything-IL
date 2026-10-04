"""The demonstration pipeline with stub sources: acceptance strategies, episode targets, resume, worker processes."""

import itertools
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from markovian_policy.data.generation import AcceptancePolicy, Candidate, DatasetBuilder, KeepAll, SuccessOnly


@dataclass(frozen=True)
class CountingSource:
    """Job i yields an episode of i + 1 steps that succeeds when i is even. Endless unless `n_jobs` is given."""

    n_jobs: int | None = None

    def jobs(self) -> Iterator[int]:
        return itertools.count() if self.n_jobs is None else iter(range(self.n_jobs))

    def key(self, job: int) -> str:
        return f"job-{job}"

    def worker(self) -> Callable[[int], Candidate]:
        def make(job: int) -> Candidate:
            episode = {"action": np.full((job + 1, 9), float(job)), "state": np.zeros((job + 1, 60))}
            return Candidate(self.key(job), episode, job % 2 == 0, "even" if job % 2 == 0 else "odd")

        return make

    def report(self) -> Mapping[str, Any]:
        return {"source": "counting"}


def _episode_ends(path: Path) -> list[int]:
    return np.asarray(zarr.open_group(str(path), mode="r")["meta/episode_ends"]).tolist()


def test_success_only_stops_at_the_episode_target(tmp_path: Path) -> None:
    summary = DatasetBuilder(CountingSource(), SuccessOnly(), tmp_path / "d.zarr", n_workers=0, target_episodes=3).build()
    assert _episode_ends(tmp_path / "d.zarr") == [1, 4, 9]  # successful jobs 0, 2, 4 (1, 3 and 5 steps)
    assert summary.n_episodes == 3 and summary.n_steps == 9 and summary.report == {"source": "counting"}
    assert summary.outcomes["odd"] == 2  # the failed jobs in between were processed and dropped


def test_keep_all_stores_failures_too(tmp_path: Path) -> None:
    acceptance: AcceptancePolicy = KeepAll()
    summary = DatasetBuilder(CountingSource(4), acceptance, tmp_path / "d.zarr", n_workers=0).build()
    assert _episode_ends(tmp_path / "d.zarr") == [1, 3, 6, 10] and summary.outcomes == {"even": 2, "odd": 2}


def test_rerun_continues_after_the_processed_jobs(tmp_path: Path) -> None:
    out = tmp_path / "d.zarr"
    DatasetBuilder(CountingSource(), SuccessOnly(), out, n_workers=0, target_episodes=2).build()
    assert _episode_ends(out) == [1, 4]
    summary = DatasetBuilder(CountingSource(), SuccessOnly(), out, n_workers=0, target_episodes=4).build()
    assert _episode_ends(out) == [1, 4, 9, 16] and summary.n_episodes == 4  # jobs 0 and 2 are not repeated


def test_worker_processes_produce_the_same_dataset(tmp_path: Path) -> None:
    DatasetBuilder(CountingSource(6), KeepAll(), tmp_path / "inline.zarr", n_workers=0).build()
    DatasetBuilder(CountingSource(6), KeepAll(), tmp_path / "pool.zarr", n_workers=2).build()
    assert _episode_ends(tmp_path / "inline.zarr") == _episode_ends(tmp_path / "pool.zarr")  # waves keep job order
