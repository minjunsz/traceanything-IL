"""`DatasetBuilder`: facade that turns a demonstration source into a dataset on disk."""

import json
import multiprocessing as mp
from collections import Counter
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np

from markovian_policy.data.episodes import EpisodeWriter
from markovian_policy.data.generation.protocols import AcceptancePolicy, Candidate, DemoSource


@dataclass(frozen=True)
class BuildSummary:
    outcomes: dict[str, int]  # label -> number of processed jobs
    n_episodes: int
    n_steps: int
    mean_episode_len: float
    report: dict[str, Any]  # the source's own report


# Worker-process state: the callable built by `source.worker()` (simulators live here).
_work: Callable[[Any], Candidate] | None = None


def _init_worker[Job](source: DemoSource[Job]) -> None:
    global _work
    _work = source.worker()


def _run_job(job: Any) -> Candidate:
    assert _work is not None
    return _work(job)


class DatasetBuilder[Job]:
    """Runs a `DemoSource` in `n_workers` processes (0: in this process), keeps what the `AcceptancePolicy` accepts
    and appends it to the zarr dataset at `out`, until `target_episodes` are stored or the jobs run out.

    Resumable: processed jobs are recorded in `<out>.manifest.json`, so rerunning continues where it stopped.
    """

    def __init__(
        self, source: DemoSource[Job], acceptance: AcceptancePolicy, out: Path, n_workers: int = 4, target_episodes: int | None = None
    ) -> None:
        self.source, self.acceptance, self.out = source, acceptance, out
        self.n_workers, self.target_episodes = n_workers, target_episodes
        self._manifest_path = out.with_suffix(".manifest.json")

    def build(self) -> BuildSummary:
        manifest: dict[str, str] = json.loads(self._manifest_path.read_text()) if self._manifest_path.exists() else {}
        writer = EpisodeWriter(self.out, resume=bool(manifest))
        pool = self._start_pool()
        try:
            for wave in self._waves(manifest):
                for candidate in pool.map(_run_job, wave) if pool else map(self._run_inline, wave):
                    if self._done(writer):
                        break
                    accepted = self.acceptance.accept(candidate)
                    if accepted:
                        assert candidate.episode is not None
                        writer.add(candidate.episode)
                    manifest[candidate.key] = candidate.label + ("" if accepted else " (dropped)")
                self._manifest_path.write_text(json.dumps(manifest, indent=1))
                print(f"{len(manifest)} jobs, {writer.n_episodes} episodes: {dict(Counter(manifest.values()))}", flush=True)
                if self._done(writer):
                    break
        finally:
            if pool:
                pool.shutdown(wait=True, cancel_futures=True)
        return self._summarize(manifest, writer)

    def _done(self, writer: EpisodeWriter) -> bool:
        return self.target_episodes is not None and writer.n_episodes >= self.target_episodes

    def _waves(self, manifest: dict[str, str]) -> Iterator[list[Job]]:
        """Jobs not yet processed, in waves that bound the backlog of finished image episodes."""
        pending = (job for job in self.source.jobs() if self.source.key(job) not in manifest)
        size = max(self.n_workers, 1) * 2
        while wave := list(islice(pending, size)):
            yield wave

    def _start_pool(self) -> ProcessPoolExecutor | None:
        if self.n_workers:
            return ProcessPoolExecutor(self.n_workers, mp.get_context("spawn"), _init_worker, (self.source,))
        _init_worker(self.source)
        return None

    @staticmethod
    def _run_inline(job: Job) -> Candidate:
        return _run_job(job)

    def _summarize(self, manifest: dict[str, str], writer: EpisodeWriter) -> BuildSummary:
        ends = np.array(writer.episode_ends) if writer.episode_ends else np.zeros(1, dtype=np.int64)
        lengths = np.diff(np.concatenate([[0], ends]))
        summary = BuildSummary(
            outcomes=dict(Counter(label.removesuffix(" (dropped)") for label in manifest.values())),
            n_episodes=writer.n_episodes, n_steps=writer.n_steps, mean_episode_len=float(lengths.mean()),
            report=dict(self.source.report()),
        )  # fmt: skip
        self.out.with_suffix(".summary.json").write_text(json.dumps(asdict(summary), indent=1, default=str))
        return summary
