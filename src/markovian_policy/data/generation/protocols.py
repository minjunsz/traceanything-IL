"""Interfaces of the demonstration-generation pipeline.

A `DemoSource` knows where demonstrations come from (scripted expert, replayed human logs, policy rollouts, ...);
an `AcceptancePolicy` decides which generated episodes enter the dataset; `DatasetBuilder` (facade) runs the
sources in parallel, applies the policy and writes the dataset.
"""

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from markovian_policy.data.episodes import EpisodeArrays


@dataclass(frozen=True)
class Candidate:
    """One generated episode awaiting the acceptance decision."""

    key: str  # unique id of the job that produced it (the resume manifest records it)
    episode: EpisodeArrays | None  # None: nothing usable was produced (e.g. the simulation diverged)
    success: bool  # the demonstration achieved its goal
    label: str  # short outcome label for logs and the summary, e.g. "ok", "incomplete", "unstable"
    info: Mapping[str, str] = field(default_factory=dict)


class DemoSource[Job](Protocol):
    """Strategy: a stream of jobs and a worker that turns a job into a `Candidate`.

    The source is sent to worker processes, so it must be picklable (a frozen dataclass of plain config).
    Everything heavy (simulators, GL contexts) is created in `worker()`, inside the worker process.
    """

    def jobs(self) -> Iterator[Job]:
        """Jobs in a deterministic order; may be infinite (the builder stops at its target)."""
        ...

    def key(self, job: Job) -> str:
        """Stable id of a job: lets an interrupted build skip what it already processed."""
        ...

    def worker(self) -> Callable[[Job], Candidate]: ...

    def report(self) -> Mapping[str, Any]:
        """Extra information for the build summary (e.g. the logs that were skipped as unusable)."""
        ...


class AcceptancePolicy(Protocol):
    """Strategy: which candidates enter the dataset."""

    def accept(self, candidate: Candidate) -> bool: ...


class SuccessOnly:
    """Keep only successful demonstrations."""

    def accept(self, candidate: Candidate) -> bool:
        return candidate.episode is not None and candidate.success


class KeepAll:
    """Keep every episode that was produced, successful or not."""

    def accept(self, candidate: Candidate) -> bool:
        return candidate.episode is not None
