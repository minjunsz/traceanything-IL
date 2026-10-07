"""Outcome of a trial and its flat record for the result files."""

from dataclasses import dataclass, field
from typing import Literal, NotRequired, TypedDict

from markovian_policy.arrays import Image
from markovian_policy.data.episodes import EpisodeArrays


@dataclass
class TrialResult:
    result: Literal["success", "failure", "timeout"]  # failure: the simulation diverged
    plan: list[str]
    completed: list[str]  # subtasks in the order they were first completed
    steps: int  # steps until success/failure, else the timeout
    completed_steps: list[int] = field(default_factory=list)  # env step count when each of `completed` was first seen
    frames: list[Image] = field(default_factory=list)  # side-by-side camera frames (if requested)
    episode: EpisodeArrays | None = None  # (o_t, a_t) rows in the dataset layout (if requested)

    @property
    def plan_progress(self) -> int:
        """Number of leading plan subtasks that were completed first and in plan order."""
        n = 0
        while n < min(len(self.plan), len(self.completed)) and self.plan[n] == self.completed[n]:
            n += 1
        return n


class TrialRecord(TypedDict):
    """One row of results.csv / results.json."""

    trial: int
    result: str
    reward: float  # number of completed subtasks
    trial_time: int
    plan: str  # "+"-joined
    completed: str  # "+"-joined
    plan_progress: int
    completed_steps: NotRequired[str]  # "+"-joined, aligned with `completed` (absent in results written before it existed)


def make_record(number: int, result: TrialResult) -> TrialRecord:
    return TrialRecord(
        trial=number, result=result.result, reward=float(len(result.completed)), trial_time=result.steps,
        plan="+".join(result.plan), completed="+".join(result.completed), plan_progress=result.plan_progress,
        completed_steps="+".join(str(step) for step in result.completed_steps),
    )  # fmt: skip
