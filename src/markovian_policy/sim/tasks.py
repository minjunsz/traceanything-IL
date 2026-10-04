"""The 7 kitchen subtasks: when each counts as completed, and how plans (subtask sequences) are encoded.

One implementation shared by the environment, the scripted expert, the data pipeline and the evaluation.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from markovian_policy.arrays import FloatArray

BONUS_THRESH: Final = 0.3  # a subtask is done when its joints are this close (L2) to the goal


@dataclass(frozen=True)
class Subtask:
    """A subtask is completed when `qpos[joint_indices]` is within `BONUS_THRESH` of `goal`."""

    name: str
    joint_indices: NDArray[np.intp]
    goal: FloatArray

    def is_done(self, qpos: FloatArray) -> bool:
        """qpos: (>=30,) joint positions (noisy or not)."""
        return bool(np.linalg.norm(qpos[self.joint_indices] - self.goal) < BONUS_THRESH)


def _subtask(name: str, indices: Sequence[int], goal: Sequence[float]) -> Subtask:
    return Subtask(name, np.array(indices), np.array(goal))


# Goals as in D4RL franka_kitchen; indices into qpos[:30]. Iteration order is the tie-break for completions in one step.
SUBTASKS: Final[Mapping[str, Subtask]] = {
    s.name: s
    for s in (
        _subtask("bottomknob", [11, 12], [-0.88, -0.01]),
        _subtask("topknob", [15, 16], [-0.92, -0.01]),
        _subtask("light", [17, 18], [-0.69, -0.05]),
        _subtask("slide", [19], [0.37]),
        _subtask("hinge", [20, 21], [0.0, 1.45]),
        _subtask("microwave", [22], [-0.75]),
        _subtask("kettle", range(23, 30), [-0.23, 0.75, 1.62, 0.99, 0.0, 0.0, -0.06]),
    )
}


@dataclass(frozen=True)
class PlanEncoding:
    """One-hot encoding of subtasks and plans. The ids are fixed: recorded datasets use them. Do not renumber."""

    ids: Mapping[str, int]
    max_len: int = 4  # plans are padded with all-zero slots to this length

    @property
    def n_subtasks(self) -> int:
        return len(self.ids)

    @property
    def plan_dim(self) -> int:
        return self.max_len * self.n_subtasks

    def subtask_onehot(self, name: str) -> FloatArray:
        """(n_subtasks,) float64."""
        onehot = np.zeros(self.n_subtasks)
        onehot[self.ids[name]] = 1.0
        return onehot

    def plan_onehot(self, plan: Sequence[str]) -> FloatArray:
        """(max_len * n_subtasks,) float64: one slot per plan entry, unused slots all zeros."""
        assert len(plan) <= self.max_len
        onehot = np.zeros((self.max_len, self.n_subtasks))
        for slot, name in enumerate(plan):
            onehot[slot, self.ids[name]] = 1.0
        return onehot.ravel()


PLAN_ENCODING: Final = PlanEncoding({"microwave": 0, "kettle": 1, "slide": 2, "light": 3, "topknob": 4, "bottomknob": 5, "hinge": 6})
SUBTASK_IDS: Final = PLAN_ENCODING.ids


def is_done(qpos: FloatArray, name: str) -> bool:
    return SUBTASKS[name].is_done(qpos)


def completed_subtasks(qpos: FloatArray) -> list[str]:
    """Names of all currently completed subtasks. qpos: (>=30,) joint positions."""
    return [name for name, subtask in SUBTASKS.items() if subtask.is_done(qpos)]


def sample_plan(rng: np.random.Generator, chain_len: int) -> list[str]:
    """Random plan: `chain_len` distinct subtasks in random order."""
    return [str(name) for name in rng.choice(list(SUBTASK_IDS), size=chain_len, replace=False)]
