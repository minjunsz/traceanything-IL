"""Markovian scripted expert: a plan (subtask sequence) -> per-step action.

An oracle advances to the next subtask once the current one is complete and the arm is back home. Within a
subtask, the FSM phase and the action are functions of the current simulation state only.
"""

from collections.abc import Sequence
from typing import NamedTuple, Protocol

from markovian_policy.arrays import FloatArray
from markovian_policy.experts import controller
from markovian_policy.experts.fsm import SUBTASK_FSMS, GripperCmd, at_home
from markovian_policy.experts.fsm.base import RESET_ARM_QPOS
from markovian_policy.sim.physics import KitchenPhysics
from markovian_policy.sim.tasks import is_done

RETURN_TO_HOME = "return-to-home"


class ExpertStep(NamedTuple):
    action: FloatArray  # (9,) in [-1, 1]
    subtask: str
    phase: str


class Expert(Protocol):
    """Drives the kitchen through a plan. Demonstration sources depend on this interface only."""

    @property
    def finished(self) -> bool: ...

    @property
    def subtask(self) -> str: ...

    @property
    def oscillation(self) -> str | None:
        """ "subtask:phase" of the first FSM phase that was re-entered after being left (dithering), if any."""
        ...

    def act(self, sim: KitchenPhysics) -> ExpertStep: ...

    def advance(self, sim: KitchenPhysics) -> None: ...


class ScriptedExpert:
    """`Expert` built from the per-subtask FSMs and the operational-space controller."""

    def __init__(self, plan: Sequence[str]) -> None:
        self.plan = list(plan)
        self.index = 0
        self._oscillation: str | None = None
        self._last_key: tuple[int, str] | None = None
        self._left_keys: set[tuple[int, str]] = set()

    @property
    def finished(self) -> bool:
        return self.index >= len(self.plan)

    @property
    def subtask(self) -> str:
        return self.plan[self.index]

    @property
    def oscillation(self) -> str | None:
        return self._oscillation

    def act(self, sim: KitchenPhysics) -> ExpertStep:
        subtask, fsm = self.subtask, SUBTASK_FSMS[self.subtask]
        phase = fsm.fsm_state(sim)
        gripper, target_pos, target_rot = fsm.targets(sim, phase)
        gripper_target = "open" if gripper == GripperCmd.OPEN else "close"
        if phase == RETURN_TO_HOME:
            action = controller.joint_pose_to_action(sim, RESET_ARM_QPOS, gripper_target)
        else:
            action = controller.pose_to_action(sim, target_pos, target_rot, gripper_target)
        self._track_phase((self.index, phase))
        return ExpertStep(action, subtask, phase)

    def advance(self, sim: KitchenPhysics) -> None:
        """Move to the next subtask once the current one is done and the arm is home."""
        if is_done(sim.data.qpos, self.subtask) and at_home(sim):
            self.index += 1

    def _track_phase(self, key: tuple[int, str]) -> None:
        if key == self._last_key:
            return
        if self._last_key is not None:
            self._left_keys.add(self._last_key)
        if key in self._left_keys and self._oscillation is None:
            self._oscillation = f"{self.subtask}:{key[1]}"
        self._last_key = key
