"""Shared pieces of the per-subtask FSMs: home pose, rotation helpers and the SubtaskFSM interface."""

import enum
from abc import ABC, abstractmethod

import numpy as np

from markovian_policy.arrays import FloatArray
from markovian_policy.sim.physics import KitchenPhysics


class GripperCmd(enum.Enum):
    OPEN = 1
    CLOSE = 2


# --------------------------------------------------------------------------- #
# Fixed home pose (a function of the env's fixed init_qpos, so it is constant).
# Measured at reset; the EE z-axis points roughly +world-y (the approach
# direction) and the fingers open along +-world-x.
# --------------------------------------------------------------------------- #
HOME_EE_POS = np.array([-0.432, 0.119, 2.025])
# HOME_EE_POS = np.array([-0.432, 0.219, 2.225])
HOME_EE_ROT = np.array(
    [
        [-0.036, 0.997, 0.073],
        [-0.049, -0.075, 0.996],
        [0.998, 0.032, 0.052],
    ]
)

# The 7 arm joint positions at the env's true reset qpos. EE pose alone doesn't
# pin down the arm's (redundant) null-space configuration, so two subtasks can
# each consider themselves "home" at the same EE pose via different joint
# configs -- one of which may be collision-prone for the next subtask's
# approach. return-to-home drives the arm to this exact joint configuration
# instead, so every subtask hands off from the identical posture.
RESET_ARM_QPOS = np.array([0.14838802, -1.76848573, 1.84390296, -2.4768576, 0.26025203, 0.7125331, 1.59515394])


def qpos(sim: KitchenPhysics) -> np.ndarray:
    """Noise-free qpos (30,) of the sim."""
    return sim.data.qpos


def rz(theta: float) -> FloatArray:
    """Rotation matrix for a yaw of `theta` radians about the world z axis."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rx(theta: float) -> FloatArray:
    """Rotation matrix for a pitch of `theta` radians about the world x axis."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def ry(theta: float) -> FloatArray:
    """Rotation matrix for a roll of `theta` radians about the world y axis."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def at_home(sim: KitchenPhysics, qpos_tol: float = 0.06) -> bool:
    """True when the arm's joints are back at the reset configuration."""
    return float(np.linalg.norm(qpos(sim)[:7] - RESET_ARM_QPOS)) < qpos_tol


class SubtaskFSM(ABC):
    """A subtask FSM whose state is a pure function of the current sim state (keeps the expert Markovian)."""

    @abstractmethod
    def fsm_state(self, sim: KitchenPhysics) -> str:
        """Current phase, recomputed from the live state each step."""

    @abstractmethod
    def targets(self, sim: KitchenPhysics, state: str) -> tuple[GripperCmd, np.ndarray, np.ndarray]:
        """(gripper command, EE target position (3,), EE target rotation (3, 3)) for the phase."""
