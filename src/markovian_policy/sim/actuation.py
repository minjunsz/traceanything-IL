"""Franka velocity actuation, joint limits and observation noise (from the original franka_config.xml).

The policy commands normalized joint velocities in [-1, 1]. They are integrated into position targets for the
actuators from the last *observed* (noisy) joint positions, and clipped to the joint limits.
"""

from typing import Final

import numpy as np

from markovian_policy.arrays import FloatArray

N_ROBOT: Final = 9  # 7 arm joints + 2 fingers
N_OBJECT: Final = 21
N_DOF: Final = N_ROBOT + N_OBJECT
ACT_AMP: Final = 2.0  # normalized action in [-1, 1] -> joint velocity in rad/s
VEL_LIMIT: Final = 10.0

# Arm/finger joint position limits (9, 2); applied to the actuator targets only.
POS_BOUND: Final[FloatArray] = np.array(
    [(-2.9, 2.9), (-1.8, 1.8), (-2.9, 2.9), (-3.1, 0.0), (-2.9, 2.9), (0.0, 3.8), (-2.9, 2.9), (0.0, 0.04), (0.0, 0.04)]
)
# Observation-noise amplitude per qpos (30,): arm, then desk slides, buttons, blocks/knobs, kettle.
_OBJECT_NOISE: Final = [(2, 0.005), (6, 0.0005), (3, 0.005), (3, 0.1), (3, 0.005), (3, 0.1), (1, 0.005)]
NOISE_AMP: Final[FloatArray] = np.array([0.1] * N_ROBOT + [noise for count, noise in _OBJECT_NOISE for _ in range(count)])


def action_to_ctrl(action: FloatArray, qpos_obs: FloatArray, dt: float) -> FloatArray:
    """(9,) action in [-1, 1] -> (9,) position-actuator targets: observed qpos + velocity * dt, within joint limits."""
    velocity = np.clip(ACT_AMP * np.clip(action, -1.0, 1.0), -VEL_LIMIT, VEL_LIMIT)
    return np.clip(qpos_obs[:N_ROBOT] + velocity * dt, POS_BOUND[:, 0], POS_BOUND[:, 1])


def ctrl_to_action(ctrl: FloatArray, qpos_obs: FloatArray, dt: float) -> FloatArray:
    """The normalized action whose position target is `ctrl`: the inverse of `action_to_ctrl` (before clipping)."""
    return np.clip((ctrl - qpos_obs[:N_ROBOT]) / dt / ACT_AMP, -0.999, 0.999)


def observe(qpos: FloatArray, noise_ratio: float, rng: np.random.Generator) -> FloatArray:
    """(30,) qpos with uniform observation noise scaled by `noise_ratio` (0 disables the noise)."""
    return qpos + noise_ratio * NOISE_AMP * rng.uniform(-1.0, 1.0, N_DOF)
