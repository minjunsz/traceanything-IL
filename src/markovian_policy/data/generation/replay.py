"""Physical replay of a human teleoperation log in the kitchen sim.

The log holds the position targets the teleoperator commanded at every control step. Replaying turns them into
the env's normalized joint-velocity actions (given the current observation) and steps the sim, so the recorded
(state, images, action) are consistent with the simulator the policy is trained and evaluated in.
"""

import numpy as np

from markovian_policy.arrays import Array
from markovian_policy.data.episodes import EpisodeArrays
from markovian_policy.data.generation.mjl import MjlLog
from markovian_policy.sim import actuation
from markovian_policy.sim.env import KitchenEnv
from markovian_policy.sim.tasks import completed_subtasks


def replay_log(env: KitchenEnv, log: MjlLog) -> tuple[EpisodeArrays, list[str]] | None:
    """Replay `log`; returns the episode arrays (o_t, a_t per step) and the subtasks completed, in the order they
    were first completed. None if the simulation diverged."""
    if log.qpos.shape[1] != actuation.N_DOF or log.ctrl.shape[1] != actuation.N_ROBOT:
        raise ValueError("log does not match the kitchen model")
    obs, info = env.reset(options={"qpos": log.qpos[0], "qvel": log.qvel[0]})
    rows: dict[str, list[Array]] = {key: [] for key in (*obs, "action")}
    completed = completed_subtasks(info["qpos"])
    for target in log.ctrl[:-1]:  # the last logged target has no following state
        action = actuation.ctrl_to_action(target, obs["state"], env.sim.dt)
        for key, value in {**obs, "action": action}.items():
            rows[key].append(value)
        obs, _, _, truncated, info = env.step(action)
        if truncated:
            return None
        completed += [s for s in completed_subtasks(info["qpos"]) if s not in completed]
    return {key: np.stack(values) for key, values in rows.items()}, completed
