"""Record reference trajectories from the ORIGINAL adept_envs env (run in the `franka-kitchen` pixi env).

Replays the actions of demo episode 0 with observation noise off and saves ground-truth
qpos/qvel per step plus a few camera frames. `tests/test_kitchen_sim.py` compares src/kitchen against it.
"""

import sys
from pathlib import Path

import adept_envs  # noqa: F401  (registers kitchen envs)
import gym
import numpy as np
import zarr

DEMOS = Path("output/kitchen_demos_markovian_scripted_expert.zarr")
OUT = Path("output/kitchen_reference/legacy.npz")
FRAME_STEPS = (0, 50, 100, 200)

root = zarr.open_group(str(DEMOS), mode="r")
n = int(root["meta/episode_ends"][0])
actions = np.asarray(root["data/action"][:n])

env = gym.make("kitchen_relax-v1").unwrapped
env.robot_noise_ratio = 0.0
env.reset()
qpos, qvel, scene, wrist = [env.sim.data.qpos.copy()], [env.sim.data.qvel.copy()], [], []
for t, a in enumerate(actions):
    if t in FRAME_STEPS:
        cams = env.render_cameras()
        scene.append(cams["scene"]), wrist.append(cams["wrist"])
    env.step(a)
    qpos.append(env.sim.data.qpos.copy()), qvel.append(env.sim.data.qvel.copy())

OUT.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    OUT,
    actions=actions,
    qpos=np.stack(qpos),
    qvel=np.stack(qvel),
    scene=np.stack(scene),
    wrist=np.stack(wrist),
    frame_steps=np.array(FRAME_STEPS),
)
print("saved", OUT, "steps", n, file=sys.stderr)
