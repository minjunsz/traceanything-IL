"""Franka kitchen as a gymnasium environment."""

from dataclasses import dataclass
from typing import Any, Final, TypedDict

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from markovian_policy.arrays import Array, FloatArray, Image
from markovian_policy.sim import actuation
from markovian_policy.sim.physics import Camera, KitchenPhysics, SimUnstable

# Fixed reset pose.
INIT_QPOS: Final[FloatArray] = np.array(
    [
        1.48388023e-01, -1.76848573e00, 1.84390296e00, -2.47685760e00, 2.60252026e-01, 7.12533105e-01,
        1.59515394e00, 4.79267505e-02, 3.71350919e-02, -2.66279850e-04, -5.18043486e-05, 3.12877220e-05,
        -4.51199853e-05, -3.90842156e-06, -4.22629655e-05, 6.28065475e-05, 4.04984708e-05, 4.62730939e-04,
        -2.26906415e-04, -4.65501369e-04, -6.44129196e-03, -1.77048263e-03, 1.08009684e-03, -2.69397440e-01,
        3.50383255e-01, 1.61944683e00, 1.00618764e00, 4.06395120e-03, -6.62095997e-03, -2.68278933e-04,
    ]
)  # fmt: skip
GOAL: Final[FloatArray] = np.zeros(30)  # constant placeholder kept so the state layout stays [qpos(30), goal(30)]
STATE_DIM: Final = actuation.N_DOF + len(GOAL)

type Observation = dict[str, Array]  # "state": (60,) float64 plus one (H, W, 3) uint8 image per camera


class StepInfo(TypedDict):
    qpos: FloatArray  # noise-free joint positions (30,)
    qvel: FloatArray  # noise-free joint velocities (29,)
    unstable: bool  # the simulation diverged in this step


@dataclass(frozen=True)
class KitchenEnvConfig:
    frame_skip: int = 40  # physics steps per control step (control period 0.08 s)
    noise_ratio: float = 0.1  # observation noise scale; 0 disables it
    cameras: tuple[Camera, ...] = ("scene", "wrist")
    image_size: tuple[int, int] = (240, 320)  # (height, width)


class KitchenEnv(gym.Env[Observation, FloatArray]):
    """Action: (9,) in [-1, 1] joint velocities. Observation: `Observation`, state = [noisy qpos(30), goal(30)].

    A diverging simulation truncates the episode (`info["unstable"]`). `reset(options={"qpos": ..., "qvel": ...})`
    starts from a recorded state instead of the fixed initial pose.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 12}

    def __init__(self, config: KitchenEnvConfig | None = None, render_mode: str | None = None) -> None:
        self.config = config or KitchenEnvConfig()
        self.render_mode = render_mode
        self.sim = KitchenPhysics(self.config.frame_skip, self.config.image_size)
        self.action_space = spaces.Box(-1.0, 1.0, (actuation.N_ROBOT,))
        image_space = spaces.Box(0, 255, (*self.config.image_size, 3), np.uint8)
        obs_spaces: dict[str, spaces.Space[Any]] = {"state": spaces.Box(-np.inf, np.inf, (STATE_DIM,), np.float64)}
        obs_spaces |= dict.fromkeys(self.config.cameras, image_space)
        self.observation_space = spaces.Dict(obs_spaces)
        self._qpos_obs: FloatArray = np.zeros(actuation.N_DOF)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Observation, StepInfo]:  # type: ignore[override]
        super().reset(seed=seed)
        options = options or {}
        self.sim.reset(options.get("qpos", INIT_QPOS), options.get("qvel"))
        return self._observe(), self._info()

    def step(self, action: FloatArray) -> tuple[Observation, float, bool, bool, StepInfo]:  # type: ignore[override]
        try:
            self.sim.step(actuation.action_to_ctrl(action, self._qpos_obs, self.sim.dt))
        except SimUnstable:
            return self._observe(), 0.0, False, True, self._info(unstable=True)
        return self._observe(), 0.0, False, False, self._info()

    def render(self) -> Image:  # type: ignore[override]
        return self.sim.render("scene")

    def close(self) -> None:
        self.sim.close()

    def _observe(self) -> Observation:
        self._qpos_obs = actuation.observe(self.sim.data.qpos, self.config.noise_ratio, self.np_random)
        images: dict[str, Array] = {camera: self.sim.render(camera) for camera in self.config.cameras}
        return {"state": np.concatenate([self._qpos_obs, GOAL]), **images}

    def _info(self, unstable: bool = False) -> StepInfo:
        return StepInfo(qpos=self.sim.data.qpos.copy(), qvel=self.sim.data.qvel.copy(), unstable=unstable)
