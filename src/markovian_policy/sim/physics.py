# pyright: reportAttributeAccessIssue=false
# (mujoco assembles its namespace dynamically, which static analysis cannot follow)
"""Low-level MuJoCo access: model/data, frame-skipped stepping and offscreen cameras."""

from collections.abc import Mapping
from pathlib import Path
from typing import Final, Literal

import mujoco

from markovian_policy.arrays import FloatArray, Image

type Camera = Literal["scene", "wrist"]

MODEL_PATH: Final = Path(__file__).parent / "assets" / "franka_kitchen.xml"
CAMERA_NAMES: Final[Mapping[Camera, str]] = {"scene": "scene_cam", "wrist": "wrist_cam"}
# MuJoCo does not raise on a diverging simulation: it warns and silently resets the state.
INSTABILITY_WARNINGS: Final = (
    mujoco.mjtWarning.mjWARN_BADQPOS,
    mujoco.mjtWarning.mjWARN_BADQVEL,
    mujoco.mjtWarning.mjWARN_BADQACC,
)


class SimUnstable(RuntimeError):
    """The simulation diverged during a step (the state after it is meaningless)."""


class KitchenPhysics:
    """The kitchen model with `frame_skip` physics steps per control step."""

    def __init__(self, frame_skip: int, image_size: tuple[int, int]) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.data = mujoco.MjData(self.model)
        self.frame_skip = frame_skip
        self.image_size = image_size  # (height, width)
        self._renderer: mujoco.Renderer | None = None

    @property
    def dt(self) -> float:
        """Duration of one control step in seconds."""
        return self.frame_skip * float(self.model.opt.timestep)

    def reset(self, qpos: FloatArray, qvel: FloatArray | None = None) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = qpos
        if qvel is not None:
            self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)

    def step(self, ctrl: FloatArray) -> None:
        """Advance one control step with position targets `ctrl`; raises `SimUnstable` if the physics diverged."""
        self.data.ctrl[:] = ctrl
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        if any(self.data.warning[w].number for w in INSTABILITY_WARNINGS):
            raise SimUnstable
        mujoco.mj_forward(self.model, self.data)  # refresh poses/Jacobians for the expert and the renderer

    def render(self, camera: Camera) -> Image:
        """(H, W, 3) uint8 image from the named camera."""
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, *self.image_size)
        self._renderer.update_scene(self.data, camera=CAMERA_NAMES[camera])
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
