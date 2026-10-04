"""src/kitchen vs. reference trajectories from the original adept_envs (experiments/016_kitchen_reference_legacy.py).

The reference comes from MuJoCo 2.3.5; MuJoCo 3.x drifts through contacts, so tolerances are loose.
"""

from pathlib import Path

import numpy as np
import pytest

from markovian_policy.sim.env import KitchenEnv, KitchenEnvConfig
from markovian_policy.sim.tasks import PLAN_ENCODING, completed_subtasks

REFERENCE = Path("output/kitchen_reference/legacy.npz")
needs_reference = pytest.mark.skipif(not REFERENCE.exists(), reason="run experiments/016 first")


@pytest.fixture(scope="module")
def reference() -> dict[str, np.ndarray]:
    return dict(np.load(REFERENCE))


@needs_reference
def test_replay_matches_reference(reference: dict[str, np.ndarray]) -> None:
    env = KitchenEnv(KitchenEnvConfig(noise_ratio=0.0, cameras=()))
    env.reset(seed=0)
    qpos = [env.sim.data.qpos.copy()]
    for action in reference["actions"]:
        env.step(action)
        qpos.append(env.sim.data.qpos.copy())
    drift = np.abs(np.stack(qpos) - reference["qpos"])
    assert drift[0, :26].max() < 1e-8  # identical reset (kettle quaternion is normalized by the old sim)
    assert drift[1, :9].max() < 1e-6
    assert drift[100, :9].max() < 0.03
    assert drift[-1].max() < 0.15


@needs_reference
@pytest.mark.gpu
def test_rendered_frames_match_reference(reference: dict[str, np.ndarray]) -> None:
    env = KitchenEnv(KitchenEnvConfig(noise_ratio=0.0))
    env.reset(seed=0)
    frames = {}
    for t, action in enumerate(reference["actions"][: reference["frame_steps"].max() + 1]):
        if t in reference["frame_steps"]:
            frames[t] = (env.sim.render("scene"), env.sim.render("wrist"))
        env.step(action)
    for i, t in enumerate(reference["frame_steps"]):
        for camera, image in zip(("scene", "wrist"), frames[t], strict=False):
            ref = reference[camera][i]
            assert image.shape == ref.shape and image.dtype == np.uint8
            # Identical rendering at t=0; later frames only differ through trajectory drift (the wrist cam is sensitive).
            assert np.abs(image.astype(int) - ref).mean() < (1.0 if t == 0 else 5.0 if camera == "scene" else 20.0)


def test_noise_off_is_deterministic() -> None:
    a, b = (KitchenEnv(KitchenEnvConfig(noise_ratio=0.0, cameras=())) for _ in range(2))
    obs_a, _ = a.reset(seed=1)
    obs_b, _ = b.reset(seed=2)
    action = np.full(9, 0.3)
    np.testing.assert_array_equal(a.step(action)[0]["state"], b.step(action)[0]["state"])
    np.testing.assert_array_equal(obs_a["state"], obs_b["state"])


def test_observation_noise_is_bounded_and_seeded() -> None:
    env = KitchenEnv(KitchenEnvConfig(noise_ratio=0.1, cameras=()))
    obs, info = env.reset(seed=3)
    assert obs["state"].shape == (60,)
    assert np.abs(obs["state"][:9] - info["qpos"][:9]).max() <= 0.1 * 0.1
    again, _ = env.reset(seed=3)
    np.testing.assert_array_equal(obs["state"], again["state"])


def test_initial_state_has_no_completed_subtask() -> None:
    _, info = KitchenEnv(KitchenEnvConfig(cameras=())).reset(seed=0)
    assert completed_subtasks(info["qpos"]) == []


def test_sequence_onehot_layout() -> None:
    v = PLAN_ENCODING.plan_onehot(["kettle", "microwave"])
    assert v.shape == (28,) and v.sum() == 2 and v[1] == 1 and v[7] == 1
