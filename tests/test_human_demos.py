"""Human-demo pipeline on CPU: log parsing, zip cleaning, replay and the stage that builds the dataset."""

import json
import struct
import zipfile
from pathlib import Path

import numpy as np
import pytest
import zarr

from markovian_policy.data.generation.human_source import list_logs, plan_from_folder
from markovian_policy.data.generation.mjl import parse_mjl
from markovian_policy.data.generation.replay import replay_log
from markovian_policy.sim import actuation
from markovian_policy.sim.env import INIT_QPOS, KitchenEnv, KitchenEnvConfig
from markovian_policy.stages import generate_human

NQ, NV, NU = 30, 29, 9
ENV = KitchenEnvConfig(cameras=(), frame_skip=5, noise_ratio=0.0)  # short control steps keep the test fast


def mjl_bytes(n_records: int, ctrl_offset: float = 0.0, truncate: int = 0) -> bytes:
    """A synthetic log with nmocap=1, nsensordata=2, nuserdata=1 (so the record layout is exercised)."""
    name = b"kitchen"
    header = struct.pack("7i", NQ, NV, NU, 1, 2, 1, len(name)) + name
    record = 1 + NQ + NV + NU + 7 + 2 + 1
    data = np.zeros((n_records, record), dtype=np.float32)
    data[:, 0] = np.arange(n_records) * 0.002
    data[:, 1 : 1 + NQ] = INIT_QPOS
    data[:, 1 + NQ + NV : 1 + NQ + NV + NU] = INIT_QPOS[:NU] + ctrl_offset
    body = data.tobytes()
    return header + body[: len(body) - truncate]


def test_parse_mjl_slices_the_record_layout_and_skips_frames() -> None:
    log = parse_mjl(mjl_bytes(12), skip=5)
    assert log.qpos.shape == (3, NQ) and log.qvel.shape == (3, NV) and log.ctrl.shape == (3, NU)
    np.testing.assert_allclose(log.qpos[1], INIT_QPOS, atol=1e-6)
    np.testing.assert_allclose(log.ctrl[2], INIT_QPOS[:NU], atol=1e-6)


def test_parse_mjl_rejects_truncated_logs() -> None:
    with pytest.raises(ValueError):
        parse_mjl(mjl_bytes(12, truncate=8))


def test_plan_from_folder_maps_switch_to_light() -> None:
    assert plan_from_folder("postcorl_microwave_bottomknob_switch_slide") == ["microwave", "bottomknob", "light", "slide"]
    assert plan_from_folder("friday_microwave_kettle") is None


def test_ctrl_to_action_inverts_the_velocity_controller() -> None:
    rng = np.random.default_rng(0)
    qpos, dt = rng.normal(size=30), 0.08
    ctrl = qpos[:9] + rng.uniform(-0.1, 0.1, 9)  # a feasible step (|velocity| < 2 * 0.999)
    reached = actuation.action_to_ctrl(actuation.ctrl_to_action(ctrl, qpos, dt), qpos, dt)
    np.testing.assert_allclose(reached, np.clip(ctrl, actuation.POS_BOUND[:, 0], actuation.POS_BOUND[:, 1]))


def test_replay_starts_from_the_logged_state_and_recovers_actions() -> None:
    env = KitchenEnv(ENV)
    log = parse_mjl(mjl_bytes(4 * 5, ctrl_offset=0.002), skip=5)  # 4 control steps, target 2 mm above the start pose
    result = replay_log(env, log)
    assert result is not None
    episode, completed = result
    assert completed == [] and len(episode["action"]) == 3  # the last target has no following state
    np.testing.assert_allclose(episode["state"][0][:30], INIT_QPOS, atol=1e-6)  # noise 0: the logged start state
    assert np.abs(episode["action"]).max() <= 0.999 and episode["state"].shape == (3, 60)


def _zip(path: Path, files: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)


def test_list_logs_drops_duplicates_truncated_and_non_plan_folders(tmp_path: Path) -> None:
    root = "kitchen_demos_multitask"
    plan_a, plan_b = f"{root}/friday_microwave_kettle_bottomknob_slide", f"{root}/postcorl_kettle_topknob_switch_hinge"
    log = mjl_bytes(10)
    _zip(
        tmp_path / "demos.zip",
        {
            f"{plan_a}/a.mjl": log, f"{plan_a}/a (1).mjl": log, f"{plan_a}/b.mjl": mjl_bytes(10, 0.1),
            f"{plan_a}/broken.mjl": mjl_bytes(10, truncate=4), f"{plan_b}/c.mjl": log,
            f"{plan_a}/{plan_b.split('/')[-1]}/nested.mjl": mjl_bytes(10, 0.2),  # nested: belongs to its own folder
            f"{root}/notes/x.mjl": log,
        },
    )  # fmt: skip
    usable, skipped = list_logs(tmp_path / "demos.zip")
    assert sorted(Path(n).name for n in usable) == ["a.mjl", "b.mjl", "c.mjl", "nested.mjl"]
    reasons = {Path(s["file"]).name: s["reason"] for s in skipped}
    assert reasons["a (1).mjl"].startswith("duplicate of") and "truncated" in reasons["broken.mjl"] and "plan" in reasons["x.mjl"]


def test_stage_keeps_only_completed_plans_and_resumes(tmp_path: Path) -> None:
    folder = "kitchen_demos_multitask/friday_microwave_kettle_bottomknob_slide"
    _zip(tmp_path / "demos.zip", {f"{folder}/a.mjl": mjl_bytes(4 * 5), f"{folder}/b.mjl": mjl_bytes(4 * 5, 0.01)})
    base = {"zip_path": tmp_path / "demos.zip", "env": ENV, "n_workers": 0}

    summary = generate_human.run(generate_human.Config(out=tmp_path / "strict.zarr", **base))
    assert summary.n_episodes == 0 and summary.outcomes == {"incomplete": 2}  # nothing completes 4 subtasks

    out = tmp_path / "loose.zarr"
    summary = generate_human.run(generate_human.Config(out=out, keep_incomplete=True, **base))
    root = zarr.open_group(str(out), mode="r")
    assert np.asarray(root["meta/episode_ends"]).tolist() == [3, 6] and root["data/subtask_sequence"].shape == (6, 28)
    assert set(root["data"]) == {"state", "action", "subtask_sequence"}  # no camera arrays: the test env renders none
    assert len(json.loads(out.with_suffix(".manifest.json").read_text())) == 2

    again = generate_human.run(generate_human.Config(out=out, keep_incomplete=True, **base))  # resume: all done
    assert again.n_episodes == 2
