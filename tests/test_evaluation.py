"""Rollouts, the Evaluator (resume, outputs) and plots on CPU: physics-only envs (no cameras) and a stub policy."""

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
import torch
import zarr

import markovian_policy.evaluation.rollout as rollout_module
from markovian_policy.evaluation import DistinctSubtasks, EvalConfig, Evaluator, RandomPlans, RolloutConfig, TrialResult, run_rollout
from markovian_policy.evaluation.plots import HorizonResult, collect_best, wilson_interval
from markovian_policy.evaluation.results import make_record
from markovian_policy.sim.env import KitchenEnv, KitchenEnvConfig
from markovian_policy.sim.physics import KitchenPhysics, SimUnstable
from markovian_policy.sim.vector import make_vector_env

CPU = torch.device("cpu")
ENV = KitchenEnvConfig(cameras=())


class ZeroPolicy:
    """Predicts zero joint velocities; records every observation history it is asked about."""

    n_obs_steps, n_action_steps = 2, 4
    cameras: Sequence[str] = ()

    def __init__(self) -> None:
        self.calls: list[dict[str, torch.Tensor]] = []

    def predict_action(self, obs_dict: dict, use_ddim: bool = False) -> dict[str, torch.Tensor]:
        self.calls.append(obs_dict["obs"])
        return {"action_pred": torch.zeros(len(obs_dict["obs"]["agent_pos"]), 6, 9)}


@pytest.fixture
def envs():
    envs = make_vector_env(ENV, 2, asynchronous=False)
    yield envs
    envs.close()


PLANS = [["microwave", "kettle", "slide", "hinge"], ["light", "slide", "kettle", "topknob"]]


def test_rollout_times_out_and_feeds_observation_history(envs) -> None:
    policy = ZeroPolicy()
    results = run_rollout(policy, envs, PLANS, RolloutConfig(task_timeout=10), DistinctSubtasks(4), CPU, seeds=[1, 2])
    assert [r.result for r in results] == ["timeout", "timeout"] and [r.steps for r in results] == [10, 10]
    assert len(policy.calls) == 3  # a chunk of 4 actions per call: steps 0, 4, 8
    first = policy.calls[0]
    assert first["agent_pos"].shape == (2, 2, 9) and first["subtask_sequence"].shape == (2, 2, 28)
    torch.testing.assert_close(first["agent_pos"][:, 0], first["agent_pos"][:, 1])  # first obs repeated at the start
    assert not torch.equal(policy.calls[1]["agent_pos"][:, 0], policy.calls[1]["agent_pos"][:, 1])


def test_success_needs_enough_distinct_completed_subtasks(envs, monkeypatch: pytest.MonkeyPatch) -> None:
    sequence = iter([["microwave"], ["microwave", "kettle"], ["microwave", "kettle"]] + [["microwave", "kettle"]] * 20)
    monkeypatch.setattr(rollout_module, "completed_subtasks", lambda state: next(sequence) if state[0] < 100 else [])
    results = run_rollout(ZeroPolicy(), envs, PLANS, RolloutConfig(task_timeout=10), DistinctSubtasks(2), CPU, seeds=[1, 2])
    assert all(r.result == "success" for r in results)
    assert all(r.completed[:2] == ["microwave", "kettle"] for r in results)
    assert [r.completed_steps for r in results] == [[1, 2], [1, 1]]  # env 1 had both after its first step
    assert make_record(1, results[0])["completed_steps"] == "1+2"


def test_diverging_sim_fails_only_that_trial(envs, monkeypatch: pytest.MonkeyPatch) -> None:
    original, calls = KitchenPhysics.step, {"n": 0}

    def flaky(self: KitchenPhysics, ctrl: np.ndarray) -> None:
        calls["n"] += 1
        if calls["n"] == 5 and self is envs.envs[0].sim:  # the third step of env 0
            raise SimUnstable
        original(self, ctrl)

    monkeypatch.setattr(KitchenPhysics, "step", flaky)
    results = run_rollout(ZeroPolicy(), envs, PLANS, RolloutConfig(task_timeout=6), DistinctSubtasks(4), CPU, seeds=[1, 2])
    assert (results[0].result, results[0].steps) == ("failure", 3)
    assert (results[1].result, results[1].steps) == ("timeout", 6)  # the other env keeps running


def test_env_reports_instability_as_truncation(monkeypatch: pytest.MonkeyPatch) -> None:
    env = KitchenEnv(ENV)
    env.reset(seed=0)
    monkeypatch.setattr(KitchenPhysics, "step", lambda self, ctrl: (_ for _ in ()).throw(SimUnstable()))
    _, _, terminated, truncated, info = env.step(np.zeros(9))
    assert truncated and not terminated and info["unstable"]


def test_recorded_episode_has_dataset_layout(envs) -> None:
    results = run_rollout(
        ZeroPolicy(), envs, PLANS, RolloutConfig(task_timeout=5), DistinctSubtasks(4), CPU, seeds=[1, 2], record_episodes=True
    )  # fmt: skip
    episode = results[0].episode
    assert episode is not None
    shapes = {k: v.shape for k, v in episode.items()}
    assert shapes["action"] == (5, 9) and shapes["state"] == (5, 60) and shapes["qpos"] == (5, 30) and shapes["qvel"] == (5, 29)
    assert shapes["subtask_sequence"] == (5, 28) and shapes["current_subtask"] == (5, 7)
    np.testing.assert_array_equal(episode["action"], 0.0)


def test_plan_progress_counts_leading_plan_subtasks_done_in_order() -> None:
    def result(completed: list[str]) -> TrialResult:
        return TrialResult("timeout", ["a", "b", "c"], completed, 1)

    assert result(["a", "b"]).plan_progress == 2
    assert result(["b", "a", "c"]).plan_progress == 0
    assert result(["a", "x", "b"]).plan_progress == 1


def test_distinct_subtasks_criterion() -> None:
    assert not DistinctSubtasks(3).is_success(["a", "b"]) and DistinctSubtasks(3).is_success(["a", "b", "c"])


def _config(tmp_path: Path, n_rollouts: int, **kwargs) -> EvalConfig:
    return EvalConfig(
        checkpoint=Path("unused"), out_dir=tmp_path / "eval", n_rollouts=n_rollouts, n_envs=2, env=ENV, async_envs=False,
        rollout=RolloutConfig(task_timeout=4), n_video_trials=0, device="cpu", **kwargs,
    )  # fmt: skip


def test_eval_writes_outputs_and_resumes_where_it_stopped(tmp_path: Path) -> None:
    Evaluator(ZeroPolicy(), _config(tmp_path, 2)).run()
    first = json.loads((tmp_path / "eval" / "results.json").read_text())["trials"]
    assert len(first) == 2

    config = _config(tmp_path, 5, record_zarr=tmp_path / "rollouts.zarr")
    outcome = Evaluator(ZeroPolicy(), config).run()
    assert [t["trial"] for t in outcome.trials] == [1, 2, 3, 4, 5] and outcome.trials[:2] == first
    expected = ["+".join(RandomPlans(config.seed, config.chain_len).plan(k)) for k in range(1, 6)]
    assert [t["plan"] for t in outcome.trials] == expected  # plans depend on the trial only
    csv_lines = (tmp_path / "eval" / "results.csv").read_text().splitlines()
    assert len(csv_lines) == 6 and csv_lines[0].startswith("trial,result,reward")
    assert "Success rate" in (tmp_path / "eval" / "summary.txt").read_text()
    root = zarr.open_group(str(tmp_path / "rollouts.zarr"), mode="r")
    assert len(np.asarray(root["meta/episode_ends"])) == 3  # the resumed run recorded trials 3-5 (trial 6 is dropped)


def test_wilson_interval_and_best_checkpoint_selection(tmp_path: Path) -> None:
    low, high = wilson_interval(5, 10)
    assert (round(low, 3), round(high, 3)) == (0.237, 0.763)
    for horizon, ckpt, n_success, n_total in [(4, "a", 3, 10), (4, "b", 6, 10), (4, "c", 9, 5), (8, "a", 2, 10)]:
        path = tmp_path / f"T_a_{horizon}" / ckpt
        path.mkdir(parents=True)
        (path / "results.json").write_text(json.dumps({"n_success": n_success, "n_total": n_total}))
    best = collect_best(tmp_path)
    assert best == [HorizonResult(4, 6, 10, "b"), HorizonResult(8, 2, 10, "a")]  # most trials first, then success rate
