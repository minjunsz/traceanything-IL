"""Demonstrations from the scripted Markovian expert.

Each job is one attempt: a random plan (sampled from the attempt number, so reproducible whatever the worker
layout) is executed by the expert. An episode succeeds if every subtask of the plan is done and no FSM phase
dithered (was re-entered after being left). Layout: state (T, 60), action (T, 9), scene/wrist images,
current_subtask (T, 7) and subtask_sequence (T, 28); a step stores (o_t, a_t), the noisy observation before the
action with the action taken.
"""

import itertools
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from markovian_policy.arrays import Array
from markovian_policy.data.episodes import EpisodeArrays
from markovian_policy.data.generation.protocols import Candidate
from markovian_policy.experts import Expert, ScriptedExpert
from markovian_policy.sim.env import KitchenEnv, KitchenEnvConfig
from markovian_policy.sim.tasks import PLAN_ENCODING, is_done, sample_plan


def record_expert_episode(env: KitchenEnv, expert: Expert, plan: Sequence[str], max_steps: int, seed: int) -> tuple[EpisodeArrays, bool]:
    """Run `expert` on `plan` from a fresh reset; returns the episode and whether it succeeded.

    A diverging simulation discards the attempt: an empty episode and False.
    """
    obs, _ = env.reset(seed=seed)
    plan_onehot = PLAN_ENCODING.plan_onehot(plan)
    rows: dict[str, list[Array]] = {k: [] for k in ("state", "action", "current_subtask", "subtask_sequence", *env.config.cameras)}
    for _ in range(max_steps):
        if expert.finished:
            break
        step = expert.act(env.sim)
        for key in ("state", *env.config.cameras):
            rows[key].append(obs[key])
        rows["action"].append(step.action)
        rows["current_subtask"].append(PLAN_ENCODING.subtask_onehot(step.subtask))
        rows["subtask_sequence"].append(plan_onehot)
        obs, _, _, truncated, _ = env.step(step.action)
        if truncated:
            return {}, False
        expert.advance(env.sim)
    done = all(is_done(env.sim.data.qpos, subtask) for subtask in plan)
    return {key: np.stack(values) for key, values in rows.items()}, done and expert.oscillation is None


@dataclass(frozen=True)
class ExpertDemoSource:
    """`DemoSource` of expert attempts: an endless stream, so the builder stops at its episode target."""

    env: KitchenEnvConfig = field(default_factory=KitchenEnvConfig)
    chain_len: int = 4  # subtasks per plan
    seed: int = 0
    max_steps_per_subtask: int = 200
    make_expert: Callable[[Sequence[str]], Expert] = ScriptedExpert

    def jobs(self) -> Iterator[int]:
        return itertools.count()

    def key(self, job: int) -> str:
        return f"attempt-{job}"

    def worker(self) -> Callable[[int], Candidate]:
        env = KitchenEnv(self.env)

        def attempt(job: int) -> Candidate:
            plan = sample_plan(np.random.default_rng([self.seed, job]), self.chain_len)
            env_seed = int(np.random.default_rng([self.seed, job, 1]).integers(2**31))
            episode, success = record_expert_episode(
                env, self.make_expert(plan), plan, self.max_steps_per_subtask * len(plan), env_seed
            )  # fmt: skip
            return Candidate(self.key(job), episode or None, success, "ok" if success else "failed", {"plan": "+".join(plan)})

        return attempt

    def report(self) -> Mapping[str, Any]:
        return {}
