"""Closed-loop rollouts of a policy in batched kitchen envs.

Each env gets its own plan (the subtask sequence the policy is conditioned on). The policy sees the last
`n_obs_steps` observations (the first one repeated at the start) and predicts a chunk of actions that is executed
open-loop. Completion is judged on the noisy observation, like the policy sees it; a `SuccessCriterion` decides
when a trial has succeeded.
"""

from collections import defaultdict, deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
from gymnasium.vector import VectorEnv

from markovian_policy.arrays import Array, FloatArray
from markovian_policy.data.episodes import EpisodeArrays
from markovian_policy.evaluation.protocols import RolloutPolicy, SuccessCriterion
from markovian_policy.evaluation.results import TrialResult
from markovian_policy.sim.tasks import PLAN_ENCODING, completed_subtasks

N_QPOS = 30  # leading entries of the state observation that are joint positions (the goal part follows)


@dataclass(frozen=True)
class RolloutConfig:
    task_timeout: int = 600  # max env steps per trial
    n_action_steps: int | None = None  # actions executed per policy call; None: the policy's own


class ObservationHistory:
    """The last `n_obs_steps` policy inputs of every env; the first observation fills the window at the start."""

    def __init__(self, policy: RolloutPolicy, plans: Sequence[Sequence[str]], device: torch.device) -> None:
        self.cameras = tuple(policy.cameras)
        self.n_obs_steps = policy.n_obs_steps
        self.device = device
        self.plan_onehot = torch.from_numpy(np.stack([PLAN_ENCODING.plan_onehot(p) for p in plans])).float().to(device)
        self._window: deque[dict[str, torch.Tensor]] = deque(maxlen=policy.n_obs_steps)

    def reset(self, raw: dict[str, Array]) -> None:
        self._window.extend([self._policy_inputs(raw)] * self.n_obs_steps)

    def push(self, raw: dict[str, Array]) -> None:
        self._window.append(self._policy_inputs(raw))

    def stacked(self) -> dict[str, torch.Tensor]:
        """{key: (n_envs, n_obs_steps, ...)} oldest to newest."""
        return {key: torch.stack([step[key] for step in self._window], dim=1) for key in self._window[0]}

    def _policy_inputs(self, raw: dict[str, Array]) -> dict[str, torch.Tensor]:
        inputs = {"agent_pos": torch.from_numpy(raw["state"][:, :9]).float().to(self.device), "subtask_sequence": self.plan_onehot}
        return inputs | {camera: torch.from_numpy(raw[camera]).to(self.device) for camera in self.cameras}


class ActionChunks:
    """Executes the policy's predicted chunks open-loop, calling the policy only when the chunk is used up."""

    def __init__(self, policy: RolloutPolicy, history: ObservationHistory, n_action_steps: int | None) -> None:
        self.policy, self.history = policy, history
        self.n_action_steps = n_action_steps or policy.n_action_steps
        self._queue: deque[torch.Tensor] = deque()

    def next_actions(self) -> FloatArray:
        """(n_envs, action_dim) actions for the next step."""
        if not self._queue:
            n_obs = self.policy.n_obs_steps
            predicted = self.policy.predict_action({"obs": self.history.stacked()}, use_ddim=True)["action_pred"]
            self._queue.extend(predicted[:, n_obs - 1 : n_obs - 1 + self.n_action_steps].unbind(1))
        return self._queue.popleft().float().cpu().numpy()


class TrialTracker:
    """Per-env progress: completed subtasks, divergence, and when each trial finished."""

    def __init__(self, n_envs: int, plans: Sequence[Sequence[str]], criterion: SuccessCriterion, timeout: int) -> None:
        self.plans, self.criterion, self.timeout = [list(p) for p in plans], criterion, timeout
        self.completed: list[list[str]] = [[] for _ in range(n_envs)]
        self.completed_steps: list[list[int]] = [[] for _ in range(n_envs)]  # env step count at each first completion
        self.done = np.zeros(n_envs, dtype=bool)
        self.unstable = np.zeros(n_envs, dtype=bool)
        self.steps = np.full(n_envs, timeout)

    @property
    def all_done(self) -> bool:
        return bool(self.done.all())

    def update(self, step: int, state: FloatArray, unstable: Array) -> None:
        """After env step number `step`: state (n_envs, 60) noisy observations, unstable (n_envs,) divergence flags."""
        for i in np.flatnonzero(~self.done):
            if unstable[i]:
                self.unstable[i] = self.done[i] = True
            else:
                new = [s for s in completed_subtasks(state[i, :N_QPOS]) if s not in self.completed[i]]
                self.completed[i] += new
                self.completed_steps[i] += [step + 1] * len(new)
                self.done[i] = self.criterion.is_success(self.completed[i])
            if self.done[i]:
                self.steps[i] = step + 1

    def result(self, i: int) -> TrialResult:
        success = self.criterion.is_success(self.completed[i])
        outcome = "success" if success else "failure" if self.unstable[i] else "timeout"
        return TrialResult(outcome, self.plans[i], self.completed[i], int(self.steps[i]), self.completed_steps[i])


class EpisodeBuffer:
    """Collects (o_t, a_t) rows per env, until that env's trial is done, in the dataset layout."""

    def __init__(self, n_envs: int, plans: Sequence[Sequence[str]]) -> None:
        self.plans = [list(p) for p in plans]
        self._rows: list[dict[str, list[Array]]] = [defaultdict(list) for _ in range(n_envs)]

    def add(self, i: int, raw: dict[str, Array], info: dict[str, Any], action: FloatArray) -> None:
        for key, value in {
            **{k: v[i] for k, v in raw.items()},
            "qpos": info["qpos"][i],
            "qvel": info["qvel"][i],
            "action": action[i],
        }.items():
            self._rows[i][key].append(value)

    def episode(self, i: int) -> EpisodeArrays | None:
        rows = self._rows[i]
        if not rows:
            return None
        episode: EpisodeArrays = {key: np.stack(values) for key, values in rows.items()}
        length = len(episode["action"])
        episode["subtask_sequence"] = np.tile(PLAN_ENCODING.plan_onehot(self.plans[i]), (length, 1))
        episode["current_subtask"] = np.zeros((length, PLAN_ENCODING.n_subtasks))  # no oracle label for a policy rollout
        return episode


@torch.no_grad()
def run_rollout(
    policy: RolloutPolicy,
    envs: VectorEnv[Any, Any, Any],
    plans: Sequence[Sequence[str]],
    config: RolloutConfig,
    criterion: SuccessCriterion,
    device: torch.device,
    seeds: Sequence[int] | None = None,
    record_video: bool = False,
    record_episodes: bool = False,
) -> list[TrialResult]:
    """One trial per env, all stepped in lockstep until every trial has finished or timed out."""
    n_envs = envs.num_envs
    history = ObservationHistory(policy, plans, device)
    chunks = ActionChunks(policy, history, config.n_action_steps)
    tracker = TrialTracker(n_envs, plans, criterion, config.task_timeout)
    buffer = EpisodeBuffer(n_envs, plans) if record_episodes else None
    frames: list[list[Any]] = [[] for _ in range(n_envs)]

    raw, info = envs.reset(seed=cast(Any, list(seeds) if seeds is not None else None))  # one seed per env
    history.reset(raw)

    def add_frames() -> None:
        images = [key for key in raw if key != "state"]
        for i in range(n_envs):
            frames[i].append(np.concatenate([raw[key][i] for key in images], axis=1))

    if record_video:
        add_frames()
    for step in range(config.task_timeout):
        if tracker.all_done:
            break
        action = chunks.next_actions()
        if buffer is not None:
            for i in np.flatnonzero(~tracker.done):
                buffer.add(int(i), raw, info, action)  # (o_t, a_t): what the policy saw, with the action it took
        raw, _, _, _, info = envs.step(action)
        tracker.update(step, raw["state"], info["unstable"])
        history.push(raw)
        if record_video:
            add_frames()

    results = [tracker.result(i) for i in range(n_envs)]
    for i, result in enumerate(results):
        result.frames = frames[i]
        result.episode = buffer.episode(i) if buffer is not None else None
    return results
