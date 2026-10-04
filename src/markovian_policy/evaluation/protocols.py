"""Interfaces of the closed-loop evaluation."""

from collections.abc import Sequence
from typing import Any, Protocol

import torch

from markovian_policy.evaluation.results import TrialRecord, TrialResult


class RolloutPolicy(Protocol):
    """What a rollout needs from a policy: its observation window, chunk length and the cameras it reads."""

    n_obs_steps: int
    n_action_steps: int

    @property
    def cameras(self) -> Sequence[str]: ...

    def predict_action(self, obs_dict: dict[str, Any], use_ddim: bool = False) -> dict[str, torch.Tensor]:
        """{"obs": {key: (B, To, ...)}} -> {"action_pred": (B, horizon, action_dim), ...}."""
        ...


class PlanSource(Protocol):
    """Strategy: the plan (subtask sequence) a trial is conditioned on. Depends on the trial number only."""

    def plan(self, trial: int) -> list[str]: ...


class SuccessCriterion(Protocol):
    """Strategy: when a trial counts as successful, given the subtasks completed so far (in completion order)."""

    def is_success(self, completed: Sequence[str]) -> bool: ...


class TrialSink(Protocol):
    """Observer of an evaluation: receives every finished trial and each finished round."""

    @property
    def wants_video(self) -> bool: ...

    @property
    def wants_episodes(self) -> bool: ...

    def wants_video_for(self, first_trial: int) -> bool:
        """Whether the round starting at (1-based) trial `first_trial` should keep frames."""
        ...

    def on_trial(self, number: int, result: TrialResult) -> None: ...

    def on_round_end(self, trials: Sequence[TrialRecord]) -> None: ...

    def close(self) -> None: ...
