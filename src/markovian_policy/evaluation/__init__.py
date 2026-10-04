"""Closed-loop evaluation: rollouts, the `Evaluator` facade, result sinks and plots."""

from markovian_policy.evaluation.criteria import DistinctSubtasks
from markovian_policy.evaluation.loading import load_policy
from markovian_policy.evaluation.plans import FixedPlan, RandomPlans
from markovian_policy.evaluation.results import TrialRecord, TrialResult
from markovian_policy.evaluation.rollout import RolloutConfig, run_rollout
from markovian_policy.evaluation.runner import EvalConfig, EvalOutcome, Evaluator

__all__ = [
    "DistinctSubtasks", "EvalConfig", "EvalOutcome", "Evaluator", "FixedPlan", "RandomPlans", "RolloutConfig",
    "TrialRecord", "TrialResult", "load_policy", "run_rollout",
]  # fmt: skip
