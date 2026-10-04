"""Stage: closed-loop evaluation of a trained policy in the kitchen sim (needs a GPU node with EGL)."""

import torch

from markovian_policy.evaluation import EvalConfig, EvalOutcome, Evaluator, load_policy

Config = EvalConfig


def run(config: Config) -> EvalOutcome:
    policy = load_policy(config.checkpoint, torch.device(config.device), config.mixed_precision, config.trace_weights)
    return Evaluator(policy, config).run()
