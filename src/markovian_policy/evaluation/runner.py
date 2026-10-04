"""`Evaluator`: facade that runs rounds of parallel trials and feeds the observers.

Plans and env seeds depend only on (seed, trial number), so a resumed run continues exactly where the interrupted
one stopped. What happens with the trials (result files, videos, recorded rollouts) is up to the `TrialSink`s.
"""

import dataclasses
import json
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch

from markovian_policy.evaluation.criteria import DistinctSubtasks
from markovian_policy.evaluation.plans import FixedPlan, RandomPlans, trial_seed
from markovian_policy.evaluation.protocols import PlanSource, RolloutPolicy, SuccessCriterion, TrialSink
from markovian_policy.evaluation.results import TrialRecord, make_record
from markovian_policy.evaluation.rollout import RolloutConfig, run_rollout
from markovian_policy.evaluation.sinks import EpisodeSink, ResultFiles, VideoSink
from markovian_policy.sim.env import KitchenEnvConfig
from markovian_policy.sim.vector import make_vector_env


@dataclass(frozen=True)
class EvalConfig:
    checkpoint: Path
    out_dir: Path
    n_rollouts: int = 50
    n_envs: int = 10  # trials per round
    n_required_subtasks: int = 4  # distinct subtasks for a trial to succeed
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    env: KitchenEnvConfig = field(default_factory=KitchenEnvConfig)  # noise_ratio 0 evaluates without observation noise
    chain_len: int = 4  # random plans of this many subtasks ...
    sequence: tuple[str, ...] | None = None  # ... unless one fixed plan is given
    seed: int = 0
    n_video_trials: int = 20  # videos of the first N trials; -1: all, 0: none
    record_failures: bool = False  # videos of all failed trials instead
    record_zarr: Path | None = None  # also record every trial's rollout as a dataset (incl. qpos/qvel)
    async_envs: bool = True
    device: str = "cuda"
    mixed_precision: Literal["no", "fp16", "bf16"] = "no"  # of the policy while sampling
    trace_weights: Path | None = None  # frozen encoder weights, if not where the checkpoint was trained
    resume: bool = True  # continue from results.json in out_dir if present


@dataclass(frozen=True)
class EvalOutcome:
    n_success: int
    n_total: int
    trials: list[TrialRecord]

    @property
    def success_rate(self) -> float:
        return self.n_success / self.n_total if self.n_total else 0.0


def default_plan_source(config: EvalConfig) -> PlanSource:
    return FixedPlan(config.sequence) if config.sequence is not None else RandomPlans(config.seed, config.chain_len)


def default_sinks(config: EvalConfig, resumed: bool) -> list[TrialSink]:
    config_dict = json.loads(json.dumps(dataclasses.asdict(config), default=str))
    sinks: list[TrialSink] = [ResultFiles(config.out_dir, config.n_rollouts, config_dict)]
    sinks.append(VideoSink(config.out_dir, config.n_video_trials, config.n_rollouts, config.record_failures))
    if config.record_zarr is not None:
        sinks.append(EpisodeSink(config.record_zarr, resume=resumed))
    return sinks


class Evaluator:
    def __init__(
        self,
        policy: RolloutPolicy,
        config: EvalConfig,
        plans: PlanSource | None = None,
        criterion: SuccessCriterion | None = None,
        sinks: Sequence[TrialSink] | None = None,
    ) -> None:
        self.policy, self.config = policy, config
        self.plans = plans or default_plan_source(config)
        self.criterion = criterion or DistinctSubtasks(config.n_required_subtasks)
        self.device = torch.device(config.device)
        self._sinks = sinks

    def run(self) -> EvalOutcome:
        config = self.config
        files = ResultFiles(config.out_dir, config.n_rollouts, {})
        trials: list[TrialRecord] = files.load_trials() if config.resume else []
        if trials:
            print(f"resuming: {len(trials)}/{config.n_rollouts} trials done")
        sinks = self._sinks if self._sinks is not None else default_sinks(config, resumed=bool(trials))
        envs = make_vector_env(config.env, config.n_envs, config.async_envs)
        try:
            for round_idx in range(math.ceil(len(trials) / config.n_envs), math.ceil(config.n_rollouts / config.n_envs)):
                first = len(trials) + 1  # 1-based number of this round's first trial
                numbers = [first + i for i in range(config.n_envs)]
                started = time.time()
                results = run_rollout(
                    self.policy, envs, [self.plans.plan(n) for n in numbers], config.rollout, self.criterion, self.device,
                    seeds=[trial_seed(config.seed, n) for n in numbers],
                    record_video=any(s.wants_video_for(first) for s in sinks), record_episodes=any(s.wants_episodes for s in sinks),
                )  # fmt: skip
                for number, result in zip(numbers, results, strict=False):
                    if number > config.n_rollouts:
                        break
                    trials.append(make_record(number, result))
                    for sink in sinks:
                        sink.on_trial(number, result)
                for sink in sinks:
                    sink.on_round_end(trials)
                n_success = sum(t["result"] == "success" for t in trials)
                print(
                    f"round {round_idx + 1}: {time.time() - started:.0f}s, {[r.result for r in results]}  running {n_success}/{len(trials)}"
                )
        finally:
            envs.close()
            for sink in sinks:
                sink.close()
        return EvalOutcome(sum(t["result"] == "success" for t in trials), len(trials), trials)
