"""Expert success rate over random plans (no rendering): overall and per subtask."""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import tyro

from markovian_policy.experts import ScriptedExpert
from markovian_policy.sim.env import KitchenEnv, KitchenEnvConfig
from markovian_policy.sim.tasks import is_done, sample_plan


@dataclass
class Config:
    n_plans: int = 100
    chain_len: int = 4
    seed: int = 0
    n_workers: int = 4
    max_steps_per_subtask: int = 200


def run(args: tuple[Config, int]) -> tuple[list[str], list[bool], str | None, str]:
    config, attempt = args
    plan = sample_plan(np.random.default_rng([config.seed, attempt]), config.chain_len)
    env = KitchenEnv(KitchenEnvConfig(cameras=()))
    env.reset(seed=attempt)
    expert = ScriptedExpert(plan)
    phase = ""
    for _ in range(config.max_steps_per_subtask * len(plan)):
        if expert.finished:
            break
        step = expert.act(env.sim)
        phase = step.phase
        env.step(step.action)
        expert.advance(env.sim)
    return plan, [is_done(env.sim.data.qpos, s) for s in plan], expert.oscillation, phase


def main(config: Config) -> None:
    with ProcessPoolExecutor(config.n_workers) as pool:
        results = list(pool.map(run, [(config, i) for i in range(config.n_plans)]))
    seen, done = Counter(), Counter()
    for plan, flags, *_ in results:
        seen.update(plan)
        done.update(s for s, ok in zip(plan, flags, strict=False) if ok)
    n_success = sum(all(flags) and osc is None for _, flags, osc, _ in results)
    print(f"success {n_success}/{len(results)} (oscillated: {sum(osc is not None for _, _, osc, _ in results)})")
    # The oracle only advances once a subtask is done, so the first unfinished one is what blocked the plan.
    blockers = Counter(plan[flags.index(False)] for plan, flags, *_ in results if not all(flags))
    print("oscillation at:", dict(Counter(osc for _, _, osc, _ in results if osc)))
    for plan, flags, osc, phase in results:
        if not all(flags):
            print(f"FAIL {'+'.join(plan)} done={[int(f) for f in flags]} osc={osc} last_phase={phase}")
    print("per subtask: done/seen | blocked a plan")
    for name in seen:
        print(f"  {name:11s} {done[name]}/{seen[name]} | {blockers[name]}")


if __name__ == "__main__":
    main(tyro.cli(Config))
