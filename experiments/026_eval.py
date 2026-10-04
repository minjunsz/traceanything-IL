"""Closed-loop evaluation of a trained policy in the kitchen sim (needs a GPU node with EGL).

    python experiments/026_eval.py --checkpoint CKPT --out-dir OUT --n-rollouts 50 --n-envs 10

Rerunning the same command resumes from OUT/results.json. Try `--help` for all options (execution horizon:
`--rollout.n-action-steps`, no observation noise: `--env.noise-ratio 0`, ...).
"""

import tyro

from markovian_policy.stages import evaluate

if __name__ == "__main__":
    outcome = evaluate.run(tyro.cli(evaluate.Config))
    print(f"final success rate: {outcome.n_success}/{outcome.n_total}")
