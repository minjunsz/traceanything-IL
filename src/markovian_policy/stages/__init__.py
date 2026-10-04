"""One facade per pipeline stage. Each module exposes a `Config` dataclass and a `run(config)` function:

- `download_assets`: fetch the pretrained weights and the human teleoperation logs
- `generate_expert`, `generate_human`: build a demonstration dataset (scripted expert / replayed human logs)
- `cache_tokens`: precompute the frozen visual encoder's tokens for a dataset
- `train`: train the policy
- `evaluate`: closed-loop evaluation of a checkpoint

The scripts in `experiments/` only parse a `Config` from the command line and call `run`.
"""

from markovian_policy.stages import cache_tokens, download_assets, evaluate, generate_expert, generate_human, train

__all__ = ["cache_tokens", "download_assets", "evaluate", "generate_expert", "generate_human", "train"]
