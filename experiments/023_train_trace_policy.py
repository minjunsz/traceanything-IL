"""Train the TraceAnything-conditioned diffusion policy (scene + wrist cameras) on a recorded dataset.

Launched through the sbatch script next to this file (multi-GPU). Typical flags:
    --data.trace-cache-dirs scene CACHE_SCENE wrist CACHE_WRIST --train.output-dir RUN

Everything (horizon, windows, batch size, ...) is a dataclass field: see `--help`. Without trace caches the
frozen encoder runs online (slow), so build them first with experiments/022_cache_trace_tokens.py.
"""

import tyro

from markovian_policy.stages import train

if __name__ == "__main__":
    train.run(tyro.cli(train.Config))
