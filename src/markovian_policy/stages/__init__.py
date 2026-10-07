"""One facade per pipeline stage. Each module exposes a `Config` dataclass and a `run(config)` function:

- `download_assets`: fetch the pretrained weights and the human teleoperation logs
- `generate_expert`, `generate_human`: build a demonstration dataset (scripted expert / replayed human logs)
- `cache_tokens`: precompute the frozen visual encoder's tokens for a dataset
- `export_frames`: export camera frames as `.npy` for memory-mapped (RAM-shared) training
- `train`: train the TraceAnything policy; `train_image`: train the image-based baseline (R3M ResNet18, raw frames); `train_image_attention`: the same with a double encoder and cross-attention (long histories)
- `evaluate`: closed-loop evaluation of a checkpoint
- `collect_results`: gather training logs and evaluation results of all runs into tidy CSV tables

The scripts in `experiments/` only parse a `Config` from the command line and call `run`.
"""

from markovian_policy.stages import (
    cache_tokens,
    collect_results,
    download_assets,
    evaluate,
    export_frames,
    generate_expert,
    generate_human,
    train,
    train_image,
    train_image_attention,
)

__all__ = [
    "cache_tokens",
    "collect_results",
    "download_assets",
    "evaluate",
    "export_frames",
    "generate_expert",
    "generate_human",
    "train",
    "train_image",
    "train_image_attention",
]
