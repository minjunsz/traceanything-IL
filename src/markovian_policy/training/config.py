"""Hyperparameters of the `Trainer` (nothing policy- or dataset-specific)."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal

WARMUP_REFERENCE: Final = (500, 64)  # (steps, batch size) the default warmup was tuned at


@dataclass(frozen=True)
class EMAConfig:
    update_after_step: int = 0
    inv_gamma: float = 1.0
    power: float = 0.75
    max_value: float = 0.9999


@dataclass(frozen=True)
class TrainConfig:
    output_dir: Path = Path("output/runs/default")
    batch_size: int = 64  # global; split across GPUs
    num_workers: int = 8
    lr: float = 1e-4
    betas: tuple[float, float] = (0.95, 0.999)
    weight_decay: float = 1e-6
    # Warmup is 500 steps at batch 64 and scales inversely with batch size (same number of samples).
    lr_warmup_steps: int | None = None
    total_train_steps: int | None = 300_000  # converted to whole epochs; ignored if num_epochs is set
    num_epochs: int | None = None
    max_train_steps_per_epoch: int | None = None  # cap for smoke tests
    grad_clip: float | None = 1.0
    mixed_precision: Literal["no", "fp16", "bf16"] = "bf16"
    ema: EMAConfig = field(default_factory=EMAConfig)
    eval_every_epochs: int = 1  # validation loss, DDPM/DDIM action MSE, checkpoint
    max_val_steps: int | None = None  # cap validation batches per rank
    checkpoint_every_frac: float | None = None  # extra latest.ckpt every this fraction of an epoch
    topk: int = 5  # best checkpoints kept, by validation DDIM action MSE (0 disables)
    # Also keep a permanent snapshot every this many epochs (and after the last epoch), spread over training.
    keep_every_epochs: int | None = None
    echo_factor: int = 1  # data echoing: each sample used this many times per epoch
    echo_block: int = 5000
    seed: int = 41
    resume: bool = True
    nccl_timeout_minutes: int = 45
    cpu: bool = False  # force CPU (tests)

    @property
    def warmup_steps(self) -> int:
        """LR warmup steps; by default the same number of *samples* as 500 steps at batch 64."""
        if self.lr_warmup_steps is not None:
            return self.lr_warmup_steps
        steps, batch = WARMUP_REFERENCE
        return round(steps * batch / self.batch_size)
