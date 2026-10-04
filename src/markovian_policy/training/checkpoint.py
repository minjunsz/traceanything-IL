"""Training checkpoints: payload layout, atomic files, and the store that applies the retention policy.

The payload keys are a stable on-disk format (evaluation and resume read them); `CheckpointPayload` documents it.
"""

import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Self

import torch


@dataclass(frozen=True)
class CheckpointPayload:
    model: dict[str, torch.Tensor]  # policy weights (the normalizer is part of them)
    ema_model: dict[str, torch.Tensor]  # EMA policy weights: what evaluation loads
    optimizer: dict[str, Any]
    lr_scheduler: dict[str, Any]
    ema_step: int  # EMA decay-warmup counter (restored on resume)
    global_step: int
    epoch: int  # the last completed epoch, or the current one when epoch_step > 0
    epoch_step: int  # batches already done in `epoch` (0 at an epoch boundary)
    topk: dict[str, float]  # retained best checkpoints: file name -> validation metric
    train_config: dict[str, Any]  # so eval/resume can rebuild without CLI arguments
    policy_config: dict[str, Any] | None  # constructor config of the policy

    def to_dict(self) -> dict[str, Any]:
        """Shallow dict (tensors are not copied)."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(**{f.name: data[f.name] for f in fields(cls)})


def save_atomic(payload: CheckpointPayload, path: Path) -> None:
    """Write to a temp file and rename, so a job killed mid-write never leaves a truncated checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload.to_dict(), tmp)
    os.replace(tmp, path)


def load_payload(path: Path, map_location: str | torch.device = "cpu") -> CheckpointPayload:
    return CheckpointPayload.from_dict(torch.load(path, map_location=map_location, weights_only=False))


class TopKCheckpoints:
    """Keeps the `k` checkpoints with the lowest monitored metric; its state is saved inside each checkpoint."""

    def __init__(self, directory: Path, k: int) -> None:
        self.directory, self.k = directory, k
        self.best: dict[str, float] = {}  # file name -> metric

    def path_for(self, epoch: int, metric: float) -> Path | None:
        """Path to write the checkpoint of this epoch to, or None if it is not among the best k."""
        if self.k <= 0:
            return None
        if len(self.best) >= self.k and metric >= max(self.best.values()):
            return None
        if len(self.best) >= self.k:
            worst = max(self.best, key=self.best.__getitem__)
            (self.directory / worst).unlink(missing_ok=True)
            del self.best[worst]
        name = f"epoch={epoch:03d}-val_ddim_mse={metric:.6f}.ckpt"
        self.best[name] = metric
        return self.directory / name


class CheckpointStore:
    """Where a run keeps its checkpoints: `latest.ckpt`, the top-k by validation metric, permanent snapshots."""

    def __init__(self, run_dir: Path, topk: int) -> None:
        self.directory = run_dir / "checkpoints"
        self.topk = TopKCheckpoints(self.directory, topk)

    @property
    def latest(self) -> Path:
        return self.directory / "latest.ckpt"

    def snapshot_path(self, epoch: int) -> Path:
        return self.directory / f"snapshot_epoch={epoch:03d}.ckpt"
