"""Observers of the training loop: the trainer reports steps and epoch ends, callbacks decide what to do."""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from markovian_policy.training.checkpoint import save_atomic
from markovian_policy.training.json_logger import JsonLogger

if TYPE_CHECKING:
    from markovian_policy.training.loop import Trainer


@dataclass(frozen=True)
class StepEvent:
    epoch: int
    global_step: int
    epoch_batches_done: int  # batches finished in this epoch, counting those before a mid-epoch resume
    loss: float  # mean over ranks
    lr: float


@dataclass(frozen=True)
class EpochEvent:
    epoch: int
    global_step: int
    metrics: dict[str, float]  # train_loss, plus val_* when validated


class TrainingCallback(Protocol):
    def on_step(self, trainer: "Trainer", event: StepEvent) -> None: ...

    def on_epoch_end(self, trainer: "Trainer", event: EpochEvent) -> None: ...

    def on_fit_end(self, trainer: "Trainer") -> None: ...


class JsonLogCallback:
    """Appends one row per step and one per epoch (with the validation metrics) to a JSON-lines file; rank 0 only."""

    def __init__(self, logger: JsonLogger) -> None:
        self.logger = logger

    def on_step(self, trainer: "Trainer", event: StepEvent) -> None:
        if trainer.is_main_process:
            self.logger.log({"train_loss": event.loss, "global_step": event.global_step, "epoch": event.epoch, "lr": event.lr})

    def on_epoch_end(self, trainer: "Trainer", event: EpochEvent) -> None:
        if trainer.is_main_process:
            self.logger.log({"epoch": event.epoch, "global_step": event.global_step, **event.metrics})

    def on_fit_end(self, trainer: "Trainer") -> None:
        self.logger.close()


class CheckpointCallback:
    """Writes `latest.ckpt` every epoch (and every fraction of an epoch), the best-k by validation DDIM action MSE,
    and optionally permanent snapshots every `keep_every_epochs` epochs and after the last one."""

    def __init__(self, every_frac: float | None, keep_every_epochs: int | None) -> None:
        self.every_frac, self.keep_every = every_frac, keep_every_epochs

    def on_step(self, trainer: "Trainer", event: StepEvent) -> None:
        frac_steps = max(1, int(trainer.steps_per_epoch * self.every_frac)) if self.every_frac else None
        done = event.epoch_batches_done
        if frac_steps and done % frac_steps == 0 and done < trainer.steps_per_epoch:
            trainer.state.epoch_step = done  # a checkpoint here resumes mid-epoch
            self._save(trainer, trainer.store.latest)

    def on_epoch_end(self, trainer: "Trainer", event: EpochEvent) -> None:
        metric = event.metrics.get("val_ddim_mse")
        if metric is not None and (path := trainer.store.topk.path_for(event.epoch, metric)):
            self._save(trainer, path)
        last = event.epoch == trainer.num_epochs - 1
        if self.keep_every and ((event.epoch + 1) % self.keep_every == 0 or last):
            self._save(trainer, trainer.store.snapshot_path(event.epoch))
        self._save(trainer, trainer.store.latest)

    def on_fit_end(self, trainer: "Trainer") -> None:
        pass

    @staticmethod
    def _save(trainer: "Trainer", path: Path) -> None:
        if trainer.is_main_process:
            save_atomic(trainer.payload(), path)
