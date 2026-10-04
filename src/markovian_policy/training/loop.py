"""Epoch-based (single- or multi-GPU) trainer for `DiffusionPolicy`s.

Per epoch: train (cosine LR with warmup, grad clipping, EMA), then validate the EMA policy. Everything around the
loop is an observer or a strategy: `TrainingCallback`s log and write checkpoints (`latest.ckpt` every epoch and
optionally mid-epoch, the best-k by validation action error, permanent snapshots), an `EpochEvaluator` computes the
validation metrics. Resuming from `latest.ckpt` continues exactly where it stopped, also mid-epoch.
"""

import copy
import dataclasses
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, cast

import torch
import tqdm
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs, set_seed
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from markovian_policy.nn.ema import EMAModel
from markovian_policy.policies.base import DiffusionPolicy
from markovian_policy.training.callbacks import CheckpointCallback, EpochEvent, JsonLogCallback, StepEvent, TrainingCallback
from markovian_policy.training.checkpoint import CheckpointPayload, CheckpointStore, load_payload
from markovian_policy.training.config import TrainConfig
from markovian_policy.training.json_logger import JsonLogger
from markovian_policy.training.samplers import build_train_sampler
from markovian_policy.training.validation import EpochEvaluator, Validator
from markovian_policy.utils import dict_apply


@dataclass
class TrainState:
    epoch: int = 0
    epoch_step: int = 0  # batches already done in `epoch` (non-zero only after a mid-epoch checkpoint)
    global_step: int = 0


class Trainer:
    """`policy` must already have its normalizer set. Defaults: JSON log + checkpoint callbacks and a `Validator`."""

    def __init__(
        self,
        config: TrainConfig,
        policy: DiffusionPolicy,
        train_set: Dataset[Any],
        val_set: Dataset[Any] | None = None,
        callbacks: Sequence[TrainingCallback] | None = None,
        evaluator: EpochEvaluator | None = None,
    ) -> None:
        self.config = config
        self.state = TrainState()
        self.store = CheckpointStore(config.output_dir, config.topk)
        self.acc = Accelerator(
            cpu=config.cpu,
            mixed_precision=config.mixed_precision,
            kwargs_handlers=[
                DistributedDataParallelKwargs(find_unused_parameters=True),
                InitProcessGroupKwargs(timeout=timedelta(minutes=config.nccl_timeout_minutes)),
            ],
        )
        set_seed(config.seed, device_specific=True)
        assert config.batch_size % self.acc.num_processes == 0, "batch_size must be divisible by the number of GPUs"
        self.batch_size = config.batch_size // self.acc.num_processes

        self.sampler = build_train_sampler(
            train_set, self.acc.num_processes, self.acc.process_index, config.seed, config.echo_factor, config.echo_block
        )
        self.train_loader = self._loader(train_set, self.sampler)
        val_loader = None
        if val_set is not None and len(val_set) > 0:  # pyright: ignore[reportArgumentType]
            val_sampler = DistributedSampler(val_set, self.acc.num_processes, self.acc.process_index, shuffle=False)
            val_loader = self._loader(val_set, val_sampler)

        self.steps_per_epoch = min(len(self.train_loader), config.max_train_steps_per_epoch or len(self.train_loader))
        self.num_epochs = config.num_epochs or math.ceil((config.total_train_steps or 1) / self.steps_per_epoch)

        optimizer = torch.optim.AdamW(policy.parameters(), lr=config.lr, betas=config.betas, weight_decay=config.weight_decay)
        self.lr_scheduler = get_scheduler(
            "cosine", optimizer, num_warmup_steps=config.warmup_steps, num_training_steps=self.steps_per_epoch * self.num_epochs
        )
        self.ema = EMAModel(copy.deepcopy(policy), **dataclasses.asdict(config.ema))
        if config.resume and self.store.latest.exists():
            self._resume(policy, optimizer)

        self.policy, self.optimizer = self.acc.prepare(policy, optimizer)  # the scheduler steps once per step: unprepared
        self.unwrapped = cast(DiffusionPolicy, self.acc.unwrap_model(self.policy))
        self.ema_policy = cast(DiffusionPolicy, self.ema.averaged_model.to(self.acc.device))
        for model in (self.unwrapped, self.ema_policy):
            model.mixed_precision = config.mixed_precision

        self.evaluator = evaluator or (Validator(self.acc, val_loader, config.max_val_steps) if val_loader else None)
        self.callbacks: list[TrainingCallback] = (
            list(callbacks)
            if callbacks is not None
            else [
                CheckpointCallback(config.checkpoint_every_frac, config.keep_every_epochs),
                JsonLogCallback(JsonLogger(config.output_dir / "logs.json.txt")),
            ]
        )

    @property
    def is_main_process(self) -> bool:
        return self.acc.is_main_process

    def payload(self) -> CheckpointPayload:
        """Everything needed to resume or to evaluate, as it is now."""
        policy_config = getattr(self.unwrapped, "config", None)
        return CheckpointPayload(
            model=self.unwrapped.state_dict(),
            ema_model=self.ema_policy.state_dict(),
            optimizer=self.optimizer.state_dict(),
            lr_scheduler=self.lr_scheduler.state_dict(),
            ema_step=self.ema.optimization_step,
            global_step=self.state.global_step,
            epoch=self.state.epoch,
            epoch_step=self.state.epoch_step,
            topk=dict(self.store.topk.best),
            train_config=dataclasses.asdict(self.config),
            policy_config=dataclasses.asdict(policy_config) if policy_config is not None else None,
        )

    def fit(self) -> None:
        acc = self.acc
        if self.is_main_process:
            self.config.output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(self.unwrapped.normalizer.state_dict(), self.config.output_dir / "normalizer.pt")
        acc.print(
            f"{acc.num_processes} process(es), batch {self.batch_size}/process, {self.steps_per_epoch} steps/epoch, {self.num_epochs} epochs"
        )
        for epoch in range(self.state.epoch, self.num_epochs):
            self.state.epoch = epoch
            metrics = {"train_loss": self._train_epoch()}
            self.state.epoch_step = 0  # epoch boundary: checkpoints written now resume at the next epoch
            self.sampler.skip_samples = 0
            if self.evaluator is not None and self.state.epoch % self.config.eval_every_epochs == 0:
                metrics |= self.evaluator(self.ema_policy)
            event = EpochEvent(self.state.epoch, self.state.global_step, metrics)
            for callback in self.callbacks:
                callback.on_epoch_end(self, event)
            acc.wait_for_everyone()
        for callback in self.callbacks:
            callback.on_fit_end(self)

    # -- internals --------------------------------------------------------------------------------------------

    def _loader(self, dataset: Dataset[Any], sampler: DistributedSampler[Any]) -> DataLoader[Any]:
        workers = self.config.num_workers
        return DataLoader(
            dataset, batch_size=self.batch_size, sampler=sampler, num_workers=workers,
            pin_memory=self.acc.device.type == "cuda", persistent_workers=workers > 0,
        )  # fmt: skip

    def _resume(self, policy: DiffusionPolicy, optimizer: torch.optim.Optimizer) -> None:
        payload = load_payload(self.store.latest)
        policy.load_state_dict(payload.model)
        self.ema.averaged_model.load_state_dict(payload.ema_model)
        optimizer.load_state_dict(payload.optimizer)
        self.lr_scheduler.load_state_dict(payload.lr_scheduler)
        self.ema.optimization_step = payload.ema_step  # else the EMA decay warmup would restart
        self.store.topk.best = dict(payload.topk)
        epoch = payload.epoch + (1 if payload.epoch_step == 0 else 0)  # a boundary checkpoint holds the last *completed* epoch
        self.state = TrainState(epoch, payload.epoch_step, payload.global_step)
        print(f"resumed from {self.store.latest}: epoch {epoch}, batch {payload.epoch_step}, step {payload.global_step}")

    def _train_epoch(self) -> float:
        acc, config, state = self.acc, self.config, self.state
        self.sampler.set_epoch(state.epoch)
        self.sampler.skip_samples = state.epoch_step * self.batch_size  # same permutation as the interrupted run
        start = state.epoch_step
        self.policy.train()
        losses: list[float] = []
        progress = tqdm.tqdm(self.train_loader, desc=f"epoch {state.epoch}", leave=False, disable=not self.is_main_process, mininterval=30)
        for i, batch in enumerate(progress):
            batch = dict_apply(batch, lambda x: x.to(acc.device, non_blocking=True))
            loss = self.policy(batch)
            acc.backward(loss)
            if config.grad_clip is not None:
                acc.clip_grad_norm_(self.policy.parameters(), config.grad_clip)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.lr_scheduler.step()
            self.ema.step(self.unwrapped)

            losses.append(float(cast(torch.Tensor, acc.reduce(loss.detach(), reduction="mean"))))  # global mean over ranks
            state.global_step += 1
            event = StepEvent(state.epoch, state.global_step, start + i + 1, losses[-1], float(self.lr_scheduler.get_last_lr()[0]))
            for callback in self.callbacks:
                callback.on_step(self, event)
            if event.epoch_batches_done >= self.steps_per_epoch:
                break
        return sum(losses) / max(len(losses), 1)
