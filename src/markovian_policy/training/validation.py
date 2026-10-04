"""Validation of the EMA policy: loss over the whole validation set, sampled-action error on one fixed batch."""

from typing import Any, Protocol, cast

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader

from markovian_policy.policies.base import DiffusionPolicy
from markovian_policy.utils import dict_apply


class EpochEvaluator(Protocol):
    """Strategy: the metrics computed with the EMA policy at the end of an epoch."""

    def __call__(self, policy: DiffusionPolicy) -> dict[str, float]: ...


class Validator:
    """`val_loss` is sharded across ranks and averaged; `val_ddpm_mse`/`val_ddim_mse` (rank 0) compare sampled
    action chunks with the ground truth on the first validation batch, which stays the same for the whole run."""

    def __init__(self, accelerator: Accelerator, loader: DataLoader[Any], max_steps: int | None = None) -> None:
        self.acc, self.loader, self.max_steps = accelerator, loader, max_steps
        self._fixed_batch: dict[str, Any] | None = None

    @torch.no_grad()
    def __call__(self, policy: DiffusionPolicy) -> dict[str, float]:
        acc = self.acc
        policy.eval()
        totals = torch.zeros(2, device=acc.device, dtype=torch.float64)  # (sum of batch losses, n batches)
        with acc.autocast():
            for i, batch in enumerate(self.loader):
                batch = dict_apply(batch, lambda x: x.to(acc.device, non_blocking=True))
                if self._fixed_batch is None:
                    self._fixed_batch = batch
                totals[0] += policy.compute_loss(batch).double()
                totals[1] += 1
                if self.max_steps and i + 1 >= self.max_steps:
                    break
        totals = cast(torch.Tensor, acc.reduce(totals, reduction="sum"))
        metrics = {"val_loss": float(totals[0] / totals[1])} if totals[1] > 0 else {}
        if acc.is_main_process and self._fixed_batch is not None:
            for name, use_ddim in (("ddpm", False), ("ddim", True)):
                predicted = policy.predict_action({"obs": self._fixed_batch["obs"]}, use_ddim=use_ddim)["action_pred"]
                metrics[f"val_{name}_mse"] = F.mse_loss(predicted, self._fixed_batch["action"]).item()
        return metrics
