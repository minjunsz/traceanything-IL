"""Exponential moving average of model weights.

Ported from third_party/diffusion-policy-experiments/diffusion_policy/model/diffusion/ema_model.py.
Pure PyTorch, no legacy dependency.
"""

import torch
from torch.nn.modules.batchnorm import _BatchNorm


class EMAModel:
    """Maintains an EMA copy of a model's weights, updated by calling `.step(model)`.

    Decay follows a warmup schedule (see `get_decay`): 0 until `update_after_step`,
    then rising toward `max_value`. With the defaults (inv_gamma=1, power=2/3),
    decay reaches 0.999 at ~31.6k steps and 0.9999 at ~1M steps -- tuned for
    million-plus-step training runs.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        update_after_step: int = 0,
        inv_gamma: float = 1.0,
        power: float = 2 / 3,
        min_value: float = 0.0,
        max_value: float = 0.9999,
    ):
        self.averaged_model = model
        self.averaged_model.eval()
        self.averaged_model.requires_grad_(False)

        self.update_after_step = update_after_step
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value
        self.max_value = max_value

        self.decay = 0.0
        self.optimization_step = 0

    def get_decay(self, optimization_step: int) -> float:
        step = max(0, optimization_step - self.update_after_step - 1)
        if step <= 0:
            return 0.0
        value = 1 - (1 + step / self.inv_gamma) ** -self.power
        return max(self.min_value, min(value, self.max_value))

    @torch.no_grad()
    def step(self, new_model: torch.nn.Module) -> None:
        self.decay = self.get_decay(self.optimization_step)

        for module, ema_module in zip(new_model.modules(), self.averaged_model.modules(), strict=False):
            for param, ema_param in zip(module.parameters(recurse=False), ema_module.parameters(recurse=False), strict=False):
                if isinstance(module, _BatchNorm):
                    # BatchNorm affine params must track running_mean/running_var exactly
                    # (hard-copied as buffers below), so EMA-blending them would desync.
                    ema_param.copy_(param.to(dtype=ema_param.dtype).data)
                elif not param.requires_grad:
                    ema_param.copy_(param.to(dtype=ema_param.dtype).data)
                else:
                    ema_param.mul_(self.decay)
                    ema_param.add_(param.data.to(dtype=ema_param.dtype), alpha=1 - self.decay)

            for buf, ema_buf in zip(module.buffers(recurse=False), ema_module.buffers(recurse=False), strict=False):
                ema_buf.copy_(buf.to(dtype=ema_buf.dtype).data)

        self.optimization_step += 1
