"""Common policy interface: every concrete policy is a thin adapter around `diffusion.sampling` and
`diffusion.loss`, responsible only for (a) how it builds conditioning from raw observations and (b) normalization.
`compute_loss` and `predict_action` are the only two methods the trainer and the evaluation rely on.
"""

from abc import ABC, abstractmethod
from typing import Any, Literal

import torch
import torch.nn as nn

from markovian_policy.nn.normalizer import LinearNormalizer


class DiffusionPolicy(nn.Module, ABC):
    """Base class for a diffusion-based action policy.

    Subclasses own a conditioning strategy, a `NoisePredictor` UNet and a `LinearNormalizer`. They delegate the
    diffusion math to `diffusion.loss.diffusion_training_loss` and `diffusion.sampling.run_reverse_diffusion`.
    """

    normalizer: LinearNormalizer
    mixed_precision: Literal["no", "fp16", "bf16"] = "no"  # set by the trainer; predict_action autocasts accordingly

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def set_normalizer(self, normalizer: LinearNormalizer) -> None:
        self.normalizer.load_state_dict(normalizer.state_dict())

    @abstractmethod
    def compute_loss(self, batch: dict[str, Any]) -> torch.Tensor:
        """batch: {"obs": {key: (B, T, ...)}, "action": (B, T, action_dim)} -> scalar loss."""

    @abstractmethod
    def predict_action(self, obs_dict: dict[str, Any], use_ddim: bool = False) -> dict[str, torch.Tensor]:
        """obs_dict: {"obs": {key: (B, T_obs, ...)}} -> {"action": (B, n_action_steps, D), "action_pred": (B, T, D)}."""

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:  # pyright: ignore[reportIncompatibleMethodOverride]
        return self.compute_loss(batch)
