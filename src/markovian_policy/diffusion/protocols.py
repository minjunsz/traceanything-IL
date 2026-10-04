"""Structural interfaces shared across the diffusion-policy stack.

These exist so that the training loop and policy classes can stay written
against a stable "shape" of collaborator, independent of which concrete
strategy is plugged in (FiLM vs. attention conditioning). Read this file first to see how the pieces fit together before
diving into any single implementation.
"""

from typing import Any, Protocol, runtime_checkable

import torch


@runtime_checkable
class NoisePredictor(Protocol):
    """Common call signature shared by every conditional-UNet noise predictor.

    Both `models.unet1d.ConditionalUnet1D` (FiLM conditioning) and
    `models.unet1d_attention.StaticAttentionConditionalUnet1D` (cross-attention
    conditioning) satisfy this protocol, which is all `diffusion.sampling` and
    `diffusion.loss` depend on -- neither cares which conditioning strategy is
    behind the model it was handed.
    """

    def __call__(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | float | int,
        local_cond: torch.Tensor | None = None,
        global_cond: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor: ...
