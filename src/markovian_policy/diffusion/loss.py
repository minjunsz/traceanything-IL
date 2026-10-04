"""Shared diffusion training loss: sample noise/timestep, corrupt, predict, compare."""

from typing import Any, cast

import torch
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from einops import reduce

from markovian_policy.diffusion.protocols import NoisePredictor


def diffusion_training_loss(
    model: NoisePredictor,
    ddpm_scheduler: DDPMScheduler,
    trajectory: torch.Tensor,
    model_kwargs: dict[str, Any] | None = None,
    loss_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """One diffusion training step: corrupt `trajectory` with random noise at a random
    timestep, ask `model` to recover the noise (or the clean sample, depending on the
    scheduler's `prediction_type`), and return the (optionally masked) MSE loss.

    `loss_mask`, if given, zeroes out loss contributions from positions that were
    inpainted from known values rather than actually predicted (e.g. `past_action_visible`).
    """
    model_kwargs = model_kwargs or {}
    noise = torch.randn(trajectory.shape, device=trajectory.device)
    timesteps = torch.randint(0, ddpm_scheduler.config["num_train_timesteps"], (trajectory.shape[0],), device=trajectory.device).long()
    noisy_trajectory = ddpm_scheduler.add_noise(trajectory, noise, cast(torch.IntTensor, timesteps))

    pred = model(noisy_trajectory, timesteps, **model_kwargs)

    prediction_type = ddpm_scheduler.config["prediction_type"]
    if prediction_type == "epsilon":
        target = noise
    elif prediction_type == "sample":
        target = trajectory
    else:
        raise ValueError(f"Unsupported prediction type {prediction_type}")

    loss = F.mse_loss(pred, target, reduction="none")
    if loss_mask is not None:
        loss = loss * loss_mask.type(loss.dtype)
    return reduce(loss, "b ... -> b (...)", "mean").mean()
