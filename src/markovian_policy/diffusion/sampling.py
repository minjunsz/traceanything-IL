"""Shared reverse-diffusion sampling loop, with naive inpainting.

Both the lowdim and hybrid-attention policies need this exact loop (only the
`model_kwargs` they pass differ), so it lives here once instead of being
copy-pasted per policy.
"""

from typing import Any, cast

import torch

from markovian_policy.diffusion.protocols import NoisePredictor
from markovian_policy.diffusion.scheduler import DiffusionSchedulers


@torch.no_grad()
def run_reverse_diffusion(
    model: NoisePredictor,
    schedulers: DiffusionSchedulers,
    inpaint_data: torch.Tensor,
    inpaint_mask: torch.Tensor,
    model_kwargs: dict[str, Any] | None = None,
    use_ddim: bool = False,
    generator: torch.Generator | None = None,
    **step_kwargs: Any,
) -> torch.Tensor:
    """Denoise random noise into a trajectory, holding `inpaint_mask` positions fixed to `inpaint_data`.

    Args:
        model: a NoisePredictor (either UNet variant) called as `model(trajectory, t, **model_kwargs)`.
        inpaint_data: known values to hold fixed, shape (B, T, D). All-zeros with an
            all-False mask (the common case) means "nothing is fixed, predict everything".
        inpaint_mask: bool mask, same shape as inpaint_data; True = held fixed.
        model_kwargs: extra kwargs forwarded to `model` every step (e.g. global_cond,
            or the attention UNet's temporal_positions/modality_indices/range_indices).
        step_kwargs: forwarded to `scheduler.step` (e.g. eta for DDIM).

    Returns:
        Denoised trajectory, shape (B, T, D).
    """
    model_kwargs = model_kwargs or {}
    scheduler = schedulers.get(use_ddim)
    trajectory = torch.randn(size=inpaint_data.shape, dtype=inpaint_data.dtype, device=inpaint_data.device, generator=generator)

    for t in scheduler.timesteps:
        if inpaint_mask.any():
            # Re-noise the known values to the current timestep's noise level so the
            # model sees a trajectory that's internally consistent, then overwrite.
            t_batch = torch.full((inpaint_data.shape[0],), int(t), device=inpaint_data.device, dtype=torch.long)
            noised_inpaint = scheduler.add_noise(inpaint_data, torch.randn_like(inpaint_data), cast(torch.IntTensor, t_batch))
            trajectory[inpaint_mask] = noised_inpaint[inpaint_mask]

        model_output = model(trajectory, t, **model_kwargs)
        trajectory = cast(Any, scheduler.step(model_output, cast(int, t), trajectory, generator=generator, **step_kwargs)).prev_sample

    trajectory[inpaint_mask] = inpaint_data[inpaint_mask]
    return trajectory
