"""Diffusion policy over purely low-dimensional observations (no vision encoder).

Conditioning is a single FiLM vector built by flattening the observation history (see
`nn.unet1d.ConditionalUnet1D`). It validates the training stack end to end on the simplest policy variant.
"""

from collections.abc import Sequence
from typing import Any

import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from markovian_policy.diffusion.loss import diffusion_training_loss
from markovian_policy.diffusion.sampling import run_reverse_diffusion
from markovian_policy.diffusion.scheduler import DiffusionSchedulers
from markovian_policy.nn.normalizer import LinearNormalizer
from markovian_policy.nn.unet1d import ConditionalUnet1D
from markovian_policy.policies.base import DiffusionPolicy


class DiffusionUnetLowdimPolicy(DiffusionPolicy):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        ddpm_scheduler: DDPMScheduler,
        horizon: int,
        n_action_steps: int,
        n_obs_steps: int,
        num_ddpm_inference_steps: int | None = None,
        num_ddim_inference_steps: int = 10,
        diffusion_step_embed_dim: int = 128,
        down_dims: Sequence[int] = (256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
    ) -> None:
        """obs_dim: total size of all low-dim observation keys at one step."""
        super().__init__()
        self.action_dim, self.obs_dim = action_dim, obs_dim
        self.model = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=obs_dim * n_obs_steps,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
        )
        self.schedulers = DiffusionSchedulers.from_ddpm(ddpm_scheduler, num_ddpm_inference_steps, num_ddim_inference_steps)
        self.normalizer = LinearNormalizer()
        self.horizon, self.n_action_steps, self.n_obs_steps = horizon, n_action_steps, n_obs_steps

    def _flatten_obs_history(self, nobs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Concatenate normalized obs keys (sorted, for determinism) into one (B, T, obs_dim) tensor."""
        return torch.cat([nobs[key] for key in sorted(nobs)], dim=-1)

    def compute_loss(self, batch: dict[str, Any]) -> torch.Tensor:
        nobs = self._flatten_obs_history(self.normalizer.normalize(batch["obs"]))
        trajectory = self.normalizer["action"].normalize(batch["action"])
        global_cond = nobs[:, : self.n_obs_steps].reshape(nobs.shape[0], -1)
        return diffusion_training_loss(self.model, self.schedulers.ddpm, trajectory, model_kwargs={"global_cond": global_cond})

    def predict_action(self, obs_dict: dict[str, Any], use_ddim: bool = False) -> dict[str, torch.Tensor]:
        nobs = self._flatten_obs_history(self.normalizer.normalize(obs_dict["obs"]))
        batch, n_obs = nobs.shape[0], self.n_obs_steps
        global_cond = nobs[:, :n_obs].reshape(batch, -1)

        shape = (batch, self.horizon, self.action_dim)
        inpaint_data = torch.zeros(shape, device=self.device, dtype=self.dtype)
        inpaint_mask = torch.zeros_like(inpaint_data, dtype=torch.bool)
        trajectory = run_reverse_diffusion(
            self.model, self.schedulers, inpaint_data, inpaint_mask, model_kwargs={"global_cond": global_cond}, use_ddim=use_ddim
        )
        action_pred = self.normalizer["action"].unnormalize(trajectory)
        start = n_obs - 1  # re-predict actions from the last observed step onward
        return {"action": action_pred[:, start : start + self.n_action_steps], "action_pred": action_pred}
