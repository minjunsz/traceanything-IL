"""Pairs a DDPM training scheduler with a matching DDIM scheduler for fast inference."""

from dataclasses import dataclass

from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


@dataclass
class DiffusionSchedulers:
    """Bundles the DDPM scheduler used for training with a DDIM scheduler for
    faster sampling, so policy code asks for "the scheduler" without caring
    which sampler variant is active.
    """

    ddpm: DDPMScheduler
    ddim: DDIMScheduler
    num_ddpm_inference_steps: int
    num_ddim_inference_steps: int

    @classmethod
    def from_ddpm(
        cls, ddpm: DDPMScheduler, num_ddpm_inference_steps: int | None = None, num_ddim_inference_steps: int = 10
    ) -> "DiffusionSchedulers":
        """Build the DDIM scheduler by copying the DDPM's beta schedule config."""
        ddim_config = dict(ddpm.config)
        ddim_config.pop("variance_type", None)  # DDPM-only option
        return cls(
            ddpm=ddpm,
            ddim=DDIMScheduler(**ddim_config),
            num_ddpm_inference_steps=num_ddpm_inference_steps or ddpm.config["num_train_timesteps"],
            num_ddim_inference_steps=num_ddim_inference_steps,
        )

    def get(self, use_ddim: bool):
        """Return the requested scheduler, with its inference timesteps already set."""
        scheduler = self.ddim if use_ddim else self.ddpm
        scheduler.set_timesteps(self.num_ddim_inference_steps if use_ddim else self.num_ddpm_inference_steps)
        return scheduler
