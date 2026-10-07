"""Diffusion policy conditioned on a window of raw camera frames (the image-based baselines).

Re-implementation of upstream's `DiffusionUnetHybridImageTargetedPolicy`: every observed step is encoded by a trainable
R3M ResNet18 per camera (`perception.r3m.MultiCameraEncoder`), concatenated with the normalized low-dim state, and the
`n_obs_steps` per-step features are flattened into one FiLM vector (`nn.unet1d.ConditionalUnet1D`).

`n_obs_steps` is the history length: 2 reproduces the paper's Markovian baseline; larger values give the
non-Markovian (human data) baselines with a multi-frame history.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Self

import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from markovian_policy.diffusion.loss import diffusion_training_loss
from markovian_policy.diffusion.sampling import run_reverse_diffusion
from markovian_policy.diffusion.scheduler import DiffusionSchedulers
from markovian_policy.nn.normalizer import LinearNormalizer
from markovian_policy.nn.unet1d import ConditionalUnet1D
from markovian_policy.paths import R3M_WEIGHTS
from markovian_policy.perception.r3m import MultiCameraEncoder
from markovian_policy.policies.base import DiffusionPolicy


@dataclass(frozen=True)
class ImagePolicyConfig:
    kind: Literal["image_film"] = "image_film"  # discriminates policy configs in checkpoints
    horizon: int = 22  # n_obs_steps + 14 in the paper's configs
    n_action_steps: int = 8
    n_obs_steps: int = 2
    action_dim: int = 9
    lowdim_keys: tuple[str, ...] = ("agent_pos", "subtask_sequence")  # concatenated per step
    lowdim_dim: int = 9 + 28
    cameras: tuple[str, ...] = ("scene", "wrist")
    front_cameras: tuple[str, ...] = ("scene",)  # cameras with crop augmentation; the others are resized
    projection_dim: int = 128
    freeze_encoder: bool = False
    r3m_weights: Path | None = R3M_WEIGHTS  # None: random init (a checkpoint carries the trained weights)
    diffusion_step_embed_dim: int = 128
    down_dims: tuple[int, ...] = (256, 512, 1024)
    kernel_size: int = 5
    n_groups: int = 8
    num_train_timesteps: int = 100
    prediction_type: Literal["epsilon", "sample"] = "epsilon"
    num_ddpm_inference_steps: int = 100
    num_ddim_inference_steps: int = 10

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Rebuild from `dataclasses.asdict` output (e.g. a checkpoint payload)."""
        kwargs: dict[str, Any] = {k: tuple(v) if isinstance(v, list) else v for k, v in data.items()}
        return cls(**kwargs)


class ImageFilmPolicy(DiffusionPolicy):
    """Parameter names (`encoder`, `model`, `normalizer`) are part of the checkpoint format."""

    def __init__(self, config: ImagePolicyConfig) -> None:
        super().__init__()
        self.config = config
        self.horizon, self.n_action_steps, self.n_obs_steps = config.horizon, config.n_action_steps, config.n_obs_steps
        self.action_dim = config.action_dim
        self.encoder = MultiCameraEncoder(
            config.cameras, config.front_cameras, config.projection_dim, config.r3m_weights, config.freeze_encoder
        )
        step_dim = self.encoder.output_dim + config.lowdim_dim
        self.model = ConditionalUnet1D(
            input_dim=config.action_dim,
            global_cond_dim=step_dim * config.n_obs_steps,
            diffusion_step_embed_dim=config.diffusion_step_embed_dim,
            down_dims=config.down_dims,
            kernel_size=config.kernel_size,
            n_groups=config.n_groups,
        )
        ddpm = DDPMScheduler(
            num_train_timesteps=config.num_train_timesteps, beta_start=1e-4, beta_end=0.02,
            beta_schedule="squaredcos_cap_v2", clip_sample=True, prediction_type=config.prediction_type,
            variance_type="fixed_small",
        )  # fmt: skip
        self.schedulers = DiffusionSchedulers.from_ddpm(ddpm, config.num_ddpm_inference_steps, config.num_ddim_inference_steps)
        self.normalizer = LinearNormalizer()

    @property
    def cameras(self) -> tuple[str, ...]:
        """Image observation keys the policy consumes."""
        return self.config.cameras

    def global_cond(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """(B, n_obs_steps * (n_cameras * projection_dim + lowdim_dim)) from obs {key: (B, T, ...)}, T >= n_obs_steps."""
        n_obs = self.n_obs_steps
        images = self.encoder({c: obs[c][:, :n_obs] for c in self.cameras})
        lowdim = torch.cat([self.normalizer[k].normalize(obs[k][:, :n_obs]).float() for k in self.config.lowdim_keys], dim=-1)
        return torch.cat([images, lowdim], dim=-1).reshape(images.shape[0], -1)

    def compute_loss(self, batch: dict[str, Any]) -> torch.Tensor:
        """batch: {"obs": {key: (B, T, ...)}, "action": (B, horizon, action_dim)}."""
        trajectory = self.normalizer["action"].normalize(batch["action"])
        return diffusion_training_loss(
            self.model, self.schedulers.ddpm, trajectory, model_kwargs={"global_cond": self.global_cond(batch["obs"])}
        )

    @torch.no_grad()
    def predict_action(self, obs_dict: dict[str, Any], use_ddim: bool = False) -> dict[str, torch.Tensor]:
        """obs_dict: {"obs": {key: (B, To, ...)}} -> {"action": (B, n_action_steps, D), "action_pred": (B, horizon, D)}."""
        dtype = torch.bfloat16 if self.mixed_precision == "bf16" else torch.float16
        with torch.autocast(self.device.type, dtype, enabled=self.mixed_precision != "no"):
            global_cond = self.global_cond(obs_dict["obs"])
            shape = (len(global_cond), self.horizon, self.action_dim)
            inpaint_data = torch.zeros(shape, device=self.device, dtype=self.dtype)
            trajectory = run_reverse_diffusion(
                self.model, self.schedulers, inpaint_data, torch.zeros_like(inpaint_data, dtype=torch.bool),
                model_kwargs={"global_cond": global_cond}, use_ddim=use_ddim,
            )  # fmt: skip
        action_pred = self.normalizer["action"].unnormalize(trajectory)
        start = self.n_obs_steps - 1  # actions are re-predicted from the last observed step onward
        return {"action": action_pred[:, start : start + self.n_action_steps], "action_pred": action_pred}
