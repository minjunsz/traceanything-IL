"""Diffusion policy conditioned on a frozen TraceAnything encoding of the observation window.

Conditioning tokens for the cross-attention UNet: one token per observed low-dim step, followed by the last
frame's TraceAnything patch tokens for the whole window, one block per camera (cached, or encoded online).
"""

from dataclasses import dataclass, field
from typing import Any, Literal, Self

import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch import nn

from markovian_policy.diffusion.loss import diffusion_training_loss
from markovian_policy.diffusion.sampling import run_reverse_diffusion
from markovian_policy.diffusion.scheduler import DiffusionSchedulers
from markovian_policy.nn.normalizer import LinearNormalizer
from markovian_policy.nn.unet1d_attention import StaticAttentionConditionalUnet1D
from markovian_policy.perception.trace_anything import TraceAnythingConfig, TraceAnythingWindowEncoder
from markovian_policy.policies.base import DiffusionPolicy
from markovian_policy.policies.conditioning import MODALITY_TRACE, Conditioning, TraceTokenSource, assemble_conditioning


@dataclass(frozen=True)
class TracePolicyConfig:
    horizon: int = 22
    n_action_steps: int = 8
    n_obs_steps: int = 8
    action_dim: int = 9
    lowdim_keys: tuple[str, ...] = ("agent_pos", "subtask_sequence")  # concatenated per step
    lowdim_dim: int = 9 + 28
    cameras: tuple[str, ...] = ("scene", "wrist")  # rgb obs keys encoded by TraceAnything, one token block each
    obs_feature_dim: int = 128  # shared cross-attention token dim
    diffusion_step_embed_dim: int = 128
    down_dims: tuple[int, ...] = (256, 512, 1024)
    kernel_size: int = 5
    n_groups: int = 8
    num_attention_heads: int = 8
    attention_dropout: float = 0.1
    num_train_timesteps: int = 100
    prediction_type: Literal["epsilon", "sample"] = "epsilon"
    num_ddpm_inference_steps: int = 100
    num_ddim_inference_steps: int = 10
    trace: TraceAnythingConfig = field(default_factory=TraceAnythingConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Rebuild from `dataclasses.asdict` output (e.g. a checkpoint payload)."""
        kwargs: dict[str, Any] = {k: tuple(v) if isinstance(v, list) else v for k, v in data.items()}
        kwargs["trace"] = TraceAnythingConfig(**data["trace"])
        return cls(**kwargs)


class TraceAttentionPolicy(DiffusionPolicy):
    """Parameter names (`model`, `lowdim_proj`, `trace_proj`, `normalizer`) are part of the checkpoint format."""

    def __init__(self, config: TracePolicyConfig) -> None:
        super().__init__()
        self.config = config
        self.horizon, self.n_action_steps, self.n_obs_steps = config.horizon, config.n_action_steps, config.n_obs_steps
        self.action_dim = config.action_dim
        self.model = StaticAttentionConditionalUnet1D(
            input_dim=config.action_dim,
            global_cond_dim=config.obs_feature_dim,
            diffusion_step_embed_dim=config.diffusion_step_embed_dim,
            down_dims=config.down_dims,
            kernel_size=config.kernel_size,
            n_groups=config.n_groups,
            num_attention_heads=config.num_attention_heads,
            attention_dropout=config.attention_dropout,
            max_modalities=MODALITY_TRACE + len(config.cameras),  # null, low-dim, one per camera
            max_ranges=3,
        )
        self.lowdim_proj = nn.Linear(config.lowdim_dim, config.obs_feature_dim)
        self.trace_proj = nn.Linear(config.trace.embed_dim, config.obs_feature_dim)
        ddpm = DDPMScheduler(
            num_train_timesteps=config.num_train_timesteps, beta_start=1e-4, beta_end=0.02,
            beta_schedule="squaredcos_cap_v2", clip_sample=True, prediction_type=config.prediction_type,
            variance_type="fixed_small",
        )  # fmt: skip
        self.schedulers = DiffusionSchedulers.from_ddpm(ddpm, config.num_ddpm_inference_steps, config.num_ddim_inference_steps)
        self.normalizer = LinearNormalizer()
        # A plain attribute (not a module): the frozen encoder stays out of state_dict, EMA copies and DDP.
        self.token_source = TraceTokenSource(config.cameras, lambda device: TraceAnythingWindowEncoder(config.trace).to(device))

    @property
    def cameras(self) -> tuple[str, ...]:
        """Image observation keys the policy consumes."""
        return self.config.cameras

    def conditioning(self, obs: dict[str, torch.Tensor]) -> Conditioning:
        """Conditioning tokens from obs {key: (B, T, ...)} (T >= n_obs_steps) and optionally cached trace tokens."""
        n_obs = self.n_obs_steps
        lowdim = torch.cat([self.normalizer[k].normalize(obs[k][:, :n_obs]).float() for k in self.config.lowdim_keys], dim=-1)
        trace_blocks = [self.trace_proj(self.token_source.tokens(obs, camera, n_obs)) for camera in self.config.cameras]
        return assemble_conditioning(self.lowdim_proj(lowdim), trace_blocks)

    def compute_loss(self, batch: dict[str, Any]) -> torch.Tensor:
        """batch: {"obs": {key: (B, T, ...)}, "action": (B, horizon, action_dim)}."""
        trajectory = self.normalizer["action"].normalize(batch["action"])
        conditioning = self.conditioning(batch["obs"])
        return diffusion_training_loss(self.model, self.schedulers.ddpm, trajectory, model_kwargs=conditioning.as_kwargs())

    @torch.no_grad()
    def predict_action(self, obs_dict: dict[str, Any], use_ddim: bool = False) -> dict[str, torch.Tensor]:
        """obs_dict: {"obs": {key: (B, To, ...)}} -> {"action": (B, n_action_steps, D), "action_pred": (B, horizon, D)}."""
        dtype = torch.bfloat16 if self.mixed_precision == "bf16" else torch.float16
        with torch.autocast(self.device.type, dtype, enabled=self.mixed_precision != "no"):
            conditioning = self.conditioning(obs_dict["obs"])
            shape = (len(conditioning.tokens), self.horizon, self.action_dim)
            inpaint_data = torch.zeros(shape, device=self.device, dtype=self.dtype)
            trajectory = run_reverse_diffusion(
                self.model, self.schedulers, inpaint_data, torch.zeros_like(inpaint_data, dtype=torch.bool),
                model_kwargs=conditioning.as_kwargs(), use_ddim=use_ddim,
            )  # fmt: skip
        action_pred = self.normalizer["action"].unnormalize(trajectory)
        start = self.n_obs_steps - 1  # actions are re-predicted from the last observed step onward
        return {"action": action_pred[:, start : start + self.n_action_steps], "action_pred": action_pred}
