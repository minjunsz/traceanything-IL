"""Diffusion policy over a window of raw camera frames, conditioned through cross-attention (long-history baselines).

Re-implementation of upstream's `DiffusionUnetHybridImageAttentionPolicy` with the double encoder: each observed step is
one token [R3M ResNet18 feature of every camera, normalized low-dim state]. The `n_obs_steps` frames are all encoded by
the long-range encoder (range 1); the most recent `short_range_obs_horizon` frames are encoded *again* by a separate
short-range encoder (range 2). During training the short-range tokens of a sample are replaced, with probability
`short_range_dropout`, by one learned null token. The UNet (`StaticAttentionConditionalUnet1D`) attends to the
`n_obs_steps + short_range_obs_horizon` tokens at every residual block; `short_range_obs_horizon=None` is a single encoder.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Self

import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch import nn

from markovian_policy.diffusion.loss import diffusion_training_loss
from markovian_policy.diffusion.sampling import run_reverse_diffusion
from markovian_policy.diffusion.scheduler import DiffusionSchedulers
from markovian_policy.nn.normalizer import LinearNormalizer
from markovian_policy.nn.unet1d_attention import StaticAttentionConditionalUnet1D
from markovian_policy.paths import R3M_WEIGHTS
from markovian_policy.perception.r3m import MultiCameraEncoder
from markovian_policy.policies.base import DiffusionPolicy

# Ids in the UNet's modality / range embedding tables (0 = the diffusion-timestep token).
MODALITY_OBS = 1
RANGE_LONG = 1
RANGE_SHORT = 2


@dataclass(frozen=True)
class ImageAttentionPolicyConfig:
    kind: Literal["image_attention"] = "image_attention"  # discriminates policy configs in checkpoints
    horizon: int = 22  # n_obs_steps + 14 in the paper's configs
    n_action_steps: int = 8
    n_obs_steps: int = 8
    action_dim: int = 9
    lowdim_keys: tuple[str, ...] = ("agent_pos", "subtask_sequence")  # concatenated per step
    lowdim_dim: int = 9 + 28
    cameras: tuple[str, ...] = ("scene", "wrist")
    front_cameras: tuple[str, ...] = ("scene",)  # cameras with crop augmentation; the others are resized
    projection_dim: int = 128
    freeze_encoder: bool = False
    r3m_weights: Path | None = R3M_WEIGHTS  # None: random init (a checkpoint carries the trained weights)
    short_range_obs_horizon: int | None = 2  # most recent frames given to the second encoder; None: single encoder
    short_range_dropout: float = 0.3
    diffusion_step_embed_dim: int = 128
    down_dims: tuple[int, ...] = (256, 512, 1024)
    kernel_size: int = 5
    n_groups: int = 8
    num_attention_heads: int = 8
    attention_dropout: float = 0.1
    use_temporal_pos_emb: bool = True
    use_modality_emb: bool = True
    use_range_emb: bool = True
    num_train_timesteps: int = 100
    prediction_type: Literal["epsilon", "sample"] = "epsilon"
    num_ddpm_inference_steps: int = 100
    num_ddim_inference_steps: int = 10

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Rebuild from `dataclasses.asdict` output (e.g. a checkpoint payload)."""
        kwargs: dict[str, Any] = {k: tuple(v) if isinstance(v, list) else v for k, v in data.items()}
        return cls(**kwargs)


class ImageAttentionPolicy(DiffusionPolicy):
    """Parameter names (`encoder`, `short_encoder`, `short_null_token`, `model`, `normalizer`) are checkpoint format."""

    def __init__(self, config: ImageAttentionPolicyConfig) -> None:
        super().__init__()
        short = config.short_range_obs_horizon
        if short is not None:
            assert 0 <= short <= config.n_obs_steps, f"short_range_obs_horizon {short} must be in [0, {config.n_obs_steps}]"
            assert 0.0 <= config.short_range_dropout <= 1.0
        self.config = config
        self.horizon, self.n_action_steps, self.n_obs_steps = config.horizon, config.n_action_steps, config.n_obs_steps
        self.action_dim = config.action_dim

        def make_encoder() -> MultiCameraEncoder:
            return MultiCameraEncoder(
                config.cameras, config.front_cameras, config.projection_dim, config.r3m_weights, config.freeze_encoder
            )

        self.encoder = make_encoder()
        self.token_dim = self.encoder.output_dim + config.lowdim_dim
        self.short_encoder = make_encoder() if short is not None else None
        # Learned token replacing the short-range tokens under short-range dropout (the same for every position).
        self.short_null_token = nn.Parameter(torch.zeros(1, self.token_dim)) if short is not None else None
        self.model = StaticAttentionConditionalUnet1D(
            input_dim=config.action_dim,
            global_cond_dim=self.token_dim,
            diffusion_step_embed_dim=config.diffusion_step_embed_dim,
            down_dims=config.down_dims,
            kernel_size=config.kernel_size,
            n_groups=config.n_groups,
            num_attention_heads=config.num_attention_heads,
            attention_dropout=config.attention_dropout,
            use_temporal_pos_emb=config.use_temporal_pos_emb,
            use_modality_emb=config.use_modality_emb,
            max_modalities=MODALITY_OBS + 1,  # 0 = diffusion timestep, 1 = observation tokens
            use_range_emb=config.use_range_emb,
            max_ranges=3,  # 0 = diffusion timestep, 1 = long, 2 = short
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

    def _tokens(self, encoder: MultiCameraEncoder, obs: dict[str, torch.Tensor], start: int) -> torch.Tensor:
        """(B, n_obs_steps - start, token_dim): [camera features, normalized low-dim] of steps start..n_obs_steps-1."""
        window = slice(start, self.n_obs_steps)
        images = encoder({c: obs[c][:, window] for c in self.cameras})
        lowdim = torch.cat([self.normalizer[k].normalize(obs[k][:, window]).float() for k in self.config.lowdim_keys], dim=-1)
        return torch.cat([images, lowdim], dim=-1)

    def conditioning(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """UNet conditioning kwargs from obs {key: (B, T, ...)}, T >= n_obs_steps.

        Tokens: the n_obs_steps long-range tokens (oldest first), then the short-range ones (if any); every token
        carries the temporal index of its frame and its range id.
        """
        n_obs, short = self.n_obs_steps, self.config.short_range_obs_horizon
        tokens = self._tokens(self.encoder, obs, 0)
        batch, device = tokens.shape[0], tokens.device
        positions = [torch.arange(n_obs, device=device)]
        ranges = [torch.full((n_obs,), RANGE_LONG, device=device)]
        if self.short_encoder is not None and self.short_null_token is not None and short is not None:
            short_tokens = self._tokens(self.short_encoder, obs, n_obs - short)
            if self.training and self.config.short_range_dropout > 0.0:
                dropped = torch.rand(batch, device=device) < self.config.short_range_dropout  # per sample
                null = self.short_null_token.view(1, 1, -1).expand_as(short_tokens)
                short_tokens = torch.where(dropped.view(batch, 1, 1), null.to(short_tokens.dtype), short_tokens)
            tokens = torch.cat([tokens, short_tokens], dim=1)
            positions.append(torch.arange(n_obs - short, n_obs, device=device))
            ranges.append(torch.full((short,), RANGE_SHORT, device=device))
        position, range_ids = torch.cat(positions), torch.cat(ranges)
        return {
            "global_cond": tokens,
            "temporal_positions": position.expand(batch, -1),
            "modality_indices": torch.full_like(range_ids, MODALITY_OBS).expand(batch, -1),
            "range_indices": range_ids.expand(batch, -1),
        }

    def compute_loss(self, batch: dict[str, Any]) -> torch.Tensor:
        """batch: {"obs": {key: (B, T, ...)}, "action": (B, horizon, action_dim)}."""
        trajectory = self.normalizer["action"].normalize(batch["action"])
        return diffusion_training_loss(self.model, self.schedulers.ddpm, trajectory, model_kwargs=self.conditioning(batch["obs"]))

    @torch.no_grad()
    def predict_action(self, obs_dict: dict[str, Any], use_ddim: bool = False) -> dict[str, torch.Tensor]:
        """obs_dict: {"obs": {key: (B, To, ...)}} -> {"action": (B, n_action_steps, D), "action_pred": (B, horizon, D)}."""
        dtype = torch.bfloat16 if self.mixed_precision == "bf16" else torch.float16
        with torch.autocast(self.device.type, dtype, enabled=self.mixed_precision != "no"):
            conditioning = self.conditioning(obs_dict["obs"])
            shape = (len(conditioning["global_cond"]), self.horizon, self.action_dim)
            inpaint_data = torch.zeros(shape, device=self.device, dtype=self.dtype)
            trajectory = run_reverse_diffusion(
                self.model, self.schedulers, inpaint_data, torch.zeros_like(inpaint_data, dtype=torch.bool),
                model_kwargs=conditioning, use_ddim=use_ddim,
            )  # fmt: skip
        action_pred = self.normalizer["action"].unnormalize(trajectory)
        start = self.n_obs_steps - 1  # actions are re-predicted from the last observed step onward
        return {"action": action_pred[:, start : start + self.n_action_steps], "action_pred": action_pred}
