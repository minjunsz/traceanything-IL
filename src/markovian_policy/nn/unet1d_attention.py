"""Cross-attention-conditioned 1D UNet noise predictor.

Ported from third_party/diffusion-policy-experiments/diffusion_policy/model/diffusion/conditional_unet_1d_static_attention.py.
Same down/mid/up ResNet-Conv1D topology as `models.unet1d.ConditionalUnet1D`,
but conditioning is injected via cross-attention over a token sequence
(observation-per-timestep tokens + one diffusion-timestep token) instead of a
single flattened FiLM vector -- used by the hybrid image + attention policy.
"""

import itertools
import logging
import math
from collections.abc import Sequence
from typing import Any, cast

import torch
import torch.nn as nn

from markovian_policy.nn.blocks import Conv1dBlock, Downsample1d, SinusoidalPosEmb, Upsample1d

logger = logging.getLogger(__name__)


class TemporalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for fixed-length observation token sequences."""

    pe: torch.Tensor

    def __init__(self, embed_dim: int, max_position: int = 1000):
        super().__init__()
        position = torch.arange(max_position).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-(math.log(10000.0) / embed_dim)))
        pe = torch.zeros(max_position, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor, positions: torch.Tensor, token_mask: torch.Tensor | None = None) -> torch.Tensor:
        pos_embed = self.pe[positions.long()].to(dtype=x.dtype, device=x.device)
        if token_mask is not None:
            pos_embed = pos_embed * token_mask.unsqueeze(-1).to(dtype=x.dtype)
        return x + pos_embed


class CrossAttentionConditioning(nn.Module):
    """Trajectory features attend to conditioning tokens (obs + diffusion timestep)."""

    def __init__(
        self,
        embed_dim: int,
        cond_token_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_temporal_pos_emb: bool = True,
        max_temporal_position: int = 1000,
        use_modality_emb: bool = True,
        max_modalities: int = 8,
        use_range_emb: bool = True,
        max_ranges: int = 3,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        self.cond_proj = nn.Linear(cond_token_dim, embed_dim)
        self.use_temporal_pos_emb = use_temporal_pos_emb
        self.use_modality_emb = use_modality_emb
        self.use_range_emb = use_range_emb

        if use_temporal_pos_emb:
            self.temporal_pos_emb = TemporalPositionalEncoding(embed_dim, max_temporal_position)
        if use_modality_emb:
            self.modality_emb = nn.Embedding(max_modalities, embed_dim)
        if use_range_emb:
            self.range_emb = nn.Embedding(max_ranges, embed_dim)  # 0: NULL, 1: LONG, 2: SHORT

        self.query_norm = nn.LayerNorm(embed_dim)
        self.cond_norm = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        cond_tokens: torch.Tensor,
        temporal_positions: torch.Tensor | None = None,
        modality_indices: torch.Tensor | None = None,
        range_indices: torch.Tensor | None = None,
        obs_token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cond = self.cond_proj(cond_tokens)
        if self.use_modality_emb:
            cond = cond + self.modality_emb(modality_indices).to(dtype=cond.dtype)
        if self.use_range_emb:
            cond = cond + self.range_emb(range_indices).to(dtype=cond.dtype)
        if self.use_temporal_pos_emb:
            cond = self.temporal_pos_emb(cond, temporal_positions, token_mask=obs_token_mask)

        cond_normed = self.cond_norm(cond)
        attn_out, _ = self.cross_attn(query=self.query_norm(x), key=cond_normed, value=cond_normed, need_weights=False)
        x = x + attn_out
        return x + self.ffn(self.ffn_norm(x))


class AttentionConditionalResidualBlock1D(nn.Module):
    """Residual Conv1D block with cross-attention conditioning sandwiched between the convs."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_token_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
        num_attention_heads: int = 8,
        attention_dropout: float = 0.1,
        **attn_kwargs: Any,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
                Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
            ]
        )
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.cross_attention = CrossAttentionConditioning(
            embed_dim=out_channels,
            cond_token_dim=cond_token_dim,
            num_heads=num_attention_heads,
            dropout=attention_dropout,
            **attn_kwargs,
        )

    def forward(self, x: torch.Tensor, cond_tokens: torch.Tensor, **cond_kwargs: Any) -> torch.Tensor:
        out = self.blocks[0](x)
        out = self.cross_attention(out.transpose(1, 2).contiguous(), cond_tokens, **cond_kwargs)
        out = out.transpose(1, 2).contiguous()
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class StaticAttentionConditionalUnet1D(nn.Module):
    """1D UNet with cross-attention conditioning for fixed, static observation horizons."""

    def __init__(
        self,
        input_dim: int,
        global_cond_dim: int,
        local_cond_dim: int | None = None,
        diffusion_step_embed_dim: int = 256,
        down_dims: Sequence[int] = (256, 512, 1024),
        kernel_size: int = 3,
        n_groups: int = 8,
        num_attention_heads: int = 8,
        attention_dropout: float = 0.1,
        use_temporal_pos_emb: bool = True,
        max_temporal_position: int = 1000,
        use_modality_emb: bool = True,
        max_modalities: int = 8,
        use_range_emb: bool = True,
        max_ranges: int = 3,
    ):
        super().__init__()
        self.global_cond_dim = global_cond_dim
        self.use_temporal_pos_emb = use_temporal_pos_emb
        self.use_modality_emb = use_modality_emb
        self.use_range_emb = use_range_emb

        # All conditioning tokens (obs + timestep) are projected to this dimension ONCE
        # here, then shared across every attention block below -- rather than each of the
        # ~12 blocks having its own cond_proj from the raw encoder dim. This keeps backward
        # gradients into the obs encoder bounded (unbounded per-block projections were
        # observed to amplify gradients toward NaN).
        unified_token_dim = down_dims[0]
        self.unified_token_dim = unified_token_dim

        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim),
        )
        self.timestep_to_token = nn.Linear(diffusion_step_embed_dim, unified_token_dim)
        self.global_to_token = nn.Linear(global_cond_dim, unified_token_dim)

        attn_kwargs: dict[str, Any] = {
            "num_attention_heads": num_attention_heads,
            "attention_dropout": attention_dropout,
            "use_temporal_pos_emb": use_temporal_pos_emb,
            "max_temporal_position": max_temporal_position,
            "use_modality_emb": use_modality_emb,
            "max_modalities": max_modalities,
            "use_range_emb": use_range_emb,
            "max_ranges": max_ranges,
        }

        def block(dim_in: int, dim_out: int) -> AttentionConditionalResidualBlock1D:
            return AttentionConditionalResidualBlock1D(
                dim_in, dim_out, cond_token_dim=unified_token_dim, kernel_size=kernel_size, n_groups=n_groups, **attn_kwargs
            )

        all_dims = [input_dim, *list(down_dims)]
        start_dim = down_dims[0]
        in_out = list(itertools.pairwise(all_dims))

        self.local_cond_encoder = None
        if local_cond_dim is not None:
            _, dim_out = in_out[0]
            self.local_cond_encoder = nn.ModuleList([block(local_cond_dim, dim_out), block(local_cond_dim, dim_out)])

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([block(mid_dim, mid_dim), block(mid_dim, mid_dim)])

        self.down_modules = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(
                nn.ModuleList([block(dim_in, dim_out), block(dim_out, dim_out), Downsample1d(dim_out) if not is_last else nn.Identity()])
            )

        self.up_modules = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(
                nn.ModuleList([block(dim_out * 2, dim_in), block(dim_in, dim_in), Upsample1d(dim_in) if not is_last else nn.Identity()])
            )

        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        logger.info("StaticAttentionConditionalUnet1D parameters: %e", sum(p.numel() for p in self.parameters()))

    def _expand_timesteps(self, timesteps: torch.Tensor | float | int, batch_size: int, device: torch.device) -> torch.Tensor:
        """Normalize a scalar/0-d/1-d diffusion timestep into a (batch_size,) long tensor."""
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=device)
        elif timesteps.ndim == 0:
            timesteps = timesteps[None].to(device)
        else:
            timesteps = timesteps.to(device)
        return timesteps.expand(batch_size)

    def _prepare_cond_tokens(
        self,
        timestep: torch.Tensor | float | int,
        global_cond: torch.Tensor | None,
        temporal_positions: torch.Tensor | None,
        modality_indices: torch.Tensor | None,
        range_indices: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build the [timestep_token, obs_tokens...] sequence and its position/modality/range metadata.

        Projects obs tokens from the encoder's raw dim into `unified_token_dim` exactly
        once here (see the __init__ comment for why), so every attention block downstream
        works in one shared token space.
        """
        if global_cond is None:
            model_dtype = self.timestep_to_token.weight.dtype
            obs_tokens = torch.zeros(batch_size, 0, self.unified_token_dim, device=device, dtype=model_dtype)
            obs_temporal_positions = torch.zeros(batch_size, 0, dtype=torch.long, device=device)
            obs_modality_indices = torch.zeros(batch_size, 0, dtype=torch.long, device=device)
            obs_range_indices = torch.zeros(batch_size, 0, dtype=torch.long, device=device)
        else:
            if global_cond.ndim == 2:
                global_cond = global_cond.unsqueeze(1)
            obs_tokens = self.global_to_token(global_cond)  # (B, N, unified_token_dim)
            n_obs_tokens = obs_tokens.shape[1]
            obs_temporal_positions = (
                cast(torch.Tensor, temporal_positions)
                if self.use_temporal_pos_emb
                else torch.zeros(batch_size, n_obs_tokens, dtype=torch.long, device=device)
            )
            obs_modality_indices = (
                cast(torch.Tensor, modality_indices)
                if self.use_modality_emb
                else torch.zeros(batch_size, n_obs_tokens, dtype=torch.long, device=device)
            )
            obs_range_indices = (
                cast(torch.Tensor, range_indices)
                if self.use_range_emb
                else torch.zeros(batch_size, n_obs_tokens, dtype=torch.long, device=device)
            )

        timesteps = self._expand_timesteps(timestep, batch_size, device)
        timestep_token = self.timestep_to_token(self.diffusion_step_encoder(timesteps)).unsqueeze(1)
        timestep_zeros = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        timestep_obs_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)

        cond_tokens = torch.cat([timestep_token, obs_tokens], dim=1)
        temporal_positions = torch.cat([timestep_zeros, obs_temporal_positions], dim=1)
        modality_indices = torch.cat([timestep_zeros, obs_modality_indices], dim=1)
        range_indices = torch.cat([timestep_zeros, obs_range_indices], dim=1)
        obs_token_mask = torch.cat([timestep_obs_mask, torch.ones(batch_size, obs_tokens.shape[1], dtype=torch.bool, device=device)], dim=1)
        return cond_tokens, temporal_positions, modality_indices, range_indices, obs_token_mask

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | float | int,
        local_cond: torch.Tensor | None = None,
        global_cond: torch.Tensor | None = None,
        temporal_positions: torch.Tensor | None = None,
        modality_indices: torch.Tensor | None = None,
        range_indices: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        sample: (B, T, input_dim); global_cond: (B, N, global_cond_dim) observation tokens;
        temporal_positions/modality_indices/range_indices: (B, N) metadata per obs token.
        returns: (B, T, input_dim)
        """
        # Cross-attention adds two extra residual paths per block (attention + FFN) on top
        # of the ResNet skip; with 12 blocks stacked those accumulate enough to overflow
        # bf16 in deep conv layers. FiLM (models/unet1d.py) has no such extra path and
        # stays bf16-stable, so only this UNet is forced to run in fp32.
        orig_dtype = sample.dtype
        with torch.autocast(device_type=sample.device.type, enabled=False):
            sample = sample.float()
            global_cond = global_cond.float() if global_cond is not None else None
            local_cond = local_cond.float() if local_cond is not None else None

            sample = sample.permute(0, 2, 1).contiguous()
            batch_size = sample.shape[0]
            original_horizon = sample.shape[-1]

            cond_tokens, cond_positions, cond_modalities, cond_ranges, cond_mask = self._prepare_cond_tokens(
                timestep, global_cond, temporal_positions, modality_indices, range_indices, batch_size, sample.device
            )
            cond_tokens = cond_tokens.to(dtype=sample.dtype, device=sample.device)
            cond_kwargs = {
                "temporal_positions": cond_positions,
                "modality_indices": cond_modalities,
                "range_indices": cond_ranges,
                "obs_token_mask": cond_mask,
            }

            h_local = []
            if local_cond is not None and self.local_cond_encoder is not None:
                local_cond = local_cond.permute(0, 2, 1).contiguous()
                resnet, resnet2 = self.local_cond_encoder
                h_local.append(resnet(local_cond, cond_tokens, **cond_kwargs))
                h_local.append(resnet2(local_cond, cond_tokens, **cond_kwargs))

            x = sample
            h = []
            for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):  # pyright: ignore[reportGeneralTypeIssues]
                x = resnet(x, cond_tokens, **cond_kwargs)
                if idx == 0 and h_local:
                    x = x + h_local[0]
                x = resnet2(x, cond_tokens, **cond_kwargs)
                h.append(x)
                x = downsample(x)

            for mid_module in self.mid_modules:
                x = mid_module(x, cond_tokens, **cond_kwargs)

            for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):  # pyright: ignore[reportGeneralTypeIssues]
                h_pop = h.pop()
                if x.shape[-1] != h_pop.shape[-1]:
                    x = x[..., : h_pop.shape[-1]]
                x = torch.cat((x, h_pop), dim=1)
                x = resnet(x, cond_tokens, **cond_kwargs)
                if idx == (len(self.up_modules) - 1) and h_local:
                    x = x + h_local[1]
                x = resnet2(x, cond_tokens, **cond_kwargs)
                x = upsample(x)

            if x.shape[-1] != original_horizon:
                x = x[..., :original_horizon]

            x = self.final_conv(x)
            x = x.permute(0, 2, 1).contiguous()
        return x.to(orig_dtype)
