"""FiLM-conditioned 1D UNet noise predictor.

Ported from third_party/diffusion-policy-experiments/diffusion_policy/model/diffusion/conditional_unet1d.py
(dropping the unused/dead `VariableConditionalUnet1D` subclass). Pure PyTorch,
no legacy dependency -- logic unchanged from upstream.
"""

import itertools
import logging
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn

from markovian_policy.nn.blocks import Conv1dBlock, Downsample1d, SinusoidalPosEmb, Upsample1d

logger = logging.getLogger(__name__)


class ConditionalResidualBlock1D(nn.Module):
    """Two Conv1dBlocks with a FiLM (scale/shift) modulation from `cond` in between."""

    def __init__(self, in_channels: int, out_channels: int, cond_dim: int, kernel_size: int = 3, n_groups: int = 8):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
                Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
            ]
        )
        self.out_channels = out_channels
        # FiLM modulation https://arxiv.org/abs/1709.07871
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, out_channels * 2))
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """x: (B, in_channels, T), cond: (B, cond_dim) -> (B, out_channels, T)"""
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond).reshape(x.shape[0], 2, self.out_channels, 1)
        scale, bias = embed[:, 0, ...], embed[:, 1, ...]
        out = scale * out + bias
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class ConditionalUnet1D(nn.Module):
    """1D UNet with FiLM conditioning, for diffusing action trajectories.

    Structure: encoder (down_modules) -> bottleneck (mid_modules) -> decoder
    (up_modules) with skip connections, mirroring a standard image UNet but
    over the temporal axis. All conditioning (diffusion timestep, observation
    history, optional target) is flattened into one vector and injected via
    FiLM at every residual block -- see `ConditionalResidualBlock1D`.
    """

    def __init__(
        self,
        input_dim: int,
        target_dim: int | None = None,
        local_cond_dim: int | None = None,
        global_cond_dim: int | None = None,
        diffusion_step_embed_dim: int = 256,
        down_dims: Sequence[int] = (256, 512, 1024),
        kernel_size: int = 3,
        n_groups: int = 8,
    ):
        super().__init__()
        all_dims = [input_dim, *list(down_dims)]
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )

        cond_dim = dsed
        if global_cond_dim is not None:
            cond_dim += global_cond_dim
        if target_dim is not None:
            cond_dim += target_dim

        in_out = list(itertools.pairwise(all_dims))

        self.local_cond_encoder = None
        if local_cond_dim is not None:
            _, dim_out = in_out[0]
            self.local_cond_encoder = nn.ModuleList(
                [
                    ConditionalResidualBlock1D(local_cond_dim, dim_out, cond_dim, kernel_size, n_groups),
                    ConditionalResidualBlock1D(local_cond_dim, dim_out, cond_dim, kernel_size, n_groups),
                ]
            )

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim, kernel_size, n_groups),
                ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim, kernel_size, n_groups),
            ]
        )

        self.down_modules = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(dim_in, dim_out, cond_dim, kernel_size, n_groups),
                        ConditionalResidualBlock1D(dim_out, dim_out, cond_dim, kernel_size, n_groups),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.up_modules = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(dim_out * 2, dim_in, cond_dim, kernel_size, n_groups),
                        ConditionalResidualBlock1D(dim_in, dim_in, cond_dim, kernel_size, n_groups),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )
        self.global_cond_dim = global_cond_dim

        logger.info("ConditionalUnet1D parameters: %e", sum(p.numel() for p in self.parameters()))

    def _build_timestep_and_global_feature(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | float | int,
        global_cond: torch.Tensor | None,
        target_cond: torch.Tensor | None,
    ) -> torch.Tensor:
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif timesteps.ndim == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)
        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], dim=-1)
        if target_cond is not None:
            global_feature = torch.cat([global_feature, target_cond], dim=-1)
        return global_feature

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | float | int,
        local_cond: torch.Tensor | None = None,
        global_cond: torch.Tensor | None = None,
        target_cond: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        sample: (B, T, input_dim), global_cond: (B, global_cond_dim), target_cond: (B, target_dim)
        returns: (B, T, input_dim)
        """
        sample = sample.permute(0, 2, 1)
        # Two stride-2 downsamples followed by two stride-2 upsamples can add one extra
        # element back for odd T (e.g. 17 -> 9 -> 5 -> 10 -> 18); truncate to restore it.
        original_horizon = sample.shape[-1]

        global_feature = self._build_timestep_and_global_feature(sample, timestep, global_cond, target_cond)

        h_local = []
        if local_cond is not None:
            local_cond = local_cond.permute(0, 2, 1)
            assert self.local_cond_encoder is not None
            resnet, resnet2 = self.local_cond_encoder  # pyright: ignore[reportGeneralTypeIssues]
            h_local.append(resnet(local_cond, global_feature))
            h_local.append(resnet2(local_cond, global_feature))

        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):  # pyright: ignore[reportGeneralTypeIssues]
            x = resnet(x, global_feature)
            if idx == 0 and h_local:
                x = x + h_local[0]
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):  # pyright: ignore[reportGeneralTypeIssues]
            h_pop = h.pop()
            if x.shape[-1] != h_pop.shape[-1]:
                x = x[..., : h_pop.shape[-1]]
            x = torch.cat((x, h_pop), dim=1)
            x = resnet(x, global_feature)
            if idx == len(self.up_modules) and h_local:
                x = x + h_local[1]
            x = resnet2(x, global_feature)
            x = upsample(x)

        if x.shape[-1] != original_horizon:
            x = x[..., :original_horizon]

        x = self.final_conv(x)
        return x.permute(0, 2, 1)
