"""Conditioning of the cross-attention UNet: where observation tokens come from and how they are laid out."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import torch

from markovian_policy.data.trace_cache import tokens_obs_key
from markovian_policy.perception import FrameWindowEncoder

# Ids in the UNet's modality / range embedding tables (0 = the diffusion-timestep token).
# Camera i gets modality MODALITY_TRACE + i, so attention can tell the cameras' token blocks apart.
MODALITY_LOWDIM: Final = 1
MODALITY_TRACE: Final = 2
RANGE_LOWDIM: Final = 1
RANGE_TRACE: Final = 2


@dataclass(frozen=True)
class Conditioning:
    """Conditioning tokens with per-token metadata for the UNet; N = n_obs_steps + sum of trace tokens per camera."""

    tokens: torch.Tensor  # (B, N, D)
    temporal_positions: torch.Tensor  # (B, N) long
    modality_indices: torch.Tensor  # (B, N) long
    range_indices: torch.Tensor  # (B, N) long

    def as_kwargs(self) -> dict[str, torch.Tensor]:
        """Keyword arguments of `StaticAttentionConditionalUnet1D.forward`."""
        return {
            "global_cond": self.tokens,
            "temporal_positions": self.temporal_positions,
            "modality_indices": self.modality_indices,
            "range_indices": self.range_indices,
        }


def assemble_conditioning(lowdim_tokens: torch.Tensor, trace_blocks: Sequence[torch.Tensor]) -> Conditioning:
    """Concatenate (B, To, D) low-dim tokens and one (B, P, D) block per camera.

    Low-dim tokens sit at their own step; all trace tokens summarize the window at its most recent step.
    """
    batch, n_obs = lowdim_tokens.shape[:2]
    device = lowdim_tokens.device

    def full(n: int, value: int) -> torch.Tensor:
        return torch.full((batch, n), value, dtype=torch.long, device=device)

    steps = torch.arange(n_obs, device=device).expand(batch, -1)
    return Conditioning(
        tokens=torch.cat([lowdim_tokens, *trace_blocks], dim=1),
        temporal_positions=torch.cat([steps, *(full(b.shape[1], n_obs - 1) for b in trace_blocks)], dim=1),
        modality_indices=torch.cat(
            [full(n_obs, MODALITY_LOWDIM), *(full(b.shape[1], MODALITY_TRACE + i) for i, b in enumerate(trace_blocks))], dim=1
        ),
        range_indices=torch.cat([full(n_obs, RANGE_LOWDIM), *(full(b.shape[1], RANGE_TRACE) for b in trace_blocks)], dim=1),
    )


class TraceTokenSource:
    """Strategy for the frozen-encoder tokens of a camera: precomputed in the batch, else encoded online.

    The encoder is only built when a batch lacks cached tokens, so training on a token cache never loads it.
    """

    def __init__(self, cameras: Sequence[str], make_encoder: Callable[[torch.device], FrameWindowEncoder]) -> None:
        self.cameras = tuple(cameras)
        self._make_encoder = make_encoder
        self._encoder: FrameWindowEncoder | None = None

    @property
    def encoder_loaded(self) -> bool:
        return self._encoder is not None

    def tokens(self, obs: Mapping[str, torch.Tensor], camera: str, n_obs_steps: int) -> torch.Tensor:
        """(B, P, token_dim) float tokens for `camera` from obs {key: (B, T, ...)} or {cache key: (B, P, token_dim)}."""
        cached = obs.get(tokens_obs_key(camera, len(self.cameras)))
        if cached is not None:
            return cached.float()
        frames = obs[camera][:, :n_obs_steps].float()  # (B, To, H, W, 3) in [0, 255]
        if self._encoder is None:
            self._encoder = self._make_encoder(frames.device)
        return self._encoder(frames)
