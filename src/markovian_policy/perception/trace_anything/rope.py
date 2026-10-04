"""2D rotary position embedding (pure PyTorch)."""

import torch
from torch import nn


class RoPE2D(nn.Module):
    """Rotates the first half of the head dim by the y position and the second half by the x position."""

    def __init__(self, freq: float = 100.0, max_pos: int = 128) -> None:
        super().__init__()
        self.freq, self.max_pos = freq, max_pos
        self.cos: torch.Tensor
        self.sin: torch.Tensor
        self.register_buffer("cos", torch.empty(0), persistent=False)
        self.register_buffer("sin", torch.empty(0), persistent=False)

    def _tables(self, half_dim: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cos.device != device or self.cos.numel() == 0:
            inv_freq = 1.0 / self.freq ** (torch.arange(0, half_dim, 2, device=device).float() / half_dim)
            freqs = torch.outer(torch.arange(self.max_pos, device=device).float(), inv_freq)
            freqs = torch.cat((freqs, freqs), dim=-1)
            self.cos, self.sin = freqs.cos(), freqs.sin()
        return self.cos, self.sin

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """tokens: (B, heads, N, D) float; positions: (B, N, 2) int (y, x). Returns tokens rotated."""
        cos, sin = self._tables(tokens.shape[-1] // 2, tokens.device)
        halves = []
        for half, pos in zip(tokens.chunk(2, dim=-1), positions.unbind(-1), strict=False):
            c, s = cos[pos][:, None].to(half.dtype), sin[pos][:, None].to(half.dtype)
            halves.append(half * c + self._rotate_half(half) * s)
        return torch.cat(halves, dim=-1)
