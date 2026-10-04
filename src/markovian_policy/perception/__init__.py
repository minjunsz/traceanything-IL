"""Frozen visual encoders."""

from typing import Protocol

import torch


class FrameWindowEncoder(Protocol):
    """Encodes a window of frames into patch tokens of its last frame.

    Policies and the token-cache builder depend on this interface, not on a concrete encoder.
    """

    @property
    def token_dim(self) -> int:
        """Channels D of each token."""
        ...

    def num_tokens(self, height: int, width: int) -> int:
        """Tokens P produced for frames of the given size."""
        ...

    def __call__(self, window: torch.Tensor) -> torch.Tensor:
        """(B, T, H, W, 3) frames in [0, 255] -> (B, P, D) tokens of the last frame (no gradients)."""
        ...
