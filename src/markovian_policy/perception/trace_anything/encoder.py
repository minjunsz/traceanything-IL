"""Frozen TraceAnything encoder: a window of frames -> last-frame patch tokens."""

import torch
import torch.nn.functional as F
from torch import nn

from markovian_policy.perception.trace_anything.model import TraceAnythingBackbone, TraceAnythingConfig

ENCODE_CHUNK = 400  # frames per encoder forward, to bound memory


class TraceAnythingWindowEncoder(nn.Module):
    """(B, T, H, W, 3) frames in [0, 255] -> (B, P, D) tokens of the last frame after the last decoder layer.

    All T frames go through the model (frame t gets time t / (T - 1)); only the last frame's tokens are returned.
    """

    def __init__(self, config: TraceAnythingConfig = TraceAnythingConfig()) -> None:
        super().__init__()
        self.config = config
        self.model = TraceAnythingBackbone.from_pretrained(config)

    @property
    def token_dim(self) -> int:
        return self.config.embed_dim

    def train(self, mode: bool = True) -> "TraceAnythingWindowEncoder":
        return super().train(False)  # always frozen

    def num_tokens(self, height: int, width: int) -> int:
        h, w = self._target_size(height, width)
        return (h // self.config.patch_size) * (w // self.config.patch_size)

    def _target_size(self, height: int, width: int) -> tuple[int, int]:
        assert width >= height, "expected landscape frames"
        patch = self.config.patch_size
        h = round(height * self.config.long_side / width)
        return h - h % patch, self.config.long_side - self.config.long_side % patch

    def _preprocess(self, frames: torch.Tensor) -> torch.Tensor:
        """(N, H, W, 3) [0, 255] -> (N, 3, h, w) in [-1, 1]."""
        x = frames.permute(0, 3, 1, 2).float() / 255.0
        x = F.interpolate(x, size=self._target_size(*frames.shape[1:3]), mode="bilinear", antialias=True)
        return (x - 0.5) / 0.5

    @torch.no_grad()
    def forward(self, window: torch.Tensor) -> torch.Tensor:
        B, T = window.shape[:2]
        # The model handles its own precision; run it outside any ambient autocast.
        with torch.autocast(window.device.type, enabled=False):
            images = self._preprocess(window.transpose(0, 1).reshape(T * B, *window.shape[2:]))  # (T*B, 3, h, w)
            tokens, pos = zip(*(self.model.encoder(chunk) for chunk in images.split(ENCODE_CHUNK)), strict=False)
            feats = torch.cat(tokens).reshape(T, B, *tokens[0].shape[1:]).transpose(0, 1)  # (B, T, P, D)
            times = torch.tensor([t / max(T - 1, 1) for t in range(T)], device=window.device)  # float32 from doubles
            return self.model.decoder(feats, pos[0][:B], times)[:, -1]
