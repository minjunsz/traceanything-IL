"""Inference-only TraceAnything backbone (CroCo encoder + time-conditioned decoder, no heads).

Parameter names match the released checkpoint (`encoder.*`, `decoder.*`).
"""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from markovian_policy.paths import TRACE_WEIGHTS
from markovian_policy.perception.trace_anything.rope import RoPE2D

DEFAULT_CKPT = TRACE_WEIGHTS
HEAD_PREFIXES = ("ds_head", "time_head", "track_head", "local_head")  # checkpoint keys we do not need
N_TIME_EMBEDDINGS = 1000


@dataclass(frozen=True)
class TraceAnythingConfig:
    ckpt_path: Path = DEFAULT_CKPT
    patch_size: int = 16
    embed_dim: int = 1024
    num_heads: int = 16
    depth: int = 24
    mlp_ratio: float = 4.0
    rope_freq: float = 100.0
    long_side: int = 512  # frames are resized so the long side has this many pixels


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, scale: float, rope: RoPE2D | None) -> None:
        super().__init__()
        self.num_heads, self.scale, self.rope = num_heads, scale, rope
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, pos: torch.Tensor | None) -> torch.Tensor:
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        if self.rope is not None and pos is not None:
            q, k = self.rope(q, pos), self.rope(k, pos)
        # Attention core runs in bf16 on GPU, matching the released model.
        with torch.autocast(x.device.type, torch.bfloat16, enabled=x.is_cuda):
            out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        return self.proj(out.to(x.dtype).transpose(1, 2).reshape(B, N, C))


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, scale: float, rope: RoPE2D | None = None) -> None:
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, scale, rope)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor, pos: torch.Tensor | None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), pos)
        return x + self.mlp(self.norm2(x))


class PatchEmbed(nn.Module):
    def __init__(self, patch_size: int, embed_dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, embed_dim, patch_size, patch_size)

    def forward(self, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """img (B, 3, H, W) -> tokens (B, h*w, D), positions (B, h*w, 2) as (y, x)."""
        x = self.proj(img)
        h, w = x.shape[-2:]
        pos = torch.cartesian_prod(torch.arange(h, device=img.device), torch.arange(w, device=img.device))
        return x.flatten(2).transpose(1, 2), pos[None].expand(len(img), -1, -1)


class Encoder(nn.Module):
    def __init__(self, cfg: TraceAnythingConfig) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(cfg.patch_size, cfg.embed_dim)
        rope = RoPE2D(cfg.rope_freq)
        scale = (cfg.embed_dim // cfg.num_heads) ** -0.5
        self.enc_blocks = nn.ModuleList(Block(cfg.embed_dim, cfg.num_heads, cfg.mlp_ratio, scale, rope) for _ in range(cfg.depth))
        self.enc_norm = nn.LayerNorm(cfg.embed_dim, eps=1e-6)

    def forward(self, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x, pos = self.patch_embed(img)
        for block in self.enc_blocks:
            x = block(x, pos)
        return self.enc_norm(x), pos


def _sincos_1d(dim: int, pos: np.ndarray) -> np.ndarray:
    omega = 1.0 / 10000 ** (np.arange(dim // 2, dtype=float) / (dim / 2.0))
    out = np.einsum("m,d->md", pos.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


class Decoder(nn.Module):
    """Joint attention over all views' tokens, conditioned on each view's normalized time in [0, 1]."""

    def __init__(self, cfg: TraceAnythingConfig) -> None:
        super().__init__()
        self.decoder_embed = nn.Linear(cfg.embed_dim, cfg.embed_dim)
        head_dim = cfg.embed_dim // cfg.num_heads
        # Attention temperature rescaled for inference sequences longer than in training (train 20 -> 137 views).
        scale = head_dim**-0.5 * math.sqrt(math.log(137) / math.log(20))
        self.dec_blocks = nn.ModuleList(Block(cfg.embed_dim, cfg.num_heads, cfg.mlp_ratio, scale) for _ in range(cfg.depth))
        self.dec_norm = nn.LayerNorm(cfg.embed_dim, eps=1e-6)
        time_emb = torch.from_numpy(_sincos_1d(cfg.embed_dim, np.arange(N_TIME_EMBEDDINGS))).float()
        self.register_buffer("time_emb", time_emb, persistent=False)
        self.time_emb: torch.Tensor

    def forward(self, feats: torch.Tensor, pos: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        """feats (B, T, P, D), pos (B, P, 2), times (T,) float32 in [0, 1] -> (B, T, P, D)."""
        B, T, P, _ = feats.shape
        x = self.decoder_embed(feats.reshape(B, T * P, -1))
        x = x + self.time_emb[(times * (N_TIME_EMBEDDINGS - 1)).long()].repeat_interleave(P, dim=0)
        pos = pos.repeat(1, T, 1)
        for block in self.dec_blocks:
            x = block(x, pos)
        return self.dec_norm(x).reshape(B, T, P, -1)


class TraceAnythingBackbone(nn.Module):
    def __init__(self, cfg: TraceAnythingConfig) -> None:
        super().__init__()
        self.encoder, self.decoder = Encoder(cfg), Decoder(cfg)

    @classmethod
    def from_pretrained(cls, cfg: TraceAnythingConfig) -> "TraceAnythingBackbone":
        model = cls(cfg)
        state = torch.load(cfg.ckpt_path, map_location="cpu")
        state = state.get("state_dict", state)
        state = {k.removeprefix("net."): v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        assert not missing, f"checkpoint lacks encoder/decoder weights: {missing[:5]}"
        assert all(k.startswith(HEAD_PREFIXES) for k in unexpected), f"unexpected keys: {unexpected[:5]}"
        return model.eval().requires_grad_(False)
