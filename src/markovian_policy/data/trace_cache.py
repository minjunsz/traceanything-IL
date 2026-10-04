"""Layout of precomputed TraceAnything window tokens.

TraceAnythingWindowEncoder is frozen and deterministic, so its output for an
observation window can be computed once and reused every epoch (written by
`TokenCacheBuilder` (stage `cache_tokens`), read by
FrankaKitchenDataset(trace_cache_dir=...)).

A window is the n_obs_steps obs frames a sample conditions on. With
SequenceSampler's padding (pad_before = n_obs_steps - 1, first frame
repeated before the episode start; last frame repeated past the end), a
window is fully determined by its episode and its start i relative to the
episode start, i in [-(n_obs_steps - 1), L - 1]. Episode e owns cache rows
[offsets[e], offsets[e + 1]), one per start i, in increasing i.
"""

import json
import os
from typing import Any, Literal, cast

import numpy as np

# obs key carrying a window's precomputed tokens in dataset samples
TRACE_TOKENS_KEY = "trace_tokens"
TOKENS_FILE = "tokens.npy"
META_FILE = "meta.json"
# written by the cache builder's verify step once every window is encoded
COMPLETE_FILE = "COMPLETE"


def tokens_obs_key(camera: str, n_cameras: int) -> str:
    """obs key of a camera's cached tokens in dataset samples. A single cached
    camera keeps the original plain key, so single-camera runs are unchanged."""
    return TRACE_TOKENS_KEY if n_cameras == 1 else f"{TRACE_TOKENS_KEY}_{camera}"


def window_layout(episode_ends: np.ndarray, n_obs_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns (episode lengths, row offsets of shape (E + 1,))."""
    lengths = np.diff(np.concatenate([[0], episode_ends])).astype(np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths + n_obs_steps - 1)]).astype(np.int64)
    return lengths, offsets


def episode_window_frames(episode_ends: np.ndarray, n_obs_steps: int, episode: int) -> np.ndarray:
    """(L + n_obs_steps - 1, n_obs_steps) global frame indices of every window of `episode`."""
    start = 0 if episode == 0 else int(episode_ends[episode - 1])
    length = int(episode_ends[episode]) - start
    rel = np.arange(-(n_obs_steps - 1), length)[:, None] + np.arange(n_obs_steps)[None, :]
    return start + np.clip(rel, 0, length - 1)


def sampler_window_rows(sampler_indices: np.ndarray, episode_ends: np.ndarray, n_obs_steps: int) -> np.ndarray:
    """Cache row for every SequenceSampler index row
    (buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx)."""
    if len(sampler_indices) == 0:
        return np.zeros(0, dtype=np.int64)
    buffer_start = sampler_indices[:, 0].astype(np.int64)
    sample_start = sampler_indices[:, 2].astype(np.int64)
    episode_starts = np.concatenate([[0], episode_ends[:-1]]).astype(np.int64)
    episode = np.searchsorted(episode_ends, buffer_start, side="right")
    # create_indices: buffer_start = episode_start + max(i, 0), sample_start = -min(i, 0)
    rel_start = buffer_start - episode_starts[episode] - sample_start
    lengths, offsets = window_layout(episode_ends, n_obs_steps)
    assert np.all(rel_start >= -(n_obs_steps - 1)) and np.all(rel_start < lengths[episode]), (
        "sampler window outside the cached range -- pad_before/pad_after differ from the cache layout"
    )
    return offsets[episode] + rel_start + (n_obs_steps - 1)


def load_meta(cache_dir: str) -> dict[str, Any]:
    with open(os.path.join(cache_dir, META_FILE)) as f:
        return json.load(f)


def open_tokens(cache_dir: str, mode: Literal["r", "r+"] = "r") -> np.memmap:
    """(n_windows, P, D) float16 memmap."""
    return cast(np.memmap, np.load(os.path.join(cache_dir, TOKENS_FILE), mmap_mode=mode))
