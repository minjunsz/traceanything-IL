"""Windowing: slice a replay buffer's flat (T_total, ...) arrays into fixed-length, padded training samples.

One sample per valid window start. A window that extends before an episode's start pre-pads observation keys by
repeating the first real frame (other keys are left as zeros: there is no meaningful "prior" action); a window
extending past an episode's end post-pads every key by repeating the last real frame.
"""

from collections.abc import Collection, Mapping
from typing import Any

import numpy as np

from markovian_policy.arrays import Array
from markovian_policy.data.replay_buffer import ReplayBuffer

type Sample = dict[str, Any]  # {"obs": {key: (sequence_length, ...)}, <other key>: (sequence_length, ...)}


def _create_window_indices(
    episode_ends: Array, sequence_length: int, episode_mask: Array, pad_before: int = 0, pad_after: int = 0
) -> Array:
    """For every kept episode, every window of `sequence_length` that overlaps it (including partially, in the
    episode's start/end padding regions).

    Returns (N, 4): (buffer_start, buffer_end, sample_start, sample_end). The first two index the buffer's flat
    arrays; the last two say where in the (sequence_length,)-shaped window the real (unpadded) data lands.
    """
    pad_before = min(max(pad_before, 0), sequence_length - 1)
    pad_after = min(max(pad_after, 0), sequence_length - 1)
    starts = np.concatenate([[0], episode_ends[:-1]])
    windows: list[Array] = []
    for start, end, keep in zip(starts, episode_ends, episode_mask, strict=False):
        if not keep:
            continue
        length = end - start
        idx = np.arange(-pad_before, length - sequence_length + pad_after + 1)
        buffer_start, buffer_end = np.maximum(idx, 0), np.minimum(idx + sequence_length, length)
        sample_start = buffer_start - idx
        sample_end = sample_start + (buffer_end - buffer_start)
        windows.append(np.stack([buffer_start + start, buffer_end + start, sample_start, sample_end], axis=1))
    return np.concatenate(windows) if windows else np.zeros((0, 4), dtype=np.int64)


def get_val_mask(n_episodes: int, val_ratio: float, seed: int = 0) -> Array:
    """Randomly select a validation subset of episodes (>=1 val and >=1 train, if val_ratio > 0)."""
    val_mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio <= 0:
        return val_mask
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes - 1)
    rng = np.random.default_rng(seed=seed)
    val_mask[rng.choice(n_episodes, size=n_val, replace=False)] = True
    return val_mask


class SequenceSampler:
    """Produces one padded (sequence_length, ...)-shaped window per call to `sample_data(idx)`."""

    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        sequence_length: int,
        obs_keys: Collection[str],
        rgb_keys: Collection[str] = (),
        *,
        pad_before: int = 0,
        pad_after: int = 0,
        key_first_k: Mapping[str, int] | None = None,
        episode_mask: Array | None = None,
        key_map: Mapping[str, str] | None = None,
    ) -> None:
        """
        obs_keys: names (after `key_map`) of the keys that are observations; all other keys are returned top-level.
        rgb_keys: observation keys that are uint8 images (the rest become float32).
        key_first_k: {buffer_key: k}: only read the first k frames of a window for this key (observations typically
            only need n_obs_steps, not the full horizon).
        key_map: {buffer_key: obs_key} for keys whose buffer name differs from the name the model expects.
        """
        assert sequence_length >= 1
        self.replay_buffer, self.sequence_length = replay_buffer, sequence_length
        self.keys = list(replay_buffer.keys())
        self.obs_keys, self.rgb_keys = frozenset(obs_keys), frozenset(rgb_keys)
        self.key_first_k = dict(key_first_k or {})
        self.key_map = dict(key_map or {})
        episode_ends = replay_buffer.episode_ends
        mask = episode_mask if episode_mask is not None else np.ones(episode_ends.shape, dtype=bool)
        self.indices = _create_window_indices(episode_ends, sequence_length, mask, pad_before, pad_after)

    def __len__(self) -> int:
        return len(self.indices)

    def sample_data(self, idx: int) -> Sample:
        buffer_start, buffer_end, sample_start, sample_end = (int(i) for i in self.indices[idx])
        sample: Sample = {"obs": {}}
        for key in self.keys:
            array = self.replay_buffer[key]
            window = self._read_window(array, key, buffer_start, buffer_end)
            data = self._pad(window, array, key, sample_start, sample_end)
            name = self.key_map.get(key, key)
            if name in self.obs_keys:
                sample["obs"][name] = data.astype(np.uint8 if name in self.rgb_keys else np.float32)
            else:
                sample[name] = data.astype(np.float32)
        return sample

    def _read_window(self, array: Array, key: str, buffer_start: int, buffer_end: int) -> Array:
        if key not in self.key_first_k:
            return array[buffer_start:buffer_end]
        # Only load the first k frames actually needed; fill the rest with NaN so any
        # accidental use of the un-loaded region is caught rather than silently wrong.
        n_frames = buffer_end - buffer_start
        k = min(self.key_first_k[key], n_frames)
        window = np.full((n_frames, *array.shape[1:]), np.nan, dtype=array.dtype)
        window[:k] = array[buffer_start : buffer_start + k]
        return window

    def _pad(self, window: Array, array: Array, key: str, sample_start: int, sample_end: int) -> Array:
        if sample_start == 0 and sample_end == self.sequence_length:
            return window  # fully inside the episode, no padding needed
        data = np.zeros((self.sequence_length, *array.shape[1:]), dtype=array.dtype)
        if sample_start > 0 and self.key_map.get(key, key) in self.obs_keys:
            data[:sample_start] = window[0]
        if sample_end < self.sequence_length:
            data[sample_end:] = window[-1]
        data[sample_start:sample_end] = window
        return data
