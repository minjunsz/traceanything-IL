"""Replay buffer, window sampler and trace-cache layout."""

from pathlib import Path

import numpy as np

from markovian_policy.data import trace_cache
from markovian_policy.data.replay_buffer import ReplayBuffer
from markovian_policy.data.sampler import SequenceSampler, _create_window_indices


def _reference_indices(episode_ends, sequence_length, mask, pad_before, pad_after):
    """Straightforward loop version of the original numba implementation."""
    out = []
    for i, end in enumerate(episode_ends):
        if not mask[i]:
            continue
        start = episode_ends[i - 1] if i > 0 else 0
        length = end - start
        for idx in range(-pad_before, length - sequence_length + pad_after + 1):
            b0, b1 = max(idx, 0) + start, min(idx + sequence_length, length) + start
            out.append([b0, b1, b0 - (idx + start), sequence_length - ((idx + sequence_length + start) - b1)])
    return np.array(out)


def test_window_indices_match_reference() -> None:
    ends = np.array([30, 55, 90])
    mask = np.array([True, False, True])
    got = _create_window_indices(ends, 10, mask, pad_before=7, pad_after=9)
    np.testing.assert_array_equal(got, _reference_indices(ends, 10, mask, 7, 9))


def test_sampler_pads_first_obs_frame_and_last_frame() -> None:
    ends = np.array([12])
    frames = np.arange(12, dtype=np.float32)[:, None]
    buffer = ReplayBuffer({"frame_id": frames, "action": frames.copy()}, ends)
    sampler = SequenceSampler(buffer, 6, ("frame_id",), pad_before=2, pad_after=5, key_first_k={"frame_id": 3})
    first = sampler.sample_data(0)["obs"]["frame_id"][:3, 0]
    np.testing.assert_array_equal(first, [0, 0, 0])  # pre-padded with the first frame
    last = sampler.sample_data(len(sampler) - 1)["action"][:, 0]
    np.testing.assert_array_equal(last, [11] * 6)  # post-padded with the last frame


def test_trace_cache_rows_cover_sampler_windows() -> None:
    ends, n_obs, horizon = np.array([20, 45]), 4, 8
    frames = np.arange(45, dtype=np.float32)[:, None]
    buffer = ReplayBuffer({"frame_id": frames, "action": frames.copy()}, ends)
    sampler = SequenceSampler(buffer, horizon, ("frame_id",), pad_before=n_obs - 1, pad_after=horizon - 1, key_first_k={"frame_id": n_obs})
    rows = trace_cache.sampler_window_rows(sampler.indices, ends, n_obs)
    _, offsets = trace_cache.window_layout(ends, n_obs)
    assert rows.max() < offsets[-1] and len(np.unique(rows)) == offsets[-1]  # every cached window is used once
    for i in (0, 7, len(sampler) - 1):
        e = np.searchsorted(offsets, rows[i], side="right") - 1
        want = trace_cache.episode_window_frames(ends, n_obs, e)[rows[i] - offsets[e]]
        np.testing.assert_array_equal(sampler.sample_data(i)["obs"]["frame_id"][:n_obs, 0], want)


def test_exported_frames_are_memory_mapped_and_equal_to_the_zarr(demo_zarr: Path, tmp_path: Path) -> None:
    from markovian_policy.data.replay_buffer import export_npy

    paths = export_npy(str(demo_zarr), tmp_path / "raw", ["scene", "wrist"])
    assert [p.name for p in paths] == ["scene.npy", "wrist.npy"] and not list((tmp_path / "raw").glob("*.tmp"))
    plain = ReplayBuffer.from_zarr(str(demo_zarr), keys=["scene", "wrist", "action"])
    mapped = ReplayBuffer.from_zarr(str(demo_zarr), keys=["scene", "wrist", "action"], mmap_dir=tmp_path / "raw")
    assert isinstance(mapped["scene"], np.memmap) and not isinstance(mapped["action"], np.memmap)  # only exported keys
    for key in ("scene", "wrist", "action"):
        np.testing.assert_array_equal(mapped[key], plain[key])
