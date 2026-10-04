"""Precompute the frozen encoder's tokens for every observation window a dataset can serve (layout: `trace_cache`).

Training then reads the tokens instead of running the encoder on every sample. The builder works in four steps
(`init`, `encode`, `verify`, `bench`) and is resumable and shardable over ranks: an episode is recorded in
`progress/rank<r>.txt` only after its rows are flushed to disk.
"""

import dataclasses
import json
import os
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import zarr

from markovian_policy.arrays import Array
from markovian_policy.data import trace_cache
from markovian_policy.data.replay_buffer import ReplayBuffer
from markovian_policy.data.sampler import SequenceSampler, get_val_mask
from markovian_policy.perception import FrameWindowEncoder
from markovian_policy.perception.trace_anything import TraceAnythingConfig, TraceAnythingWindowEncoder

type EncoderFactory = Callable[[], FrameWindowEncoder]


@dataclass(frozen=True)
class TokenCacheConfig:
    zarr_path: Path
    out_dir: Path
    camera: str = "scene"
    n_obs_steps: int = 8
    horizon: int = 22  # policy horizon (sampler sequence length)
    val_ratio: float = 0.03
    val_seed: int = 42
    trace: TraceAnythingConfig = field(default_factory=TraceAnythingConfig)
    batch_size: int = 32
    rank: int = 0
    world: int = 1
    n_verify: int = 16
    bench_rows: int = 2048
    bench_threads: int = 16


def _default_encoder(config: TokenCacheConfig) -> FrameWindowEncoder:
    return TraceAnythingWindowEncoder(config.trace).cuda()


class TokenCacheBuilder:
    def __init__(self, config: TokenCacheConfig, make_encoder: EncoderFactory | None = None, device: torch.device | None = None) -> None:
        self.config = config
        self._make_encoder = make_encoder or (lambda: _default_encoder(config))
        self.device = device or torch.device("cuda")
        group = zarr.open_group(str(config.zarr_path), mode="r")
        self.episode_ends = np.asarray(group["meta/episode_ends"], dtype=np.int64)
        data = group["data"]
        assert isinstance(data, zarr.Group)
        frames = data[config.camera]
        assert isinstance(frames, zarr.Array)
        self.frames = frames
        _, self.offsets = trace_cache.window_layout(self.episode_ends, config.n_obs_steps)
        self.n_windows = int(self.offsets[-1])

    # -- steps ------------------------------------------------------------------------------------------------

    def init(self) -> None:
        """Check the window layout against the sampler, then create meta.json and an empty float16 tokens memmap."""
        self._check_layout()
        out = self.config.out_dir
        encoder = self._make_encoder()
        token_shape = [encoder.num_tokens(*self.frames.shape[1:3]), encoder.token_dim]
        (out / "progress").mkdir(parents=True, exist_ok=True)
        tokens_path = out / trace_cache.TOKENS_FILE
        if not tokens_path.exists():
            np.lib.format.open_memmap(tokens_path, "w+", np.float16, (self.n_windows, *token_shape)).flush()
        meta = {
            "zarr": str(self.config.zarr_path.resolve()), "scene_key": self.config.camera, "n_obs_steps": self.config.n_obs_steps,
            "n_windows": self.n_windows, "token_shape": token_shape, "dtype": "float16",
            "trace": dataclasses.asdict(self.config.trace), "episode_ends": self.episode_ends.tolist(),
        }  # fmt: skip
        (out / trace_cache.META_FILE).write_text(json.dumps(meta, default=str))
        print(f"initialized {tokens_path}: {self.n_windows} windows x {token_shape} float16")

    def encode(self) -> None:
        """Encode this rank's episodes (episode % world == rank) that no rank has finished yet."""
        c = self.config
        encoder = self._make_encoder()
        tokens = trace_cache.open_tokens(str(c.out_dir), mode="r+")
        done = self._finished_episodes()
        mine = [e for e in range(len(self.episode_ends)) if e % c.world == c.rank and e not in done]
        print(f"rank {c.rank}/{c.world}: {len(mine)} episodes to encode ({len(done)} done)", flush=True)
        start_time, n_encoded = time.time(), 0
        for k, episode in enumerate(mine):
            frames, start = self._episode_frames(episode)
            windows = trace_cache.episode_window_frames(self.episode_ends, c.n_obs_steps, episode) - start
            for b in range(0, len(windows), c.batch_size):
                batch = windows[b : b + c.batch_size]
                row = self.offsets[episode] + b
                tokens[row : row + len(batch)] = self._encode(encoder, frames, batch)
            tokens.flush()
            with open(c.out_dir / "progress" / f"rank{c.rank}.txt", "a") as f:
                f.write(f"{episode}\n")
                f.flush()
                os.fsync(f.fileno())
            n_encoded += len(windows)
            print(
                f"rank {c.rank}: episode {episode} done ({k + 1}/{len(mine)}), {n_encoded / (time.time() - start_time):.1f} windows/s",
                flush=True,
            )

    def verify(self) -> float:
        """Re-encode random finished windows online and compare with the cache; marks the cache COMPLETE when every
        window is encoded. Returns the worst max-relative difference."""
        c = self.config
        encoder = self._make_encoder()
        tokens = trace_cache.open_tokens(str(c.out_dir))
        rows = np.random.default_rng(0).choice(self._finished_rows(), c.n_verify, replace=False)
        worst = 0.0
        for row in rows:
            episode = int(np.searchsorted(self.offsets, row, side="right") - 1)
            frames, start = self._episode_frames(episode)
            window = trace_cache.episode_window_frames(self.episode_ends, c.n_obs_steps, episode)[row - self.offsets[episode]] - start
            online = self._encode(encoder, frames, window[None])[0].astype(np.float32)
            cached = np.asarray(tokens[row], dtype=np.float32)
            worst = max(worst, float(np.abs(online - cached).max() / (np.abs(online).max() + 1e-6)))
        assert worst < 1e-2, f"cache mismatch: worst max rel diff {worst:.2e}"
        print(f"verify ok: worst max rel diff {worst:.2e}")
        if len(self._finished_rows()) == self.n_windows:  # training refuses a cache without this marker
            (c.out_dir / trace_cache.COMPLETE_FILE).write_text(f"{self.n_windows} windows verified\n")
            print(f"cache complete: {self.n_windows} windows")
        return worst

    def bench(self) -> float:
        """Random-row read throughput (rows/s) from finished episodes: what the training dataloader will see."""
        c = self.config
        tokens = trace_cache.open_tokens(str(c.out_dir))
        rows = np.random.default_rng(0).choice(self._finished_rows(), size=c.bench_rows)
        start = time.time()
        with ThreadPoolExecutor(c.bench_threads) as pool:
            n_bytes = sum(pool.map(lambda r: np.array(tokens[r]).nbytes, rows))
        elapsed = time.time() - start
        print(f"bench: {len(rows)} rows ({n_bytes / 2**30:.2f} GiB) in {elapsed:.1f}s = {len(rows) / elapsed:.0f} rows/s")
        return len(rows) / elapsed

    # -- internals --------------------------------------------------------------------------------------------

    def _episode_frames(self, episode: int) -> tuple[Array, int]:
        start = 0 if episode == 0 else int(self.episode_ends[episode - 1])
        return np.asarray(self.frames[start : int(self.episode_ends[episode])]), start

    @torch.no_grad()
    def _encode(self, encoder: FrameWindowEncoder, frames: Array, windows: Array) -> Array:
        """frames: (L, H, W, 3) uint8 of an episode; windows: (B, T) indices into it -> (B, P, D) float16."""
        batch = torch.from_numpy(frames[windows]).to(self.device).float()
        return encoder(batch).half().cpu().numpy()

    def _finished_episodes(self) -> set[int]:
        """Episodes some rank has recorded as done (union over ranks, so a rerun may use another world size)."""
        progress = self.config.out_dir / "progress"
        return {int(x) for f in progress.glob("rank*.txt") for x in f.read_text().split()}

    def _finished_rows(self) -> Array:
        done = sorted(self._finished_episodes())
        if not done:
            raise SystemExit("no finished episodes yet")
        return np.concatenate([np.arange(self.offsets[e], self.offsets[e + 1]) for e in done])

    def _check_layout(self) -> None:
        """Every window the SequenceSampler produces (train and val masks) must equal the cached row's frames."""
        c = self.config
        n_obs, horizon = c.n_obs_steps, c.horizon
        frame_id = np.arange(self.episode_ends[-1], dtype=np.float32)[:, None]
        buffer = ReplayBuffer({"frame_id": frame_id, "action": np.zeros_like(frame_id)}, self.episode_ends)
        val_mask = get_val_mask(len(self.episode_ends), c.val_ratio, c.val_seed)
        for mask in (~val_mask, val_mask):
            sampler = SequenceSampler(
                buffer, horizon, ("frame_id",), pad_before=n_obs - 1, pad_after=horizon - 1, episode_mask=mask,
                key_first_k={"frame_id": n_obs, "action": horizon},
            )  # fmt: skip
            rows = trace_cache.sampler_window_rows(sampler.indices, self.episode_ends, n_obs)
            episodes = np.searchsorted(self.offsets, rows, side="right") - 1
            for e in np.unique(episodes):
                expected = trace_cache.episode_window_frames(self.episode_ends, n_obs, int(e))
                for i in np.nonzero(episodes == e)[0]:
                    got = sampler.sample_data(int(i))["obs"]["frame_id"][:n_obs, 0].astype(np.int64)
                    assert np.array_equal(got, expected[rows[i] - self.offsets[e]]), (i, got)
        print(f"layout check ok: {len(self.episode_ends)} episodes, sampler windows match cache rows")
