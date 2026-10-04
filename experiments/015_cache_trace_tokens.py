"""Precompute frozen TraceAnything window tokens for the kitchen demos.

DiffusionUnetTraceAnythingAttentionPolicy conditions on the frozen
TraceAnythingWindowEncoder's output for each n_obs_steps scene window, which
dominated training time (~11 s/step at batch 128 on 4x L40S, almost all of it
this forward pass). This writes that output once for every window the dataset
can sample (layout: diffusion_policy/common/trace_cache.py), so training can
read it via FrankaKitchenDataset(trace_cache_dir=...).

Modes (see experiments/015_cache_trace_tokens.sbatch):
  init    check the window layout against the real ImprovedDatasetSampler,
          then create meta.json + an empty float16 tokens.npy memmap.
  encode  encode this rank's episodes (episode % world == rank); resumable:
          finished episodes are appended to progress/rank<r>.txt only after
          their rows are flushed.
  verify  re-encode a few random windows (from finished episodes only, so it
          also works mid-run) online and compare to the cache.
  bench   CPU only: random-row read throughput from finished episodes, i.e.
          what the training dataloader will see (unwritten rows are sparse
          holes and read unrealistically fast).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import zarr
from diffusion_policy.model.vision.trace_anything_encoder import TraceAnythingWindowEncoder

from markovian_policy.utils import trace_cache
from markovian_policy.utils.replay_buffer import ReplayBuffer
from markovian_policy.utils.sampler import ImprovedDatasetSampler, get_val_mask

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TA_ROOT = PROJECT_ROOT / "third_party/TraceAnything"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["init", "encode", "verify", "bench"])
    p.add_argument("--zarr", default=str(PROJECT_ROOT / "output/kitchen_demos_markovian_scripted_expert.zarr"))
    p.add_argument("--out-dir", required=True)
    p.add_argument("--scene-key", default="scene")
    p.add_argument("--n-obs-steps", type=int, default=8)
    p.add_argument("--horizon", type=int, default=22, help="policy horizon (sampler sequence length)")
    p.add_argument("--ta-config", default=str(TA_ROOT / "configs/eval.yaml"))
    p.add_argument("--ta-ckpt", default=str(TA_ROOT / "checkpoints/trace_anything.pt"))
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world", type=int, default=1)
    p.add_argument("--n-verify", type=int, default=16)
    p.add_argument("--bench-rows", type=int, default=2048)
    p.add_argument("--bench-threads", type=int, default=16)
    return p.parse_args()


def check_layout(episode_ends: np.ndarray, n_obs_steps: int, horizon: int):
    """Every window ImprovedDatasetSampler produces (train and val masks, same
    padding as the training config) must equal the cached row's frames."""
    buf = ReplayBuffer.create_empty_numpy()
    start = 0
    for end in episode_ends:
        frames = np.arange(start, end, dtype=np.float32)[:, None]
        buf.add_episode({"frame_id": frames, "action": np.zeros((end - start, 1), np.float32)})
        start = end
    shape_meta = {"obs": {"frame_id": {"shape": [1], "type": "low_dim"}}, "action": {"shape": [1]}}
    val_mask = get_val_mask(n_episodes=len(episode_ends), val_ratio=0.03, seed=41)
    for mask in (~val_mask, val_mask):
        sampler = ImprovedDatasetSampler(
            replay_buffer=buf,
            sequence_length=horizon,
            shape_meta=shape_meta,
            pad_before=n_obs_steps - 1,
            pad_after=horizon - 1,
            episode_mask=mask,
            key_first_k={"frame_id": n_obs_steps, "action": horizon},
        )
        rows = trace_cache.sampler_window_rows(sampler.indices, episode_ends, n_obs_steps)
        _, offsets = trace_cache.window_layout(episode_ends, n_obs_steps)
        episode = np.searchsorted(offsets, rows, side="right") - 1
        for e in np.unique(episode):
            expected = trace_cache.episode_window_frames(episode_ends, n_obs_steps, e)
            for idx in np.nonzero(episode == e)[0]:
                got = sampler.sample_data(idx)["obs"]["frame_id"][:n_obs_steps, 0].astype(np.int64)
                want = expected[rows[idx] - offsets[e]]
                assert np.array_equal(got, want), (idx, got, want)
    print(f"layout check ok: {len(episode_ends)} episodes, sampler windows match cache rows")


def finished_rows(out_dir: Path, offsets: np.ndarray) -> np.ndarray:
    """Cache rows of every episode some rank has recorded as done."""
    done = set()
    for f in (out_dir / "progress").glob("rank*.txt"):
        done.update(int(x) for x in f.read_text().split())
    if not done:
        raise SystemExit("no finished episodes yet")
    return np.concatenate([np.arange(offsets[e], offsets[e + 1]) for e in sorted(done)])


def build_encoder(args, device):
    enc = TraceAnythingWindowEncoder(config_path=args.ta_config, ckpt_path=args.ta_ckpt)
    return enc.to(device).eval()


def encode_windows(enc, scene: np.ndarray, frames: np.ndarray, device) -> np.ndarray:
    """scene: (N, H, W, C) uint8 of the episode; frames: (B, T) indices into it."""
    window = torch.from_numpy(scene[frames]).to(device).float()  # (B, T, H, W, C)
    return enc(window).to(torch.float16).cpu().numpy()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    group = zarr.open(args.zarr, mode="r")
    episode_ends = group["meta/episode_ends"][:].astype(np.int64)
    scene_arr = group["data"][args.scene_key]
    _, offsets = trace_cache.window_layout(episode_ends, args.n_obs_steps)
    n_windows = int(offsets[-1])

    if args.mode == "init":
        check_layout(episode_ends, args.n_obs_steps, args.horizon)
        enc = TraceAnythingWindowEncoder(config_path=args.ta_config, ckpt_path=args.ta_ckpt)
        H, W = scene_arr.shape[1:3]
        P, D = enc.num_tokens(H, W), enc.output_token_dim()
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "progress").mkdir(exist_ok=True)
        meta = {
            "zarr": os.path.abspath(args.zarr),
            "scene_key": args.scene_key,
            "n_obs_steps": args.n_obs_steps,
            "n_windows": n_windows,
            "token_shape": [P, D],
            "dtype": "float16",
            "ta_config": args.ta_config,
            "ta_ckpt": args.ta_ckpt,
            "episode_ends": episode_ends.tolist(),
        }
        tokens_path = out_dir / trace_cache.TOKENS_FILE
        if not tokens_path.exists():
            np.lib.format.open_memmap(tokens_path, mode="w+", dtype=np.float16, shape=(n_windows, P, D)).flush()
        with open(out_dir / trace_cache.META_FILE, "w") as f:
            json.dump(meta, f)
        print(f"initialized {tokens_path}: {n_windows} windows x ({P}, {D}) float16 = {n_windows * P * D * 2 / 2**30:.1f} GiB")
        return

    if args.mode == "bench":
        from concurrent.futures import ThreadPoolExecutor

        tokens = trace_cache.open_tokens(str(out_dir))
        rows = np.random.default_rng(0).choice(finished_rows(out_dir, offsets), size=args.bench_rows)
        t0 = time.time()
        with ThreadPoolExecutor(args.bench_threads) as pool:
            n_bytes = sum(pool.map(lambda r: np.array(tokens[r]).nbytes, rows))
        dt = time.time() - t0
        print(
            f"bench: {len(rows)} random rows ({n_bytes / 2**30:.2f} GiB) in {dt:.1f}s = "
            f"{n_bytes / 2**20 / dt:.0f} MiB/s, {len(rows) / dt:.0f} rows/s "
            f"({args.bench_threads} threads)"
        )
        return

    device = torch.device("cuda")
    enc = build_encoder(args, device)
    tokens = trace_cache.open_tokens(str(out_dir), mode="r+" if args.mode == "encode" else "r")

    if args.mode == "verify":
        rng = np.random.default_rng(0)
        worst = 0.0
        for row in rng.choice(finished_rows(out_dir, offsets), size=args.n_verify, replace=False):
            e = int(np.searchsorted(offsets, row, side="right") - 1)
            start = 0 if e == 0 else int(episode_ends[e - 1])
            scene = scene_arr[start : int(episode_ends[e])]
            frames = trace_cache.episode_window_frames(episode_ends, args.n_obs_steps, e)[row - offsets[e]] - start
            online = encode_windows(enc, scene, frames[None], device)[0].astype(np.float32)
            cached = np.asarray(tokens[row], dtype=np.float32)
            rel = np.abs(online - cached).max() / (np.abs(online).max() + 1e-6)
            worst = max(worst, rel)
            print(f"row {row} (episode {e}): max rel diff {rel:.2e}")
        assert worst < 1e-2, f"cache mismatch: worst max rel diff {worst:.2e}"
        print(f"verify ok: worst max rel diff {worst:.2e}")
        n_done = len(finished_rows(out_dir, offsets))
        if n_done == n_windows:
            # Training refuses a cache without this marker (FrankaKitchenDataset),
            # so a partially encoded cache can never be trained on by accident.
            (out_dir / trace_cache.COMPLETE_FILE).write_text(f"{n_windows} windows verified\n")
            print(f"cache complete: {n_windows} windows")
        else:
            print(f"cache incomplete: {n_done}/{n_windows} windows encoded")
        return

    progress = out_dir / "progress" / f"rank{args.rank}.txt"
    # Union over every rank's record, so a later run may use a different world size.
    done = set()
    for f in (out_dir / "progress").glob("rank*.txt"):
        done.update(int(x) for x in f.read_text().split())
    mine = [e for e in range(len(episode_ends)) if e % args.world == args.rank and e not in done]
    print(f"rank {args.rank}/{args.world}: {len(mine)} episodes to encode ({len(done)} already done)", flush=True)
    t0 = time.time()
    n_done_windows = 0
    for k, e in enumerate(mine):
        start = 0 if e == 0 else int(episode_ends[e - 1])
        scene = scene_arr[start : int(episode_ends[e])]  # (L, H, W, C) uint8, one episode
        frames = trace_cache.episode_window_frames(episode_ends, args.n_obs_steps, e) - start
        for b in range(0, len(frames), args.batch_size):
            batch = frames[b : b + args.batch_size]
            row0 = int(offsets[e]) + b
            tokens[row0 : row0 + len(batch)] = encode_windows(enc, scene, batch, device)
        tokens.flush()
        with open(progress, "a") as f:
            f.write(f"{e}\n")
            f.flush()
            os.fsync(f.fileno())
        n_done_windows += len(frames)
        rate = n_done_windows / (time.time() - t0)
        print(f"rank {args.rank}: episode {e} done ({k + 1}/{len(mine)}), {rate:.1f} windows/s", flush=True)
    print(f"rank {args.rank}: finished", flush=True)


if __name__ == "__main__":
    sys.exit(main())
