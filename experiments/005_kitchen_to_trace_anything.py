"""Extract a few Franka Kitchen episodes as PNG frame sequences for TraceAnything.

Reads output/kitchen_demos_markovian_scripted_expert.zarr (produced by
third_party/relay-policy-learning/experts/record_demos.py) and writes each
selected episode's `scene` camera frames to
output/trace_anything_kitchen_input/<episode_name>/{i:03d}.png, matching the
naming convention TraceAnything's scripts/infer.py expects (see its
examples/input/*/NNN.png scenes).

Run in the `franka-kitchen` env (needs zarr + opencv, not present in
`trace-anything`):
    pixi run -e franka-kitchen python experiments/005_kitchen_to_trace_anything.py
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import zarr

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent

DEFAULT_ZARR = _REPO / "output" / "kitchen_demos_markovian_scripted_expert.zarr"
DEFAULT_OUT = _REPO / "output" / "trace_anything_kitchen_input"


def extract_episode(root: zarr.Group, episode_ends: np.ndarray, ep_idx: int, camera: str, out_dir: Path) -> int:
    start = 0 if ep_idx == 0 else int(episode_ends[ep_idx - 1])
    end = int(episode_ends[ep_idx])
    frames = root[f"data/{camera}"][start:end]  # [T, H, W, 3] uint8, RGB

    scene_name = f"kitchen_ep{ep_idx:03d}_{camera}"
    scene_dir = out_dir / scene_name
    scene_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        # frame is RGB (per record_demos.py's docstring); infer.py's
        # _load_images does cv2.imread (BGR) -> cvtColor(BGR2RGB), so write
        # BGR here to round-trip correctly.
        cv2.imwrite(str(scene_dir / f"{i:03d}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    return len(frames)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zarr", type=str, default=str(DEFAULT_ZARR))
    p.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    p.add_argument("--episodes", type=int, nargs="+", default=[0], help="episode indices to extract (0-based)")
    p.add_argument("--camera", type=str, default="scene", choices=["scene", "wrist"])
    args = p.parse_args()

    root = zarr.open_group(args.zarr, mode="r")
    episode_ends = root["meta/episode_ends"][:]
    print(f"dataset: {len(episode_ends)} episodes, {episode_ends[-1]} steps total")

    out_dir = Path(args.out)
    for ep in args.episodes:
        n = extract_episode(root, episode_ends, ep, args.camera, out_dir)
        print(f"episode {ep}: {n} frames -> {out_dir / f'kitchen_ep{ep:03d}_{args.camera}'}")


if __name__ == "__main__":
    main()
