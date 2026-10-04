"""Save legacy vs. src frames side by side (needs a GPU node): output/kitchen_reference/compare_<cam>_<t>.png"""

import imageio.v3 as iio
import numpy as np

from markovian_policy.sim.env import KitchenEnv, KitchenEnvConfig

ref = np.load("output/kitchen_reference/legacy.npz")
env = KitchenEnv(KitchenEnvConfig(noise_ratio=0.0))
env.reset(seed=0)
for t, action in enumerate(ref["actions"][: ref["frame_steps"].max() + 1]):
    if t in ref["frame_steps"]:
        i = list(ref["frame_steps"]).index(t)
        for cam in ("scene", "wrist"):
            mine, theirs = env.sim.render(cam), ref[cam][i]
            diff = np.abs(mine.astype(int) - theirs).clip(0, 255).astype(np.uint8)
            iio.imwrite(f"output/kitchen_reference/compare_{cam}_{t}.png", np.concatenate([theirs, mine, diff], axis=1))
            print(cam, t, "mean abs diff", np.abs(mine.astype(int) - theirs).mean())
    env.step(action)
