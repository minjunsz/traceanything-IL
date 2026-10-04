"""Replay the human teleoperation logs in the kitchen sim and build the human demonstration dataset.

    python experiments/028_build_human_demos.py [--zip-path ZIP] [--out OUT.zarr] [--n-workers 4]

Needs a GPU node (EGL rendering of both cameras). Resumable: rerun the same command after a timeout.
Then train with plan-free conditioning, as in the original human-data configs, e.g.
    --data.zarr-path OUT.zarr --data.lowdim-keys agent_pos --policy.lowdim-keys agent_pos --policy.lowdim-dim 9
"""

import tyro

from markovian_policy.stages import generate_human

if __name__ == "__main__":
    generate_human.run(tyro.cli(generate_human.Config))
