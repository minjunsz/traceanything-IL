"""Record scripted-expert demos to zarr: python experiments/021_record_expert.py --n-episodes 581 --out ..."""

import tyro

from markovian_policy.stages import generate_expert

if __name__ == "__main__":
    generate_expert.run(tyro.cli(generate_expert.Config))
