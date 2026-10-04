"""Turn the weights saved by 024_policy_reference_legacy.py into a src-format checkpoint that evaluation can load.

    python experiments/025_convert_fork_checkpoint.py --out output/policy_reference/fork_epoch001.ckpt

The fork policy conditions on the scene camera only, so the converted policy config has cameras=("scene",).
"""

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro

from markovian_policy.policies.trace_attention import TracePolicyConfig


@dataclass
class Config:
    reference: Path = Path("output/policy_reference/fork_policy.pt")
    out: Path = Path("output/policy_reference/fork_epoch001.ckpt")


def main(config: Config) -> None:
    reference = torch.load(config.reference, map_location="cpu", weights_only=False)
    policy_config = TracePolicyConfig(cameras=("scene",))
    torch.save(
        {"ema_model": reference["state_dict"], "policy_config": dataclasses.asdict(policy_config), "source": reference["ckpt"]}, config.out
    )
    print("saved", config.out, "from", reference["ckpt"], f"({reference.get('weights', 'model')} weights)")


if __name__ == "__main__":
    main(tyro.cli(Config))
