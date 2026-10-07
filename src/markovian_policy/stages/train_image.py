"""Stage: train the image-based baseline (R3M ResNet18 on raw frames, FiLM UNet) on a recorded dataset."""

from dataclasses import dataclass, field

from markovian_policy.data.dataset import DataConfig
from markovian_policy.policies.image_film import ImagePolicyConfig
from markovian_policy.stages.train import fit
from markovian_policy.training import TrainConfig


@dataclass
class Config:
    # n_obs_steps is the history length: keep data.n_obs_steps / policy.n_obs_steps (and horizon) in sync.
    data: DataConfig = field(default_factory=lambda: DataConfig(n_obs_steps=2))
    policy: ImagePolicyConfig = field(default_factory=ImagePolicyConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def run(config: Config) -> None:
    assert not config.data.trace_cache_dirs, "the image policy reads raw frames; do not pass trace caches"
    fit(config.data, config.policy, config.train)
