"""Stage: train the image-based attention baseline (double R3M encoder + cross-attention UNet) on a recorded dataset."""

from dataclasses import dataclass, field

from markovian_policy.data.dataset import DataConfig
from markovian_policy.policies.image_attention import ImageAttentionPolicyConfig
from markovian_policy.stages.train import fit
from markovian_policy.training import TrainConfig


@dataclass
class Config:
    # n_obs_steps is the history length: keep data.n_obs_steps / policy.n_obs_steps (and horizon) in sync.
    data: DataConfig = field(default_factory=lambda: DataConfig(n_obs_steps=8))
    policy: ImageAttentionPolicyConfig = field(default_factory=ImageAttentionPolicyConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def run(config: Config) -> None:
    assert not config.data.trace_cache_dirs, "the image policy reads raw frames; do not pass trace caches"
    fit(config.data, config.policy, config.train)
