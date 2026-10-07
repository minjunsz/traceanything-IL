"""Stage: train the TraceAnything-conditioned diffusion policy."""

from dataclasses import dataclass, field

from markovian_policy.data.dataset import DataConfig, FrankaKitchenDataset
from markovian_policy.policies.factory import PolicyConfig, build_policy
from markovian_policy.policies.trace_attention import TracePolicyConfig
from markovian_policy.training import TrainConfig, Trainer


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    policy: TracePolicyConfig = field(default_factory=TracePolicyConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def fit(data: DataConfig, policy_config: PolicyConfig, train: TrainConfig) -> None:
    """Train any policy kind; shared by the TraceAnything and the image-baseline stages."""
    # The policy's windows, cameras and low-dim inputs must match what the dataset serves.
    assert (data.horizon, data.n_obs_steps) == (policy_config.horizon, policy_config.n_obs_steps)
    assert data.cameras == policy_config.cameras and data.lowdim_keys == policy_config.lowdim_keys

    train_set = FrankaKitchenDataset(data)
    policy = build_policy(policy_config)
    policy.set_normalizer(train_set.get_normalizer())
    Trainer(train, policy, train_set, train_set.get_validation_dataset()).fit()


def run(config: Config) -> None:
    fit(config.data, config.policy, config.train)
