"""Stage: train the TraceAnything-conditioned diffusion policy."""

from dataclasses import dataclass, field

from markovian_policy.data.dataset import DataConfig, FrankaKitchenDataset
from markovian_policy.policies.trace_attention import TraceAttentionPolicy, TracePolicyConfig
from markovian_policy.training import TrainConfig, Trainer


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    policy: TracePolicyConfig = field(default_factory=TracePolicyConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def run(config: Config) -> None:
    data, policy_config = config.data, config.policy
    # The policy's windows, cameras and low-dim inputs must match what the dataset serves.
    assert (data.horizon, data.n_obs_steps) == (policy_config.horizon, policy_config.n_obs_steps)
    assert data.cameras == policy_config.cameras and data.lowdim_keys == policy_config.lowdim_keys

    train_set = FrankaKitchenDataset(data)
    policy = TraceAttentionPolicy(policy_config)
    policy.set_normalizer(train_set.get_normalizer())
    Trainer(config.train, policy, train_set, train_set.get_validation_dataset()).fit()
