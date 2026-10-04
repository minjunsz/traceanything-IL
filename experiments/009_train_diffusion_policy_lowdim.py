"""Train the low-dim (no vision) diffusion policy: a small end-to-end check of the training stack.

pixi run -e port python experiments/009_train_diffusion_policy_lowdim.py --train.num-epochs 2
"""

from dataclasses import dataclass, field
from pathlib import Path

import tyro
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from markovian_policy.data.dataset import DataConfig, FrankaKitchenDataset
from markovian_policy.policies.lowdim import DiffusionUnetLowdimPolicy
from markovian_policy.training.config import TrainConfig
from markovian_policy.training.loop import Trainer


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=lambda: DataConfig(lowdim_keys=("agent_pos",), cameras=()))
    train: TrainConfig = field(
        default_factory=lambda: TrainConfig(
            output_dir=Path("output/runs/lowdim"), total_train_steps=None, num_epochs=5, mixed_precision="no"
        )
    )
    n_action_steps: int = 8
    diffusion_step_embed_dim: int = 128
    down_dims: tuple[int, ...] = (256, 512, 1024)
    num_train_timesteps: int = 100
    num_ddim_inference_steps: int = 10


def main(config: ExperimentConfig) -> None:
    data = config.data
    train_set = FrankaKitchenDataset(data)
    ddpm = DDPMScheduler(num_train_timesteps=config.num_train_timesteps, beta_schedule="squaredcos_cap_v2", prediction_type="epsilon")
    policy = DiffusionUnetLowdimPolicy(
        obs_dim=9,
        action_dim=9,
        ddpm_scheduler=ddpm,
        horizon=data.horizon,
        n_action_steps=config.n_action_steps,
        n_obs_steps=data.n_obs_steps,
        num_ddim_inference_steps=config.num_ddim_inference_steps,
        diffusion_step_embed_dim=config.diffusion_step_embed_dim,
        down_dims=config.down_dims,
    )
    policy.set_normalizer(train_set.get_normalizer())
    Trainer(config.train, policy, train_set, train_set.get_validation_dataset()).fit()


if __name__ == "__main__":
    main(tyro.cli(ExperimentConfig))
