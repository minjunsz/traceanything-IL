"""End-to-end rollout with rendered cameras and the real TraceAnything encoder (needs a GPU node with EGL)."""

import numpy as np
import pytest
import torch

from markovian_policy.evaluation import DistinctSubtasks, RolloutConfig, run_rollout
from markovian_policy.nn.normalizer import LinearNormalizer
from markovian_policy.policies.trace_attention import TraceAttentionPolicy, TracePolicyConfig
from markovian_policy.sim.env import KitchenEnvConfig
from markovian_policy.sim.vector import make_vector_env

pytestmark = pytest.mark.gpu


def test_rollout_with_online_trace_encoding_on_both_cameras() -> None:
    config = TracePolicyConfig(
        horizon=6, n_obs_steps=2, n_action_steps=4, obs_feature_dim=32, diffusion_step_embed_dim=16, down_dims=(32, 64),
        num_attention_heads=4, num_train_timesteps=10, num_ddpm_inference_steps=4, num_ddim_inference_steps=2,
    )  # fmt: skip
    rng = np.random.default_rng(0)
    policy = TraceAttentionPolicy(config)
    normalizer = LinearNormalizer()
    normalizer.fit(
        {"action": rng.normal(size=(50, 9)), "agent_pos": rng.normal(size=(50, 9)), "subtask_sequence": rng.normal(size=(50, 28))},
        last_n_dims=1,
        mode="limits",
    )
    policy.set_normalizer(normalizer)
    device = torch.device("cuda")
    policy.to(device).eval()

    envs = make_vector_env(KitchenEnvConfig(), 2, asynchronous=False)
    try:
        plans = [["microwave", "kettle", "slide", "hinge"]] * 2
        results = run_rollout(
            policy,
            envs,
            plans,
            RolloutConfig(task_timeout=5),
            DistinctSubtasks(4),
            device,
            seeds=[1, 2],
            record_video=True,
            record_episodes=True,
        )
    finally:
        envs.close()
    assert [r.result for r in results] == ["timeout", "timeout"]
    assert results[0].frames[0].shape == (240, 640, 3)  # scene | wrist side by side
    episode = results[0].episode
    assert episode is not None and episode["scene"].shape == (5, 240, 320, 3) and episode["wrist"].dtype == np.uint8
