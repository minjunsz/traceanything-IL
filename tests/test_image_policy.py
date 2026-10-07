"""ImageFilmPolicy (CPU, tiny frames): loss/backward, history length, sampling, config/checkpoint round trip."""

import dataclasses

import numpy as np
import pytest
import torch

from markovian_policy.nn.normalizer import LinearNormalizer
from markovian_policy.policies import (
    ImageAttentionPolicy,
    ImageAttentionPolicyConfig,
    ImageFilmPolicy,
    ImagePolicyConfig,
    build_policy,
    config_from_dict,
)

B = 2


def _config(n_obs_steps: int) -> ImagePolicyConfig:
    return ImagePolicyConfig(
        horizon=n_obs_steps + 6, n_obs_steps=n_obs_steps, n_action_steps=3, r3m_weights=None, projection_dim=16,
        diffusion_step_embed_dim=16, down_dims=(32, 64), num_train_timesteps=10, num_ddpm_inference_steps=4,
        num_ddim_inference_steps=2,
    )  # fmt: skip


def _policy(n_obs_steps: int) -> tuple[ImageFilmPolicy, dict]:
    config, rng = _config(n_obs_steps), np.random.default_rng(0)
    policy = ImageFilmPolicy(config)
    normalizer = LinearNormalizer()
    data = {"action": rng.normal(size=(50, 9)), "agent_pos": rng.normal(size=(50, 9)), "subtask_sequence": rng.normal(size=(50, 28))}
    normalizer.fit(data, last_n_dims=1, mode="limits")
    policy.set_normalizer(normalizer)
    obs = {
        "agent_pos": torch.randn(B, config.horizon, 9),
        "subtask_sequence": torch.randn(B, config.horizon, 28),
        "scene": torch.randint(0, 255, (B, config.horizon, 24, 32, 3), dtype=torch.uint8),
        "wrist": torch.randint(0, 255, (B, config.horizon, 24, 32, 3), dtype=torch.uint8),
    }
    return policy, {"obs": obs, "action": torch.randn(B, config.horizon, 9)}


@pytest.mark.parametrize("n_obs_steps", [2, 4])
def test_history_length_sets_conditioning_size_and_gradients_reach_encoder(n_obs_steps: int) -> None:
    policy, batch = _policy(n_obs_steps)
    config = policy.config
    cond = policy.global_cond(batch["obs"])
    assert cond.shape == (B, n_obs_steps * (2 * config.projection_dim + config.lowdim_dim))
    loss = policy(batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in policy.encoder.backbones["scene"].parameters())


def test_only_the_first_n_obs_steps_condition_the_policy() -> None:
    policy, batch = _policy(2)
    policy.eval()
    later = {k: v.clone() for k, v in batch["obs"].items()}
    for v in later.values():
        v[:, 2:] = 0
    assert torch.equal(policy.global_cond(batch["obs"]), policy.global_cond(later))


def test_predict_action_shapes_for_both_samplers() -> None:
    policy, batch = _policy(3)
    policy.eval()
    obs = {k: v[:, :3] for k, v in batch["obs"].items()}
    for use_ddim in (False, True):
        out = policy.predict_action({"obs": obs}, use_ddim=use_ddim)
        assert out["action"].shape == (B, 3, 9) and out["action_pred"].shape == (B, policy.horizon, 9)
        assert torch.isfinite(out["action_pred"]).all()


def test_config_round_trips_through_checkpoint_dict() -> None:
    config = _config(4)
    rebuilt = config_from_dict(dataclasses.asdict(config))
    assert rebuilt == config and isinstance(build_policy(rebuilt), ImageFilmPolicy)


def _attention_policy(n_obs_steps: int, short: int | None) -> tuple[ImageAttentionPolicy, dict]:
    config = ImageAttentionPolicyConfig(
        horizon=n_obs_steps + 6, n_obs_steps=n_obs_steps, n_action_steps=3, r3m_weights=None, projection_dim=16,
        short_range_obs_horizon=short, short_range_dropout=0.0, diffusion_step_embed_dim=16, down_dims=(32, 64), num_attention_heads=4,
        num_train_timesteps=10, num_ddpm_inference_steps=4, num_ddim_inference_steps=2,
    )  # fmt: skip
    policy, batch = _policy(n_obs_steps)
    policy = ImageAttentionPolicy(config)
    rng = np.random.default_rng(0)
    normalizer = LinearNormalizer()
    data = {"action": rng.normal(size=(50, 9)), "agent_pos": rng.normal(size=(50, 9)), "subtask_sequence": rng.normal(size=(50, 28))}
    normalizer.fit(data, last_n_dims=1, mode="limits")
    policy.set_normalizer(normalizer)
    batch["obs"] = {k: v[:, : config.horizon] if v.shape[1] >= config.horizon else v for k, v in batch["obs"].items()}
    return policy, batch


@pytest.mark.parametrize(("n_obs_steps", "short"), [(4, 2), (8, 2), (4, None)])
def test_attention_token_layout_long_then_short(n_obs_steps: int, short: int | None) -> None:
    policy, batch = _attention_policy(n_obs_steps, short)
    cond = policy.conditioning(batch["obs"])
    n_short = short or 0
    assert cond["global_cond"].shape == (B, n_obs_steps + n_short, policy.token_dim)
    assert cond["temporal_positions"][0].tolist() == [*range(n_obs_steps), *range(n_obs_steps - n_short, n_obs_steps)]
    assert cond["range_indices"][0].tolist() == [1] * n_obs_steps + [2] * n_short
    loss = policy(batch)
    loss.backward()
    assert torch.isfinite(loss)
    if short:
        assert policy.short_null_token is not None and policy.short_encoder is not None
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in policy.short_encoder.parameters())


def test_short_range_dropout_replaces_short_tokens_with_the_null_token_in_training_only() -> None:
    policy, batch = _attention_policy(4, 2)
    assert policy.short_null_token is not None
    with torch.no_grad():
        policy.short_null_token.fill_(7.0)
    object.__setattr__(policy.config, "short_range_dropout", 1.0)
    policy.train()
    assert (policy.conditioning(batch["obs"])["global_cond"][:, 4:] == 7.0).all()
    policy.eval()
    assert not (policy.conditioning(batch["obs"])["global_cond"][:, 4:] == 7.0).any()


def test_attention_predict_action_and_config_round_trip() -> None:
    policy, batch = _attention_policy(4, 2)
    policy.eval()
    obs = {k: v[:, :4] for k, v in batch["obs"].items()}
    out = policy.predict_action({"obs": obs}, use_ddim=True)
    assert out["action"].shape == (B, 3, 9) and torch.isfinite(out["action_pred"]).all()
    rebuilt = config_from_dict(dataclasses.asdict(policy.config))
    assert rebuilt == policy.config and isinstance(build_policy(rebuilt), ImageAttentionPolicy)
