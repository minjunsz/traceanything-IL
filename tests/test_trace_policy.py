"""TraceAttentionPolicy on cached tokens (CPU): loss/backward, sampling, token layout, config round trip."""

import dataclasses

import numpy as np
import torch

from markovian_policy.data.trace_cache import tokens_obs_key
from markovian_policy.nn.normalizer import LinearNormalizer
from markovian_policy.policies.conditioning import MODALITY_LOWDIM, MODALITY_TRACE
from markovian_policy.policies.trace_attention import TraceAttentionPolicy, TracePolicyConfig

B, P, D = 3, 6, 1024  # batch, trace tokens per camera, TraceAnything token dim
CONFIG = TracePolicyConfig(
    horizon=10, n_obs_steps=4, n_action_steps=3, obs_feature_dim=32, diffusion_step_embed_dim=16, down_dims=(32, 64),
    num_attention_heads=4, num_train_timesteps=10, num_ddpm_inference_steps=4, num_ddim_inference_steps=2,
)  # fmt: skip


def _policy() -> tuple[TraceAttentionPolicy, dict]:
    rng = np.random.default_rng(0)
    policy = TraceAttentionPolicy(CONFIG)
    normalizer = LinearNormalizer()
    normalizer.fit(
        {"action": rng.normal(size=(50, 9)), "agent_pos": rng.normal(size=(50, 9)), "subtask_sequence": rng.normal(size=(50, 28))},
        last_n_dims=1, mode="limits",
    )  # fmt: skip
    policy.set_normalizer(normalizer)
    obs = {
        "agent_pos": torch.randn(B, CONFIG.horizon, 9),
        "subtask_sequence": torch.randn(B, CONFIG.horizon, 28),
        **{tokens_obs_key(camera, 2): torch.randn(B, P, D) for camera in CONFIG.cameras},
    }
    return policy, {"obs": obs, "action": torch.randn(B, CONFIG.horizon, 9)}


def test_loss_backward_reaches_all_trainable_parts() -> None:
    policy, batch = _policy()
    loss = policy(batch)
    loss.backward()
    assert torch.isfinite(loss)
    for module in (policy.model, policy.lowdim_proj, policy.trace_proj):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())


def test_predict_action_shapes_for_both_samplers() -> None:
    policy, batch = _policy()
    for use_ddim in (False, True):
        out = policy.predict_action({"obs": batch["obs"]}, use_ddim=use_ddim)
        assert out["action"].shape == (B, CONFIG.n_action_steps, 9) and out["action_pred"].shape == (B, CONFIG.horizon, 9)
        assert torch.isfinite(out["action_pred"]).all()


def test_token_layout_covers_lowdim_steps_then_one_block_per_camera() -> None:
    policy, batch = _policy()
    cond = policy.conditioning(batch["obs"])
    To, n = CONFIG.n_obs_steps, CONFIG.n_obs_steps + 2 * P
    assert cond.tokens.shape == (B, n, CONFIG.obs_feature_dim)
    assert cond.temporal_positions[0].tolist() == [*range(To), *[To - 1] * (2 * P)]
    assert cond.modality_indices[0].tolist() == [MODALITY_LOWDIM] * To + [MODALITY_TRACE] * P + [MODALITY_TRACE + 1] * P


def test_frozen_encoder_stays_out_of_the_module_tree() -> None:
    policy, batch = _policy()
    policy(batch)  # cached tokens: the encoder must not even be built
    assert not policy.token_source.encoder_loaded
    assert not any("trace_encoder" in key for key in policy.state_dict())


def test_config_round_trips_through_a_checkpoint_dict() -> None:
    assert TracePolicyConfig.from_dict(dataclasses.asdict(CONFIG)) == CONFIG
