"""src TraceAttentionPolicy vs. the fork's policy, with the same weights (reference: experiments/024_*).

Run on a compute node (batch check): sbatch experiments/024_policy_equivalence.sbatch
"""

from pathlib import Path

import pytest
import torch

from markovian_policy.policies.trace_attention import TraceAttentionPolicy, TracePolicyConfig

REFERENCE = Path("output/policy_reference/fork_policy.pt")

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not REFERENCE.exists(), reason="run experiments/024_policy_reference_legacy.py")]


@pytest.fixture(scope="module")
def reference() -> dict:
    return torch.load(REFERENCE, map_location="cpu", weights_only=False)


@pytest.fixture(scope="module")
def policy(reference: dict) -> TraceAttentionPolicy:
    policy = TraceAttentionPolicy(TracePolicyConfig(cameras=("scene",)))  # the fork policy conditions on the scene camera only
    result = policy.load_state_dict(reference["state_dict"], strict=False)
    assert not result.unexpected_keys, result.unexpected_keys[:5]
    assert not result.missing_keys, result.missing_keys[:5]
    return policy.eval()


def test_conditioning_tokens_match(policy: TraceAttentionPolicy, reference: dict) -> None:
    cond = policy.conditioning(reference["obs"])
    torch.testing.assert_close(cond.tokens, reference["cond_tokens"], rtol=1e-4, atol=1e-5)


@torch.no_grad()
def test_noise_prediction_matches(policy: TraceAttentionPolicy, reference: dict) -> None:
    cond = policy.conditioning(reference["obs"])
    pred = policy.model(reference["noisy"], reference["timesteps"], **cond.as_kwargs())
    torch.testing.assert_close(pred, reference["noise_pred"], rtol=1e-4, atol=1e-5)


def test_ddim_action_matches_with_fixed_noise(policy: TraceAttentionPolicy, reference: dict) -> None:
    torch.manual_seed(0)
    out = policy.predict_action({"obs": reference["obs"]}, use_ddim=True)
    torch.testing.assert_close(out["action"], reference["action"], rtol=1e-3, atol=1e-4)
    torch.testing.assert_close(out["action_pred"], reference["action_pred"], rtol=1e-3, atol=1e-4)
