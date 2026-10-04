"""Checkpoints written before the package restructuring still load and give bit-identical outputs.

tests/data/legacy_policy.ckpt was written by the pre-refactor code (Trainer payload layout) together with the
policy outputs for fixed inputs.
"""

from pathlib import Path

import torch

from markovian_policy.evaluation.loading import load_policy
from markovian_policy.training import load_payload

LEGACY = Path("tests/data/legacy_policy.ckpt")


def test_legacy_checkpoint_loads_and_reproduces_outputs() -> None:
    reference = torch.load(LEGACY, map_location="cpu", weights_only=False)["reference"]
    policy = load_policy(LEGACY, torch.device("cpu"))
    cond = policy.conditioning(reference["obs"])
    noise_pred = policy.model(reference["noisy"], reference["timesteps"], **cond.as_kwargs())
    torch.testing.assert_close(noise_pred, reference["noise_pred"], rtol=0, atol=0)
    torch.manual_seed(0)
    action = policy.predict_action({"obs": reference["obs"]}, use_ddim=True)
    torch.testing.assert_close(action["action"], reference["action"], rtol=0, atol=0)
    torch.testing.assert_close(action["action_pred"], reference["action_pred"], rtol=0, atol=0)


def test_legacy_payload_has_the_documented_layout() -> None:
    payload = load_payload(LEGACY)
    assert (payload.epoch, payload.epoch_step, payload.global_step, payload.ema_step) == (0, 0, 3, 3)
    assert payload.policy_config is not None and tuple(payload.policy_config["cameras"]) == ("scene", "wrist")
    assert payload.train_config["batch_size"] == 64
    assert any(key.startswith("normalizer.params_dict.action") for key in payload.ema_model)  # the normalizer travels along
