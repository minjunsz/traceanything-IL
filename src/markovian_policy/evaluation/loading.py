"""Rebuild a trained policy from a training checkpoint (no CLI arguments needed)."""

from dataclasses import replace
from pathlib import Path
from typing import Literal

import torch

from markovian_policy.policies.trace_attention import TraceAttentionPolicy, TracePolicyConfig
from markovian_policy.training.checkpoint import load_payload


def load_policy(
    checkpoint: Path,
    device: torch.device,
    mixed_precision: Literal["no", "fp16", "bf16"] = "no",
    trace_weights: Path | None = None,
) -> TraceAttentionPolicy:
    """The EMA policy of `checkpoint`, on `device`, in eval mode (the normalizer is part of the weights).

    `trace_weights` overrides where the frozen encoder's weights are read from; a checkpoint otherwise records the
    path it was trained with, which may not exist on another machine.
    """
    payload = load_payload(checkpoint)
    assert payload.policy_config is not None, "checkpoint does not record its policy config"
    config = TracePolicyConfig.from_dict(payload.policy_config)
    if trace_weights is not None:
        config = replace(config, trace=replace(config.trace, ckpt_path=trace_weights))
    policy = TraceAttentionPolicy(config)
    policy.load_state_dict(payload.ema_model)
    policy.mixed_precision = mixed_precision
    return policy.to(device).eval()
