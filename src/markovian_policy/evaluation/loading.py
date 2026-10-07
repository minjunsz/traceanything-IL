"""Rebuild a trained policy from a training checkpoint (no CLI arguments needed)."""

from dataclasses import replace
from pathlib import Path
from typing import Literal

import torch

from markovian_policy.policies.factory import Policy, build_policy, config_from_dict
from markovian_policy.policies.image_attention import ImageAttentionPolicyConfig
from markovian_policy.policies.image_film import ImagePolicyConfig
from markovian_policy.policies.trace_attention import TracePolicyConfig
from markovian_policy.training.checkpoint import load_payload


def load_policy(
    checkpoint: Path,
    device: torch.device,
    mixed_precision: Literal["no", "fp16", "bf16"] = "no",
    trace_weights: Path | None = None,
) -> Policy:
    """The EMA policy of `checkpoint`, on `device`, in eval mode (the normalizer is part of the weights).

    `trace_weights` overrides where a TraceAnything policy's frozen encoder reads its weights from; a checkpoint
    otherwise records the path it was trained with, which may not exist on another machine. An image policy's
    checkpoint holds all its weights, so the pretrained R3M file is not needed (nor read) at evaluation.
    """
    payload = load_payload(checkpoint)
    assert payload.policy_config is not None, "checkpoint does not record its policy config"
    config = config_from_dict(payload.policy_config)
    if isinstance(config, TracePolicyConfig) and trace_weights is not None:
        config = replace(config, trace=replace(config.trace, ckpt_path=trace_weights))
    elif isinstance(config, ImagePolicyConfig | ImageAttentionPolicyConfig):
        config = replace(config, r3m_weights=None)
    policy = build_policy(config)
    policy.load_state_dict(payload.ema_model)
    policy.mixed_precision = mixed_precision
    return policy.to(device).eval()
