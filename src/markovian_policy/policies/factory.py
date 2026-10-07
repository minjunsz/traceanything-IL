"""Build a policy from its config, and recover the config of a checkpoint (policy kinds share one training/eval path)."""

from typing import Any

from markovian_policy.policies.image_attention import ImageAttentionPolicy, ImageAttentionPolicyConfig
from markovian_policy.policies.image_film import ImageFilmPolicy, ImagePolicyConfig
from markovian_policy.policies.trace_attention import TraceAttentionPolicy, TracePolicyConfig

type PolicyConfig = TracePolicyConfig | ImagePolicyConfig | ImageAttentionPolicyConfig
type Policy = TraceAttentionPolicy | ImageFilmPolicy | ImageAttentionPolicy


def build_policy(config: PolicyConfig) -> Policy:
    if isinstance(config, ImageAttentionPolicyConfig):
        return ImageAttentionPolicy(config)
    if isinstance(config, ImagePolicyConfig):
        return ImageFilmPolicy(config)
    return TraceAttentionPolicy(config)


def config_from_dict(data: dict[str, Any]) -> PolicyConfig:
    """Checkpoints written before there were several kinds carry no `kind`: they are TraceAnything policies."""
    match data.get("kind"):
        case "image_film":
            return ImagePolicyConfig.from_dict(data)
        case "image_attention":
            return ImageAttentionPolicyConfig.from_dict(data)
        case _:
            return TracePolicyConfig.from_dict(data)
