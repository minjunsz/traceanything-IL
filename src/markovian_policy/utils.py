"""Small helpers shared across layers."""

from collections.abc import Callable, Mapping
from typing import Any

import torch


def dict_apply(x: Mapping[str, Any], func: Callable[[torch.Tensor], torch.Tensor]) -> dict[str, Any]:
    """Apply `func` to every leaf tensor of a (possibly nested) mapping."""
    return {key: dict_apply(value, func) if isinstance(value, Mapping) else func(value) for key, value in x.items()}
