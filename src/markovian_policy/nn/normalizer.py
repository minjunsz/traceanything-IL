"""Linear (scale + offset) normalizer, checkpointed as part of the model state_dict.

The parameters live in nested `nn.ParameterDict`s (key -> scale, offset, input_stats.{min,max,mean,std}); this layout
is part of the checkpoint format.
"""

from collections.abc import Mapping
from typing import Any, Literal, cast

import numpy as np
import torch
from torch import nn

type ArrayLike = torch.Tensor | np.ndarray[Any, Any]


class _ParamsModule(nn.Module):
    """Stores a nested dict of tensors that survives `state_dict()` save/load.

    A plain `nn.ParameterDict` flattens nested dicts into dotted keys on save; this rebuilds the nesting on load.
    """

    def __init__(self, params_dict: nn.ParameterDict | None = None) -> None:
        super().__init__()
        self.params_dict = params_dict if params_dict is not None else nn.ParameterDict()

    @property
    def device(self) -> torch.device:
        return next(iter(self.parameters())).device

    def _load_from_state_dict(
        self,
        state_dict: Mapping[str, Any],
        prefix: str,
        local_metadata: Any,
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        def add(dest: nn.ParameterDict, keys: list[str], value: torch.Tensor) -> None:
            if len(keys) == 1:
                dest[keys[0]] = value
                return
            if keys[0] not in dest:
                dest[keys[0]] = nn.ParameterDict()
            add(cast(nn.ParameterDict, dest[keys[0]]), keys[1:], value)

        loaded = nn.ParameterDict()
        full_prefix = prefix + "params_dict"
        for key, value in state_dict.items():
            if key.startswith(full_prefix):
                add(loaded, key[len(full_prefix) :].split(".")[1:], value.clone())
        self.params_dict = loaded
        self.params_dict.requires_grad_(False)


class FieldNormalizer(_ParamsModule):
    """Normalizer of one field: x -> x * scale + offset over the last dimension."""

    def normalize(self, x: ArrayLike) -> torch.Tensor:
        return _apply(x, self.params_dict, forward=True)

    def unnormalize(self, x: ArrayLike) -> torch.Tensor:
        return _apply(x, self.params_dict, forward=False)


class LinearNormalizer(_ParamsModule):
    """A dict of per-key `FieldNormalizer`s, fit from data statistics.

    "limits" mode maps the input [min, max] to [output_min, output_max] (default [-1, 1]); "gaussian" mode maps to
    zero mean and unit variance.
    """

    @torch.no_grad()
    def fit(
        self,
        data: Mapping[str, ArrayLike],
        last_n_dims: int = 1,
        mode: Literal["limits", "gaussian"] = "limits",
        output_min: float = -1.0,
        output_max: float = 1.0,
        range_eps: float = 1e-4,
        fit_offset: bool = True,
    ) -> None:
        for key, values in data.items():
            self.params_dict[key] = _fit(values, last_n_dims, mode, output_min, output_max, range_eps, fit_offset)

    def __getitem__(self, key: str) -> FieldNormalizer:
        return FieldNormalizer(cast(nn.ParameterDict, self.params_dict[key]))

    def normalize(self, x: Mapping[str, ArrayLike]) -> dict[str, torch.Tensor]:
        return {key: _apply(value, cast(nn.ParameterDict, self.params_dict[key]), forward=True) for key, value in x.items()}

    def unnormalize(self, x: Mapping[str, ArrayLike]) -> dict[str, torch.Tensor]:
        return {key: _apply(value, cast(nn.ParameterDict, self.params_dict[key]), forward=False) for key, value in x.items()}


def _fit(
    data: ArrayLike, last_n_dims: int, mode: Literal["limits", "gaussian"], output_min: float, output_max: float,
    range_eps: float, fit_offset: bool,
) -> nn.ParameterDict:  # fmt: skip
    assert output_max > output_min
    values = torch.from_numpy(data) if isinstance(data, np.ndarray) else data
    values = values.type(torch.float32)
    values = values.reshape(-1, int(np.prod(values.shape[-last_n_dims:])) if last_n_dims > 0 else 1)

    input_min, input_max = values.min(dim=0).values, values.max(dim=0).values
    input_mean, input_std = values.mean(dim=0), values.std(dim=0)

    if mode == "limits":
        if fit_offset:
            input_range = input_max - input_min
            ignore = input_range < range_eps
            input_range[ignore] = output_max - output_min
            scale = (output_max - output_min) / input_range
            offset = output_min - scale * input_min
            offset[ignore] = (output_max + output_min) / 2 - input_min[ignore]
        else:
            assert output_max > 0 > output_min
            output_abs = min(abs(output_min), abs(output_max))
            input_abs = torch.maximum(input_min.abs(), input_max.abs())
            input_abs[input_abs < range_eps] = output_abs
            scale = output_abs / input_abs
            offset = torch.zeros_like(input_mean)
    else:
        ignore = input_std < range_eps
        scale = input_std.clone()
        scale[ignore] = 1
        scale = 1 / scale
        offset = -input_mean * scale if fit_offset else torch.zeros_like(input_mean)

    params = nn.ParameterDict(
        {
            "scale": scale,
            "offset": offset,
            "input_stats": nn.ParameterDict({"min": input_min, "max": input_max, "mean": input_mean, "std": input_std}),
        }
    )
    params.requires_grad_(False)
    return params


def _apply(x: ArrayLike, params: nn.ParameterDict, forward: bool) -> torch.Tensor:
    tensor = torch.from_numpy(x) if isinstance(x, np.ndarray) else x
    scale, offset = cast(torch.Tensor, params["scale"]), cast(torch.Tensor, params["offset"])
    tensor = tensor.to(device=scale.device, dtype=scale.dtype)
    shape = tensor.shape
    tensor = tensor.reshape(-1, scale.shape[0])
    tensor = tensor * scale + offset if forward else (tensor - offset) / scale
    return tensor.reshape(shape)
