"""Stage: precompute the frozen encoder's tokens for a dataset (init -> encode -> verify, see `TokenCacheBuilder`)."""

from dataclasses import dataclass
from typing import Literal

import tyro

from markovian_policy.data.token_cache_builder import TokenCacheBuilder, TokenCacheConfig


@dataclass
class Config:
    mode: tyro.conf.Positional[Literal["init", "encode", "verify", "bench"]]
    cache: tyro.conf.OmitArgPrefixes[TokenCacheConfig]  # flags without a "--cache." prefix


def run(config: Config) -> None:
    builder = TokenCacheBuilder(config.cache)
    {"init": builder.init, "encode": builder.encode, "verify": builder.verify, "bench": builder.bench}[config.mode]()
