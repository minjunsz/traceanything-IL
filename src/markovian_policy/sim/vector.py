"""Batched kitchen environments (gymnasium vector API)."""

from functools import partial
from typing import Any

from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv, VectorEnv

from markovian_policy.sim.env import KitchenEnv, KitchenEnvConfig


def make_vector_env(config: KitchenEnvConfig, n_envs: int, asynchronous: bool = True) -> VectorEnv[Any, Any, Any]:
    """`n_envs` kitchen envs, in spawned subprocesses (CUDA-safe, one GL context each) or in this process."""
    env_fns = [partial(KitchenEnv, config)] * n_envs
    return AsyncVectorEnv(env_fns, context="spawn") if asynchronous else SyncVectorEnv(env_fns)
