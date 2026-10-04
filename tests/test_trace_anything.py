"""Ported TraceAnything tokens vs. the original implementation (experiments/019_*)."""

from pathlib import Path

import numpy as np
import pytest
import torch

from markovian_policy.perception.trace_anything.encoder import TraceAnythingWindowEncoder

WINDOW = Path("output/trace_reference/window.npz")
TOKENS = Path("output/trace_reference/tokens.npz")

needs_reference = pytest.mark.skipif(not TOKENS.exists(), reason="run experiments/019_trace_reference.sbatch first")


def _rel(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    return ((a - b).abs().max() / b.abs().max()).item(), ((a - b).abs().mean() / b.abs().mean()).item()


@needs_reference
@pytest.mark.gpu  # runs on CPU but is too heavy for the login node: submit via a compute-node job
def test_tokens_match_original_fp32() -> None:
    window = torch.from_numpy(np.load(WINDOW)["window"][:1, :4]).float()
    expected = torch.from_numpy(np.load(TOKENS)["tokens_cpu"])
    tokens = TraceAnythingWindowEncoder()(window)
    assert tokens.shape == expected.shape == (1, 768, 1024)
    assert _rel(tokens, expected)[0] < 1e-4


@needs_reference
@pytest.mark.gpu
def test_tokens_match_original_bf16_attention() -> None:
    window = torch.from_numpy(np.load(WINDOW)["window"]).float().cuda()
    expected = torch.from_numpy(np.load(TOKENS)["tokens"]).cuda()
    tokens = TraceAnythingWindowEncoder().cuda()(window)
    assert tokens.shape == expected.shape == (2, 768, 1024)
    max_rel, mean_rel = _rel(tokens, expected)  # bf16 attention: different kernels round differently
    assert max_rel < 2e-2 and mean_rel < 1e-2
