"""Reference TraceAnything tokens from the ORIGINAL code (run in the `trace-anything` pixi env).

Reads output/trace_reference/window.npz (made by 019_make_trace_window.py) and writes tokens.npz:
  tokens      GPU (bf16 attention core), full window
  tokens_cpu  CPU fp32 (exact-logic check), first window's first 4 frames
"""

import contextlib
import importlib.util
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import torch

sys.path[:0] = ["third_party/TraceAnything", "third_party"]  # original trace_anything + fast3r
WRAPPER = "third_party/diffusion-policy-experiments/diffusion_policy/model/vision/trace_anything_encoder.py"
spec = importlib.util.spec_from_file_location("legacy_encoder", WRAPPER)
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)

root = Path("third_party/TraceAnything")


def build() -> torch.nn.Module:  # one instance per device: the original caches patch positions regardless of device
    return legacy.TraceAnythingWindowEncoder(str(root / "configs/eval.yaml"), str(root / "checkpoints/trace_anything.pt"), long_side=512)


window = torch.from_numpy(np.load("output/trace_reference/window.npz")["window"]).float()

with mock.patch("torch.nn.attention.sdpa_kernel", lambda *a, **k: contextlib.nullcontext()):  # no EFFICIENT backend on CPU
    tokens_cpu = build()(window[:1, :4]).numpy()
tokens = build().cuda()(window.cuda()).cpu().numpy()
np.savez("output/trace_reference/tokens.npz", tokens=tokens, tokens_cpu=tokens_cpu)
print("tokens", tokens.shape, tokens_cpu.shape)
