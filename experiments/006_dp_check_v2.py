"""Sanity check for the `dpv2` pixi environment on an actual GPU node.

Beyond import + cuda.is_available(), this specifically exercises conv1d and
scaled_dot_product_attention forward+backward on GPU, since the old
PyPI-pinned torch==2.7.1+cu128 stack (feature.diffusion-policy-pro6000) hit a
CUDA "illegal memory access" inside SDPA on the RTX PRO 6000 (Blackwell/sm_120)
-- see experiments/008_train_diffusion_policy.sbatch's comments for that history.

Run via: pixi run -e dpv2 python experiments/006_dp_check_v2.py
Only meaningful on a GPU node (login node has no GPU) -- submit through
experiments/006_dp_check_v2.sbatch.
"""

import accelerate
import diffusers
import timm
import torch
import torch.nn.functional as F
import transformers

print(f"torch {torch.__version__} (built for cuda {torch.version.cuda})")
print(
    f"diffusers {diffusers.__version__}, transformers {transformers.__version__}, "
    f"accelerate {accelerate.__version__}, timm {timm.__version__}"
)
print(f"cuda available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    raise SystemExit("CUDA not available -- check gres/qos and driver CUDA version")

for i in range(torch.cuda.device_count()):
    name = torch.cuda.get_device_name(i)
    cap = torch.cuda.get_device_capability(i)
    print(f"  device {i}: {name} (sm_{cap[0]}{cap[1]})")

device = "cuda"

x = torch.randn(4096, 4096, device=device)
y = x @ x
torch.cuda.synchronize()
print(f"matmul smoke test ok, result mean={y.mean().item():.4f}")

conv = torch.nn.Conv1d(64, 128, kernel_size=5, padding=2).to(device)
inp = torch.randn(8, 64, 32, device=device, requires_grad=True)
out = conv(inp)
out.sum().backward()
torch.cuda.synchronize()
print(f"conv1d forward+backward ok, out shape={tuple(out.shape)}, grad ok={inp.grad is not None}")

B, H, T, D = 8, 8, 32, 64
q = torch.randn(B, H, T, D, device=device, requires_grad=True)
k = torch.randn(B, H, T, D, device=device, requires_grad=True)
v = torch.randn(B, H, T, D, device=device, requires_grad=True)
attn_out = F.scaled_dot_product_attention(q, k, v)
attn_out.sum().backward()
torch.cuda.synchronize()
print(f"sdpa forward+backward ok, out shape={tuple(attn_out.shape)}, grad ok={q.grad is not None}")

print("dp-check-v2 PASSED")
