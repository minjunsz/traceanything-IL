"""Sanity check for the `trace-anything` pixi environment on an actual GPU node.

Run via: pixi run -e trace-anything python experiments/000_env_check.py
Only meaningful on a GPU node (login node has no GPU) -- submit through
experiments/000_env_check.sbatch.
"""

import torch

print(f"torch {torch.__version__} (built for cuda {torch.version.cuda})")
print(f"compiled arch flags: {torch._C._cuda_getArchFlags()}")
print(f"cuda available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        cap = torch.cuda.get_device_capability(i)
        print(f"  device {i}: {name} (sm_{cap[0]}{cap[1]})")
    x = torch.randn(4096, 4096, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print(f"matmul smoke test ok, result mean={y.mean().item():.4f}")
else:
    raise SystemExit("CUDA not available -- check gres/qos and driver CUDA version")
