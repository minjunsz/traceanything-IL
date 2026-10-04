#!/bin/bash
# Verify the diffusion-policy-l40 env (PyPI torch==2.3.1+cu121) on each GPU
# type of this cluster before the real training run (TRANSFER_README.md §1).
# Beyond `dp-check` (imports + cuda.is_available), runs matmul / conv2d /
# bf16 SDPA forward+backward so a missing kernel image for the arch fails here.
#
# Pick the GPU type at submit time (a100 only exists in mit_preemptable):
#   sbatch -p mit_normal_gpu  --gres=gpu:l40s:1 experiments/011_dp_check.sbatch
#   sbatch -p mit_normal_gpu  --gres=gpu:h100:1 experiments/011_dp_check.sbatch
#   sbatch -p mit_normal_gpu  --gres=gpu:h200:1 experiments/011_dp_check.sbatch
#   sbatch -p mit_preemptable --gres=gpu:a100:1 experiments/011_dp_check.sbatch

#SBATCH --job-name=dp_check
#SBATCH --output=output/logs/%x_%j.log
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00

set -euo pipefail
export PATH="$HOME/.pixi/bin:$PATH"

echo "job $SLURM_JOB_ID on $(hostname) partition=$SLURM_JOB_PARTITION"
nvidia-smi --query-gpu=index,name,compute_cap,driver_version --format=csv

pixi run -e diffusion-policy-l40 dp-check
pixi run -e diffusion-policy-l40 python - <<'EOF'
import torch
import torch.nn.functional as F

dev = "cuda"
print("device:", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
print("arch list:", torch.cuda.get_arch_list())
a = torch.randn(1024, 1024, device=dev)
print("matmul ok:", (a @ a).sum().item() == (a @ a).sum().item())
x = torch.randn(8, 3, 64, 64, device=dev)
w = torch.randn(16, 3, 3, 3, device=dev)
print("conv2d ok:", tuple(F.conv2d(x, w).shape))
q, k, v = (torch.randn(4, 8, 256, 64, device=dev, dtype=torch.bfloat16, requires_grad=True) for _ in range(3))
out = F.scaled_dot_product_attention(q, k, v)
out.float().sum().backward()
torch.cuda.synchronize()
print("sdpa bf16 fwd+bwd ok:", tuple(out.shape), q.grad.abs().sum().item() > 0)
print("ALL OK")
EOF
echo "finished at $(date)"
