#!/bin/bash
# Verify the new `dpv2` pixi env (recent torch/diffusers/transformers/accelerate/timm,
# conda-forge pytorch-gpu instead of the old PyPI-pinned robomimic/r3m stack) actually
# works on a GPU node -- especially the scaled_dot_product_attention forward+backward,
# since torch==2.7.1+cu128 (feature.diffusion-policy-pro6000) hit a CUDA "illegal memory
# access" inside SDPA on the RTX PRO 6000 (Blackwell/sm_120). See
# experiments/006_dp_check_v2.py and experiments/008_train_diffusion_policy.sbatch's
# comments for that history.
#
# Submit from the project root:
#   cd /home/minjunsz/codes/1_pai/markovian-policy && sbatch experiments/006_dp_check_v2.sbatch
#
# Run this once per GPU type before trusting dpv2 for real training. Swap the qos/gres
# pair below to test the other GPU type:
#   high_v1 + gpu:l40:1      -- team v1 pool
#   high_v  + gpu:pro6000:1  -- shared pool (RTX PRO 6000 Blackwell)
# `low` (below) grabs whichever type is idle on the cluster right now, regardless of team
# quota -- fine here since the job is short and cheap to lose to preemption/requeue.

#SBATCH --job-name=dp_check_v2
#SBATCH --output=output/logs/%x_%j.log
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40:1
#SBATCH --time=00:05:00

set -euo pipefail

# Don't rely on sbatch inheriting an interactive shell's PATH.
export PATH="$HOME/.pixi/bin:$PATH"

mkdir -p output/logs

echo "job $SLURM_JOB_ID running as $USER on $(hostname)"
echo "qos=$SLURM_JOB_QOS gpus_on_node=$SLURM_GPUS_ON_NODE"
nvidia-smi --query-gpu=index,name,compute_cap,driver_version --format=csv

pixi run -e dpv2 python experiments/006_dp_check_v2.py

echo "finished at $(date)"
