#!/bin/bash
# Verify the `trace-anything` pixi env + CUDA toolchain actually work on a GPU node
# (the login node has no GPU, so this can't be checked without submitting a job).
#
# Submit from the project root:
#   cd /home/minjunsz/codes/1_pai/markovian-policy && sbatch experiments/000_env_check.sbatch
#
# Swap the qos/gres pair below to test the other GPU type:
#   high_v1 + gpu:l40:1      -- team v1 pool
#   high_v  + gpu:pro6000:1  -- shared pool (RTX PRO 6000 Blackwell)

#SBATCH --job-name=env_check
#SBATCH --output=output/logs/%x_%j.log
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40:1
#SBATCH --time=00:10:00

set -euo pipefail

# Don't rely on sbatch inheriting an interactive shell's PATH.
export PATH="$HOME/.pixi/bin:$PATH"

mkdir -p output/logs

echo "job $SLURM_JOB_ID running as $USER on $(hostname)"
echo "qos=$SLURM_JOB_QOS gpus_on_node=$SLURM_GPUS_ON_NODE"
nvidia-smi --query-gpu=index,name,compute_cap,driver_version --format=csv

pixi run -e trace-anything python experiments/000_env_check.py

echo "finished at $(date)"
