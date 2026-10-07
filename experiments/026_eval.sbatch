#!/bin/bash
# Evaluate one checkpoint (one GPU, one subprocess env per trial slot). Resumable: rerun the same command.
#   sbatch experiments/026_eval.sbatch --checkpoint CKPT --out-dir OUT [--n-rollouts 50 --n-envs 10 ...]
#SBATCH --job-name=eval_kitchen
#SBATCH --output=output/logs/%x_%j.log
#SBATCH --open-mode=append
#SBATCH --qos=high_v
#SBATCH --gres=gpu:l40:1
#SBATCH --time=04:00:00
#SBATCH --requeue

set -euo pipefail
PIXI_ENV=port
source scripts/slurm_env.sh
echo "job $SLURM_JOB_ID on $(hostname), restart ${SLURM_RESTART_COUNT:-0}, ${SLURM_CPUS_ON_NODE:-?} cpus"
pyrun experiments/026_eval.py "$@"
