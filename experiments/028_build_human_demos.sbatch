#!/bin/bash
# Replay the human logs with physics (one replay process per CPU, rendering on one GPU). Resumable: if the job is
# preempted (requeued) or times out (resubmit), the same command continues where it stopped.
#   sbatch experiments/028_build_human_demos.sbatch [--out ...] [--n-workers N]
#SBATCH --job-name=human_demos
#SBATCH --output=output/logs/%x_%j.log
#SBATCH --open-mode=append
#SBATCH --qos=high_v
#SBATCH --gres=gpu:l40:1
#SBATCH --time=06:00:00
#SBATCH --requeue

set -euo pipefail
PIXI_ENV=port
source scripts/slurm_env.sh
N_WORKERS=$(( ${SLURM_CPUS_ON_NODE:-4} > 2 ? ${SLURM_CPUS_ON_NODE:-4} - 2 : 1 ))  # leave cores for the main process
echo "job $SLURM_JOB_ID on $(hostname), restart ${SLURM_RESTART_COUNT:-0}, $N_WORKERS replay workers"
pyrun experiments/028_build_human_demos.py --n-workers "$N_WORKERS" "$@"
