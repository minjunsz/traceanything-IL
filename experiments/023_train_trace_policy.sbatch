#!/bin/bash
# Multi-GPU training of the TraceAnything-conditioned policy on cached tokens. Resumable: the job is requeued after a
# preemption; after a timeout resubmit the same command (same output dir) to continue from checkpoints/latest.ckpt.
# n_obs_steps is the history length (the cache must have been built for it); horizon = n_obs_steps + 14. Example:
#
#   sbatch experiments/023_train_trace_policy.sbatch --data.zarr-path data/human.zarr \
#       --data.n-obs-steps 16 --policy.n-obs-steps 16 --data.horizon 30 --policy.horizon 30 \
#       --data.lowdim-keys agent_pos --policy.lowdim-keys agent_pos --policy.lowdim-dim 9 \
#       --data.trace-cache-dirs scene data/trace_cache/human_obs16_scene wrist data/trace_cache/human_obs16_wrist \
#       --train.output-dir runs/human_trace16 --train.batch-size 64
#SBATCH --job-name=trace_policy
#SBATCH --output=output/logs/%x_%j.log
#SBATCH --open-mode=append
#SBATCH --qos=high_v
#SBATCH --gres=gpu:l40:4
#SBATCH --time=2-00:00:00
#SBATCH --requeue

set -euo pipefail
PIXI_ENV=port
source scripts/slurm_env.sh
export NCCL_P2P_DISABLE=1
echo "job $SLURM_JOB_ID on $(hostname), restart ${SLURM_RESTART_COUNT:-0}"
pxrun torchrun --standalone --nproc_per_node="$SLURM_GPUS_ON_NODE" experiments/023_train_trace_policy.py "$@"
