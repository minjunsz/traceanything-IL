#!/bin/bash
# Multi-GPU training of the image-based attention baseline (double encoder) (raw frames, R3M ResNet18). Resumable: the job is requeued after a
# preemption; after a timeout resubmit the same command (same output dir) to continue from checkpoints/latest.ckpt.
# n_obs_steps is the history length; horizon = n_obs_steps + 14, and --data.* / --policy.* must agree. Example:
#
#   sbatch experiments/031_train_image_attention_policy.sbatch --data.zarr-path data/human.zarr \
#       --data.n-obs-steps 16 --policy.n-obs-steps 16 --data.horizon 30 --policy.horizon 30 \
#       --data.lowdim-keys agent_pos --policy.lowdim-keys agent_pos --policy.lowdim-dim 9 \
#       --train.output-dir runs/human_image_attention --train.batch-size 64
#SBATCH --job-name=image_attention
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
# Optional: WARM_DIR=data/human.raw sbatch ... reads the exported frames once, sequentially, so the memory-mapped
# training reads hit the node's page cache instead of faulting pages in randomly from NFS.
if [ -n "${WARM_DIR:-}" ]; then echo "warming $WARM_DIR"; cat "$WARM_DIR"/*.npy > /dev/null; fi
echo "job $SLURM_JOB_ID on $(hostname), restart ${SLURM_RESTART_COUNT:-0}"
pxrun torchrun --standalone --nproc_per_node="$SLURM_GPUS_ON_NODE" experiments/031_train_image_attention_policy.py "$@"
