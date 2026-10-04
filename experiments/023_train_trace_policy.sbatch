#!/bin/bash
# Multi-GPU training of the TraceAnything-conditioned policy. Resumable: resubmit the same command (same output dir)
# to continue from checkpoints/latest.ckpt after a preemption/timeout.
#
#   sbatch experiments/023_train_trace_policy.sbatch \
#       --data.zarr-path data/kitchen_demos_expert.zarr \
#       --data.trace-cache-dirs scene CACHE_SCENE wrist CACHE_WRIST \
#       --train.output-dir runs/trace_policy --train.batch-size 128
#SBATCH --job-name=trace_policy
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
export PATH="$HOME/.pixi/bin:$PATH"
[ -d "$HOME/orcd/scratch" ] && export PIXI_CACHE_DIR=$HOME/orcd/scratch/cache/pixi  # ORCD Engaging; elsewhere pixi's default cache
export CONDA_OVERRIDE_CUDA=12.9
export NCCL_P2P_DISABLE=1
cd "$SLURM_SUBMIT_DIR"

pixi run -e port torchrun --standalone --nproc_per_node="$SLURM_GPUS_ON_NODE" \
    experiments/023_train_trace_policy.py "$@"
