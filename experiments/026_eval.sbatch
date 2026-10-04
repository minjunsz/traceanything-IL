#!/bin/bash
# Evaluate one checkpoint (one GPU, one subprocess env per trial slot). Resumable: rerun the same command.
#   sbatch experiments/026_eval.sbatch --checkpoint CKPT --out-dir OUT [--n-rollouts 50 --n-envs 10 ...]
#SBATCH --job-name=eval_kitchen
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
export PATH="$HOME/.pixi/bin:$PATH"
[ -d "$HOME/orcd/scratch" ] && export PIXI_CACHE_DIR=$HOME/orcd/scratch/cache/pixi  # ORCD Engaging; elsewhere pixi's default cache
export CONDA_OVERRIDE_CUDA=12.9
cd "$SLURM_SUBMIT_DIR"
pixi run -e port python experiments/026_eval.py "$@"
