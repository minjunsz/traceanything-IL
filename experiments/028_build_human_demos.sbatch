#!/bin/bash
# Replay the human logs (one replay process per CPU, rendering on one GPU). Resumable: resubmit to continue.
#   sbatch experiments/028_build_human_demos.sbatch [--out ...] [--n-workers N]
#SBATCH --job-name=human_demos
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
export PATH="$HOME/.pixi/bin:$PATH"
[ -d "$HOME/orcd/scratch" ] && export PIXI_CACHE_DIR=$HOME/orcd/scratch/cache/pixi  # ORCD Engaging; elsewhere pixi's default cache
export CONDA_OVERRIDE_CUDA=12.9
cd "$SLURM_SUBMIT_DIR"
pixi run -e port python experiments/028_build_human_demos.py --n-workers 6 "$@"
