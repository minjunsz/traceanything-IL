#!/bin/bash
#SBATCH --job-name=expert_success
#SBATCH --partition=mit_quicktest
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
export PATH="$HOME/.pixi/bin:$PATH"
[ -d "$HOME/orcd/scratch" ] && export PIXI_CACHE_DIR=$HOME/orcd/scratch/cache/pixi  # ORCD Engaging; elsewhere pixi's default cache
export CONDA_OVERRIDE_CUDA=12.9
cd "$SLURM_SUBMIT_DIR"
pixi run -e port python experiments/018_expert_success.py "$@"
