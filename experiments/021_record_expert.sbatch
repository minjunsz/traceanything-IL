#!/bin/bash
# Usage: sbatch [--time=..] experiments/021_record_expert.sbatch [record flags]  (default: 4-episode smoke test)
#SBATCH --job-name=record_expert
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
export PATH="$HOME/.pixi/bin:$PATH"
[ -d "$HOME/orcd/scratch" ] && export PIXI_CACHE_DIR=$HOME/orcd/scratch/cache/pixi  # ORCD Engaging; elsewhere pixi's default cache
export CONDA_OVERRIDE_CUDA=12.9
cd "$SLURM_SUBMIT_DIR"
ARGS=("$@")
[ ${#ARGS[@]} -eq 0 ] && ARGS=(--n-episodes 4 --out output/kitchen_demos_smoke.zarr)
pixi run -e port python experiments/021_record_expert.py "${ARGS[@]}"
