#!/bin/bash
# Install the three pixi environments needed on the ORCD Engaging cluster
# (see TRANSFER_README.md §1). Runs on a CPU node to keep the login node light.
# CONDA_OVERRIDE_CUDA satisfies the linux-cuda platform's __cuda virtual
# package on a node without GPUs.
#
#   cd $PROJECT_ROOT && sbatch experiments/010_setup_pixi_envs.sbatch

#SBATCH --job-name=pixi_setup
#SBATCH --output=output/logs/%x_%j.log
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -euo pipefail
export PATH="$HOME/.pixi/bin:$PATH"
[ -d "$HOME/orcd/scratch" ] && export PIXI_CACHE_DIR=$HOME/orcd/scratch/cache/pixi  # ORCD Engaging; elsewhere pixi's default cache
export CONDA_OVERRIDE_CUDA=12.9

echo "job $SLURM_JOB_ID on $(hostname)"
for env in franka-kitchen trace-anything diffusion-policy-l40; do
    echo "=== pixi install -e $env ($(date))"
    time pixi install -e "$env"
done
echo "finished at $(date)"
