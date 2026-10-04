#!/bin/bash
# Install the dp-kitchen-eval pixi environment (diffusion-policy + kitchen sim
# in one Python 3.10 env, for third_party/relay-policy-learning/eval). CPU only:
# mujoco_py's extension is compiled on first import, which must happen on a GPU
# node (EGL path) -- see experiments/014_eval_kitchen.sbatch.
#
#   cd $PROJECT_ROOT && sbatch experiments/013_setup_eval_env.sbatch

#SBATCH --job-name=pixi_eval_setup
#SBATCH --output=output/logs/%x_%j.log
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail
export PATH="$HOME/.pixi/bin:$PATH"
[ -d "$HOME/orcd/scratch" ] && export PIXI_CACHE_DIR=$HOME/orcd/scratch/cache/pixi  # ORCD Engaging; elsewhere pixi's default cache
export CONDA_OVERRIDE_CUDA=12.9

echo "job $SLURM_JOB_ID on $(hostname)"
time pixi install -e dp-kitchen-eval
pixi run -e dp-kitchen-eval python -c "import gym, dm_control, mujoco, torch, diffusers; print('imports ok', torch.__version__)"
echo "finished at $(date)"
