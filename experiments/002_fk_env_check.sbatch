#!/bin/bash
# Verify the `franka-kitchen` pixi env can actually render (mujoco-py's cymj
# extension needs a real GPU to pick the GPU/EGL rendering backend on this
# cluster -- conda-forge's mesalib dropped classic OSMesa, so the CPU-only
# path is not available to us; see pixi.toml's franka-kitchen comment).
#
# Submit from the project root:
#   cd /home/minjunsz/codes/1_pai/markovian-policy && sbatch experiments/002_fk_env_check.sbatch
#
# First run on a GPU node triggers mujoco-py's one-time cymj extension build
# (compiles gl/eglshim.c against MuJoCo 2.1.0 + EGL) -- should take well under
# a minute; the .so is then cached in site-packages for later runs.
#
# Job 37760 (first attempt) failed here: mujoco_py's GPU/EGL vs CPU/OSMesa
# picker only trusts nvidia-smi + a hardcoded path (/usr/lib/nvidia etc, a
# Docker-era assumption); confirmed /usr/lib/nvidia doesn't exist on gpu3 even
# with a real L40S + driver present, so it silently fell back to the
# (unusable here) CPU/OSMesa path. Patched via
# third_party/patch_mujoco_py_gpu_detect.py, auto-applied on every
# `franka-kitchen` activation -- this run re-validates that fix for real.

#SBATCH --job-name=fk_env_check
#SBATCH --output=output/logs/%x_%j.log
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40:1
#SBATCH --time=00:10:00

set -euo pipefail

# Don't rely on sbatch inheriting an interactive shell's PATH.
export PATH="$HOME/.pixi/bin:$PATH"

mkdir -p output/logs

echo "job $SLURM_JOB_ID running as $USER on $(hostname)"
echo "qos=$SLURM_JOB_QOS gpus_on_node=$SLURM_GPUS_ON_NODE"
nvidia-smi --query-gpu=index,name,compute_cap,driver_version --format=csv

pixi run -e franka-kitchen fk-diagnose

echo "finished at $(date)"
