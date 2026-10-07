#!/bin/bash
# CPU job: decompress the camera frames of a zarr dataset into .npy files (see 032_export_frames.py).
#   sbatch experiments/032_export_frames.sbatch --zarr-path data/human.zarr
#SBATCH --job-name=export_frames
#SBATCH --output=output/logs/%x_%j.log
#SBATCH --open-mode=append
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --requeue

set -euo pipefail
PIXI_ENV=port
source scripts/slurm_env.sh
echo "job $SLURM_JOB_ID on $(hostname), restart ${SLURM_RESTART_COUNT:-0}"
pyrun experiments/032_export_frames.py "$@"
