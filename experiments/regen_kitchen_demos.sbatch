#!/bin/bash
# Regenerate output/kitchen_demos_markovian_scripted_expert.zarr on a NEW
# cluster (this file is not present on the source machine -- written for the
# transfer bundle, see ../TRANSFER_README.md). Same underlying task as the
# source project's experiments/004_fk_record_demos_full.sbatch
# (pixi's `fk-record-demos` task: --randomize --chain-len 4 --seed 0
# --n-workers 4), deterministic given the seed, ~47min on 1 GPU / ~45GB output.
#
# ORCD Engaging: mit_normal_gpu (<=6h), 1x L40S. --n-workers 4 in the pixi
# task -> 8 CPUs. Output (~45GB zarr) goes to output/ under $HOME (200G quota).
#
# Submit from the project root (this bundle's root, wherever you scp'd it to):
#   cd $PROJECT_ROOT && sbatch experiments/regen_kitchen_demos.sbatch

#SBATCH --job-name=fk_demos_full
#SBATCH --output=output/logs/%x_%j.log
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
# mem: job 24107029 peaked at the old 48G limit, so 64G for headroom.
#SBATCH --mem=64G
#SBATCH --time=03:00:00

set -euo pipefail

export PATH="$HOME/.pixi/bin:$PATH"

mkdir -p output/logs

echo "job $SLURM_JOB_ID running as $USER on $(hostname)"
echo "qos=$SLURM_JOB_QOS gpus_on_node=$SLURM_GPUS_ON_NODE"
nvidia-smi --query-gpu=index,name,compute_cap,driver_version --format=csv

time pixi run -e franka-kitchen fk-record-demos

echo "finished at $(date)"
