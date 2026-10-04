#!/bin/bash
# Train the low-dim diffusion policy (experiments/009_train_diffusion_policy_lowdim.py)
# on the new `dpv2` pixi environment (recent torch/diffusers/transformers/accelerate,
# no robomimic/r3m/Hydra) -- see
# /home/minjunsz/.claude/plans/diffusion-policy-unified-donut.md for context. This is
# the first real GPU validation of the new src/diffusion_policy training stack, so it
# defaults to a short `low`-qos smoke run against the small smoke zarr, not a full
# training job.
#
# Submit from the project root:
#   cd /home/minjunsz/codes/1_pai/markovian-policy && sbatch experiments/009_train_diffusion_policy_lowdim.sbatch
#
# For a real training run (full dataset, many more steps), override via --export and
# switch to the team's L40 pool, e.g.:
#   sbatch --qos=high_v --gres=gpu:l40:1 \
#     --export=ALL,ZARR_PATH=output/kitchen_demos_markovian_scripted_expert.zarr,NUM_TRAIN_STEPS=50000 \
#     experiments/009_train_diffusion_policy_lowdim.sbatch
#
# Resume: this script always passes the same --output-dir, and Trainer
# (src/diffusion_policy/training/loop.py) auto-resumes from
# <output-dir>/latest.ckpt if present -- re-submitting this same script continues
# training, whether it stopped due to preemption (--requeue resubmits automatically)
# or hitting the walltime.
#
# GPU type: L40 only (`high_v` qos, up to 8x). Do NOT use pro6000, even if
# allocated/visible -- project policy as of 2026-09-07, independent of quota
# (there was also an unresolved SDPA CUDA bug on pro6000 under the old
# dependency stack; see experiments/006_dp_check_v2.sbatch history).

#SBATCH --job-name=dp_train_lowdim
#SBATCH --output=output/logs/%x_%j.log
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40:1
#SBATCH --time=00:30:00
#SBATCH --requeue

set -euo pipefail

export PATH="$HOME/.pixi/bin:$PATH"

ZARR_PATH="${ZARR_PATH:-output/kitchen_demos_smoke.zarr}"
OUTPUT_DIR="${OUTPUT_DIR:-output/checkpoints/diffusion_policy_lowdim}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-200}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"

mkdir -p output/logs "$OUTPUT_DIR"

echo "job $SLURM_JOB_ID running as $USER on $(hostname)"
echo "qos=$SLURM_JOB_QOS gpus_on_node=$SLURM_GPUS_ON_NODE"
nvidia-smi --query-gpu=index,name,compute_cap,driver_version --format=csv

pixi run -e dpv2 python experiments/009_train_diffusion_policy_lowdim.py \
  --zarr-path "$ZARR_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --num-train-steps "$NUM_TRAIN_STEPS" \
  --batch-size "$BATCH_SIZE" \
  --mixed-precision "$MIXED_PRECISION" \
  --checkpoint-every-steps 50 \
  --val-every-steps 50

echo "finished at $(date)"
