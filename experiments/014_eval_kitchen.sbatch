#!/bin/bash
# Closed-loop Franka Kitchen evaluation of a diffusion-policy checkpoint with
# third_party/relay-policy-learning/eval/evaluate_kitchen.py, in the
# dp-kitchen-eval pixi env (experiments/013_setup_eval_env.sbatch).
#
# Success = >= N_REQUIRED distinct kitchen subtasks achieved before the
# timeout; the policy is conditioned on a random 4-subtask plan per trial
# (subtask_sequence), like the recorded demos. TASK_TIMEOUT defaults to 600
# steps, not the script's D4RL 280: the scripted-expert demos average ~460
# steps for a 4-subtask chain, so 280 would cap success well below the data.
#
#   cd $PROJECT_ROOT && sbatch --export=ALL,CKPT=<path/to/.ckpt> experiments/014_eval_kitchen.sbatch
# Optional: N_ROLLOUTS (50), N_ENVS (10), TASK_TIMEOUT (600), N_REQUIRED (4),
#           N_VIDEOS (10), OUT_DIR (output/eval/<run>/<ckpt stem>), EXTRA_ARGS.

#SBATCH --job-name=dp_eval
#SBATCH --output=output/logs/%x_%j.log
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=03:00:00

set -euo pipefail
export PATH="$HOME/.pixi/bin:$PATH"

if [ -z "${CKPT:-}" ]; then
    echo "ERROR: set CKPT=<path/to/checkpoint.ckpt>" >&2
    exit 1
fi
CKPT="$(realpath "$CKPT")"
RUN_NAME="$(basename "$(dirname "$(dirname "$CKPT")")")"
CKPT_STEM="$(basename "$CKPT" .ckpt)"
OUT_DIR="${OUT_DIR:-$PWD/output/eval/$RUN_NAME/$CKPT_STEM}"
mkdir -p output/logs "$OUT_DIR"

echo "job $SLURM_JOB_ID on $(hostname)"
echo "CKPT=$CKPT"
echo "OUT_DIR=$OUT_DIR"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

RESUME_ARG=()
if [ -f "$OUT_DIR/results.pkl" ]; then
    RESUME_ARG=(--resume)
fi

pixi run -e dp-kitchen-eval python third_party/relay-policy-learning/eval/evaluate_kitchen.py \
    --checkpoint "$CKPT" \
    --headless \
    --n-rollouts "${N_ROLLOUTS:-50}" \
    --n-envs "${N_ENVS:-10}" \
    --task-timeout "${TASK_TIMEOUT:-600}" \
    --n-required-subtasks "${N_REQUIRED:-4}" \
    --n-video-trials "${N_VIDEOS:-10}" \
    --output-dir "$OUT_DIR" \
    "${RESUME_ARG[@]}" \
    ${EXTRA_ARGS:-}

echo "finished at $(date)"
