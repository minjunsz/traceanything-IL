#!/bin/bash
# Submit one eval job per (checkpoint, execution horizon):  output/eval/<run>/T_a_<H>/<checkpoint stem>/
# Usage: experiments/026_eval_sweep.sh RUN_DIR [H_MIN H_MAX]   (RUN_DIR contains checkpoints/*.ckpt)
# Prints the submissions; with 8 horizons x k checkpoints this is 8k single-GPU jobs, so check k first.
set -euo pipefail
run_dir=$(realpath "$1"); h_min=${2:-1}; h_max=${3:-8}
run=$(basename "$run_dir")
for ckpt in "$run_dir"/checkpoints/epoch=*.ckpt; do
    stem=$(basename "$ckpt" .ckpt)
    for h in $(seq "$h_min" "$h_max"); do
        sbatch --job-name="eval_${run}_T${h}" experiments/026_eval.sbatch --checkpoint "$ckpt" \
            --out-dir "output/eval/$run/T_a_$h/$stem" --rollout.n-action-steps "$h"
    done
done
