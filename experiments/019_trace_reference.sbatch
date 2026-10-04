#!/bin/bash
# Original TraceAnything tokens (trace-anything env), then the ported-vs-original test (port env).
#SBATCH --job-name=trace_ref
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
export PATH="$HOME/.pixi/bin:$PATH"
[ -d "$HOME/orcd/scratch" ] && export PIXI_CACHE_DIR=$HOME/orcd/scratch/cache/pixi  # ORCD Engaging; elsewhere pixi's default cache
export CONDA_OVERRIDE_CUDA=12.9
cd "$SLURM_SUBMIT_DIR"
pixi run -e trace-anything python experiments/019_trace_reference_legacy.py
pixi run -e port pytest -q tests/test_trace_anything.py
