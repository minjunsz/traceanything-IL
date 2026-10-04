#!/bin/bash
# Fork policy reference (fork env) then the src-vs-fork equivalence tests (port env). CPU only.
#   sbatch experiments/024_policy_equivalence.sbatch [--ckpt PATH]
#SBATCH --job-name=policy_equiv
#SBATCH --partition=mit_quicktest
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
export PATH="$HOME/.pixi/bin:$PATH"
[ -d "$HOME/orcd/scratch" ] && export PIXI_CACHE_DIR=$HOME/orcd/scratch/cache/pixi  # ORCD Engaging; elsewhere pixi's default cache
export CONDA_OVERRIDE_CUDA=12.9
export OMP_NUM_THREADS=8
cd "$SLURM_SUBMIT_DIR"
pixi run -e diffusion-policy-l40 python experiments/024_policy_reference_legacy.py "$@"
pixi run -e port pytest -q tests/test_policy_equivalence.py
