#!/bin/bash
# Build the TraceAnything token cache for a recorded dataset, one encoder process per GPU. Resumable: rerun the
# same command after a timeout/preemption (finished episodes are skipped), then it verifies and writes COMPLETE.
#
#   ZARR=data/kitchen_demos_expert.zarr sbatch experiments/022_cache_trace_tokens.sbatch
#   (optional: CAMERA=scene|wrist, CACHE_DIR=..., WORLD=<total ranks> RANK_OFFSET=<first rank> to split over jobs)
#SBATCH --job-name=trace_cache
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
export PATH="$HOME/.pixi/bin:$PATH"
[ -d "$HOME/orcd/scratch" ] && export PIXI_CACHE_DIR=$HOME/orcd/scratch/cache/pixi  # ORCD Engaging; elsewhere pixi's default cache
export CONDA_OVERRIDE_CUDA=12.9
cd "$SLURM_SUBMIT_DIR"

ZARR="${ZARR:-data/kitchen_demos_expert.zarr}"
CAMERA="${CAMERA:-scene}"
CACHE_DIR="${CACHE_DIR:-data/trace_cache/$(basename "$ZARR" .zarr)_obs8_$CAMERA}"
PY=(pixi run -e port python experiments/022_cache_trace_tokens.py)
COMMON=(--out-dir "$CACHE_DIR" --zarr-path "$ZARR" --camera "$CAMERA")

echo "job $SLURM_JOB_ID on $(hostname): ZARR=$ZARR CAMERA=$CAMERA CACHE_DIR=$CACHE_DIR"
[ -f "$CACHE_DIR/meta.json" ] || "${PY[@]}" init "${COMMON[@]}"

NGPU=$SLURM_GPUS_ON_NODE
WORLD=${WORLD:-$NGPU}
RANK_OFFSET=${RANK_OFFSET:-0}
pids=()
for g in $(seq 0 $((NGPU - 1))); do
    r=$((RANK_OFFSET + g))
    CUDA_VISIBLE_DEVICES=$g "${PY[@]}" encode "${COMMON[@]}" --rank "$r" --world "$WORLD" \
        >> "logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}-rank${r}.log" 2>&1 &
    pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done

"${PY[@]}" verify "${COMMON[@]}"
