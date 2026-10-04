#!/bin/bash
# Precompute TraceAnything window tokens for training (see
# experiments/015_cache_trace_tokens.py): init (layout check + empty memmap,
# skipped if already initialized) -> one encode process per GPU -> verify.
# Resumable across preemption/requeue: finished episodes are recorded in
# $CACHE_DIR/progress/ and skipped.
#
#   cd $PROJECT_ROOT && sbatch experiments/015_cache_trace_tokens.sbatch
# mit_normal_gpu caps jobs at 6h and 2 GPUs; the full cache takes longer on
# 2x L40S, so chain a few submissions (each resumes where the last stopped):
#   j=$(sbatch --parsable experiments/015_cache_trace_tokens.sbatch)
#   sbatch --dependency=afterany:$j experiments/015_cache_trace_tokens.sbatch
# Then train with TRACE_CACHE_DIR=$CACHE_DIR (experiments/008_train_diffusion_policy.sbatch).

#SBATCH --job-name=trace_cache
#SBATCH --output=output/logs/%x_%j.log
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00

set -euo pipefail
export PATH="$HOME/.pixi/bin:$PATH"
# ZARR / CAMERA select the dataset and rgb key to encode (one cache per camera,
# ~1.5 MiB per window, e.g. ~400 GiB for the Markovian-expert zarr). Caches are
# regenerable, so they default to scratch (fast parallel I/O, not backed up).
# The first cache (Markovian zarr, scene) lives in pool at
# ~/orcd/pool/markovian-policy/trace_cache/kitchen_markovian_expert_obs8 --
# pass CACHE_DIR explicitly to reuse it.
ZARR="${ZARR:-output/kitchen_demos_markovian_scripted_expert.zarr}"
CAMERA="${CAMERA:-scene}"
CACHE_DIR="${CACHE_DIR:-data/trace_cache/$(basename "$ZARR" .zarr)_obs8_$CAMERA}"
PY=(pixi run -e diffusion-policy-l40 python experiments/015_cache_trace_tokens.py)
COMMON=(--out-dir "$CACHE_DIR" --zarr "$ZARR" --scene-key "$CAMERA")

echo "job $SLURM_JOB_ID (restart ${SLURM_RESTART_COUNT:-0}) on $(hostname), ZARR=$ZARR CAMERA=$CAMERA CACHE_DIR=$CACHE_DIR"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

if [ ! -f "$CACHE_DIR/meta.json" ]; then
    "${PY[@]}" init "${COMMON[@]}"
fi

NGPU=$SLURM_GPUS_ON_NODE
# To split the work across concurrent jobs, give each job a disjoint rank range
# of one global WORLD, e.g. a 2-GPU job with RANK_OFFSET=0,WORLD=6 and a 4-GPU
# job with RANK_OFFSET=2,WORLD=6 (episode e goes to rank e % WORLD).
RANK_OFFSET=${RANK_OFFSET:-0}
WORLD=${WORLD:-$NGPU}
pids=()
for g in $(seq 0 $((NGPU - 1))); do
    r=$((RANK_OFFSET + g))
    CUDA_VISIBLE_DEVICES=$g "${PY[@]}" encode "${COMMON[@]}" --rank "$r" --world "$WORLD" \
        >> "output/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_rank${r}.log" 2>&1 &
    pids+=($!)
done
for pid in "${pids[@]}"; do
    wait "$pid"
done

CUDA_VISIBLE_DEVICES=0 "${PY[@]}" verify "${COMMON[@]}"
echo "finished at $(date)"
