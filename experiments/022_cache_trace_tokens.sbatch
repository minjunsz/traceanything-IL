#!/bin/bash
# Build the TraceAnything token cache for a recorded dataset, one encoder process per GPU (2 GPUs: scene and wrist caches then fit the 4-GPU l40 budget together). Resumable: preemption
# (requeued) or a timeout (resubmit) just continues: finished episodes are skipped; the last step
# verifies and writes COMPLETE. A cache is specific to (dataset, camera, n_obs_steps), the policy's history length.
#
#   ZARR=data/human.zarr CAMERA=scene N_OBS=16 HORIZON=30 sbatch experiments/022_cache_trace_tokens.sbatch
#   (optional: CACHE_DIR=..., WORLD=<total ranks> RANK_OFFSET=<first rank> to split over jobs; N_OBS=8 HORIZON=22 default)
#SBATCH --job-name=trace_cache
#SBATCH --output=output/logs/%x_%j.log
#SBATCH --open-mode=append
#SBATCH --qos=high_v
#SBATCH --gres=gpu:l40:2
#SBATCH --time=24:00:00
#SBATCH --requeue

set -euo pipefail
PIXI_ENV=port
source scripts/slurm_env.sh

ZARR="${ZARR:-data/kitchen_demos_expert.zarr}"
CAMERA="${CAMERA:-scene}"
N_OBS="${N_OBS:-8}"
HORIZON="${HORIZON:-22}"  # n_obs_steps + 14
CACHE_DIR="${CACHE_DIR:-data/trace_cache/$(basename "$ZARR" .zarr)_obs${N_OBS}_$CAMERA}"
PY=(pxrun python experiments/022_cache_trace_tokens.py)
COMMON=(--out-dir "$CACHE_DIR" --zarr-path "$ZARR" --camera "$CAMERA" --n-obs-steps "$N_OBS" --horizon "$HORIZON")

echo "job $SLURM_JOB_ID on $(hostname), restart ${SLURM_RESTART_COUNT:-0}: ZARR=$ZARR CAMERA=$CAMERA N_OBS=$N_OBS CACHE_DIR=$CACHE_DIR"
[ -f "$CACHE_DIR/meta.json" ] || "${PY[@]}" init "${COMMON[@]}"

NGPU=$SLURM_GPUS_ON_NODE
WORLD=${WORLD:-$NGPU}
RANK_OFFSET=${RANK_OFFSET:-0}
pids=()
for g in $(seq 0 $((NGPU - 1))); do
    r=$((RANK_OFFSET + g))
    CUDA_VISIBLE_DEVICES=$g "${PY[@]}" encode "${COMMON[@]}" --rank "$r" --world "$WORLD" \
        >> "output/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_rank${r}.log" 2>&1 &
    pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done

"${PY[@]}" verify "${COMMON[@]}"
