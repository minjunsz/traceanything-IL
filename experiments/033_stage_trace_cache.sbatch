#!/bin/bash
# Copy TraceAnything token caches to the local disk of the node that serves /scratch (gpu8), where training then reads
# them without going through NFS, and benchmark random-row reads before and after (rows/s of one camera's cache;
# training needs about 64 x steps/s rows per camera). Training that uses the copies must run on gpu8 (--nodelist=gpu8).
# Resumable at file level: a finished copy is kept, a partial one is redone.
#
#   sbatch experiments/033_stage_trace_cache.sbatch            # CACHES="human_obs16_scene human_obs16_wrist"
#   CACHES="..." DEST_ROOT=... ZARR=... N_OBS=16 HORIZON=30 sbatch ...
#SBATCH --job-name=stage_cache
#SBATCH --output=output/logs/%x_%j.log
#SBATCH --open-mode=append
#SBATCH --nodelist=gpu8
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --requeue

set -euo pipefail
PIXI_ENV=port
source scripts/slurm_env.sh
CACHES="${CACHES:-human_obs16_scene human_obs16_wrist}"
DEST_ROOT="${DEST_ROOT:-/scratch/$USER/traceanything-IL/trace_cache}"
ZARR="${ZARR:-data/human.zarr}"
N_OBS="${N_OBS:-16}"
HORIZON="${HORIZON:-30}"
echo "job $SLURM_JOB_ID on $(hostname), restart ${SLURM_RESTART_COUNT:-0}: $CACHES -> $DEST_ROOT"
mkdir -p "$DEST_ROOT"

bench() {  # $1 = cache dir; the camera name is only used by the builder's config, rows are read from the files
    local camera="${1##*_}"
    pxrun python experiments/022_cache_trace_tokens.py bench --out-dir "$1" --zarr-path "$ZARR" --camera "$camera" \
        --n-obs-steps "$N_OBS" --horizon "$HORIZON"
}

for name in $CACHES; do
    src="data/trace_cache/$name"
    dest="$DEST_ROOT/$name"
    [ -f "$src/COMPLETE" ] || { echo "$src is not a complete cache" >&2; exit 1; }
    echo "== $name: source read benchmark"; bench "$src"
    if [ -f "$dest/COMPLETE" ]; then
        echo "$dest already copied"
    else
        rm -rf "$dest.tmp"; mkdir -p "$dest.tmp"
        echo "copying $src ($(du -sh "$src" | cut -f1)) at $(date)"
        cp -r "$src/." "$dest.tmp/"
        rm -rf "$dest"; mv "$dest.tmp" "$dest"   # COMPLETE is part of the copy: its presence marks a finished one
        echo "copied at $(date)"
    fi
    echo "== $name: copy read benchmark"; bench "$dest"
done
