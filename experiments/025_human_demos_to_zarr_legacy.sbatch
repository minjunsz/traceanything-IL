#!/bin/bash
# Relay Policy Learning human teleop demos -> zarr, same layout as the
# Markovian-expert zarr (state/action/scene/wrist + subtask_sequence):
#   1. unzip kitchen_demos_multitask.zip (608 .mjl logs, 25 4-subtask plans)
#   2. experiments/025_prepare_human_demos_legacy.py clean: symlink only unique,
#      parseable logs (the zip has byte-identical "(1)" copies and a truncated log)
#   3. parse_demos.py --view playback --skip 40: physics replay at 12.5 Hz,
#      rendering scene + wrist cameras (EGL, hence the GPU), one process per plan folder
#   4. experiments/025_prepare_human_demos_legacy.py filter: keep replays that complete
#      all 4 planned subtasks, add the subtask_sequence one-hot
#   5. process_pickles.py -> $OUT_ZARR
#
#   cd $PROJECT_ROOT && sbatch experiments/025_human_demos_to_zarr_legacy.sbatch

#SBATCH --job-name=human_demos
#SBATCH --output=output/logs/%x_%j.log
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00

set -euo pipefail
export PATH="$HOME/.pixi/bin:$PATH"

RAW_DIR="${RAW_DIR:-$HOME/orcd/pool/markovian-policy/raw}"
UNZIPPED="$RAW_DIR/kitchen_demos_multitask"
DEMO_ROOT="$RAW_DIR/kitchen_demos_multitask_clean"
FILTERED_DIR="${FILTERED_DIR:-$RAW_DIR/kitchen_demos_multitask_filtered}"
OUT_ZARR="${OUT_ZARR:-$HOME/orcd/pool/markovian-policy/data/kitchen_demos_human.zarr}"
N_PARALLEL="${N_PARALLEL:-12}"
RPL=third_party/relay-policy-learning
LOG_DIR="output/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "$LOG_DIR" "$(dirname "$OUT_ZARR")"

echo "job $SLURM_JOB_ID on $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

if [ ! -d "$UNZIPPED" ]; then
    unzip -q "$RAW_DIR/kitchen_demos_multitask.zip" -d "$RAW_DIR"
fi
pixi run -e franka-kitchen python experiments/025_prepare_human_demos_legacy.py clean \
    --demo-root "$UNZIPPED" --out-dir "$DEMO_ROOT"
echo "$(find "$DEMO_ROOT" -maxdepth 2 -name '*.mjl' | wc -l) usable mjl logs in $(ls -d "$DEMO_ROOT"/*/ | wc -l) plan folders"

# Replay every plan folder in parallel (each writes <log>_path.pkl next to its .mjl).
export RPL LOG_DIR
ls -d "$DEMO_ROOT"/*/ | xargs -P "$N_PARALLEL" -I{} bash -c '
    name=$(basename {})
    pixi run -e franka-kitchen python $RPL/adept_envs/adept_envs/utils/parse_demos.py \
        --env kitchen_relax-v1 --demo_dir {} --view playback --skip 40 --render None \
        > "$LOG_DIR/$name.log" 2>&1 || echo "parse failed: $name"
'
echo "$(find "$DEMO_ROOT" -maxdepth 2 -name '*_path.pkl' | wc -l) replayed pkls"

pixi run -e franka-kitchen python experiments/025_prepare_human_demos_legacy.py filter \
    --demo-root "$DEMO_ROOT" --out-dir "$FILTERED_DIR"

pixi run -e franka-kitchen python $RPL/adept_envs/adept_envs/utils/process_pickles.py \
    --input_dir "$FILTERED_DIR" --output "$OUT_ZARR"

echo "finished at $(date)"
