#!/bin/bash
# Small smoke test of the Markovian scripted-expert data generation pipeline
# before committing to the full run.
#
# Why not just run the full default (581 episodes) directly:
#   1. record_demos.py's _ZarrEpisodeWriter has NO resume/checkpoint support --
#      __init__ does `if out_path.exists(): shutil.rmtree(out_path)` and
#      meta/episode_ends is only written once at the very end in finalize().
#      If a `low` qos job gets preempted mid-run, the whole run is lost, not
#      resumable.
#   2. Our team's high_v1 (l40) quota is fully used right now (4/4), so this
#      has to run under `low` (preemptible) or wait.
#   3. We've only verified rendering works (one MjRenderContextOffscreen frame
#      in 002_fk_env_check.sbatch); the actual expert/controller rollout logic
#      (experts/expert.py, experts/controller.py) hasn't been exercised yet.
#
# So: run just 8 episodes here to get real per-episode timing + confirm the
# expert controller itself works end-to-end, then size the full run's --time
# from that instead of guessing. Output goes to output/kitchen_demos_smoke.zarr
# (separate from the real output/kitchen_demos_markovian_scripted_expert.zarr).
#
# Submit from the project root:
#   cd /home/minjunsz/codes/1_pai/markovian-policy && sbatch experiments/003_fk_record_demos_smoke.sbatch

#SBATCH --job-name=fk_demos_smoke
#SBATCH --output=output/logs/%x_%j.log
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40:1
#SBATCH --time=00:20:00

set -euo pipefail

# Don't rely on sbatch inheriting an interactive shell's PATH.
export PATH="$HOME/.pixi/bin:$PATH"

mkdir -p output/logs

echo "job $SLURM_JOB_ID running as $USER on $(hostname)"
echo "qos=$SLURM_JOB_QOS gpus_on_node=$SLURM_GPUS_ON_NODE"
nvidia-smi --query-gpu=index,name,compute_cap,driver_version --format=csv

time pixi run -e franka-kitchen fk-record-demos \
  --n-episodes 8 --out ../../../output/kitchen_demos_smoke.zarr

echo "finished at $(date)"
