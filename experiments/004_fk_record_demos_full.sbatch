#!/bin/bash
# Full Markovian scripted-expert data generation run (default 581 episodes,
# chain-len 4, seed 0, 4 parallel workers -- see pixi.toml's fk-record-demos
# task). Output: output/kitchen_demos_markovian_scripted_expert.zarr
#
# Sized from the 8-episode smoke test (job 37769):
#   - 8 episodes / 9 attempts in 62.7s wall (4 workers) -> ~70 min extrapolated
#     for 581 episodes (~611 attempts at the observed 88.9% success rate)
#   - ~175 KB/step on disk (scene+wrist cameras dominate), 447.8 steps/episode
#     avg -> ~43-45 GB total, mostly image data
# --time=4:00:00 leaves a generous margin over the ~70min estimate for NFS
# write variance (HDD-backed, per CLAUDE.md) and per-run variance from a
# small (n=8) sample.
#
# Risk: record_demos.py's _ZarrEpisodeWriter has NO resume/checkpoint support
# (wipes existing output on start, meta/episode_ends only written at the very
# end) -- if this `low` qos job gets preempted, the whole run is lost and
# must restart from episode 0. Team high_v1 (l40) quota is fully used right
# now (4/4) so `low` is the only way to start immediately; 12 l40s are idle
# cluster-wide as of this writing, so preemption risk is real but not high.
#
# Submit from the project root:
#   cd /home/minjunsz/codes/1_pai/markovian-policy && sbatch experiments/004_fk_record_demos_full.sbatch

#SBATCH --job-name=fk_demos_full
#SBATCH --output=output/logs/%x_%j.log
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40:1
#SBATCH --time=04:00:00

set -euo pipefail

# Don't rely on sbatch inheriting an interactive shell's PATH.
export PATH="$HOME/.pixi/bin:$PATH"

mkdir -p output/logs

echo "job $SLURM_JOB_ID running as $USER on $(hostname)"
echo "qos=$SLURM_JOB_QOS gpus_on_node=$SLURM_GPUS_ON_NODE"
nvidia-smi --query-gpu=index,name,compute_cap,driver_version --format=csv

time pixi run -e franka-kitchen fk-record-demos

echo "finished at $(date)"
