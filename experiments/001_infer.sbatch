#!/bin/bash
# Run TraceAnything inference on the example scenes (third_party/TraceAnything/examples/input)
# on a free/preemptible L40.
#
# Submit from the project root:
#   cd /home/minjunsz/codes/1_pai/markovian-policy && sbatch experiments/001_infer.sbatch
#
# `low` qos is preemptible -- if this gets requeued, infer.py just reruns from
# scratch (it's a single short forward pass per scene, not worth checkpointing).

#SBATCH --job-name=ta_infer
#SBATCH --output=output/logs/%x_%j.log
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40:1
#SBATCH --time=00:30:00

set -euo pipefail

# Don't rely on sbatch inheriting an interactive shell's PATH.
export PATH="$HOME/.pixi/bin:$PATH"

mkdir -p output/logs

echo "job $SLURM_JOB_ID running as $USER on $(hostname)"
echo "qos=$SLURM_JOB_QOS gpus_on_node=$SLURM_GPUS_ON_NODE"
nvidia-smi --query-gpu=index,name,compute_cap,driver_version --format=csv

pixi run -e trace-anything ta-infer

echo "finished at $(date)"
