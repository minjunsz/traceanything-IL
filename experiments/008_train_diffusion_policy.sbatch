#!/bin/bash
# Train a diffusion policy on the Markovian scripted-expert kitchen dataset,
# either conditioned on a TraceAnything window latent (our variant) or the
# paper's per-frame R3M tokens (baseline) -- see
# third_party/diffusion-policy-experiments/config/franka_kitchen/*_markovian_expert.yaml.
#
# Submit from the project root, picking ONE config via CONFIG_NAME:
#   cd /home/minjunsz/codes/markovian-policy
#   sbatch --export=CONFIG_NAME=8_obs_trace_anything_markovian_expert experiments/008_train_diffusion_policy.sbatch
#   sbatch --export=CONFIG_NAME=16_obs_trace_anything_markovian_expert experiments/008_train_diffusion_policy.sbatch
#   sbatch --export=CONFIG_NAME=8_obs_attention_markovian_expert experiments/008_train_diffusion_policy.sbatch
#   sbatch --export=CONFIG_NAME=16_obs_attention_markovian_expert experiments/008_train_diffusion_policy.sbatch
#
# Checkpointing: training.checkpoint_every=1 (epoch) in the yaml. Measured
# ~5.4s/it on 4x L40 (job 37920) -> ~6.1h/epoch, 73 epochs to reach the yaml's
# total_train_steps=300000 (~18.5 days) -- checkpoint_every=5 (this repo's own
# default) would've meant ~30h to the first checkpoint, longer than `low`
# qos's 1-day MaxWall. RESUME=True + the latest.ckpt detection below means
# re-submitting this same script (same CONFIG_NAME) continues training from
# where it left off, whether it stopped because of preemption (--requeue also
# resubmits automatically on that) or because it simply hit the walltime.
#
# Optional overrides via --export:
#   NUM_EPOCHS=3    -- quick smoke test instead of the full 73-epoch run
#                      (implies training.total_train_steps=null, since that
#                      would otherwise override any num_epochs back to 73)
#   BATCH_SIZE=128   -- global batch size (default 64 in the yaml); job 37920
#                      used only ~24GB/48GB per L40 at batch=64, so there's
#                      headroom to try larger -- watch for OOM and back off
#                      if it happens.
#   PIXI_ENV=diffusion-policy-pro6000 -- MUST match whatever --gres GPU type
#                      you edit below (default env is back to
#                      diffusion-policy-l40, default --gres is l40:4, per L40
#                      capacity being available again). pro6000 is currently
#                      NOT usable for real training regardless of capacity:
#                      job 38940/38950 both hit a CUDA "illegal memory access"
#                      inside the first training step's SDPA call on torch
#                      2.7.1+cu128/Blackwell (sm_120) -- reproduced under both
#                      the FLASH_ATTENTION and EFFICIENT_ATTENTION backends
#                      (see third_party/TraceAnything/configs/eval.yaml's
#                      attn_implementation), so it isn't just a backend
#                      choice -- root cause still unresolved. Two separate
#                      pixi environments exist because they need different
#                      torch/cudnn builds -- see pixi.toml's
#                      feature.diffusion-policy comment for why (torch
#                      2.3.1/cu121 has no sm_120 kernels for pro6000 at all).
# e.g.:
#   sbatch --export=ALL,CONFIG_NAME=8_obs_trace_anything_markovian_expert,NUM_EPOCHS=3,BATCH_SIZE=128 \
#     experiments/008_train_diffusion_policy.sbatch
#
# pro6000: DO NOT USE (policy, as of 2026-09-07) -- pro6000 is off-limits for
# this project regardless of quota/allocation, on top of the still-unresolved
# SDPA illegal-memory-access bug above. L40 only from here on; up to 8x L40
# available under `high_v` (this account's whole quota, shared with
# `low`-qos jobs from other users on the same nodes).

#SBATCH --job-name=dp_train
#SBATCH --output=output/logs/%x_%j.log
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
# ORCD Engaging (2026-09-27): mit_normal_gpu caps at 6h, shorter than one
# ~6h epoch on 4x L40, so the long run goes to mit_preemptable (2 days,
# preemptible -> --requeue + latest.ckpt resume above; consider
# CHECKPOINT_FRAC so a preemption loses less than a full epoch). For a
# DEBUG smoke test, override on the command line, e.g.
#   sbatch -p mit_normal_gpu -t 00:30:00 --export=ALL,...,DEBUG=true ...
# cpus: 8 dataloader workers per rank x 4 ranks. mem: each rank copies the
# ~45GB zarr into a zarr.MemoryStore (base_dataset.py), so 4 ranks need
# ~180GB+ -- first guess, refine from `sacct -j <id> -o MaxRSS`.
# GPU type: l40s matches the L40 memory profile the batch-size notes below
# were measured on; h100/h200/a100 also exist here (compatibility is
# verified per type by experiments/011_dp_check.sbatch).

set -euo pipefail

# Confirmed (job 37915): `--export=ALL,CONFIG_NAME=...` alone is enough to
# inherit PATH -- the earlier "pixi: not found" was a missing ALL on the
# submit command, not sbatch failing to propagate the environment. Kept as a
# safety net anyway so a bare --export=CONFIG_NAME=... doesn't silently break.
export PATH="$HOME/.pixi/bin:$PATH"

if [ -z "${CONFIG_NAME:-}" ]; then
    echo "ERROR: set CONFIG_NAME, e.g.:" >&2
    echo '  sbatch --export=CONFIG_NAME=8_obs_trace_anything_markovian_expert experiments/008_train_diffusion_policy.sbatch' >&2
    exit 1
fi

mkdir -p output/logs

PIXI_ENV="${PIXI_ENV:-diffusion-policy-l40}"

echo "job $SLURM_JOB_ID running as $USER on $(hostname)"
echo "qos=$SLURM_JOB_QOS gpus_on_node=$SLURM_GPUS_ON_NODE"
echo "CONFIG_NAME=$CONFIG_NAME PIXI_ENV=$PIXI_ENV"
nvidia-smi --query-gpu=index,name,memory.total,compute_cap,driver_version --format=csv
# Per-GPU memory trace for batch-size tuning (killed with the job's cgroup).
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv -l 15 \
    > "output/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_gpumem.csv" &

cd third_party/diffusion-policy-experiments

export WANDB_MODE=disabled  # no wandb account configured on this cluster
export PYTHONPATH="$PWD:$PWD/../TraceAnything:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
export TORCH_LOGS="recompiles"
# Job 38218: NCCL ALLREDUCE timed out (2700000ms = 45min, our own raised
# limit) waiting on rank 0, which was still deep in its (rank-0-only, see
# InitProcessGroupKwargs comment in the workspace file) validation pass.
# Structurally that's consistent with validation genuinely taking that long
# under this workload (DDPM+DDIM MSE eval over the full val set) -- but
# NVIDIA has documented near-identical AllReduce hangs on shared/multi-GPU
# nodes fixed by disabling P2P (e.g. NVIDIA/nccl#1999, dual RTX PRO 6000).
# Worth ruling out before falling back to single-GPU (which sidesteps the
# whole rank-desync class of problem, but at roughly 4x less throughput).
export NCCL_P2P_DISABLE=1

NUM_GPUS=$SLURM_GPUS_ON_NODE
RUN_NAME="$CONFIG_NAME"
if [ "${DEBUG:-}" = "true" ]; then
    # Isolated from the real run dir so a debug checkpoint can never get
    # picked up as `latest.ckpt` by a real training submission.
    RUN_NAME="${CONFIG_NAME}_debug"
fi
if [ -n "${RUN_TAG:-}" ]; then
    # Separate run dir per probe (e.g. RUN_TAG=bs256), so it neither resumes
    # from nor clobbers another run's latest.ckpt.
    RUN_NAME="${RUN_NAME}_${RUN_TAG}"
fi
HYDRA_RUN_DIR="$PWD/data/outputs/franka_kitchen/${RUN_NAME}"
LATEST_CKPT="$HYDRA_RUN_DIR/checkpoints/latest.ckpt"

# Resume policy, same as upstream submit_training_cluster.sbatch:
#   latest.ckpt present (this run already made progress) -> resume from it.
#   otherwise -> fresh start (training.checkpoint_path stays null from the yaml).
CKPT_OVERRIDE=()
if [ -f "$LATEST_CKPT" ]; then
    echo "Found $LATEST_CKPT -> resuming"
    CKPT_OVERRIDE=(training.checkpoint_path=null)
else
    echo "No checkpoint found -> starting fresh"
fi

# Rendezvous port: previously a fixed `20000 + SLURM_JOB_ID % 20000`, which
# hit EADDRINUSE (job on pro6000/low qos) -- collided either with another
# job's torchrun on a shared node, or a leftover process from a preempted
# prior attempt of this same job that hadn't released the port yet. Since
# this script only ever runs single-node (--nodes=1), --standalone lets
# torchrun pick a free port itself instead of us guessing one.

EXTRA_OVERRIDES=()
if [ -n "${NUM_EPOCHS:-}" ]; then
    echo "NUM_EPOCHS=$NUM_EPOCHS (overriding total_train_steps=null so this isn't clobbered)"
    EXTRA_OVERRIDES+=(training.total_train_steps=null "training.num_epochs=$NUM_EPOCHS")
fi
if [ -z "${BATCH_SIZE:-}" ] && [ "$NUM_GPUS" -eq 1 ]; then
    # dataloader.batch_size is GLOBAL and only gets divided by GPU count when
    # accelerator.num_processes > 1 (see the workspace's dataloader setup) --
    # at 1 GPU the full yaml default (64) lands on that single GPU
    # unmodified. Job 37931 measured ~24GB/48GB per L40 at batch=16/GPU
    # during training (4-GPU run, global batch 64); 32/GPU also trained
    # cleanly (job 37931/37974, global batch 128, before their VALIDATION-side
    # OOM at a too-large val batch, since fixed separately). Default to 32
    # here so a bare single-GPU submission doesn't blindly try 64/GPU and OOM.
    BATCH_SIZE=32
    echo "NUM_GPUS=1 and no BATCH_SIZE given -> defaulting to $BATCH_SIZE (known-safe per-GPU footprint)"
fi
if [ -n "${BATCH_SIZE:-}" ]; then
    echo "BATCH_SIZE=$BATCH_SIZE"
    EXTRA_OVERRIDES+=("dataloader.batch_size=$BATCH_SIZE")
    # Keep the warmup at the same number of *samples* as the yaml's 500 steps
    # at batch 64, so a bigger batch (fewer steps per epoch) doesn't spend a
    # disproportionate share of a short run warming up. LR_WARMUP_STEPS wins.
    LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-$(( 500 * 64 / BATCH_SIZE ))}"
fi
if [ -n "${LR_WARMUP_STEPS:-}" ]; then
    echo "LR_WARMUP_STEPS=$LR_WARMUP_STEPS"
    EXTRA_OVERRIDES+=("training.lr_warmup_steps=$LR_WARMUP_STEPS")
fi
if [ -n "${LR:-}" ]; then
    echo "LR=$LR"
    EXTRA_OVERRIDES+=("optimizer.lr=$LR")
fi
# Separate from BATCH_SIZE: job 37931 ran fine at train batch=128 for a full
# epoch (6h17m) but OOM'd in end-of-epoch validation, which evaluates through
# the EMA model -- a second full parameter copy (incl. the frozen 668M-param
# TraceAnything backbone) alongside the training model, so it has less
# headroom at the same batch size. PER GPU; default (yaml null) = per-GPU train batch.
if [ -n "${VAL_BATCH_SIZE:-}" ]; then
    echo "VAL_BATCH_SIZE=$VAL_BATCH_SIZE"
    EXTRA_OVERRIDES+=("val_dataloader.batch_size=$VAL_BATCH_SIZE")
fi
if [ -n "${CHECKPOINT_FRAC:-}" ]; then
    # Saves latest.ckpt every ~this fraction of an epoch's steps (e.g. 0.2 =
    # every 20%), IN ADDITION to the end-of-epoch save -- see
    # training.checkpoint_every_frac in the yaml / the workspace's mid-epoch
    # checkpoint block. Useful when NUM_EPOCHS is small (e.g. 1) and you don't
    # want to wait a full epoch for the first safety checkpoint.
    echo "CHECKPOINT_FRAC=$CHECKPOINT_FRAC"
    EXTRA_OVERRIDES+=("training.checkpoint_every_frac=$CHECKPOINT_FRAC")
fi
if [ -n "${MAX_TRAIN_STEPS:-}" ]; then
    # Caps steps *within* each epoch (independent of DEBUG) -- for a quick
    # save-mechanism smoke test: e.g. MAX_TRAIN_STEPS=5,CHECKPOINT_FRAC=0.5
    # triggers a mid-epoch checkpoint after ~2-3 steps instead of waiting for
    # a real fraction of the full ~4000+-step epoch.
    echo "MAX_TRAIN_STEPS=$MAX_TRAIN_STEPS"
    EXTRA_OVERRIDES+=("training.max_train_steps=$MAX_TRAIN_STEPS")
fi
if [ -n "${MAX_VAL_STEPS:-}" ]; then
    # Caps validation to this many batches per dataset instead of the full
    # val_dataloader (245 batches for this config -- ~60min at the ~15s/it
    # seen in job 39138). Combine with NUM_EPOCHS + MAX_TRAIN_STEPS for
    # periodic-but-cheap validation within a single real epoch's worth of
    # steps: val_every stays epoch-granular in this codebase (no native
    # mid-epoch validation hook), so e.g. NUM_EPOCHS=4,MAX_TRAIN_STEPS=1000
    # turns 1 real ~4111-step epoch into 4 short "epochs" of 1000 steps each,
    # each followed by a validation pass (val_every=1 default).
    echo "MAX_VAL_STEPS=$MAX_VAL_STEPS"
    EXTRA_OVERRIDES+=("training.max_val_steps=$MAX_VAL_STEPS")
fi
if [ -n "${TRACE_CACHE_DIR:-}" ]; then
    # Precomputed TraceAnything window tokens (experiments/015_cache_trace_tokens.sbatch):
    # the dataset serves them instead of scene images and the policy skips the
    # frozen encoder during training/validation. Rollout evaluation is unaffected.
    # Comma-separate one cache per camera (e.g. scene,wrist caches) for configs
    # with several policy.trace_keys.
    echo "TRACE_CACHE_DIR=$TRACE_CACHE_DIR"
    if [[ "$TRACE_CACHE_DIR" == *,* ]]; then
        EXTRA_OVERRIDES+=("+task.dataset.trace_cache_dirs=[$TRACE_CACHE_DIR]")
    else
        EXTRA_OVERRIDES+=("+task.dataset.trace_cache_dir=$TRACE_CACHE_DIR")
    fi
fi
if [ -n "${ECHO:-}" ]; then
    # Data echoing (EchoDistributedSampler): each sample is trained on ECHO
    # times per epoch (fresh diffusion timestep/noise each time), with repeats
    # served from the OS page cache -- for when cache reads, not the GPU, set
    # the pace. An epoch is still one pass over the unique data (ECHO x steps).
    # Page cache budget ~2 x ECHO_BLOCK rows (1.5 MiB each) per GPU: size
    # --mem accordingly (default block 5000 -> ~15 GiB per GPU).
    echo "ECHO=$ECHO ECHO_BLOCK=${ECHO_BLOCK:-5000}"
    EXTRA_OVERRIDES+=("+training.echo_factor=$ECHO" "+training.echo_block=${ECHO_BLOCK:-5000}")
fi
if [ "${DEBUG:-}" = "true" ]; then
    # training.debug=true (see the workspace's own `if cfg.training.debug`
    # branch) sets num_epochs=2, max_train_steps=3, max_val_steps=3,
    # checkpoint_every=1 -- a few-minute run that exercises the full loop
    # (train step -> validate -> checkpoint save) without waiting hours.
    # Overrides NUM_EPOCHS/BATCH_SIZE/VAL_BATCH_SIZE above if also set.
    #
    # IMPORTANT: do NOT also pass training.total_train_steps=null here (job
    # 38131's bug). The `total_train_steps -> num_epochs` block runs BEFORE
    # the debug block in the workspace, and the LR scheduler is built
    # immediately after using cfg.training.num_epochs -- nulling
    # total_train_steps skips that first block, leaving num_epochs=None at
    # that point -> `len(train_dataloader) * None` crashes. Leaving
    # total_train_steps at the yaml's 300000 lets that block populate some
    # (unused, harmless) num_epochs=73 for the scheduler; the debug block
    # then correctly overwrites num_epochs=2 afterward, before the actual
    # training loop reads it.
    echo "DEBUG=true -> training.debug=true (few-minute checkpoint-save smoke test, isolated run dir: $RUN_NAME)"
    EXTRA_OVERRIDES+=(training.debug=True)
fi

exec pixi run --manifest-path ../../pixi.toml -e "$PIXI_ENV" \
  torchrun --nproc_per_node="$NUM_GPUS" --standalone train.py \
    --config-dir=config/franka_kitchen \
    --config-name="$CONFIG_NAME" \
    hydra.run.dir="$HYDRA_RUN_DIR" \
    training.resume=True \
    training.mixed_precision=bf16 \
    "${CKPT_OVERRIDE[@]}" \
    "${EXTRA_OVERRIDES[@]}"
