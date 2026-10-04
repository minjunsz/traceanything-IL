# markovian-policy

Diffusion policies for long-horizon manipulation in a MuJoCo Franka kitchen, with a **scripted Markovian expert**,
**human-teleoperation replay**, and a frozen **TraceAnything** visual encoder as the policy's image encoder.

The code is a small library plus one thin script per pipeline stage:

```
generate data  ->  cache encoder tokens  ->  train  ->  evaluate
(expert / human)    (frozen TraceAnything)   (DDP)      (closed loop in the sim)
```

- **Simulator** - the 7-subtask kitchen as a `gymnasium` environment on current MuJoCo (3.x), two cameras, vectorized.
- **Scripted expert** - per-subtask finite state machines whose phase is a pure function of the simulator state.
- **Data** - demonstrations from the expert or from replayed human logs, stored as zarr; cached encoder tokens.
- **Policy** - cross-attention UNet diffusion policy conditioned on low-dim history and TraceAnything tokens.
- **Training / evaluation** - resumable multi-GPU trainer; batched closed-loop evaluator with result files and videos.

## Install

Dependencies are managed with [pixi](https://pixi.sh) (environment `port`: Python 3.12, PyTorch, MuJoCo, gymnasium,
zarr 3, diffusers, tyro). On a machine without a GPU, pixi needs the CUDA virtual package:

```bash
export CONDA_OVERRIDE_CUDA=12.9
pixi install -e port
pixi run -e port pytest -q -m "not gpu" tests     # CPU test suite
```

Rendering uses `mujoco.Renderer` with EGL (`MUJOCO_GL=egl` is set by the environment), so data generation, token
caching and evaluation need a GPU node. Pretrained TraceAnything weights are expected at
`third_party/TraceAnything/checkpoints/trace_anything.pt` (see `TraceAnythingConfig.ckpt_path`).

## The pipeline

Every stage is a `Config` dataclass plus a `run(config)` function in `markovian_policy.stages`; the scripts in
`experiments/` only parse the config from the command line (via [tyro](https://brentyi.github.io/tyro/)) and call `run`.
`--help` lists every option; nested configs use dotted flags (`--train.batch-size 128`).

```bash
# 1. demonstrations (scripted expert; or: experiments/028_build_human_demos.py for replayed human logs)
python experiments/021_record_expert.py --n-episodes 581 --out output/demos.zarr

# 2. precompute the frozen encoder's tokens, once per camera (init -> encode -> verify; resumable, shardable)
python experiments/022_cache_trace_tokens.py init   --zarr-path output/demos.zarr --out-dir CACHE_SCENE --camera scene
python experiments/022_cache_trace_tokens.py encode --zarr-path output/demos.zarr --out-dir CACHE_SCENE --camera scene
python experiments/022_cache_trace_tokens.py verify --zarr-path output/demos.zarr --out-dir CACHE_SCENE --camera scene

# 3. train (multi-GPU through torchrun, see the sbatch script)
python experiments/023_train_trace_policy.py --data.zarr-path output/demos.zarr \
    --data.trace-cache-dirs scene CACHE_SCENE wrist CACHE_WRIST --train.output-dir runs/trace_policy

# 4. evaluate a checkpoint in the simulator; results.csv/json, summary.txt and videos land in --out-dir
python experiments/026_eval.py --checkpoint runs/trace_policy/checkpoints/latest.ckpt --out-dir eval/run --n-rollouts 50
```

Each experiment script has an `.sbatch` companion for Slurm clusters. Everything is resumable: rerun the same command
after a preemption or timeout.

## Layout

```
src/markovian_policy/
  sim/          MuJoCo kitchen: physics, actuation, subtask definitions, gymnasium env, vector envs
  experts/      scripted Markovian expert: subtask FSMs + operational-space controller
  perception/   frozen visual encoders (TraceAnything) behind a small `FrameWindowEncoder` protocol
  nn/           conditional 1D UNets, EMA, linear normalizer
  diffusion/    training loss, reverse sampling, scheduler pairing
  policies/     diffusion policies and the conditioning they assemble
  data/         dataset, windowing, token cache, and `generation/` (demonstration sources + builder)
  training/     trainer, config, callbacks, checkpoint format
  evaluation/   rollouts, `Evaluator`, result sinks, plots
  stages/       one facade per pipeline stage (config + run)
experiments/    thin scripts + sbatch files; numbered in the order they were run
tests/          CPU tests (`pytest -m "not gpu"`) and GPU tests (`-m gpu`, run on a compute node)
```

The design (layers, extension points, the checkpoint and dataset formats) is described in
[docs/architecture.md](docs/architecture.md).

## Development

```bash
pixi run -e port pytest -q -m "not gpu" tests     # tests
pixi run -e port pyright                           # strict type check of src/
pixi exec ruff check src tests experiments         # lint (config: ruff.toml)
pixi exec ruff format src tests experiments
```

Types are checked in strict mode (`pyrightconfig.json`); public functions, methods and dataclass fields are annotated,
tensor shapes are documented in docstrings.

## Third-party material

The kitchen scene and Franka model assets in `src/markovian_policy/sim/assets` come from the
[Relay Policy Learning](https://github.com/google-research/relay-policy-learning) kitchen (Apache 2.0; see
`LICENSE.adept_models`, `LICENSE.franka`). The TraceAnything encoder is an inference-only port of
[TraceAnything](https://github.com/ByteDance-Seed/TraceAnything); its weights and license apply to the checkpoint.
