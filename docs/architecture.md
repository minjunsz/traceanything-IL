# Architecture

The package is layered; a layer only imports from layers below it. High-level orchestration (what happens, in which
order) lives in small facades and in `stages/`; low-level code (simulation, tensors, files) lives below and knows
nothing about the pipeline.

```
stages/        facades: Config + run() per pipeline stage
─────────────────────────────────────────────────────────────────
training/   evaluation/   data/generation/   data/token_cache_builder
─────────────────────────────────────────────────────────────────
policies/   data/ (dataset, sampler, cache)   experts/
─────────────────────────────────────────────────────────────────
nn/   diffusion/   perception/   sim/
```

## Design patterns

The recurring pattern is a **Protocol** (the interface a high-level component depends on), **strategies** that implement
it, and a **facade** that wires strategies together. Protocols are structural: implementing one needs no inheritance.

| Facade | Strategies it is composed of | Protocol |
|---|---|---|
| `DatasetBuilder` (`data/generation/builder.py`) | where demonstrations come from: `ExpertDemoSource`, `HumanReplaySource` | `DemoSource` |
| | which episodes are kept: `SuccessOnly`, `KeepAll` | `AcceptancePolicy` |
| `Trainer` (`training/loop.py`) | logging, checkpointing (observers): `JsonLogCallback`, `CheckpointCallback` | `TrainingCallback` |
| | epoch-end metrics: `Validator` | `EpochEvaluator` |
| | epoch sharding: `ResumableDistributedSampler`, `EchoDistributedSampler` | (`build_train_sampler`) |
| `Evaluator` (`evaluation/runner.py`) | which plan a trial gets: `RandomPlans`, `FixedPlan` | `PlanSource` |
| | when a trial succeeded: `DistinctSubtasks` | `SuccessCriterion` |
| | what happens with trials: `ResultFiles`, `VideoSink`, `EpisodeSink` | `TrialSink` |
| `TraceAttentionPolicy` | where encoder tokens come from: cache or online encoder (`TraceTokenSource`) | `FrameWindowEncoder` |
| `ScriptedExpert` | per-subtask behaviour: `MicrowaveFSM`, `KettleFSM`, ... | `SubtaskFSM` |

### Adding your own

- **A new demonstration source** (e.g. rollouts of a policy, another dataset): write a frozen dataclass with `jobs()`,
  `key(job)`, `worker()` (builds the simulator *inside* the worker process and returns `job -> Candidate`) and `report()`.
  `DatasetBuilder(source, SuccessOnly(), out, n_workers, target_episodes).build()` then provides parallel execution,
  resume (`<out>.manifest.json`) and the summary for free.
- **A new success rule or plan distribution**: implement `SuccessCriterion` / `PlanSource` and pass it to `Evaluator`.
- **A new output of the evaluation** (a plot per round, a database): implement `TrialSink`.
- **A new visual encoder**: implement `FrameWindowEncoder` (`token_dim`, `num_tokens`, `__call__`) and build the policy's
  `TraceTokenSource` with it.
- **Training hooks** (extra logging, LR probes): implement `TrainingCallback` and pass `callbacks=[...]` to `Trainer`.

## Data flow

```
 ExpertDemoSource ─┐                      ┌── TokenCacheBuilder ──> token cache per camera ─┐
                   ├─> DatasetBuilder ──> zarr dataset ─────────────────────────────────────┤
 HumanReplaySource ┘                                                                        v
                                                    FrankaKitchenDataset ──> Trainer ──> checkpoints
                                                                                              │
                              Evaluator <── load_policy <─────────────────────────────────────┘
                                 │
                       KitchenEnv (vectorized) ──> TrialSinks: results, videos, rollouts (zarr)
```

## Formats

**Dataset (zarr)**: `data/<key>` arrays of shape `(T_total, ...)` plus `meta/episode_ends` `(E,)` (cumulative end
indices). A step stores `(o_t, a_t)`: the observation *before* the action with the action taken.

| key | shape | dtype | meaning |
|---|---|---|---|
| `action` | (T, 9) | float64 | normalized joint velocities in [-1, 1] |
| `state` | (T, 60) | float64 | `[noisy qpos(30), goal(30)]`; the policy sees the first 9 dims as `agent_pos` |
| `scene`, `wrist` | (T, 240, 320, 3) | uint8 | camera frames |
| `subtask_sequence` | (T, 28) | float64 | plan one-hot (4 slots x 7 subtasks), constant per episode |
| `current_subtask` | (T, 7) | float64 | the expert's current subtask (expert data only) |
| `qpos`, `qvel` | (T, 30), (T, 29) | float64 | noise-free simulator state (evaluation recordings only) |

**Token cache**: `<dir>/tokens.npy` float16 `(n_windows, P, D)`, `meta.json`, `progress/rank<r>.txt`, and a `COMPLETE`
marker that `verify` writes once every window is encoded (the dataset refuses an incomplete cache). Row layout:
see `data/trace_cache.py`.

**Checkpoint** (`CheckpointPayload`, written by `Trainer` as `checkpoints/latest.ckpt`, top-k and snapshot files):
`model`, `ema_model` (state dicts; the normalizer is part of them), `optimizer`, `lr_scheduler`, `ema_step`,
`global_step`, `epoch`, `epoch_step`, `topk`, `train_config`, `policy_config`. Parameter names of the policies are part of
the format; `tests/test_checkpoint_compat.py` loads a checkpoint written by an earlier version of the code and
checks that its outputs are bit-identical.

**Assets**: `data/assets.py` lists the downloadable assets (URL, expected size, default location); `download` resumes
partial files, writes atomically and verifies the size. Default locations are defined once in `paths.py`.

**Evaluation output**: `results.csv` (one row per trial), `results.json` (resume + plots), `summary.txt`, `videos/`,
optionally a zarr of the rollouts. Plans and environment seeds depend only on `(seed, trial number)`, so a resumed run
continues exactly where the interrupted one stopped.

## Conventions

- Configuration is a tree of dataclasses defined next to the component they configure; stages compose them and tyro
  turns them into command lines. Resolved configs are stored in checkpoints and result files.
- Observation tensors: images `(B, T, H, W, 3)` uint8/float in [0, 255], low-dim `(B, T, d)`; actions `(B, T, 9)`.
- Heavy resources (simulators, GL contexts, encoders) are created lazily and inside the process that uses them.
- A diverging simulation is reported, never silently continued: `KitchenEnv` truncates the episode
  (`info["unstable"]`), sources drop the attempt, evaluation marks the trial as a failure.
