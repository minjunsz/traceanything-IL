# 003 — Port the whole experiment stack into `src/` (standard, up-to-date dependencies only)

Written: 2026-09-29

## Goal
Today the experiment depends on four external repos under `third_party/`, pinned to old dependencies
(torch 2.3.1, gym 0.26, mujoco-py, dm_control 1.0.12, hydra, cython<3, numpy<2).

| Repo | Role |
|---|---|
| `diffusion-policy-experiments` | Training and policy code |
| `relay-policy-learning` | Kitchen simulation, scripted expert, evaluation |
| `TraceAnything` | Frozen visual encoder |
| `fast3r` | Optional curope extension |

Goal: rewrite the code we actually use inside `src/` and **keep only up-to-date, standard dependencies**.
What stays external is only the pretrained weights (`trace_anything.pt`) and the MuJoCo model assets
(XML and meshes, Apache 2.0).

## Target dependencies (one pixi env)
- **Keep:** python 3.12, conda-forge `pytorch-gpu` (latest), `numpy>=2`, `mujoco` (official bindings,
  latest 3.x), `gymnasium`, `zarr`, `numcodecs`, `diffusers`, `accelerate`, `einops`, `imageio` /
  `imageio-ffmpeg`, `matplotlib`, `tqdm`, `tyro` (experiment configuration)
  - Optional: `wandb`, `pytest`
- **Drop:**
  - gym, mujoco-py, dm_control, mjrl, cython, hydra-core / omegaconf, robomimic, r3m, numba, timm
  - mesalib / glew / glfw / xorg-* (`mujoco.Renderer` + `MUJOCO_GL=egl` uses the NVIDIA driver's EGL)
- **Decisions to make:**
  - **zarr version.** 2.x keeps the existing zarr readable as is. 3.x is the latest, but its storage
    API is incompatible with 2.x, so the data would be regenerated or converted.
    → Recommendation: zarr 3, because the data gets regenerated anyway (see Phase A).
  - **MuJoCo version.** The data was recorded with mujoco 2.3.5. Moving to 3.x can change the
    dynamics slightly, so **the demo data and the trace cache must be regenerated on the new sim**
    (Phase A gate).

## Coding rules
- **Configuration with tyro.** No hydra/omegaconf/yaml configs.
  - Each component's settings are a `@dataclass` defined next to that component, e.g.
    `KitchenEnvConfig`, `ExpertConfig`, `TraceAnythingConfig`, `TracePolicyConfig`, `DataConfig`,
    `TrainConfig`, `EvalConfig`.
  - An experiment config is a top-level dataclass composed of these (nested dataclasses).
  - Experiment scripts only call `config = tyro.cli(ExperimentConfig)`, then pass `config` to the
    src API.
  - Defaults are the current experiment values (the fork yaml `8_obs_trace_anything_markovian_expert`).
    Overrides use CLI flags, e.g. `--train.batch-size 128`.
  - Serialize the full resolved config into the run directory and the checkpoint payload
    (`dataclasses.asdict`), so eval and resume can rebuild the same config without CLI arguments.
  - Presets (e.g. debug / smoke / full) are handled with `tyro.extras.overridable_config_cli` or
    subcommands, not yaml files.
- **Type annotations.** Every function, method and dataclass field has type annotations (arguments
  and return values).
  - Tensor/array arguments state their shape and dtype in the docstring, e.g. `(B, T, H, W, C) uint8`.
  - Use `Protocol` / `ABC` for shared interfaces (e.g. `NoisePredictor`, `ObsEncoder`,
    `DiffusionPolicy` in the existing `src/diffusion_policy/interfaces.py`).
  - Prefer `Literal` / `Enum` over free strings (e.g. `mixed_precision: Literal["no", "fp16", "bf16"]`).
  - Type check with pyright (basic mode) on `src/`.

## Proposed package layout
```
src/
  kitchen/                 # simulation (replaces adept_envs)
    assets/                # kitchen scene + franka XML/meshes (copied, include paths fixed)
    sim.py                 # MjModel/MjData wrapper, frame skip, qpos/qvel access, offscreen cameras
    robot.py               # Franka velocity actuation, joint limits, observation noise, calibration
    env.py                 # KitchenEnv(gymnasium.Env): reset/step/render, subtask completion
    subtasks.py            # 7 subtask definitions (goal joint indices/values, 0.3 threshold), plan one-hot
    vector.py              # parallel envs (gymnasium AsyncVectorEnv or a thin subprocess wrapper)
  expert/                  # scripted Markovian expert (replaces experts/)
    controller.py          # damped-least-squares IK for EE pose → 9-D normalized joint velocity
    fsm/                   # one FSM per subtask (7) + shared base
    expert.py              # plan (subtask sequence) → per-step action
    record.py              # parallel rollout recorder → zarr (same layout as today)
  trace_anything/          # frozen encoder inference-only port (drop heads and DPT)
  diffusion_policy/        # (extends the existing dpv2 package)
    data/ models/ diffusion/ policy/ training/ ...
  evaluation/              # replaces eval/
    rollout.py             # obs deque, action queue, n_action_steps, success rule
    runner.py              # rounds, resume, results.csv/pkl/summary, videos
    recorder.py            # rollout → zarr (with qpos/qvel)
    plots.py               # per-checkpoint success-rate plots
experiments/               # thin scripts only (argparse → src API)
tests/                     # equivalence / regression tests (pytest)
```

## Code to write

### A. Kitchen simulation (`src/kitchen`), source: `relay-policy-learning/adept_envs` (~2.7k lines, about 1.5k used)
1. **Assets.**
   - Copy `adept_envs/franka/assets/franka_kitchen_jntpos_act_ab.xml`, `adept_models/{kitchen,scenes}`
     (8.4 MB) and `third_party/franka` (7.4 MB), and fix the `<include>` relative paths.
   - Check whether the XML loads in MuJoCo 3.x (deprecated attributes).
   - Keep the LICENSE files.
2. **`sim.py`.**
   - Load MjModel/MjData.
   - `frame_skip=40`, `dt = frame_skip * opt.timestep`.
   - Offscreen rendering with `mujoco.Renderer`: scene cam id 2, wrist cam id 3, 240×320.
   - This replaces the dm_control Physics path used by today's `simulation/sim_robot.py` and
     `renderer.py`.
3. **`robot.py`.** Port of `franka/robot/franka_robot.py` `Robot_VelAct` plus the `franka_config.xml`
   specs.
   - Velocity action: `q_next = q + clip(vel, vel_bound) * dt`, then position limits clip, then the
     position actuator ctrl.
   - Observation noise: `pos/vel_noise_amp * robot_noise_ratio`, default 0.1.
   - `_calib` / `_de_calib`, and the `sim_mimic_hardware` behavior.
   - Only the 9 robot DOF and 21 object DOF split.
4. **`env.py`.** Port of `KitchenV0` / `KitchenTaskRelaxV1` onto the gymnasium API.
   - Action clip [-1, 1], then `act_mid + a * act_amp`.
   - Observation = `[qp(9), obj_qp(21), goal(30)]`, plus the camera images.
   - The fixed `init_qpos` array.
   - `reset(seed)` → `(obs, info)`, `step` → `(obs, reward, terminated, truncated, info)`.
   - Noise on/off option.
   - Drop the old API leftovers: `configurable`, pickling, `parse_demos` / mjrl.
5. **`subtasks.py`.**
   - `SUBTASK_INFO` (per-subtask qpos indices and goals), `BONUS_THRESH=0.3`, completed-subtask test.
   - `SUBTASK_IDS` and `sequence_onehot` (4 slots × 7 = 28 dims), from `experts/subtasks/base.py`.
   - One implementation shared by the env, the expert and eval.
6. **`vector.py`.** Port of the subprocess batched env in `evaluate_kitchen.py`
   (`_worker` / `_BatchedEnv`), or replace it with gymnasium `AsyncVectorEnv`.

### B. Scripted expert (`src/expert`), source: `relay-policy-learning/experts` (~2.6k lines)
1. `controller.py`: EE pose / site Jacobian, body-point world coordinates, damped-least-squares IK,
   gripper saturation logic, gain constants. It already uses the `mujoco` API, so only the env
   handle needs to change.
2. `fsm/`: the 7 subtask FSMs plus the shared base (GripperCmd, HOME pose, phase transitions).
   Mostly a port.
3. `expert.py`: plan → current subtask → action (Markovian: a function of the current state only).
4. `record.py`: port of `record_demos.py`.
   - Arguments: random plans with `chain_len=4`, `seed`, `n_episodes=581`, `n_workers`.
   - Keep only successful episodes.
   - Writes the zarr layout (action/state/scene/wrist/current_subtask/subtask_sequence +
     meta/episode_ends; blosc lz4 chunks).
5. Rewrite `run_expert.py` / `chain_test.py` as `tests/` and a debugging script.

### C. TraceAnything encoder (`src/trace_anything`), source: `third_party/TraceAnything/trace_anything` (~1.8k lines)
1. Port only the encoder (patch_embed, blocks), the decoder and `_encode_images`, about 1k lines.
   - The heads and `dpt_block` are not needed for token extraction.
   - Load the weights with `strict=False` and ignore head weights, but assert that every
     encoder/decoder key loads.
2. RoPE2D: pure PyTorch implementation only. Drop the curope CUDA extension and fast3r.
3. Replace the yaml / omegaconf config with a typed `TraceAnythingConfig` dataclass (defaults =
   `configs/eval.yaml` values), so it can also be exposed through tyro.
4. `TraceAnythingWindowEncoder` wrapper: T frames → last-frame, last-layer tokens (768×1024).
   Keep the resize to 384×512, normalization to [-1, 1] and `time_step = t/(T-1)` exactly.

### D. Diffusion policy (`src/diffusion_policy`, extending dpv2)
Reuse what exists: `SequenceSampler`, `FrankaKitchenDataset`, `LinearNormalizer`, EMA,
`StaticAttentionConditionalUnet1D`, `diffusion/{loss,sampling,scheduler}`, `Checkpointer`.
New code:
1. `policy/trace_attention_policy.py`.
   - `lowdim_proj` / `trace_proj`.
   - Token layout: positions 0..To-1 plus To-1; modality and range ids 1 = low-dim, 2 = trace.
   - Cached `trace_tokens` input.
   - `predict_action(use_ddim)` returns `action_pred[To-1 : To-1+8]`.
2. `data/trace_cache.py` plus a dataset option (fork `common/trace_cache.py`, the `trace_cache_dir`
   feature of `franka_kitchen_dataset.py`).
   - Skip loading the cached rgb key; add `keys=` to `ZarrReplayBuffer.from_path`.
   - Replace numba in `SequenceSampler` with vectorized numpy.
3. `training/samplers.py`: `ResumableDistributedSampler` and `EchoDistributedSampler` (taken from the
   fork workspace).
4. `training/loop.py`: epoch-based DDP (feature parity with the fork's
   `train_diffusion_unet_hybrid_workspace_no_env.py`).
   - Cosine LR with warmup; warmup scales with batch size.
   - Grad clip 1.0, bf16.
   - EMA, with `optimization_step` restored on resume.
   - Validation loss sharded across GPUs, plus rank-0 DDPM/DDIM action MSE on one batch.
   - `logs.json.txt` (per-step and per-epoch rows).
   - Checkpoints: `latest` (end of epoch plus every `checkpoint_every_frac`), top-k by val DDIM MSE,
     `epoch_step` for exact mid-epoch resume, `normalizer.pt`, atomic saves.
   - The checkpoint payload includes the policy constructor config dict, so eval can rebuild the
     policy without hydra.
   - Keep the existing step-based `009` working, or migrate it to the new Trainer.
5. `training/config.py`: extend the `TrainConfig` dataclass (typed). Drop Hydra; each experiment
   script composes it with `tyro.cli` (see "Coding rules"). Replace the existing 009's argparse with
   tyro as well.

### E. Evaluation (`src/evaluation`), source: `relay-policy-learning/eval` (~1.8k lines)
1. `rollout.py`.
   - Initialize the obs deque by repeating the first obs n_obs_steps times.
   - Action queue with n_action_steps; policy input is agent_pos (qp[:9]), the scene image and the
     plan one-hot.
   - Success = at least N distinct subtasks.
   - Handle simulation blow-ups.
2. `runner.py`.
   - Rounds × `n_envs`, per-trial random plans (`seq_seed`), noise on/off, `task_timeout`
     (default 600 steps).
   - results.csv/pkl, summary.txt, `--resume`, mp4 of the first N trials or of failures.
   - Record completed subtasks and whether they matched the plan order. This is a new metric.
3. `recorder.py`: rollout → zarr (including qpos/qvel), port of `episode_recorder.py`.
4. `plots.py`: success-rate plot per checkpoint, port of `plot_eval_results.py`.
   - `motion_overlay.py` (visualization) is ported last, only if needed.

### F. Experiment scripts (`experiments/`, thin wrappers only)
- Data regeneration (`record`), trace cache build (port of today's 015), training (to replace 008),
  eval (to replace 014), plots (port of 012).
- Each script is roughly `main(tyro.cli(Config))`, with no argparse or env-var parsing; the sbatch
  scripts pass tyro CLI flags directly.
- sbatch scripts use one pixi env. Numbering continues from the current last number.

### G. Tests (`tests/`, pytest)
- Sim, expert, sampler, cache layout, policy equivalence (see "Verification gates" below).
- CPU-only tests are fast; GPU/EGL tests carry a marker.

## Order of work and verification gates
Each phase moves on only after its gate passes.

1. **Phase A — sim port.**
   - First port under mujoco 2.3.5 (a temporary pin). Replay the same initial state and the same
     action sequence (an existing demo's actions) and check that the qpos trajectory matches the
     original adept_envs env **bitwise**, and that rendered frames match.
   - Then upgrade to MuJoCo 3.x and quantify the drift.
2. **Phase B — expert port.**
   - Same seed / plans as the original expert: the success rate (581/629 at seed 0) and per-subtask
     counts match, or come close after the MuJoCo upgrade.
   - Regenerate the new dataset and switch zarr versions if needed.
3. **Phase C — TraceAnything port.** Tokens from the original model and the ported model on the same
   input window match (max relative diff ~1e-3 at float tolerance). Rebuild the trace cache on the new
   data.
4. **Phase D — policy/training port.**
   - Map the fork's epoch=004 weights into the src policy; noise prediction and DDIM action output
     (with fixed initial noise) match the fork.
   - src dataset windows match fork windows.
   - Short training smoke test with 2 GPUs + cache + echo, including a forced stop and resume.
5. **Phase E — eval port.**
   - Under the pinned MuJoCo, the fork's epoch=004 checkpoint evaluated through the src path matches
     the fork's result within statistical noise (2%, average 0.88 subtasks).
   - Then move to the MuJoCo 3.x sim and new data, retrain, and re-evaluate.
6. **Cleanup.** Once every experiment runs on `src/` alone, archive the fork envs
   (`diffusion-policy-l40`, `dp-kitchen-eval`, `franka-kitchen`) and `third_party/` code.
   Keep the weights and assets.

## Risks and notes
- **MuJoCo 3.x upgrade:** contact and solver defaults changed, so the expert's success rate can change.
  Since the expert FSM gains were tuned on 2.3.5, retuning may be needed.
- **Old gym API parts of adept_envs:** `configurable` pickling and `robot_noise_ratio` defaults are
  easy to miss. Pin the noise behavior against the original with a regression test.
- **TraceAnything:** without curope, the pure PyTorch RoPE is slow; it is already in use today, so
  there is no performance change. A newer torch SDPA might speed it up.
- **Comparing with existing results:** the current results (5 epochs, 2%) come from the old stack and
  old data. After new data and a new sim, run baseline evals again under the same protocol.
- **Work in progress:** everything added to the fork this session (distributed validation, mid-epoch
  resume, EMA restore, atomic saves, trace cache, echo, validation batch rules) must be carried over
  in Phase D.
