"""Collect the numbers of training runs and their evaluations into tidy (one observation per row) CSV tables.

Inputs (all optional per run):
    runs/<run>/logs.json.txt            training log: one JSON per step (train_loss, lr) and per epoch (val_*)
    runs/<run>/checkpoints/latest.ckpt  policy and train config (read memory-mapped, weights are not loaded)
    runs/<run>/meta.json                free-form facts about the run (data, encoder, job ids, ...), merged into runs.csv
    <eval_dir>/<run>/T_a_<H>/<checkpoint stem>/results.json   evaluations (see `evaluation.runner`)

Outputs in `out_dir` (every table has a `run` column; a run's evaluation rows join on `run`, `checkpoint`):
    runs.csv              one row per run: config, progress, best validation metric, meta
    training_steps.csv    run, global_step, epoch, train_loss, lr
    training_epochs.csv   run, epoch, global_step, train_loss, val_loss, val_ddpm_mse, val_ddim_mse
    eval_checkpoints.csv  one row per (run, checkpoint, horizon): success counts/rate with Wilson interval, mean subtasks, ...
    eval_trials.csv       one row per evaluated trial
    eval_subtasks.csv     one row per (run, checkpoint, horizon, subtask): how many trials completed it

Progress metrics: a trial ends at its 4th distinct subtask, so `n_completed` is capped at the success threshold
(`mean_subtasks` is censored there); `frac_ge_k` is the share of trials with at least k subtasks.
Speed metrics (evaluations that logged `completed_steps`; empty otherwise): `steps_to_<k>` is the env step at which a
trial first had k distinct subtasks (empty if it never did), `mean_steps_to_<k>` averages it over the `n_reached_<k>`
trials that got there, and `restricted_mean_steps_to_success` counts every trial that did not succeed as the timeout,
so it combines success rate and speed in one number (lower is better) without discarding the failures.

A run that was resumed re-logs the steps after its last checkpoint; the later log line wins (dedupe by step/epoch).
"""

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from markovian_policy.evaluation.plots import wilson_interval
from markovian_policy.sim.tasks import SUBTASK_IDS

STEP_COLUMNS = ["run", "global_step", "epoch", "train_loss", "lr"]
EPOCH_COLUMNS = ["run", "epoch", "global_step", "train_loss", "val_loss", "val_ddpm_mse", "val_ddim_mse"]
SUBTASKS = tuple(SUBTASK_IDS)
CHECKPOINT_STEM = re.compile(r"epoch=(\d+)-val_ddim_mse=([0-9.]+)")


@dataclass
class CollectConfig:
    runs_dir: Path = Path("runs")
    eval_dir: Path = Path("output/eval")
    out_dir: Path = Path("output/results")
    runs: tuple[str, ...] | None = None  # default: every directory of runs_dir with a training log


def read_training_log(run: str, log_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(steps, epochs) of one run; unreadable lines (e.g. one still being written) are skipped."""
    steps: list[dict[str, Any]] = []
    epochs: list[dict[str, Any]] = []
    for line in log_path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        (epochs if "val_loss" in row else steps).append({"run": run, **row})
    step_table = pd.DataFrame(steps, columns=STEP_COLUMNS).drop_duplicates("global_step", keep="last")
    epoch_table = pd.DataFrame(epochs, columns=EPOCH_COLUMNS).drop_duplicates("epoch", keep="last")
    return step_table.sort_values("global_step").reset_index(drop=True), epoch_table.sort_values("epoch").reset_index(drop=True)


def read_checkpoint_configs(checkpoint: Path) -> dict[str, Any]:
    """Policy and train config of a checkpoint, and where it stopped; the weights stay on disk (memory-mapped)."""
    if not checkpoint.exists():
        return {}
    payload = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    policy, train = payload.get("policy_config") or {}, payload.get("train_config") or {}
    return {
        "policy_kind": policy.get("kind", "trace_attention"),
        "n_obs_steps": policy.get("n_obs_steps"),
        "horizon": policy.get("horizon"),
        "n_action_steps": policy.get("n_action_steps"),
        "short_range_obs_horizon": policy.get("short_range_obs_horizon"),
        "batch_size": train.get("batch_size"),
        "total_train_steps": train.get("total_train_steps"),
        "lr": train.get("lr"),
        "checkpoint_epoch": payload.get("epoch"),
        "checkpoint_global_step": payload.get("global_step"),
    }


def run_row(run: str, run_dir: Path, steps: pd.DataFrame, epochs: pd.DataFrame) -> dict[str, Any]:
    row: dict[str, Any] = {"run": run, **read_checkpoint_configs(run_dir / "checkpoints" / "latest.ckpt")}
    row["global_step"] = int(steps["global_step"].to_numpy().max()) if len(steps) else None
    row["epochs_logged"] = len(epochs)
    if len(epochs):
        val, epoch = epochs["val_ddim_mse"].to_numpy(dtype=float), epochs["epoch"].to_numpy()
        row |= {"best_epoch": int(epoch[val.argmin()]), "best_val_ddim_mse": float(val.min())}
        row["final_val_ddim_mse"] = float(epochs["val_ddim_mse"].iloc[-1])
    meta_path = run_dir / "meta.json"
    if meta_path.exists():
        row |= {f"meta_{k}": v for k, v in json.loads(meta_path.read_text()).items()}
    return row


MAX_K = 4  # steps_to_k columns


def steps_to_k(record: dict[str, Any]) -> list[int | None]:
    """Env step at which the trial first had k = 1..MAX_K subtasks (None: never, or not logged)."""
    steps = [int(x) for x in record["completed_steps"].split("+")] if record.get("completed_steps") else []
    return [steps[k] if k < len(steps) else None for k in range(MAX_K)]


def speed_columns(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate speed metrics of one evaluation (see module docstring)."""
    per_trial = [steps_to_k(r) for r in records]
    columns: dict[str, Any] = {}
    for k in range(MAX_K):
        reached = [v for steps in per_trial if (v := steps[k]) is not None]
        columns[f"n_reached_{k + 1}"] = len(reached)
        columns[f"mean_steps_to_{k + 1}"] = sum(reached) / len(reached) if reached else None
    timeouts = [r["trial_time"] for r in records if r["result"] == "timeout"]
    horizon = max(timeouts) if timeouts else max(r["trial_time"] for r in records)
    columns["restricted_mean_steps_to_success"] = sum(r["trial_time"] if r["result"] == "success" else horizon for r in records) / len(
        records
    )
    return columns


def read_evaluations(
    run: str, run_eval_dir: Path, epochs: pd.DataFrame, latest_epoch: int | None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(per checkpoint and horizon, per trial, per subtask) tables of the evaluations under <run_eval_dir>/T_a_<H>/<stem>/."""
    summaries: list[dict[str, Any]] = []
    trials: list[dict[str, Any]] = []
    subtasks: list[dict[str, Any]] = []
    val_by_epoch = dict(zip(epochs["epoch"], epochs["val_ddim_mse"], strict=False))
    for results_path in sorted(run_eval_dir.glob("T_a_*/*/results.json")):
        horizon, stem = int(results_path.parent.parent.name.removeprefix("T_a_")), results_path.parent.name
        data = json.loads(results_path.read_text())
        records = data["trials"]
        if not records:
            continue
        match = CHECKPOINT_STEM.fullmatch(stem)
        epoch = int(match.group(1)) if match else latest_epoch if stem == "latest" else None  # latest.ckpt moves on
        n, n_success = len(records), sum(r["result"] == "success" for r in records)
        low, high = wilson_interval(n_success, n)
        successes = [r["trial_time"] for r in records if r["result"] == "success"]
        summaries.append({
            "run": run, "checkpoint": stem, "epoch": epoch, "is_latest": stem == "latest", "horizon": horizon,
            "val_ddim_mse": float(match.group(2)) if match else val_by_epoch.get(epoch),
            "n_trials": n, "n_success": n_success, "n_timeout": sum(r["result"] == "timeout" for r in records),
            "n_failure": sum(r["result"] == "failure" for r in records),
            "success_rate": n_success / n, "ci_low": low, "ci_high": high,
            "mean_subtasks": sum(r["reward"] for r in records) / n,
            "mean_subtasks_se": float(np.std([r["reward"] for r in records], ddof=1) / math.sqrt(n)) if n > 1 else None,
            **{f"frac_ge_{k}": sum(r["reward"] >= k for r in records) / n for k in (1, 2, 3)},
            "mean_plan_progress": sum(r["plan_progress"] for r in records) / n,
            "mean_success_steps": sum(successes) / len(successes) if successes else None,
            **speed_columns(records),
            "eval_dir": str(results_path.parent),
        })  # fmt: skip
        trials += [
            {"run": run, "checkpoint": stem, "horizon": horizon, "trial": r["trial"], "result": r["result"],
             "n_completed": int(r["reward"]), "trial_time": r["trial_time"], "plan": r["plan"],
             "completed": r["completed"], "plan_progress": r["plan_progress"], "completed_steps": r.get("completed_steps", ""),
             **{f"steps_to_{k + 1}": v for k, v in enumerate(steps_to_k(r))}}
            for r in records
        ]  # fmt: skip
        completed = [r["completed"].split("+") if r["completed"] else [] for r in records]
        subtasks += [
            {"run": run, "checkpoint": stem, "horizon": horizon, "subtask": name,
             "n_trials": n, "n_completed": sum(name in c for c in completed), "rate": sum(name in c for c in completed) / n}
            for name in SUBTASKS
        ]  # fmt: skip
    return pd.DataFrame(summaries), pd.DataFrame(trials), pd.DataFrame(subtasks)


def collect(config: CollectConfig) -> dict[str, Path]:
    """Write the tables (see module docstring); returns table name -> path."""
    runs = config.runs or tuple(sorted(p.name for p in config.runs_dir.iterdir() if (p / "logs.json.txt").exists()))
    run_rows: list[dict[str, Any]] = []
    steps, epochs, checkpoints, trials, subtasks = [], [], [], [], []
    for run in runs:
        run_dir = config.runs_dir / run
        run_steps, run_epochs = read_training_log(run, run_dir / "logs.json.txt")
        row = run_row(run, run_dir, run_steps, run_epochs)
        run_rows.append(row)
        run_checkpoints, run_trials, run_subtasks = read_evaluations(run, config.eval_dir / run, run_epochs, row.get("checkpoint_epoch"))
        steps.append(run_steps)
        epochs.append(run_epochs)
        checkpoints.append(run_checkpoints)
        trials.append(run_trials)
        subtasks.append(run_subtasks)
    tables = {
        "runs": pd.DataFrame(run_rows),
        "training_steps": pd.concat(steps, ignore_index=True),
        "training_epochs": pd.concat(epochs, ignore_index=True),
        "eval_checkpoints": pd.concat(checkpoints, ignore_index=True),
        "eval_trials": pd.concat(trials, ignore_index=True),
        "eval_subtasks": pd.concat(subtasks, ignore_index=True),
    }
    config.out_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: config.out_dir / f"{name}.csv" for name in tables}
    for name, table in tables.items():
        table.to_csv(paths[name], index=False)
        print(f"{paths[name]}: {len(table)} rows")
    return paths
