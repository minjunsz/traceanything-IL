"""Result collection: tidy tables from training logs and evaluation results (synthetic files, no GPU)."""

import json
from pathlib import Path

import pandas as pd

from markovian_policy.reporting import CollectConfig, collect


def _write_run(root: Path) -> None:
    run = root / "runs" / "r1"
    run.mkdir(parents=True)
    rows = [
        {"train_loss": 1.0, "global_step": 1, "epoch": 0, "lr": 1e-6},
        {"train_loss": 0.9, "global_step": 2, "epoch": 0, "lr": 2e-6},
        {"epoch": 0, "global_step": 2, "train_loss": 0.95, "val_loss": 1.0, "val_ddpm_mse": 0.5, "val_ddim_mse": 0.4},
        {"train_loss": 0.8, "global_step": 3, "epoch": 1, "lr": 3e-6},  # logged before the run was resumed ...
        {"train_loss": 0.7, "global_step": 3, "epoch": 1, "lr": 3e-6},  # ... and logged again after
        {"epoch": 1, "global_step": 3, "train_loss": 0.75, "val_loss": 0.9, "val_ddpm_mse": 0.3, "val_ddim_mse": 0.2},
    ]
    (run / "logs.json.txt").write_text("\n".join(json.dumps(r) for r in rows) + '\n{"train_loss": 0.6, "glob')  # partial line
    (run / "meta.json").write_text(json.dumps({"data": "human", "history": 16}))


def _write_eval(root: Path, stem: str, horizon: int, results: list[str]) -> None:
    out = root / "eval" / "r1" / f"T_a_{horizon}" / stem
    out.mkdir(parents=True)
    trials = [
        {"trial": i + 1, "result": r, "reward": 4.0 if r == "success" else 2.0, "trial_time": 200 if r == "success" else 600,
         "plan": "a+b+c+d", "completed": "microwave+kettle", "plan_progress": 2,
         **({"completed_steps": "10+50+90+200"} if r == "success" else {"completed_steps": "30+400"})}
        for i, r in enumerate(results)
    ]  # fmt: skip
    (out / "results.json").write_text(json.dumps({"n_success": results.count("success"), "n_total": len(results), "trials": trials}))


def test_collect_writes_tidy_tables_and_dedupes_resumed_steps(tmp_path: Path) -> None:
    _write_run(tmp_path)
    _write_eval(tmp_path, "epoch=001-val_ddim_mse=0.200000", 8, ["success", "timeout", "success", "success"])
    _write_eval(tmp_path, "latest", 8, ["timeout", "timeout"])
    config = CollectConfig(runs_dir=tmp_path / "runs", eval_dir=tmp_path / "eval", out_dir=tmp_path / "out")
    paths = collect(config)

    steps = pd.read_csv(paths["training_steps"])
    assert steps["global_step"].tolist() == [1, 2, 3] and steps.loc[2, "train_loss"] == 0.7  # the later line wins
    epochs = pd.read_csv(paths["training_epochs"])
    assert epochs["val_ddim_mse"].tolist() == [0.4, 0.2]

    runs = pd.read_csv(paths["runs"])
    assert runs.loc[0, "best_epoch"] == 1 and runs.loc[0, "meta_history"] == 16 and runs.loc[0, "meta_data"] == "human"

    checkpoints = pd.read_csv(paths["eval_checkpoints"]).set_index("checkpoint")
    best = checkpoints.loc["epoch=001-val_ddim_mse=0.200000"]
    assert (best["epoch"], best["n_trials"], best["n_success"], best["success_rate"]) == (1, 4, 3, 0.75)
    assert best["ci_low"] < 0.75 < best["ci_high"] and best["mean_success_steps"] == 200 and best["mean_subtasks"] == 3.5
    assert bool(checkpoints.loc["latest", "is_latest"]) and checkpoints.loc["latest", "n_success"] == 0

    assert best["frac_ge_3"] == 0.75 and best["mean_subtasks_se"] > 0
    subtasks = pd.read_csv(paths["eval_subtasks"])
    one = subtasks[(subtasks["checkpoint"] == "latest") & (subtasks["subtask"] == "slide")].iloc[0]
    assert len(subtasks) == 2 * 7 and one["n_completed"] == 0 and one["n_trials"] == 2
    microwave = subtasks[(subtasks["checkpoint"] == "latest") & (subtasks["subtask"] == "microwave")].iloc[0]
    assert microwave["n_completed"] == 2 and microwave["rate"] == 1.0

    assert best["n_reached_1"] == 4 and best["n_reached_3"] == 3 and best["mean_steps_to_1"] == (10 * 3 + 30) / 4
    assert best["mean_steps_to_4"] == 200 and best["restricted_mean_steps_to_success"] == (200 * 3 + 600) / 4
    assert pd.isna(checkpoints.loc["latest", "mean_steps_to_3"])

    trials = pd.read_csv(paths["eval_trials"])
    assert trials.loc[0, "steps_to_4"] == 200 and pd.isna(trials.loc[1, "steps_to_3"])
    assert len(trials) == 6 and set(trials["checkpoint"]) == {"epoch=001-val_ddim_mse=0.200000", "latest"}
