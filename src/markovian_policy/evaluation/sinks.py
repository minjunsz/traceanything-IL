"""Observers of an evaluation run: result files, videos and the recorded rollouts."""

import csv
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

from markovian_policy.arrays import Image
from markovian_policy.data.episodes import EpisodeWriter
from markovian_policy.evaluation.results import TrialRecord, TrialResult
from markovian_policy.sim.tasks import SUBTASK_IDS

CSV_FIELDS = ["trial", "result", "reward", "trial_time", "plan", "completed", "plan_progress"]


class ResultFiles:
    """results.csv (one row per trial), results.json (resume + plots), summary.txt; rewritten after every round."""

    wants_video = False
    wants_episodes = False

    def __init__(self, out_dir: Path, n_rollouts: int, config: Mapping[str, Any]) -> None:
        self.out_dir, self.n_rollouts, self.config = out_dir, n_rollouts, dict(config)
        out_dir.mkdir(parents=True, exist_ok=True)

    def load_trials(self) -> list[TrialRecord]:
        """Trials of an earlier run in this directory (to resume), or none."""
        path = self.out_dir / "results.json"
        return json.loads(path.read_text())["trials"] if path.exists() else []

    def wants_video_for(self, first_trial: int) -> bool:
        return False

    def on_trial(self, number: int, result: TrialResult) -> None:
        pass

    def on_round_end(self, trials: Sequence[TrialRecord]) -> None:
        with open(self.out_dir / "results.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(trials)
        n_success = sum(t["result"] == "success" for t in trials)
        payload = {"n_success": n_success, "n_total": len(trials), "trials": list(trials), "config": self.config}
        (self.out_dir / "results.json").write_text(json.dumps(payload, indent=1))
        (self.out_dir / "summary.txt").write_text(summarize(trials, self.n_rollouts))

    def close(self) -> None:
        pass


def summarize(trials: Sequence[TrialRecord], n_rollouts: int) -> str:
    n = len(trials)
    outcome = Counter(t["result"] for t in trials)
    per_subtask = Counter(s for t in trials for s in t["completed"].split("+") if s)
    lines = [
        f"Trials completed : {n} / {n_rollouts}",
        f"Successes        : {outcome['success']}",
        f"Failures         : {outcome['failure']}",
        f"Timeouts         : {outcome['timeout']}",
    ]
    if n:
        lines += [
            f"Success rate     : {outcome['success'] / n:.1%}",
            f"Avg trial steps  : {np.mean([t['trial_time'] for t in trials]):.1f}",
            f"Avg subtasks     : {np.mean([t['reward'] for t in trials]):.2f}",
            f"Avg plan progress: {np.mean([t['plan_progress'] for t in trials]):.2f} (leading plan subtasks done in order)",
        ]
    lines.append("Completed per subtask: " + ", ".join(f"{s}={per_subtask[s]}" for s in SUBTASK_IDS))
    return "\n".join(lines) + "\n"


class VideoSink:
    """mp4s of the first `n_trials` trials (-1: all, 0: none), or of all failed trials with `failures_only`."""

    wants_episodes = False

    def __init__(self, out_dir: Path, n_trials: int, n_rollouts: int, failures_only: bool = False) -> None:
        self.directory = out_dir / "videos"
        self.budget = n_rollouts if n_trials < 0 else n_trials
        self.failures_only = failures_only

    @property
    def wants_video(self) -> bool:
        return self.failures_only or self.budget > 0

    def wants_video_for(self, first_trial: int) -> bool:
        return self.failures_only or first_trial <= self.budget

    def on_trial(self, number: int, result: TrialResult) -> None:
        keep = result.result != "success" if self.failures_only else number <= self.budget
        if keep and result.frames:
            self.directory.mkdir(parents=True, exist_ok=True)
            write_video(result.frames, self.directory / f"trial_{number:04d}_{result.result}.mp4")

    def on_round_end(self, trials: Sequence[TrialRecord]) -> None:
        pass

    def close(self) -> None:
        pass


def write_video(frames: Sequence[Image], path: Path, fps: int = 10) -> None:
    imageio.mimwrite(path, list(frames), fps=fps, codec="libx264", pixelformat="yuv420p")


class EpisodeSink:
    """Appends every trial's rollout to a dataset (layout of the demonstrations, plus qpos/qvel)."""

    wants_video = False
    wants_episodes = True

    def __init__(self, path: Path, resume: bool) -> None:
        self.writer = EpisodeWriter(path, resume=resume)

    def wants_video_for(self, first_trial: int) -> bool:
        return False

    def on_trial(self, number: int, result: TrialResult) -> None:
        if result.episode is not None:
            self.writer.add(result.episode)

    def on_round_end(self, trials: Sequence[TrialRecord]) -> None:
        pass

    def close(self) -> None:
        pass
