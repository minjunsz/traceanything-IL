"""Success rate vs. execution horizon (actions executed per policy call), from eval runs.

Expected layout: <experiment>/T_a_<horizon>/<checkpoint>/results.json (see markovian_policy.evaluation.runner). For each horizon
the checkpoint with the most trials (then the highest success rate) is plotted, with Wilson 95% intervals.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

NAVY, DARK_RED = "#2f5fb3", "#c0392b"
Z95 = 1.959964


@dataclass
class HorizonResult:
    horizon: int
    n_success: int
    n_total: int
    checkpoint: str

    @property
    def rate(self) -> float:
        return self.n_success / self.n_total


def wilson_interval(n_success: int, n_total: int, z: float = Z95) -> tuple[float, float]:
    p = n_success / n_total
    center = (p + z**2 / (2 * n_total)) / (1 + z**2 / n_total)
    half = z * math.sqrt(p * (1 - p) / n_total + z**2 / (4 * n_total**2)) / (1 + z**2 / n_total)
    return center - half, center + half


def collect_best(experiment: Path, horizons: tuple[int, int] | None = None) -> list[HorizonResult]:
    best: list[HorizonResult] = []
    for horizon_dir in sorted(experiment.glob("T_a_*")):
        horizon = int(horizon_dir.name.removeprefix("T_a_"))
        if horizons and not horizons[0] <= horizon <= horizons[1]:
            continue
        candidates = []
        for results in sorted(horizon_dir.glob("*/results.json")):
            data = json.loads(results.read_text())
            if data["n_total"] > 0:
                candidates.append(HorizonResult(horizon, data["n_success"], data["n_total"], results.parent.name))
        if candidates:
            best.append(max(candidates, key=lambda r: (r.n_total, r.rate)))
    return sorted(best, key=lambda r: r.horizon)


def colors(n: int) -> list[str]:
    """Gradient from navy (the first experiment, the baseline) to dark red."""
    if n == 1:
        return [NAVY]
    a, b = (np.array([int(c[i : i + 2], 16) for i in (1, 3, 5)]) for c in (NAVY, DARK_RED))
    return ["#{:02x}{:02x}{:02x}".format(*tuple(np.round(a + (b - a) * t).astype(int))) for t in np.linspace(0, 1, n)]


def plot_experiments(experiments: dict[str, list[HorizonResult]], title: str | None = None) -> Figure:
    fig, ax = plt.subplots(figsize=(3.4, 2.6), layout="constrained")
    for (label, results), color in zip(experiments.items(), colors(len(experiments)), strict=False):
        x = np.array([r.horizon for r in results], dtype=float)
        rate = np.array([r.rate for r in results])
        low, high = np.array([wilson_interval(r.n_success, r.n_total) for r in results]).T
        ax.plot(x, rate, color=color, marker="o", markersize=4, markeredgecolor="white", linewidth=1.5, label=label, zorder=3)
        ax.errorbar(x, rate, yerr=[np.clip(rate - low, 0, 1), np.clip(high - rate, 0, 1)], fmt="none", ecolor=color, capsize=2, alpha=0.9)
    ax.set_xticks(sorted({r.horizon for rs in experiments.values() for r in rs}))
    ax.set_xlabel("Execution Horizon (steps)", fontsize=9)
    ax.set_ylabel("Success Rate", fontsize=9)
    ax.grid(True, color="#bdbdbd", alpha=0.6)
    ax.tick_params(direction="in", labelsize=7)
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold")
    if len(experiments) > 1:
        ax.legend(fontsize=7)
    return fig
