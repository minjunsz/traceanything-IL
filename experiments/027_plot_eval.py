"""Plot success rate vs. execution horizon from eval runs (layout: <experiment>/T_a_<H>/<ckpt>/results.json).

python experiments/027_plot_eval.py --experiments output/eval/runA output/eval/runB --labels A B --out plot.png
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import tyro

from markovian_policy.evaluation.plots import collect_best, plot_experiments


@dataclass
class Config:
    experiments: tuple[Path, ...]
    out: Path
    labels: tuple[str, ...] | None = None  # legend labels (default: directory names)
    title: str | None = None
    horizons: tuple[int, int] = (1, 8)  # inclusive range of execution horizons
    dpi: int = 300


def main(config: Config) -> None:
    labels = config.labels or tuple(p.name for p in config.experiments)
    assert len(labels) == len(config.experiments), "need one label per experiment"
    results = {label: collect_best(path, config.horizons) for label, path in zip(labels, config.experiments, strict=False)}
    for label, rs in results.items():
        print(label, [f"T_a_{r.horizon}: {r.n_success}/{r.n_total} ({r.checkpoint})" for r in rs])
    fig = plot_experiments({k: v for k, v in results.items() if v}, config.title)
    config.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.out, dpi=config.dpi, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


if __name__ == "__main__":
    main(tyro.cli(Config))
