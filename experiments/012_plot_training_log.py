"""Plot train/val curves from a diffusion-policy run's logs.json.txt.

The workspace's JsonLogger writes one JSON line per train step (train_loss, lr,
global_step, epoch) plus one end-of-epoch line that also carries the val
metrics (val_loss, val_ddpm_mse, val_ddim_mse, ...). After a preemption the run
resumes from latest.ckpt and re-logs steps it had already logged, so rows are
de-duplicated by global_step, keeping the latest write.

Usage (login node is fine -- it only reads a text file):
    pixi run -e diffusion-policy-l40 python experiments/012_plot_training_log.py \
        third_party/diffusion-policy-experiments/data/outputs/franka_kitchen/<RUN_NAME>
Writes <run_dir>/training_curves.png and <run_dir>/epoch_metrics.csv.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_log(path: Path) -> list[dict]:
    by_step = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            by_step[row["global_step"]] = row
    return [by_step[k] for k in sorted(by_step)]


def smooth(y: np.ndarray, window: int) -> np.ndarray:
    if len(y) < window or window <= 1:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="valid")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--smooth", type=int, default=100, help="moving-average window (steps) for train_loss")
    args = parser.parse_args()

    rows = load_log(args.run_dir / "logs.json.txt")
    steps = np.array([r["global_step"] for r in rows])
    train_loss = np.array([r["train_loss"] for r in rows])
    lr = np.array([r["lr"] for r in rows])
    val_rows = [r for r in rows if "val_loss" in r]
    val_keys = ["val_loss", "val_ddpm_mse", "val_ddim_mse"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax = axes[0, 0]
    ax.plot(steps, train_loss, alpha=0.25, lw=0.5, label="per step")
    sm = smooth(train_loss, args.smooth)
    ax.plot(steps[len(steps) - len(sm) :], sm, lw=1.5, label=f"moving avg ({args.smooth})")
    ax.set(title="train_loss", xlabel="global_step", yscale="log")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(steps, lr)
    ax.set(title="learning rate", xlabel="global_step")

    ax = axes[1, 0]
    if val_rows:
        ax.plot([r["global_step"] for r in val_rows], [r["val_loss"] for r in val_rows], "o-", label="val_loss")
        ax.plot([r["global_step"] for r in val_rows], [r["train_loss"] for r in val_rows], "s--", label="train_loss (epoch mean)")
    ax.set(title="epoch-end loss", xlabel="global_step")
    ax.legend()

    ax = axes[1, 1]
    for key in ("val_ddpm_mse", "val_ddim_mse"):
        pts = [(r["global_step"], r[key]) for r in val_rows if key in r]
        if pts:
            ax.plot(*zip(*pts, strict=False), "o-", label=key)
    ax.set(title="action MSE (one val batch, EMA policy)", xlabel="global_step")
    ax.legend()

    fig.suptitle(args.run_dir.name)
    fig.tight_layout()
    fig.savefig(args.run_dir / "training_curves.png", dpi=120)

    with open(args.run_dir / "epoch_metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "global_step", "train_loss_epoch_mean", *val_keys])
        for r in val_rows:
            writer.writerow([r["epoch"], r["global_step"], r["train_loss"], *(r.get(k, "") for k in val_keys)])

    print(f"{len(rows)} steps, {len(val_rows)} epoch-end rows")
    for r in val_rows:
        print(f"  epoch {r['epoch']}: " + ", ".join(f"{k}={r[k]:.5f}" for k in ["train_loss", *val_keys] if k in r))
    print(f"wrote {args.run_dir / 'training_curves.png'} and epoch_metrics.csv")


if __name__ == "__main__":
    main()
