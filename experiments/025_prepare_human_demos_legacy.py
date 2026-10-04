"""Validate and label the replayed Relay Policy Learning human demos.

The human teleop logs (kitchen_demos_multitask.zip, one folder per 4-subtask
plan, e.g. friday_microwave_kettle_bottomknob_slide) are first replayed with
physics by relay-policy-learning's parse_demos.py (--view playback --skip 40),
which writes one <demo>_path.pkl per log with observations {state, scene,
wrist} and normalized joint-velocity actions at 12.5 Hz -- the same action
space and rate as our Markovian-expert zarr.

Physics playback can drift from the original teleop trajectory, so this keeps
only episodes whose replay actually completes all 4 subtasks named by the
folder (same criterion as the Markovian data, which keeps successful episodes
only). Kept episodes are copied to --out-dir with an extra
observations["subtask_sequence"] (T, 28) one-hot of the folder's plan -- same
encoding as the Markovian data -- so a plan-conditioned variant stays possible;
the default human-data config does not use it.

Two steps (see experiments/025_human_demos_to_zarr_legacy.sbatch), in the franka-kitchen env:
    clean  -- mirror the unzipped folders as symlinks to the usable .mjl logs only:
              the zip holds byte-identical "(1)" copies (21 in
              postcorl_microwave_bottomknob_switch_slide) and at least one truncated
              log that makes parse_demos.py abort its whole folder.
    filter -- after parse_demos.py replayed the clean folders, keep episodes whose
              replay completes all 4 planned subtasks and add subtask_sequence.
    pixi run -e franka-kitchen python experiments/025_prepare_human_demos_legacy.py clean \
        --demo-root <unzipped>/kitchen_demos_multitask --out-dir <clean dir>
    pixi run -e franka-kitchen python experiments/025_prepare_human_demos_legacy.py filter \
        --demo-root <clean dir> --out-dir <filtered pkls>
"""

import argparse
import hashlib
import importlib.util
import json
import pickle
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent / "third_party/relay-policy-learning"

# Load experts/subtasks/base.py directly (numpy-only) for the exact subtask
# encoding used by the Markovian data and the eval script.
_spec = importlib.util.spec_from_file_location("_subtask_base", REPO / "experts/subtasks/base.py")
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
SUBTASK_IDS = _base.SUBTASK_IDS
sequence_onehot = _base.sequence_onehot

# Goal joint configurations, copied from eval/evaluate_kitchen.py (SUBTASK_INFO,
# BONUS_THRESH); indices are into state[:30] = [qp(9), obj_qp(21)].
SUBTASK_GOALS = {
    "bottomknob": (np.array([11, 12]), np.array([-0.88, -0.01])),
    "topknob": (np.array([15, 16]), np.array([-0.92, -0.01])),
    "light": (np.array([17, 18]), np.array([-0.69, -0.05])),
    "slide": (np.array([19]), np.array([0.37])),
    "hinge": (np.array([20, 21]), np.array([0.0, 1.45])),
    "microwave": (np.array([22]), np.array([-0.75])),
    "kettle": (np.array([23, 24, 25, 26, 27, 28, 29]), np.array([-0.23, 0.75, 1.62, 0.99, 0.0, 0.0, -0.06])),
}
BONUS_THRESH = 0.3
# Folder names say "switch" for the light switch.
NAME_ALIASES = {"switch": "light"}


def plan_from_folder(name: str) -> list:
    """friday_microwave_kettle_bottomknob_slide -> [microwave, kettle, bottomknob, slide]."""
    parts = name.split("_")[1:]  # drop the session prefix (friday / postcorl)
    plan = [NAME_ALIASES.get(p, p) for p in parts]
    assert len(plan) == 4 and all(p in SUBTASK_IDS for p in plan), (name, plan)
    return plan


def completed_subtasks(state: np.ndarray) -> list:
    """Subtasks whose goal is reached at any step of the (T, 60) state trajectory, in first-completion order."""
    first = {}
    for name, (idx, goal) in SUBTASK_GOALS.items():
        hit = np.nonzero(np.linalg.norm(state[:, idx] - goal, axis=-1) < BONUS_THRESH)[0]
        if len(hit):
            first[name] = int(hit[0])
    return sorted(first, key=first.get)


def clean(demo_root: Path, out_dir: Path) -> None:
    """Symlink every unique, parseable .mjl into out_dir/<plan folder>/."""
    sys.path.insert(0, str(REPO / "adept_envs/adept_envs/utils"))
    from parse_mjl import parse_mjl_logs

    if out_dir.exists():
        shutil.rmtree(out_dir)
    stats = Counter()
    skipped = []
    for folder in sorted(d for d in demo_root.iterdir() if d.is_dir()):
        seen = {}
        (out_dir / folder.name).mkdir(parents=True)
        # originals sort before their "(1)" copies, so the original name is kept
        for mjl in sorted(folder.glob("*.mjl"), key=lambda f: ("(1)" in f.name, f.name)):
            stats["logs"] += 1
            digest = hashlib.md5(mjl.read_bytes()).hexdigest()
            if digest in seen:
                stats["duplicate"] += 1
                skipped.append({"file": str(mjl), "reason": f"duplicate of {seen[digest]}"})
                continue
            seen[digest] = mjl.name
            try:
                parse_mjl_logs(str(mjl), 40)
            except Exception as e:  # truncated / corrupt log
                stats["unreadable"] += 1
                skipped.append({"file": str(mjl), "reason": f"{type(e).__name__}: {e}"})
                continue
            stats["kept"] += 1
            (out_dir / folder.name / mjl.name).symlink_to(mjl)
    with open(out_dir / "clean_summary.json", "w") as f:
        json.dump({**stats, "skipped": skipped}, f, indent=1)
    print(json.dumps(stats), f"({len(skipped)} skipped, see clean_summary.json)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["clean", "filter"])
    p.add_argument("--demo-root", required=True, help="dir containing one folder per plan")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    if args.mode == "clean":
        clean(Path(args.demo_root), Path(args.out_dir))
        return

    out_dir = Path(args.out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    stats = Counter()
    rejected = []
    lengths = []
    for folder in sorted(d for d in Path(args.demo_root).iterdir() if d.is_dir()):
        plan = plan_from_folder(folder.name)
        onehot = sequence_onehot(plan).astype(np.float64)
        for pkl in sorted(folder.glob("*_path.pkl")):
            with open(pkl, "rb") as f:
                path = pickle.load(f)
            state = np.asarray(path["observations"]["state"])
            done = completed_subtasks(state)
            stats["total"] += 1
            if not set(plan) <= set(done):
                stats["rejected"] += 1
                rejected.append({"file": str(pkl), "plan": plan, "completed": done})
                continue
            stats["kept"] += 1
            stats["kept_in_plan_order"] += int([s for s in done if s in plan] == plan)
            T = len(path["actions"])
            path["observations"]["subtask_sequence"] = np.repeat(onehot[None], T, axis=0)
            lengths.append(T)
            dst = out_dir / folder.name
            dst.mkdir(exist_ok=True)
            with open(dst / pkl.name, "wb") as f:
                pickle.dump(path, f)

    lengths = np.array(lengths)
    summary = {
        **stats,
        "steps_total": int(lengths.sum()),
        "episode_len_mean": float(lengths.mean()),
        "episode_len_pcts_50_90_max": np.percentile(lengths, [50, 90, 100]).tolist(),
        "rejected": rejected,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "rejected"}, indent=1))


if __name__ == "__main__":
    main()
