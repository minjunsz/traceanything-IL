"""Demonstrations from human teleoperation logs (kitchen_demos_multitask.zip).

The zip holds one folder per planned 4-subtask sequence (e.g. friday_microwave_kettle_bottomknob_slide), with
byte-identical "(1)" copies and some truncated logs. Each unique, readable log is replayed with physics
(`replay_log`). A replay succeeds if it completes every subtask of its folder's plan. Episodes get
`subtask_sequence` (the plan one-hot), so plan-conditioned policies stay possible.
"""

import hashlib
import zipfile
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Final

import numpy as np

from markovian_policy.data.generation.mjl import parse_mjl
from markovian_policy.data.generation.protocols import Candidate
from markovian_policy.data.generation.replay import replay_log
from markovian_policy.sim.env import KitchenEnv, KitchenEnvConfig
from markovian_policy.sim.tasks import PLAN_ENCODING, SUBTASK_IDS

FOLDER_ALIASES: Final = {"switch": "light"}  # folder names call the light switch "switch"


def plan_from_folder(name: str) -> list[str] | None:
    """friday_microwave_kettle_bottomknob_slide -> [microwave, kettle, bottomknob, slide] (None: not a plan folder)."""
    plan = [FOLDER_ALIASES.get(part, part) for part in name.split("_")[1:]]  # drop the session prefix
    return plan if len(plan) == 4 and all(p in SUBTASK_IDS for p in plan) else None


def list_logs(zip_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Names of the usable logs in the zip, and the skipped ones with the reason (duplicate/unreadable/no plan).

    A log's plan is its immediate folder, so logs nested in another plan's folder are attributed correctly.
    """
    usable: list[str] = []
    skipped: list[dict[str, str]] = []
    with zipfile.ZipFile(zip_path) as archive:
        names = [info.filename for info in archive.infolist() if info.filename.endswith(".mjl")]
        seen: dict[tuple[str, str], str] = {}  # (folder, content hash) -> kept log
        # originals sort before their "(1)" copies, so the original name is kept
        for name in sorted(names, key=lambda n: (str(PurePosixPath(n).parent), "(1)" in n, n)):
            folder = PurePosixPath(name).parent.name
            data = archive.read(name)
            key = (folder, hashlib.md5(data).hexdigest())
            if plan_from_folder(folder) is None:
                skipped.append({"file": name, "reason": "folder is not a 4-subtask plan"})
            elif key in seen:
                skipped.append({"file": name, "reason": f"duplicate of {seen[key]}"})
            else:
                try:
                    parse_mjl(data)
                except Exception as error:  # truncated or corrupt log
                    skipped.append({"file": name, "reason": f"{type(error).__name__}: {error}"})
                    continue
                seen[key] = name
                usable.append(name)
    return usable, skipped


@dataclass(frozen=True)
class HumanReplaySource:
    """`DemoSource` of replayed human logs: one job per usable log, identified by its name in the zip."""

    zip_path: Path
    env: KitchenEnvConfig = field(default_factory=KitchenEnvConfig)
    max_logs: int | None = None  # smoke test: only this many logs, evenly spread over the usable ones

    def jobs(self) -> Iterator[str]:
        usable, _ = list_logs(self.zip_path)
        if self.max_logs:
            usable = usable[:: max(1, len(usable) // self.max_logs)][: self.max_logs]
        return iter(usable)

    def key(self, job: str) -> str:
        return job

    def worker(self) -> Callable[[str], Candidate]:
        env = KitchenEnv(self.env)

        def replay(name: str) -> Candidate:
            plan = plan_from_folder(PurePosixPath(name).parent.name)
            assert plan is not None
            with zipfile.ZipFile(self.zip_path) as archive:
                log = parse_mjl(archive.read(name), skip=self.env.frame_skip)  # one log record per control step
            result = replay_log(env, log)
            if result is None:
                return Candidate(name, None, False, "unstable")
            episode, completed = result
            episode["subtask_sequence"] = np.tile(PLAN_ENCODING.plan_onehot(plan), (len(episode["action"]), 1))
            if not set(plan) <= set(completed):
                return Candidate(name, episode, False, "incomplete")
            in_order = [s for s in completed if s in plan] == plan
            return Candidate(name, episode, True, "kept_in_plan_order" if in_order else "kept", {"plan": "+".join(plan)})

        return replay

    def report(self) -> Mapping[str, Any]:
        usable, skipped = list_logs(self.zip_path)
        return {"usable_logs": len(usable), "skipped_logs": skipped}
