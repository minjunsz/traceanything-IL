"""Append-only JSON-lines training log (one row per step; epoch rows also carry validation metrics)."""

import json
from pathlib import Path
from typing import Any


class JsonLogger:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(path, "a")  # noqa: SIM115 (kept open for the whole run, closed in `close`)

    def log(self, row: dict[str, Any]) -> None:
        self.file.write(json.dumps(row) + "\n")
        self.file.flush()

    def close(self) -> None:
        self.file.close()
