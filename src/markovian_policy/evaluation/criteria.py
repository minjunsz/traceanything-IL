"""Success criteria for a trial."""

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class DistinctSubtasks:
    """Success once at least `required` distinct subtasks were completed, in any order."""

    required: int = 4

    def is_success(self, completed: Sequence[str]) -> bool:
        return len(set(completed)) >= self.required
