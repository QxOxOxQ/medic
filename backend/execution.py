from __future__ import annotations

from collections.abc import Callable
from typing import Protocol


Task = Callable[[], None]


class BackgroundExecutor(Protocol):
    def submit(self, task: Task) -> None: ...
