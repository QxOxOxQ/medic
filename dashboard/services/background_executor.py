from __future__ import annotations

import threading

from backend.execution import Task


class ThreadBackgroundExecutor:
    def submit(self, task: Task) -> None:
        threading.Thread(target=task, daemon=True).start()
