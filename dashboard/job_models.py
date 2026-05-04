from __future__ import annotations

import threading
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True)
class JobEvent:
    id: int
    timestamp: str
    step: str
    status: str
    message: str
    counters: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class PipelineJob:
    def __init__(self, job_id: str) -> None:
        self.id = job_id
        self.status = "queued"
        self.started_at: str | None = None
        self.finished_at: str | None = None
        self.error: str | None = None
        self._events: list[JobEvent] = []
        self._next_event_id = 1
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            self.status = "running"
            self.started_at = _timestamp()

    def finish(self, status: str, *, error: str | None = None) -> None:
        with self._lock:
            self.status = status
            self.error = error
            self.finished_at = _timestamp()

    def emit(
        self,
        *,
        step: str,
        status: str,
        message: str,
        counters: dict[str, Any] | None = None,
        result: dict[str, Any] | None = None,
        timestamp: str | None = None,
    ) -> JobEvent:
        with self._lock:
            event = JobEvent(
                id=self._next_event_id,
                timestamp=timestamp or _timestamp(),
                step=step,
                status=status,
                message=message,
                counters=counters or {},
                result=result or {},
            )
            self._next_event_id += 1
            self._events.append(event)
            return event

    def emit_progress(self, payload: dict[str, Any]) -> None:
        self.emit(
            step=str(payload.get("step", "pipeline")),
            status=str(payload.get("status", "running")),
            message=str(payload.get("message", "")),
            counters=_dict_payload(payload.get("counters")),
            result=_dict_payload(payload.get("result")),
            timestamp=str(payload.get("timestamp") or _timestamp()),
        )

    def events_after(self, event_id: int) -> list[JobEvent]:
        with self._lock:
            return [event for event in self._events if event.id > event_id]

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "id": self.id,
                "status": self.status,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "error": self.error,
                "events": [event.as_dict() for event in self._events],
            }

    @property
    def is_terminal(self) -> bool:
        return self.status in {"succeeded", "failed"}


def _timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _dict_payload(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}
