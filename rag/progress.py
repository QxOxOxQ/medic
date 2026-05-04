from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any


ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class ProgressEvent:
    timestamp: str
    step: str
    status: str
    message: str
    counters: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class ProgressEmitter:
    def __init__(self, callback: ProgressCallback | None) -> None:
        self._callback = callback

    def emit(
        self,
        *,
        step: str,
        status: str,
        message: str,
        counters: dict[str, Any] | None = None,
        result: dict[str, Any] | None = None,
    ) -> None:
        emit_progress(
            self._callback,
            step=step,
            status=status,
            message=message,
            counters=counters,
            result=result,
        )


def emit_progress(
    callback: ProgressCallback | None,
    *,
    step: str,
    status: str,
    message: str,
    counters: dict[str, Any] | None = None,
    result: dict[str, Any] | None = None,
) -> None:
    if callback is None:
        return

    callback(
        ProgressEvent(
            timestamp=_timestamp(),
            step=step,
            status=status,
            message=message,
            counters=counters or {},
            result=result or {},
        ).as_dict()
    )


def _timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")

