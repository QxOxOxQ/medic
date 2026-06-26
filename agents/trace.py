from __future__ import annotations

import logging
from collections.abc import Mapping
from threading import Lock
from typing import Any, Protocol

from agents.models import AgentTraceEvent


logger = logging.getLogger("medic.agents.trace")

_STATUS_LEVELS = {
    "failed": logging.ERROR,
    "degraded": logging.WARNING,
    "retrying": logging.WARNING,
}


class AgentTraceSink(Protocol):
    def record(self, event: AgentTraceEvent) -> None: ...


class AgentTraceRecorder:
    def __init__(self, sink: AgentTraceSink | None = None) -> None:
        self._events: list[AgentTraceEvent] = []
        self._next_sequence = 1
        self._sink = sink
        self._lock = Lock()

    def record(
        self,
        *,
        event_type: str,
        title: str,
        status: str,
        agent_name: str | None = None,
        tool_name: str | None = None,
        payload: Mapping[str, Any] | None = None,
        duration_ms: int | None = None,
    ) -> None:
        with self._lock:
            event = AgentTraceEvent(
                sequence=self._next_sequence,
                event_type=event_type,
                title=title,
                status=status,
                agent_name=agent_name,
                tool_name=tool_name,
                payload=dict(payload or {}),
                duration_ms=duration_ms,
            )
            self._events.append(event)
            self._next_sequence += 1
        _log_event(event)
        if self._sink is not None:
            self._sink.record(event)

    def events(self) -> tuple[AgentTraceEvent, ...]:
        with self._lock:
            return tuple(self._events)


def _log_event(event: AgentTraceEvent) -> None:
    level = _STATUS_LEVELS.get(event.status, logging.INFO)
    parts: list[str] = [event.event_type]
    if event.agent_name:
        parts.append(event.agent_name)
    phase = event.payload.get("phase")
    if phase:
        parts.append(str(phase))
    model = event.payload.get("model")
    if model:
        parts.append(f"model={model}")
    if event.tool_name:
        parts.append(f"tool={event.tool_name}")
    detail = event.payload.get("error") or event.payload.get("reason")
    suffix = f" — {detail}" if detail else ""
    logger.log(
        level,
        "[%02d] %s · %s (%s)%s",
        event.sequence,
        " · ".join(parts),
        event.title,
        event.status,
        suffix,
    )
