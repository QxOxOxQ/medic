from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from agents.models import AgentTraceEvent


class AgentTraceSink(Protocol):
    def record(self, event: AgentTraceEvent) -> None: ...


class AgentTraceRecorder:
    def __init__(self, sink: AgentTraceSink | None = None) -> None:
        self._events: list[AgentTraceEvent] = []
        self._next_sequence = 1
        self._sink = sink

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
        if self._sink is not None:
            self._sink.record(event)

    def events(self) -> tuple[AgentTraceEvent, ...]:
        return tuple(self._events)
