from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID


@dataclass(frozen=True)
class ChatSourceView:
    id: UUID
    source_id: str
    source: str | None
    content_hash: str | None
    document_id: UUID | None
    document_name: str | None
    relative_raw_path: str | None
    qdrant_point_id: str | None
    chunk_index: int | None
    char_start: int | None
    char_end: int | None
    retrieval_query: str | None
    score: float | None
    excerpt: str
    used: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "source_id": self.source_id,
            "source": self.source,
            "content_hash": self.content_hash,
            "document_id": str(self.document_id) if self.document_id else None,
            "document_name": self.document_name,
            "relative_raw_path": self.relative_raw_path,
            "qdrant_point_id": self.qdrant_point_id,
            "chunk_index": self.chunk_index,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "retrieval_query": self.retrieval_query,
            "score": self.score,
            "excerpt": self.excerpt,
            "used": self.used,
        }


@dataclass(frozen=True)
class ChatTraceEventView:
    id: UUID
    sequence: int
    event_type: str
    title: str
    status: str
    agent_name: str | None
    tool_name: str | None
    payload: dict[str, Any]
    duration_ms: int | None
    created_at: datetime

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "sequence": self.sequence,
            "event_type": self.event_type,
            "phase": _trace_phase(self),
            "title": self.title,
            "status": self.status,
            "agent_name": self.agent_name,
            "tool_name": self.tool_name,
            "payload": self.payload,
            "duration_ms": self.duration_ms,
            "created_at": self.created_at.isoformat(),
        }


def _trace_phase(event: ChatTraceEventView) -> str:
    if event.event_type == "coordinator":
        return "coordinator"
    if event.event_type == "tool_call":
        return "retrieval"
    if event.event_type == "source_expansion":
        return "expansion"
    if event.event_type == "review":
        return "review"
    if event.event_type == "synthesis":
        return "synthesis"
    if event.event_type == "error":
        return "error"
    return "specialist"


@dataclass(frozen=True)
class ChatMessageView:
    id: UUID
    role: str
    content: str
    sequence: int
    insufficient_context: bool
    created_at: datetime
    sources: tuple[ChatSourceView, ...] = ()
    trace_events: tuple[ChatTraceEventView, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "role": self.role,
            "content": self.content,
            "sequence": self.sequence,
            "insufficient_context": self.insufficient_context,
            "created_at": self.created_at.isoformat(),
            "sources": [source.as_dict() for source in self.sources],
            "trace_events": [event.as_dict() for event in self.trace_events],
        }


@dataclass(frozen=True)
class ChatConversationSummary:
    id: UUID
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "title": self.title,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "message_count": self.message_count,
        }


@dataclass(frozen=True)
class ChatConversationDetail:
    id: UUID
    title: str
    created_at: datetime
    updated_at: datetime
    messages: tuple[ChatMessageView, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "title": self.title,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "messages": [message.as_dict() for message in self.messages],
        }
