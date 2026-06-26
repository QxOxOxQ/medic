from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from html import escape
from uuid import UUID, uuid4


class AgentError(RuntimeError):
    """Base error for agent orchestration failures."""


class AgentExecutionError(AgentError):
    """Raised when the agent cannot produce an answer."""


class UnknownAgentError(AgentError):
    """Raised when a requested agent profile does not exist."""


@dataclass(frozen=True)
class AgentSource:
    id: str
    source: str | None
    content_hash: str | None
    document_name: str | None
    score: float | None
    excerpt: str
    qdrant_point_id: str | None = None
    document_id: UUID | None = None
    relative_raw_path: str | None = None
    chunk_index: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    retrieval_query: str | None = None
    full_content: str | None = None

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        if self.document_id is not None:
            payload["document_id"] = str(self.document_id)
        return payload

    def prompt_block(self, *, full: bool = True) -> str:
        source = self.document_name or self.source or "unknown"
        content = self.excerpt
        content_type = "excerpt"
        if full and self.full_content:
            content = self.full_content
            content_type = "full_document"
        return (
            f'<untrusted_source id="{self.id}">\n'
            f"source_name: {escape(source)}\n"
            f"content_type: {content_type}\n"
            "content:\n"
            f"{escape(content)}\n"
            "</untrusted_source>"
        )


@dataclass(frozen=True)
class ChatHistoryMessage:
    role: str
    content: str


@dataclass(frozen=True)
class AgentRequest:
    question: str
    requested_agent: str | None = None
    conversation_messages: tuple[ChatHistoryMessage, ...] = ()
    user_id: UUID | None = None
    session_id: UUID | None = None
    execution_id: UUID = field(default_factory=uuid4)


@dataclass(frozen=True)
class AgentTraceEvent:
    sequence: int
    event_type: str
    title: str
    status: str
    agent_name: str | None = None
    tool_name: str | None = None
    payload: Mapping[str, object] = field(default_factory=dict)
    duration_ms: int | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "sequence": self.sequence,
            "event_type": self.event_type,
            "title": self.title,
            "status": self.status,
            "agent_name": self.agent_name,
            "tool_name": self.tool_name,
            "payload": dict(self.payload),
            "duration_ms": self.duration_ms,
        }


@dataclass(frozen=True)
class AgentAnswer:
    answer: str
    agents: tuple[str, ...]
    sources: tuple[AgentSource, ...]
    insufficient_context: bool
    trace_events: tuple[AgentTraceEvent, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "answer": self.answer,
            "agents": list(self.agents),
            "sources": [source.as_dict() for source in self.sources],
            "insufficient_context": self.insufficient_context,
            "trace_events": [event.as_dict() for event in self.trace_events],
        }
