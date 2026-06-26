from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

from backend.chat_models import ChatConversationDetail, ChatTraceEventView


@dataclass(frozen=True)
class ChatRunStarted:
    conversation_id: UUID
    run_id: UUID

    def as_dict(self) -> dict[str, str]:
        return {
            "conversation_id": str(self.conversation_id),
            "run_id": str(self.run_id),
        }


@dataclass(frozen=True)
class ChatRunView:
    id: UUID
    conversation_id: UUID
    status: str
    question: str
    error: str | None
    started_at: datetime
    finished_at: datetime | None
    trace_events: tuple[ChatTraceEventView, ...]
    conversation: ChatConversationDetail | None

    @property
    def is_terminal(self) -> bool:
        return self.status in {"succeeded", "failed", "interrupted"}

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "conversation_id": str(self.conversation_id),
            "status": self.status,
            "question": self.question,
            "error": self.error,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "trace_events": [event.as_dict() for event in self.trace_events],
            "conversation": (
                self.conversation.as_dict() if self.conversation is not None else None
            ),
        }
