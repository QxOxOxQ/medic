from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID


@dataclass(frozen=True)
class PipelineDocumentSnapshot:
    document_id: UUID | None
    position: int
    document_name: str
    relative_raw_path: str
    status: str
    current_step: str | None
    error: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "document_id": str(self.document_id) if self.document_id else None,
            "position": self.position,
            "document_name": self.document_name,
            "relative_raw_path": self.relative_raw_path,
            "status": self.status,
            "current_step": self.current_step,
            "error": self.error,
        }


@dataclass(frozen=True)
class PipelineEventView:
    sequence: int
    timestamp: datetime
    step: str
    status: str
    message: str
    counters: dict[str, Any]
    result: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "timestamp": self.timestamp.isoformat(),
            "step": self.step,
            "status": self.status,
            "message": self.message,
            "counters": self.counters,
            "result": self.result,
        }


@dataclass(frozen=True)
class PipelineRunView:
    id: UUID
    owner_user_id: UUID
    status: str
    summary: str | None
    error: str | None
    started_at: datetime | None
    finished_at: datetime | None
    created_at: datetime
    updated_at: datetime
    documents: tuple[PipelineDocumentSnapshot, ...] = ()
    events: tuple[PipelineEventView, ...] = ()

    @property
    def is_terminal(self) -> bool:
        return self.status in {"succeeded", "failed", "interrupted"}

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "status": self.status,
            "summary": self.summary,
            "error": self.error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": (
                self.finished_at.isoformat() if self.finished_at else None
            ),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "documents": [document.as_dict() for document in self.documents],
            "events": [event.as_dict() for event in self.events],
        }


@dataclass(frozen=True)
class CreatedPipelineRun:
    run: PipelineRunView
    selected_raw_paths: tuple[str, ...]
