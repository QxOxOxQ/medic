from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, cast
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload, sessionmaker

from backend.pipeline_models import (
    CreatedPipelineRun,
    PipelineDocumentSnapshot,
    PipelineEventView,
    PipelineRunView,
)
from rag.database.models import (
    Document,
    PipelineRun,
    PipelineRunDocument,
    PipelineRunEvent,
    utc_now,
)
from rag.database.session import session_scope


class SqlAlchemyPipelineRunRepository:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def create(
        self,
        *,
        owner_user_id: UUID,
        document_ids: tuple[UUID, ...],
    ) -> CreatedPipelineRun:
        with session_scope(self._session_factory) as session:
            documents = _owned_documents(
                session,
                owner_user_id=owner_user_id,
                document_ids=document_ids,
            )
            if document_ids and len(documents) != len(set(document_ids)):
                raise ValueError("One or more selected documents were not found")
            run = PipelineRun(owner_user_id=owner_user_id)
            session.add(run)
            session.flush()
            for position, document in enumerate(documents, start=1):
                session.add(
                    PipelineRunDocument(
                        run_id=run.id,
                        document_id=document.id,
                        position=position,
                        document_name=document.original_filename,
                        relative_raw_path=document.relative_raw_path,
                    )
                )
            session.flush()
            created = _run_view(_required_run(session, run.id))
            return CreatedPipelineRun(
                run=created,
                selected_raw_paths=tuple(
                    document.relative_raw_path for document in documents
                ),
            )

    def list_for_owner(
        self,
        *,
        owner_user_id: UUID,
        limit: int,
    ) -> tuple[PipelineRunView, ...]:
        with self._session_factory() as session:
            runs = session.scalars(
                _run_query()
                .where(PipelineRun.owner_user_id == owner_user_id)
                .order_by(PipelineRun.created_at.desc())
                .limit(limit)
            ).unique()
            return tuple(_run_view(run) for run in runs)

    def get_for_owner(
        self,
        *,
        owner_user_id: UUID,
        run_id: UUID,
    ) -> PipelineRunView | None:
        with self._session_factory() as session:
            run = session.scalar(
                _run_query().where(
                    PipelineRun.id == run_id,
                    PipelineRun.owner_user_id == owner_user_id,
                )
            )
            return _run_view(run) if run is not None else None

    def events_after(
        self,
        *,
        owner_user_id: UUID,
        run_id: UUID,
        sequence: int,
    ) -> tuple[PipelineEventView, ...] | None:
        with self._session_factory() as session:
            owned = session.scalar(
                select(PipelineRun.id).where(
                    PipelineRun.id == run_id,
                    PipelineRun.owner_user_id == owner_user_id,
                )
            )
            if owned is None:
                return None
            events = session.scalars(
                select(PipelineRunEvent)
                .where(
                    PipelineRunEvent.run_id == run_id,
                    PipelineRunEvent.sequence > sequence,
                )
                .order_by(PipelineRunEvent.sequence)
            )
            return tuple(_event_view(event) for event in events)

    def has_active_run(self) -> bool:
        with self._session_factory() as session:
            active = session.scalar(
                select(PipelineRun.id)
                .where(PipelineRun.status.in_(("queued", "running")))
                .limit(1)
            )
            return active is not None

    def start(self, *, run_id: UUID) -> None:
        with session_scope(self._session_factory) as session:
            run = _required_run(session, run_id)
            run.status = "running"
            run.started_at = utc_now()
            for document in run.documents:
                document.status = "running"

    def append_event(
        self,
        *,
        run_id: UUID,
        payload: Mapping[str, Any],
    ) -> None:
        with session_scope(self._session_factory) as session:
            run = _required_run(session, run_id)
            sequence = _next_event_sequence(session, run_id)
            result = _dict_payload(payload.get("result"))
            event = PipelineRunEvent(
                run_id=run_id,
                sequence=sequence,
                timestamp=_timestamp(payload.get("timestamp")),
                step=str(payload.get("step", "pipeline")),
                status=str(payload.get("status", "running")),
                message=str(payload.get("message", "")),
                counters=_json_ready(_dict_payload(payload.get("counters"))),
                result=_json_ready(result),
            )
            session.add(event)
            _update_document_snapshots(run, event=event)

    def finish(
        self,
        *,
        run_id: UUID,
        status: str,
        summary: str | None,
        error: str | None,
    ) -> None:
        with session_scope(self._session_factory) as session:
            run = _required_run(session, run_id)
            run.status = status
            run.summary = summary
            run.error = error
            run.finished_at = utc_now()
            for document in run.documents:
                if document.status in {"queued", "running"}:
                    document.status = status

    def interrupt_active_runs(self) -> int:
        with session_scope(self._session_factory) as session:
            runs = list(
                session.scalars(
                    select(PipelineRun)
                    .options(selectinload(PipelineRun.documents))
                    .where(PipelineRun.status.in_(("queued", "running")))
                )
            )
            for run in runs:
                run.status = "interrupted"
                run.error = "Application restarted before the pipeline completed"
                run.finished_at = utc_now()
                for document in run.documents:
                    if document.status in {"queued", "running"}:
                        document.status = "failed"
                        document.error = run.error
            return len(runs)


def _run_query() -> Any:
    return select(PipelineRun).options(
        selectinload(PipelineRun.documents),
        selectinload(PipelineRun.events),
    )


def _required_run(session: Session, run_id: UUID) -> PipelineRun:
    run = session.scalar(_run_query().where(PipelineRun.id == run_id))
    if run is None:
        raise ValueError(f"Pipeline run not found: {run_id}")
    return cast(PipelineRun, run)


def _owned_documents(
    session: Session,
    *,
    owner_user_id: UUID,
    document_ids: tuple[UUID, ...],
) -> list[Document]:
    query = select(Document).where(Document.owner_user_id == owner_user_id)
    if document_ids:
        query = query.where(Document.id.in_(document_ids))
    documents = list(session.scalars(query))
    positions = {document_id: index for index, document_id in enumerate(document_ids)}
    if document_ids:
        documents.sort(key=lambda document: positions[document.id])
    else:
        documents.sort(key=lambda document: document.relative_raw_path)
    return documents


def _next_event_sequence(session: Session, run_id: UUID) -> int:
    current = session.scalar(
        select(func.max(PipelineRunEvent.sequence)).where(
            PipelineRunEvent.run_id == run_id
        )
    )
    return int(current or 0) + 1


def _update_document_snapshots(
    run: PipelineRun,
    *,
    event: PipelineRunEvent,
) -> None:
    path = event.result.get("path") or event.result.get("source")
    for document in run.documents:
        if path and not _matches_document(document, str(path)):
            continue
        if event.step in {"prepare", "chunk", "embed", "index"}:
            document.current_step = event.step
        if event.status == "failed":
            document.status = "failed"
            document.error = str(event.result.get("error") or event.message)
        elif event.status == "skipped":
            document.status = "skipped"
        elif event.step == "index" and event.status == "succeeded":
            document.status = "succeeded"
        if path:
            return


def _matches_document(document: PipelineRunDocument, path: str) -> bool:
    return path in {
        document.relative_raw_path,
        document.document_name,
        document.relative_raw_path.removesuffix(".pdf") + ".md",
    }


def _run_view(run: PipelineRun) -> PipelineRunView:
    return PipelineRunView(
        id=run.id,
        owner_user_id=run.owner_user_id,
        status=run.status,
        summary=run.summary,
        error=run.error,
        started_at=run.started_at,
        finished_at=run.finished_at,
        created_at=run.created_at,
        updated_at=run.updated_at,
        documents=tuple(
            PipelineDocumentSnapshot(
                document_id=document.document_id,
                position=document.position,
                document_name=document.document_name,
                relative_raw_path=document.relative_raw_path,
                status=document.status,
                current_step=document.current_step,
                error=document.error,
            )
            for document in run.documents
        ),
        events=tuple(_event_view(event) for event in run.events),
    )


def _event_view(event: PipelineRunEvent) -> PipelineEventView:
    return PipelineEventView(
        sequence=event.sequence,
        timestamp=event.timestamp,
        step=event.step,
        status=event.status,
        message=event.message,
        counters=dict(event.counters or {}),
        result=dict(event.result or {}),
    )


def _timestamp(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            pass
    return datetime.now(UTC)


def _dict_payload(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _json_ready(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _json_value(item) for key, item in value.items()}


def _json_value(value: Any) -> Any:
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, Mapping):
        return _json_ready(value)
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value
