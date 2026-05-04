from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from agents.models import AgentSource, AgentTraceEvent
from backend.chat_models import (
    ChatConversationDetail,
    ChatConversationSummary,
    ChatMessageView,
    ChatSourceView,
    ChatTraceEventView,
)
from rag.database.models import (
    ChatConversation,
    ChatMessage,
    ChatMessageSource,
    ChatRun,
    ChatTraceEvent,
    utc_now,
)


class ChatRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def create_conversation(
        self,
        *,
        owner_user_id: UUID,
        title: str,
    ) -> ChatConversation:
        conversation = ChatConversation(
            owner_user_id=owner_user_id,
            title=_conversation_title(title),
        )
        self._session.add(conversation)
        self._session.flush()
        return conversation

    def list_summaries(self, *, owner_user_id: UUID) -> tuple[ChatConversationSummary, ...]:
        conversations = self._session.scalars(
            select(ChatConversation)
            .options(selectinload(ChatConversation.messages))
            .where(ChatConversation.owner_user_id == owner_user_id)
            .order_by(ChatConversation.updated_at.desc(), ChatConversation.created_at.desc())
        )
        return tuple(_summary(conversation) for conversation in conversations)

    def detail(
        self,
        *,
        owner_user_id: UUID,
        conversation_id: UUID,
    ) -> ChatConversationDetail | None:
        conversation = self._owned_conversation(
            owner_user_id=owner_user_id,
            conversation_id=conversation_id,
        )
        if conversation is None:
            return None
        return self._detail_from_conversation(conversation)

    def recent_messages(
        self,
        *,
        owner_user_id: UUID,
        conversation_id: UUID,
        limit: int,
    ) -> tuple[ChatMessageView, ...] | None:
        conversation = self._owned_conversation(
            owner_user_id=owner_user_id,
            conversation_id=conversation_id,
        )
        if conversation is None:
            return None
        messages = sorted(conversation.messages, key=lambda message: message.sequence)
        return tuple(_message_view(message) for message in messages[-limit:])

    def append_message(
        self,
        *,
        conversation_id: UUID,
        role: str,
        content: str,
        insufficient_context: bool = False,
    ) -> ChatMessage:
        conversation = self._session.get(ChatConversation, conversation_id)
        if conversation is None:
            raise ValueError(f"Conversation not found: {conversation_id}")
        message = ChatMessage(
            conversation_id=conversation_id,
            role=role,
            content=content,
            sequence=self._next_message_sequence(conversation_id),
            insufficient_context=insufficient_context,
        )
        conversation.updated_at = utc_now()
        self._session.add(message)
        self._session.flush()
        return message

    def create_run(self, *, conversation_id: UUID, question: str) -> ChatRun:
        run = ChatRun(conversation_id=conversation_id, question=question)
        self._session.add(run)
        self._session.flush()
        return run

    def complete_run(
        self,
        *,
        run_id: UUID,
        assistant_message_id: UUID,
        answer: str,
        insufficient_context: bool,
    ) -> None:
        run = self._required_run(run_id)
        run.assistant_message_id = assistant_message_id
        run.answer = answer
        run.insufficient_context = insufficient_context
        run.status = "succeeded"
        run.finished_at = utc_now()
        self._session.flush()

    def fail_run(self, *, run_id: UUID, error: str) -> None:
        run = self._required_run(run_id)
        run.status = "failed"
        run.error = error
        run.finished_at = utc_now()
        self._session.flush()

    def add_sources(
        self,
        *,
        message_id: UUID,
        run_id: UUID,
        sources: Iterable[AgentSource],
    ) -> None:
        for source in sources:
            self._session.add(
                ChatMessageSource(
                    message_id=message_id,
                    run_id=run_id,
                    source_id=source.id,
                    source=source.source,
                    content_hash=source.content_hash,
                    document_id=source.document_id,
                    document_name=source.document_name,
                    relative_raw_path=source.relative_raw_path,
                    qdrant_point_id=source.qdrant_point_id,
                    chunk_index=source.chunk_index,
                    char_start=source.char_start,
                    char_end=source.char_end,
                    retrieval_query=source.retrieval_query,
                    score=source.score,
                    excerpt=source.excerpt,
                )
            )
        self._session.flush()

    def add_trace_events(
        self,
        *,
        run_id: UUID,
        events: Iterable[AgentTraceEvent],
    ) -> None:
        for event in events:
            self._session.add(
                ChatTraceEvent(
                    run_id=run_id,
                    sequence=event.sequence,
                    event_type=event.event_type,
                    title=event.title,
                    status=event.status,
                    agent_name=event.agent_name,
                    tool_name=event.tool_name,
                    payload=_json_ready(event.payload),
                    duration_ms=event.duration_ms,
                )
            )
        self._session.flush()

    def _owned_conversation(
        self,
        *,
        owner_user_id: UUID,
        conversation_id: UUID,
    ) -> ChatConversation | None:
        return self._session.scalar(
            select(ChatConversation)
            .options(
                selectinload(ChatConversation.messages).selectinload(
                    ChatMessage.sources
                ),
                selectinload(ChatConversation.runs).selectinload(ChatRun.trace_events),
            )
            .where(
                ChatConversation.id == conversation_id,
                ChatConversation.owner_user_id == owner_user_id,
            )
        )

    def _detail_from_conversation(
        self,
        conversation: ChatConversation,
    ) -> ChatConversationDetail:
        runs_by_message_id = {
            run.assistant_message_id: run
            for run in conversation.runs
            if run.assistant_message_id is not None
        }
        messages = tuple(
            _message_view(
                message,
                trace_events=_trace_events_for_message(
                    message.id,
                    runs_by_message_id,
                ),
            )
            for message in sorted(conversation.messages, key=lambda item: item.sequence)
        )
        return ChatConversationDetail(
            id=conversation.id,
            title=conversation.title,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
            messages=messages,
        )

    def _next_message_sequence(self, conversation_id: UUID) -> int:
        current = self._session.scalar(
            select(func.max(ChatMessage.sequence)).where(
                ChatMessage.conversation_id == conversation_id
            )
        )
        if current is None:
            return 1
        return int(current) + 1

    def _required_run(self, run_id: UUID) -> ChatRun:
        run = self._session.get(ChatRun, run_id)
        if run is None:
            raise ValueError(f"Chat run not found: {run_id}")
        return run


def _summary(conversation: ChatConversation) -> ChatConversationSummary:
    return ChatConversationSummary(
        id=conversation.id,
        title=conversation.title,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        message_count=len(conversation.messages),
    )


def _message_view(
    message: ChatMessage,
    *,
    trace_events: tuple[ChatTraceEventView, ...] = (),
) -> ChatMessageView:
    return ChatMessageView(
        id=message.id,
        role=message.role,
        content=message.content,
        sequence=message.sequence,
        insufficient_context=message.insufficient_context,
        created_at=message.created_at,
        sources=tuple(_source_view(source) for source in message.sources),
        trace_events=trace_events,
    )


def _trace_events_for_message(
    message_id: UUID,
    runs_by_message_id: Mapping[UUID, ChatRun],
) -> tuple[ChatTraceEventView, ...]:
    run = runs_by_message_id.get(message_id)
    if run is None:
        return ()
    return tuple(_trace_event_view(event) for event in run.trace_events)


def _source_view(source: ChatMessageSource) -> ChatSourceView:
    return ChatSourceView(
        id=source.id,
        source_id=source.source_id,
        source=source.source,
        content_hash=source.content_hash,
        document_id=source.document_id,
        document_name=source.document_name,
        relative_raw_path=source.relative_raw_path,
        qdrant_point_id=source.qdrant_point_id,
        chunk_index=source.chunk_index,
        char_start=source.char_start,
        char_end=source.char_end,
        retrieval_query=source.retrieval_query,
        score=source.score,
        excerpt=source.excerpt,
    )


def _trace_event_view(event: ChatTraceEvent) -> ChatTraceEventView:
    return ChatTraceEventView(
        id=event.id,
        sequence=event.sequence,
        event_type=event.event_type,
        title=event.title,
        status=event.status,
        agent_name=event.agent_name,
        tool_name=event.tool_name,
        payload=dict(event.payload or {}),
        duration_ms=event.duration_ms,
        created_at=event.created_at,
    )


def _conversation_title(value: str) -> str:
    compact = " ".join(value.split())
    if not compact:
        return "Nowa rozmowa"
    if len(compact) <= 80:
        return compact
    return f"{compact[:77]}..."


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
