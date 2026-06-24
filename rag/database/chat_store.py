from __future__ import annotations

from uuid import UUID

from sqlalchemy.orm import Session, sessionmaker

from agents.models import AgentAnswer
from agents.models import AgentTraceEvent
from agents.trace import AgentTraceSink
from backend.chat_models import ChatConversationDetail, ChatConversationSummary
from backend.chat_models import ChatTraceEventView
from backend.chat_run_models import ChatRunView
from backend.chat_use_cases import ContinuedChatRun, StartedChatRun
from rag.database.chat_repositories import ChatRepository


class SqlAlchemyChatConversationStore:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def list_summaries(
        self,
        *,
        owner_user_id: UUID,
    ) -> tuple[ChatConversationSummary, ...]:
        with self._session_factory() as session:
            return ChatRepository(session).list_summaries(
                owner_user_id=owner_user_id
            )

    def load_detail(
        self,
        *,
        owner_user_id: UUID,
        conversation_id: UUID,
    ) -> ChatConversationDetail | None:
        with self._session_factory() as session:
            return ChatRepository(session).detail(
                owner_user_id=owner_user_id,
                conversation_id=conversation_id,
            )

    def start_conversation(
        self,
        *,
        owner_user_id: UUID,
        question: str,
    ) -> StartedChatRun:
        with self._session_factory() as session:
            repository = ChatRepository(session)
            conversation = repository.create_conversation(
                owner_user_id=owner_user_id,
                title=question,
            )
            repository.append_message(
                conversation_id=conversation.id,
                role="user",
                content=question,
            )
            run = repository.create_run(
                conversation_id=conversation.id,
                question=question,
            )
            session.commit()
            return StartedChatRun(conversation_id=conversation.id, run_id=run.id)

    def append_user_message(
        self,
        *,
        owner_user_id: UUID,
        conversation_id: UUID,
        question: str,
        context_limit: int,
    ) -> ContinuedChatRun | None:
        with self._session_factory() as session:
            repository = ChatRepository(session)
            previous_messages = repository.recent_messages(
                owner_user_id=owner_user_id,
                conversation_id=conversation_id,
                limit=context_limit,
            )
            if previous_messages is None:
                return None
            repository.append_message(
                conversation_id=conversation_id,
                role="user",
                content=question,
            )
            run = repository.create_run(
                conversation_id=conversation_id,
                question=question,
            )
            session.commit()
            return ContinuedChatRun(
                run_id=run.id,
                previous_messages=previous_messages,
            )

    def complete_run(
        self,
        *,
        owner_user_id: UUID,
        conversation_id: UUID,
        run_id: UUID,
        answer: AgentAnswer,
    ) -> ChatConversationDetail | None:
        with self._session_factory() as session:
            repository = ChatRepository(session)
            assistant_message = repository.append_message(
                conversation_id=conversation_id,
                role="assistant",
                content=answer.answer,
                insufficient_context=answer.insufficient_context,
            )
            repository.add_sources(
                message_id=assistant_message.id,
                run_id=run_id,
                sources=answer.sources,
            )
            repository.add_trace_events(run_id=run_id, events=answer.trace_events)
            repository.complete_run(
                run_id=run_id,
                assistant_message_id=assistant_message.id,
                answer=answer.answer,
                insufficient_context=answer.insufficient_context,
            )
            detail = repository.detail(
                owner_user_id=owner_user_id,
                conversation_id=conversation_id,
            )
            session.commit()
            return detail

    def fail_run(self, *, run_id: UUID, error: str) -> None:
        with self._session_factory() as session:
            ChatRepository(session).fail_run(run_id=run_id, error=error)
            session.commit()

    def has_active_run(
        self,
        *,
        owner_user_id: UUID,
        conversation_id: UUID,
    ) -> bool:
        with self._session_factory() as session:
            return ChatRepository(session).has_active_run(
                owner_user_id=owner_user_id,
                conversation_id=conversation_id,
            )

    def run_view(
        self,
        *,
        owner_user_id: UUID,
        run_id: UUID,
    ) -> ChatRunView | None:
        with self._session_factory() as session:
            return ChatRepository(session).run_view(
                owner_user_id=owner_user_id,
                run_id=run_id,
            )

    def trace_events_after(
        self,
        *,
        owner_user_id: UUID,
        run_id: UUID,
        sequence: int,
    ) -> tuple[ChatTraceEventView, ...] | None:
        with self._session_factory() as session:
            return ChatRepository(session).trace_events_after(
                owner_user_id=owner_user_id,
                run_id=run_id,
                sequence=sequence,
            )

    def trace_sink(self, *, run_id: UUID) -> AgentTraceSink:
        return _SqlAlchemyAgentTraceSink(
            session_factory=self._session_factory,
            run_id=run_id,
        )

    def interrupt_active_runs(self) -> int:
        with self._session_factory() as session:
            interrupted = ChatRepository(session).interrupt_active_runs()
            session.commit()
            return interrupted


class _SqlAlchemyAgentTraceSink:
    def __init__(
        self,
        *,
        session_factory: sessionmaker[Session],
        run_id: UUID,
    ) -> None:
        self._session_factory = session_factory
        self._run_id = run_id

    def record(self, event: AgentTraceEvent) -> None:
        with self._session_factory() as session:
            ChatRepository(session).add_trace_event(
                run_id=self._run_id,
                event=event,
            )
            session.commit()
