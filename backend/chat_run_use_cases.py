from __future__ import annotations

import logging
import threading
from typing import Protocol
from uuid import UUID

from agents.models import (
    AgentAnswer,
    AgentExecutionError,
    AgentRequest,
    ChatHistoryMessage,
)
from agents.trace import AgentTraceSink
from backend.chat_models import ChatMessageView, ChatTraceEventView
from backend.chat_run_models import ChatRunStarted, ChatRunView
from backend.chat_use_cases import (
    CHAT_CONTEXT_MESSAGE_LIMIT,
    ContinuedChatRun,
    ConversationNotFoundError,
    StartedChatRun,
)
from backend.execution import BackgroundExecutor
from backend.use_cases import AgentRunner, EmptyQuestionError


logger = logging.getLogger(__name__)


class ChatRunAlreadyActiveError(RuntimeError):
    pass


class ChatRunNotFoundError(LookupError):
    pass


class TraceableAgentRunnerFactory(Protocol):
    def __call__(
        self,
        *,
        owner_user_id: UUID,
        retrieval_limit: int,
        trace_sink: AgentTraceSink | None = None,
    ) -> AgentRunner: ...


class AsyncChatRunStore(Protocol):
    def start_conversation(
        self,
        *,
        owner_user_id: UUID,
        question: str,
    ) -> StartedChatRun: ...

    def append_user_message(
        self,
        *,
        owner_user_id: UUID,
        conversation_id: UUID,
        question: str,
        context_limit: int,
    ) -> ContinuedChatRun | None: ...

    def complete_run(
        self,
        *,
        owner_user_id: UUID,
        conversation_id: UUID,
        run_id: UUID,
        answer: AgentAnswer,
    ) -> object | None: ...

    def fail_run(self, *, run_id: UUID, error: str) -> None: ...

    def has_active_run(
        self,
        *,
        owner_user_id: UUID,
        conversation_id: UUID,
    ) -> bool: ...

    def run_view(
        self,
        *,
        owner_user_id: UUID,
        run_id: UUID,
    ) -> ChatRunView | None: ...

    def trace_events_after(
        self,
        *,
        owner_user_id: UUID,
        run_id: UUID,
        sequence: int,
    ) -> tuple[ChatTraceEventView, ...] | None: ...

    def trace_sink(self, *, run_id: UUID) -> AgentTraceSink: ...

    def interrupt_active_runs(self) -> int: ...


class StartChatRunUseCase:
    def __init__(
        self,
        *,
        store: AsyncChatRunStore,
        agent_runner_factory: TraceableAgentRunnerFactory,
        executor: BackgroundExecutor,
    ) -> None:
        self._store = store
        self._agent_runner_factory = agent_runner_factory
        self._executor = executor
        self._lock = threading.Lock()

    def execute(
        self,
        *,
        owner_user_id: UUID,
        question: str,
        limit: int,
        conversation_id: UUID | None,
        requested_agent: str | None,
    ) -> ChatRunStarted:
        normalized = question.strip()
        if not normalized:
            raise EmptyQuestionError("Question is required")
        bounded_limit = max(1, min(limit, 20))

        with self._lock:
            started, previous_messages = self._start(
                owner_user_id=owner_user_id,
                conversation_id=conversation_id,
                question=normalized,
            )
            self._executor.submit(
                lambda: self._execute_run(
                    owner_user_id=owner_user_id,
                    conversation_id=started.conversation_id,
                    run_id=started.run_id,
                    question=normalized,
                    limit=bounded_limit,
                    requested_agent=requested_agent,
                    previous_messages=previous_messages,
                )
            )
        return ChatRunStarted(
            conversation_id=started.conversation_id,
            run_id=started.run_id,
        )

    def _start(
        self,
        *,
        owner_user_id: UUID,
        conversation_id: UUID | None,
        question: str,
    ) -> tuple[StartedChatRun, tuple[ChatMessageView, ...]]:
        if conversation_id is None:
            return (
                self._store.start_conversation(
                    owner_user_id=owner_user_id,
                    question=question,
                ),
                (),
            )
        if self._store.has_active_run(
            owner_user_id=owner_user_id,
            conversation_id=conversation_id,
        ):
            raise ChatRunAlreadyActiveError(
                "This conversation already has an active run"
            )
        continued = self._store.append_user_message(
            owner_user_id=owner_user_id,
            conversation_id=conversation_id,
            question=question,
            context_limit=CHAT_CONTEXT_MESSAGE_LIMIT,
        )
        if continued is None:
            raise ConversationNotFoundError("Conversation not found")
        return (
            StartedChatRun(
                conversation_id=conversation_id,
                run_id=continued.run_id,
            ),
            continued.previous_messages,
        )

    def _execute_run(
        self,
        *,
        owner_user_id: UUID,
        conversation_id: UUID,
        run_id: UUID,
        question: str,
        limit: int,
        requested_agent: str | None,
        previous_messages: tuple[ChatMessageView, ...],
    ) -> None:
        try:
            runner = self._agent_runner_factory(
                owner_user_id=owner_user_id,
                retrieval_limit=limit,
                trace_sink=self._store.trace_sink(run_id=run_id),
            )
            answer = runner.answer(
                AgentRequest(
                    question=question,
                    requested_agent=requested_agent,
                    conversation_messages=_history(previous_messages),
                    user_id=owner_user_id,
                    session_id=conversation_id,
                    execution_id=run_id,
                )
            )
            self._store.complete_run(
                owner_user_id=owner_user_id,
                conversation_id=conversation_id,
                run_id=run_id,
                answer=answer,
            )
        except Exception as error:
            logger.exception("Chat run %s failed", run_id)
            message = (
                str(error)
                if isinstance(error, AgentExecutionError)
                else (
                    "The assistant couldn't complete this answer. "
                    "This is usually temporary — please try again in a moment."
                )
            )
            self._store.fail_run(run_id=run_id, error=message)


class GetChatRunUseCase:
    def __init__(self, store: AsyncChatRunStore) -> None:
        self._store = store

    def execute(self, *, owner_user_id: UUID, run_id: UUID) -> ChatRunView:
        view = self._store.run_view(
            owner_user_id=owner_user_id,
            run_id=run_id,
        )
        if view is None:
            raise ChatRunNotFoundError("Chat run not found")
        return view


class StreamChatRunEventsUseCase:
    def __init__(self, store: AsyncChatRunStore) -> None:
        self._store = store

    def execute(
        self,
        *,
        owner_user_id: UUID,
        run_id: UUID,
        after_sequence: int,
    ) -> tuple[ChatTraceEventView, ...]:
        events = self._store.trace_events_after(
            owner_user_id=owner_user_id,
            run_id=run_id,
            sequence=max(0, after_sequence),
        )
        if events is None:
            raise ChatRunNotFoundError("Chat run not found")
        return events


def _history(
    messages: tuple[ChatMessageView, ...],
) -> tuple[ChatHistoryMessage, ...]:
    return tuple(
        ChatHistoryMessage(role=message.role, content=message.content)
        for message in messages
    )
