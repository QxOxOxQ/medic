from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from agents.models import (
    AgentAnswer,
    AgentExecutionError,
    AgentRequest,
    ChatHistoryMessage,
    UnknownAgentError,
)
from backend.chat_models import (
    ChatConversationDetail,
    ChatConversationSummary,
    ChatMessageView,
)
from backend.use_cases import AgentRunnerFactory, EmptyQuestionError


CHAT_CONTEXT_MESSAGE_LIMIT = 6


class ConversationError(RuntimeError):
    """Base error for conversation use cases."""


class ConversationNotFoundError(ConversationError):
    """Raised when a conversation does not exist for the current user."""


class ConversationAccessDeniedError(ConversationError):
    """Raised when a user attempts to access another user's conversation."""


@dataclass(frozen=True)
class StartedChatRun:
    conversation_id: UUID
    run_id: UUID


@dataclass(frozen=True)
class ContinuedChatRun:
    run_id: UUID
    previous_messages: tuple[ChatMessageView, ...]


class ChatConversationStore(Protocol):
    def list_summaries(
        self,
        *,
        owner_user_id: UUID,
    ) -> tuple[ChatConversationSummary, ...]: ...

    def load_detail(
        self,
        *,
        owner_user_id: UUID,
        conversation_id: UUID,
    ) -> ChatConversationDetail | None: ...

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
    ) -> ChatConversationDetail | None: ...

    def fail_run(self, *, run_id: UUID, error: str) -> None: ...


class ChatConversationUseCase:
    def __init__(
        self,
        *,
        agent_runner_factory: AgentRunnerFactory,
        conversation_store: ChatConversationStore,
    ) -> None:
        self._agent_runner_factory = agent_runner_factory
        self._conversation_store = conversation_store

    def list_conversations(
        self,
        *,
        owner_user_id: UUID,
    ) -> tuple[ChatConversationSummary, ...]:
        return self._conversation_store.list_summaries(owner_user_id=owner_user_id)

    def load_conversation(
        self,
        *,
        owner_user_id: UUID,
        conversation_id: UUID,
    ) -> ChatConversationDetail:
        detail = self._conversation_store.load_detail(
            owner_user_id=owner_user_id,
            conversation_id=conversation_id,
        )
        if detail is None:
            raise ConversationNotFoundError("Conversation not found")
        return detail

    def create_conversation(
        self,
        *,
        owner_user_id: UUID,
        question: str,
        limit: int,
        requested_agent: str | None = None,
    ) -> ChatConversationDetail:
        normalized_question = _normalized_question(question)
        started_run = self._conversation_store.start_conversation(
            owner_user_id=owner_user_id,
            question=normalized_question,
        )

        try:
            answer = self._run_agent(
                owner_user_id=owner_user_id,
                question=normalized_question,
                limit=limit,
                requested_agent=requested_agent,
                conversation_messages=(),
                conversation_id=started_run.conversation_id,
                run_id=started_run.run_id,
            )
        except Exception as error:
            self._conversation_store.fail_run(run_id=started_run.run_id, error=str(error))
            raise
        return self._persist_answer(
            owner_user_id=owner_user_id,
            conversation_id=started_run.conversation_id,
            run_id=started_run.run_id,
            answer=answer,
        )

    def continue_conversation(
        self,
        *,
        owner_user_id: UUID,
        conversation_id: UUID,
        question: str,
        limit: int,
        requested_agent: str | None = None,
    ) -> ChatConversationDetail:
        normalized_question = _normalized_question(question)
        continued_run = self._conversation_store.append_user_message(
            owner_user_id=owner_user_id,
            conversation_id=conversation_id,
            question=normalized_question,
            context_limit=CHAT_CONTEXT_MESSAGE_LIMIT,
        )
        if continued_run is None:
            raise ConversationNotFoundError("Conversation not found")

        try:
            answer = self._run_agent(
                owner_user_id=owner_user_id,
                question=normalized_question,
                limit=limit,
                requested_agent=requested_agent,
                conversation_messages=_history_messages(continued_run.previous_messages),
                conversation_id=conversation_id,
                run_id=continued_run.run_id,
            )
        except Exception as error:
            self._conversation_store.fail_run(run_id=continued_run.run_id, error=str(error))
            raise

        return self._persist_answer(
            owner_user_id=owner_user_id,
            conversation_id=conversation_id,
            run_id=continued_run.run_id,
            answer=answer,
        )

    def _run_agent(
        self,
        *,
        owner_user_id: UUID,
        question: str,
        limit: int,
        requested_agent: str | None,
        conversation_messages: tuple[ChatHistoryMessage, ...],
        conversation_id: UUID,
        run_id: UUID,
    ) -> AgentAnswer:
        try:
            runner = self._agent_runner_factory(
                owner_user_id=owner_user_id,
                retrieval_limit=limit,
            )
            return runner.answer(
                AgentRequest(
                    question=question,
                    requested_agent=requested_agent,
                    conversation_messages=conversation_messages,
                    user_id=owner_user_id,
                    session_id=conversation_id,
                    execution_id=run_id,
                )
            )
        except UnknownAgentError:
            raise
        except AgentExecutionError:
            raise
        except Exception as error:
            raise AgentExecutionError("Agent execution failed") from error

    def _persist_answer(
        self,
        *,
        owner_user_id: UUID,
        conversation_id: UUID,
        run_id: UUID,
        answer: AgentAnswer,
    ) -> ChatConversationDetail:
        detail = self._conversation_store.complete_run(
            owner_user_id=owner_user_id,
            conversation_id=conversation_id,
            run_id=run_id,
            answer=answer,
        )
        if detail is None:
            raise ConversationNotFoundError("Conversation not found")
        return detail


def _normalized_question(question: str) -> str:
    normalized = question.strip()
    if not normalized:
        raise EmptyQuestionError("Question is required")
    return normalized


def _history_messages(
    messages: tuple[ChatMessageView, ...],
) -> tuple[ChatHistoryMessage, ...]:
    history: list[ChatHistoryMessage] = []
    for message in messages:
        history.append(ChatHistoryMessage(role=message.role, content=message.content))
    return tuple(history)
