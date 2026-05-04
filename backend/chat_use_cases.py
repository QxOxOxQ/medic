from __future__ import annotations

from uuid import UUID

from sqlalchemy.orm import Session, sessionmaker

from agents.models import (
    AgentAnswer,
    AgentExecutionError,
    AgentRequest,
    ChatHistoryMessage,
    UnknownAgentError,
)
from backend.chat_models import ChatConversationDetail, ChatConversationSummary
from backend.use_cases import AgentRunnerFactory, EmptyQuestionError
from rag.database.chat_repositories import ChatRepository


CHAT_CONTEXT_MESSAGE_LIMIT = 6


class ConversationError(RuntimeError):
    """Base error for conversation use cases."""


class ConversationNotFoundError(ConversationError):
    """Raised when a conversation does not exist for the current user."""


class ConversationAccessDeniedError(ConversationError):
    """Raised when a user attempts to access another user's conversation."""


class ChatConversationUseCase:
    def __init__(
        self,
        *,
        agent_runner_factory: AgentRunnerFactory,
        database_session_factory: sessionmaker[Session],
    ) -> None:
        self._agent_runner_factory = agent_runner_factory
        self._database_session_factory = database_session_factory

    def list_conversations(
        self,
        *,
        owner_user_id: UUID,
    ) -> tuple[ChatConversationSummary, ...]:
        with self._database_session_factory() as session:
            return ChatRepository(session).list_summaries(
                owner_user_id=owner_user_id
            )

    def load_conversation(
        self,
        *,
        owner_user_id: UUID,
        conversation_id: UUID,
    ) -> ChatConversationDetail:
        with self._database_session_factory() as session:
            detail = ChatRepository(session).detail(
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
        with self._database_session_factory() as session:
            repository = ChatRepository(session)
            conversation = repository.create_conversation(
                owner_user_id=owner_user_id,
                title=normalized_question,
            )
            repository.append_message(
                conversation_id=conversation.id,
                role="user",
                content=normalized_question,
            )
            run = repository.create_run(
                conversation_id=conversation.id,
                question=normalized_question,
            )
            conversation_id = conversation.id
            run_id = run.id
            session.commit()

        try:
            answer = self._run_agent(
                owner_user_id=owner_user_id,
                question=normalized_question,
                limit=limit,
                requested_agent=requested_agent,
                conversation_messages=(),
                conversation_id=conversation_id,
                run_id=run_id,
            )
        except Exception as error:
            self._fail_run(run_id=run_id, error=str(error))
            raise
        return self._persist_answer(
            owner_user_id=owner_user_id,
            conversation_id=conversation_id,
            run_id=run_id,
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
        with self._database_session_factory() as session:
            repository = ChatRepository(session)
            previous_messages = repository.recent_messages(
                owner_user_id=owner_user_id,
                conversation_id=conversation_id,
                limit=CHAT_CONTEXT_MESSAGE_LIMIT,
            )
            if previous_messages is None:
                raise ConversationNotFoundError("Conversation not found")
            repository.append_message(
                conversation_id=conversation_id,
                role="user",
                content=normalized_question,
            )
            run = repository.create_run(
                conversation_id=conversation_id,
                question=normalized_question,
            )
            run_id = run.id
            session.commit()

        try:
            answer = self._run_agent(
                owner_user_id=owner_user_id,
                question=normalized_question,
                limit=limit,
                requested_agent=requested_agent,
                conversation_messages=_history_messages(previous_messages),
                conversation_id=conversation_id,
                run_id=run_id,
            )
        except Exception as error:
            self._fail_run(run_id=run_id, error=str(error))
            raise

        return self._persist_answer(
            owner_user_id=owner_user_id,
            conversation_id=conversation_id,
            run_id=run_id,
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
        with self._database_session_factory() as session:
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
        if detail is None:
            raise ConversationNotFoundError("Conversation not found")
        return detail

    def _fail_run(self, *, run_id: UUID, error: str) -> None:
        with self._database_session_factory() as session:
            ChatRepository(session).fail_run(run_id=run_id, error=error)
            session.commit()


def _normalized_question(question: str) -> str:
    normalized = question.strip()
    if not normalized:
        raise EmptyQuestionError("Question is required")
    return normalized


def _history_messages(
    messages: tuple[object, ...],
) -> tuple[ChatHistoryMessage, ...]:
    history: list[ChatHistoryMessage] = []
    for message in messages:
        role = getattr(message, "role")
        content = getattr(message, "content")
        history.append(ChatHistoryMessage(role=str(role), content=str(content)))
    return tuple(history)
