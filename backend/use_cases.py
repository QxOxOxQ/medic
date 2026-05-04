from __future__ import annotations

from typing import Protocol
from uuid import UUID

from agents.models import (
    AgentAnswer,
    AgentExecutionError,
    AgentRequest,
    UnknownAgentError,
)


class BackendError(RuntimeError):
    """Base error for backend application use cases."""


class EmptyQuestionError(BackendError):
    """Raised when a user asks an empty question."""


class RetrievalError(BackendError):
    """Raised when document retrieval fails."""


class AgentRunner(Protocol):
    def answer(self, request: AgentRequest) -> AgentAnswer: ...


class AgentRunnerFactory(Protocol):
    def __call__(
        self,
        *,
        owner_user_id: UUID,
        retrieval_limit: int,
    ) -> AgentRunner: ...


class AnswerQuestionUseCase:
    def __init__(
        self,
        *,
        agent_runner_factory: AgentRunnerFactory,
    ) -> None:
        self._agent_runner_factory = agent_runner_factory

    def execute(
        self,
        *,
        question: str,
        limit: int,
        owner_user_id: UUID,
        requested_agent: str | None = None,
    ) -> AgentAnswer:
        normalized_question = question.strip()
        if not normalized_question:
            raise EmptyQuestionError("Question is required")

        request = AgentRequest(
            question=normalized_question,
            requested_agent=requested_agent,
            user_id=owner_user_id,
        )
        return self._run_agent(
            request,
            owner_user_id=owner_user_id,
            retrieval_limit=limit,
        )

    def _run_agent(
        self,
        request: AgentRequest,
        *,
        owner_user_id: UUID,
        retrieval_limit: int,
    ) -> AgentAnswer:
        try:
            runner = self._agent_runner_factory(
                owner_user_id=owner_user_id,
                retrieval_limit=retrieval_limit,
            )
            return runner.answer(request)
        except UnknownAgentError:
            raise
        except AgentExecutionError:
            raise
        except Exception as error:
            raise AgentExecutionError("Agent execution failed") from error
