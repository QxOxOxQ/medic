from __future__ import annotations

from typing import Protocol, cast
from uuid import UUID

from fastapi import Request

from backend.chat_use_cases import ChatConversationUseCase
from backend.use_cases import AnswerQuestionUseCase


class AuthenticatedUser(Protocol):
    id: UUID


class CurrentUserResolver(Protocol):
    def __call__(self, request: Request) -> AuthenticatedUser:
        ...


def answer_question_use_case(request: Request) -> AnswerQuestionUseCase:
    return cast(AnswerQuestionUseCase, request.app.state.answer_question_use_case)


def chat_conversation_use_case(request: Request) -> ChatConversationUseCase:
    return cast(ChatConversationUseCase, request.app.state.chat_conversation_use_case)


def current_user(request: Request) -> AuthenticatedUser:
    resolver = _current_user_resolver(request)
    return resolver(request)


def _current_user_resolver(request: Request) -> CurrentUserResolver:
    resolver = getattr(request.app.state, "current_user_resolver", None)
    if resolver is None:
        raise RuntimeError("Current user resolver is not configured")
    return cast(CurrentUserResolver, resolver)
