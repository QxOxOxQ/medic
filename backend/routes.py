from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import JSONResponse

from agents.models import AgentExecutionError, UnknownAgentError
from backend.chat_use_cases import ConversationNotFoundError
from backend.dependencies import (
    answer_question_use_case,
    chat_conversation_use_case,
    current_user,
)
from backend.use_cases import EmptyQuestionError, RetrievalError


router = APIRouter(prefix="/api")


@router.get("/chat/conversations")
async def list_chat_conversations(request: Request) -> JSONResponse:
    user = current_user(request)
    conversations = chat_conversation_use_case(request).list_conversations(
        owner_user_id=user.id,
    )
    return JSONResponse(
        {
            "ok": True,
            "conversations": [
                conversation.as_dict() for conversation in conversations
            ],
        }
    )


@router.get("/chat/conversations/{conversation_id}")
async def load_chat_conversation(
    request: Request,
    conversation_id: UUID,
) -> JSONResponse:
    user = current_user(request)
    try:
        conversation = chat_conversation_use_case(request).load_conversation(
            owner_user_id=user.id,
            conversation_id=conversation_id,
        )
    except ConversationNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error
    return JSONResponse({"ok": True, "conversation": conversation.as_dict()})


@router.post("/chat/conversations")
async def create_chat_conversation(request: Request) -> JSONResponse:
    user = current_user(request)
    payload = await request.json()
    try:
        conversation = chat_conversation_use_case(request).create_conversation(
            owner_user_id=user.id,
            question=str(payload.get("question", "")),
            limit=_bounded_limit(payload.get("limit", 5)),
            requested_agent=_optional_string(payload.get("specialist")),
        )
    except EmptyQuestionError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error
    except UnknownAgentError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error
    except (AgentExecutionError, RetrievalError) as error:
        return JSONResponse(
            {"ok": False, "error": str(error)},
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )
    return JSONResponse({"ok": True, "conversation": conversation.as_dict()})


@router.post("/chat/conversations/{conversation_id}/messages")
async def continue_chat_conversation(
    request: Request,
    conversation_id: UUID,
) -> JSONResponse:
    user = current_user(request)
    payload = await request.json()
    try:
        conversation = chat_conversation_use_case(request).continue_conversation(
            owner_user_id=user.id,
            conversation_id=conversation_id,
            question=str(payload.get("question", "")),
            limit=_bounded_limit(payload.get("limit", 5)),
            requested_agent=_optional_string(payload.get("specialist")),
        )
    except EmptyQuestionError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error
    except ConversationNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error
    except UnknownAgentError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error
    except (AgentExecutionError, RetrievalError) as error:
        return JSONResponse(
            {"ok": False, "error": str(error)},
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )
    return JSONResponse({"ok": True, "conversation": conversation.as_dict()})


@router.post("/ask")
async def ask(request: Request) -> JSONResponse:
    user = current_user(request)
    payload = await request.json()
    try:
        answer = answer_question_use_case(request).execute(
            question=str(payload.get("question", "")),
            limit=_bounded_limit(payload.get("limit", 5)),
            owner_user_id=user.id,
            requested_agent=_optional_string(payload.get("specialist")),
        )
    except EmptyQuestionError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error
    except UnknownAgentError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error
    except (AgentExecutionError, RetrievalError) as error:
        return JSONResponse(
            {"ok": False, "error": str(error)},
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    return JSONResponse({"ok": True, **answer.as_dict()})


def _bounded_limit(raw_limit: Any) -> int:
    try:
        limit = int(raw_limit)
    except (TypeError, ValueError):
        limit = 5
    return max(1, min(limit, 20))


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    stripped = str(value).strip()
    return stripped or None
