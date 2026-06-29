from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import JSONResponse

from backend.chat_use_cases import ConversationNotFoundError
from backend.dependencies import chat_conversation_use_case, current_user


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
