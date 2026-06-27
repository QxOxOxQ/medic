from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status

from clients.chat_models import (
    DEFAULT_CHAT_MODEL_KEY,
    SELECTABLE_CHAT_MODELS,
    is_valid_chat_model_key,
)
from dashboard.api_models import (
    ChatModelSelectionRequest,
    ChatModelSettingsResponse,
)
from dashboard.auth import verify_csrf
from dashboard.dependencies import (
    auth_settings,
    current_user,
    database_session_factory,
)
from rag.database.repositories import UserRepository


router = APIRouter(prefix="/api/settings")


def _options() -> list[dict[str, str]]:
    return [
        {"key": model.key, "label": model.label, "model_id": model.model_id}
        for model in SELECTABLE_CHAT_MODELS
    ]


@router.get("/chat-model", response_model=ChatModelSettingsResponse)
def get_chat_model_setting(request: Request) -> dict[str, object]:
    user = current_user(request)
    with database_session_factory(request)() as session:
        stored = UserRepository(session).get_by_id(user.id)
        selected = stored.preferred_chat_model if stored is not None else None
    if selected is None or not is_valid_chat_model_key(selected):
        selected = DEFAULT_CHAT_MODEL_KEY
    return {"ok": True, "options": _options(), "selected": selected}


@router.put("/chat-model", response_model=ChatModelSettingsResponse)
def update_chat_model_setting(
    request: Request,
    payload: ChatModelSelectionRequest,
) -> dict[str, object]:
    user = current_user(request)
    verify_csrf(request, auth_settings(request))
    if not is_valid_chat_model_key(payload.key):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unknown chat model",
        )
    with database_session_factory(request)() as session:
        updated = UserRepository(session).set_preferred_chat_model(
            user_id=user.id,
            model_key=payload.key,
        )
        if updated is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found",
            )
        session.commit()
    return {"ok": True, "options": _options(), "selected": payload.key}
