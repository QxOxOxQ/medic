from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from time import monotonic
from uuid import UUID

from fastapi import APIRouter, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse

from backend.chat_run_use_cases import (
    ChatRunAlreadyActiveError,
    ChatRunNotFoundError,
    GetChatRunUseCase,
    StartChatRunUseCase,
    StreamChatRunEventsUseCase,
)
from backend.chat_use_cases import ConversationNotFoundError
from backend.use_cases import EmptyQuestionError
from dashboard.api_models import (
    ChatRunCreateRequest,
    ChatRunResponse,
    ChatRunStartResponse,
)
from dashboard.auth import verify_csrf
from dashboard.dependencies import auth_settings, current_user
from dashboard.http import SSE_HEADERS, sse, sse_heartbeat


router = APIRouter(prefix="/api/chat/runs")


@router.post(
    "",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=ChatRunStartResponse,
)
def start_chat_run(
    request: Request,
    payload: ChatRunCreateRequest,
) -> JSONResponse:
    user = current_user(request)
    verify_csrf(request, auth_settings(request))
    use_case: StartChatRunUseCase = request.app.state.start_chat_run_use_case
    try:
        started = use_case.execute(
            owner_user_id=user.id,
            question=payload.question,
            limit=payload.limit,
            conversation_id=payload.conversation_id,
            requested_agent=payload.specialist,
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
    except ChatRunAlreadyActiveError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(error),
        ) from error
    return JSONResponse(
        {"ok": True, "run": started.as_dict()},
        status_code=status.HTTP_202_ACCEPTED,
    )


@router.get("/{run_id}", response_model=ChatRunResponse)
def get_chat_run(request: Request, run_id: UUID) -> dict[str, object]:
    user = current_user(request)
    use_case: GetChatRunUseCase = request.app.state.get_chat_run_use_case
    try:
        run = use_case.execute(owner_user_id=user.id, run_id=run_id)
    except ChatRunNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error
    return {"ok": True, "run": run.as_dict()}


@router.get("/{run_id}/events")
async def stream_chat_run_events(
    request: Request,
    run_id: UUID,
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
) -> StreamingResponse:
    user = current_user(request)
    event_use_case: StreamChatRunEventsUseCase = (
        request.app.state.stream_chat_run_events_use_case
    )
    run_use_case: GetChatRunUseCase = request.app.state.get_chat_run_use_case
    after_sequence = _event_sequence(last_event_id)

    async def event_stream() -> AsyncIterator[str]:
        sequence = after_sequence
        last_output_at = monotonic()
        while True:
            if await request.is_disconnected():
                return
            try:
                events = event_use_case.execute(
                    owner_user_id=user.id,
                    run_id=run_id,
                    after_sequence=sequence,
                )
                run = run_use_case.execute(owner_user_id=user.id, run_id=run_id)
            except ChatRunNotFoundError:
                yield sse("error", {"error": "Chat run not found"})
                return

            for event in events:
                sequence = event.sequence
                yield sse("trace", event.as_dict(), event_id=event.sequence)
                last_output_at = monotonic()

            if run.is_terminal:
                yield sse("done", run.as_dict())
                return
            if monotonic() - last_output_at >= 15:
                yield sse_heartbeat()
                last_output_at = monotonic()
            await asyncio.sleep(0.3)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


def _event_sequence(value: str | None) -> int:
    if value is None:
        return 0
    try:
        return max(0, int(value))
    except ValueError:
        return 0
