from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from time import monotonic
from uuid import UUID

from fastapi import APIRouter, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse

from backend.pipeline_use_cases import (
    GetPipelineRunUseCase,
    ListPipelineRunsUseCase,
    PipelineAlreadyRunningError,
    PipelineRunNotFoundError,
    StartPipelineRunUseCase,
    StreamPipelineEventsUseCase,
)
from dashboard.api_models import (
    PipelineRunCreateRequest,
    PipelineRunListResponse,
    PipelineRunResponse,
)
from dashboard.auth import verify_csrf
from dashboard.dependencies import auth_settings, current_user
from dashboard.http import SSE_HEADERS, sse, sse_heartbeat


router = APIRouter(prefix="/api/pipeline-runs")


@router.post(
    "",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=PipelineRunResponse,
)
def start_pipeline_run(
    request: Request,
    payload: PipelineRunCreateRequest,
) -> JSONResponse:
    user = current_user(request)
    verify_csrf(request, auth_settings(request))
    use_case: StartPipelineRunUseCase = request.app.state.start_pipeline_run_use_case
    try:
        run = use_case.execute(
            owner_user_id=user.id,
            document_ids=tuple(payload.document_ids),
        )
    except PipelineAlreadyRunningError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(error),
        ) from error
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error
    return JSONResponse(
        {"ok": True, "run": run.as_dict()},
        status_code=status.HTTP_202_ACCEPTED,
    )


@router.get("", response_model=PipelineRunListResponse)
def list_pipeline_runs(request: Request, limit: int = 20) -> dict[str, object]:
    user = current_user(request)
    use_case: ListPipelineRunsUseCase = request.app.state.list_pipeline_runs_use_case
    runs = use_case.execute(owner_user_id=user.id, limit=limit)
    return {"ok": True, "runs": [run.as_dict() for run in runs]}


@router.get("/{run_id}", response_model=PipelineRunResponse)
def get_pipeline_run(request: Request, run_id: UUID) -> dict[str, object]:
    user = current_user(request)
    use_case: GetPipelineRunUseCase = request.app.state.get_pipeline_run_use_case
    try:
        run = use_case.execute(owner_user_id=user.id, run_id=run_id)
    except PipelineRunNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error
    return {"ok": True, "run": run.as_dict()}


@router.get("/{run_id}/events")
async def stream_pipeline_events(
    request: Request,
    run_id: UUID,
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
) -> StreamingResponse:
    user = current_user(request)
    event_use_case: StreamPipelineEventsUseCase = (
        request.app.state.stream_pipeline_events_use_case
    )
    run_use_case: GetPipelineRunUseCase = request.app.state.get_pipeline_run_use_case
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
            except PipelineRunNotFoundError:
                yield sse("error", {"error": "Pipeline run not found"})
                return

            for event in events:
                sequence = event.sequence
                yield sse("progress", event.as_dict(), event_id=event.sequence)
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
