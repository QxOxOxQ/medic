from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse

from dashboard.auth import verify_csrf
from dashboard.dependencies import (
    auth_settings,
    current_user,
    document_catalog,
    document_settings,
    job_store,
)
from dashboard.http import sse
from dashboard.jobs import JobAlreadyRunningError
from dashboard.selection import (
    ensure_raw_documents_exist,
    json_payload,
    selected_relative_raw_paths,
)


router = APIRouter(prefix="/api/jobs")


@router.get("")
def list_jobs(request: Request) -> dict[str, Any]:
    current_user(request)
    return {"jobs": job_store(request).recent_jobs()}


@router.post("/ingest")
async def start_ingest(request: Request) -> JSONResponse:
    user = current_user(request)
    verify_csrf(request, auth_settings(request))
    settings = document_settings(request)
    selected_paths = selected_relative_raw_paths(
        await json_payload(request),
        required=False,
    )
    if selected_paths is not None:
        owned_records, _ = document_catalog(request).list_records(
            settings,
            owner_user_id=user.id,
        )
        owned_paths = {record.relative_raw_path for record in owned_records}
        ensure_raw_documents_exist(selected_paths, settings=settings)
        unowned_paths = sorted(set(selected_paths) - owned_paths)
        if unowned_paths:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Documents belong to another user: {', '.join(unowned_paths)}",
            )
    try:
        job = job_store(request).start_ingest(
            settings,
            selected_raw_paths=selected_paths,
            owner_user_id=user.id,
        )
    except JobAlreadyRunningError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(error),
        ) from error
    return JSONResponse({"ok": True, "job": job.snapshot()})


@router.get("/{job_id}")
def get_job(request: Request, job_id: str) -> dict[str, Any]:
    current_user(request)
    job = job_store(request).get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found",
        )
    return job.snapshot()


@router.get("/{job_id}/events")
async def job_events(request: Request, job_id: str) -> StreamingResponse:
    current_user(request)
    job = job_store(request).get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found",
        )

    async def event_stream() -> AsyncIterator[str]:
        last_event_id = 0
        while True:
            if await request.is_disconnected():
                break

            events = job.events_after(last_event_id)
            for event in events:
                last_event_id = event.id
                yield sse("progress", event.as_dict(), event_id=event.id)

            if job.is_terminal and not events:
                yield sse("done", job.snapshot())
                break

            await asyncio.sleep(0.25)

    return StreamingResponse(event_stream(), media_type="text/event-stream")
