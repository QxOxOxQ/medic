from __future__ import annotations

import json
from typing import Any

from fastapi import Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates


SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def template_response(
    templates: Jinja2Templates,
    request: Request,
    name: str,
    context: dict[str, Any],
    *,
    status_code: int = status.HTTP_200_OK,
) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name=name,
        context=context,
        status_code=status_code,
    )


def redirect(path: str) -> RedirectResponse:
    return RedirectResponse(path, status_code=status.HTTP_303_SEE_OTHER)


def sse(event: str, payload: dict[str, Any], *, event_id: int | None = None) -> str:
    data = json.dumps(payload, ensure_ascii=False)
    event_id_line = f"id: {event_id}\n" if event_id is not None else ""
    return f"{event_id_line}event: {event}\ndata: {data}\n\n"


def sse_heartbeat() -> str:
    return ": keep-alive\n\n"


def bounded_limit(raw_limit: Any) -> int:
    try:
        limit = int(raw_limit)
    except (TypeError, ValueError):
        limit = 10
    return max(1, min(limit, 20))
