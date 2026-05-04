from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import JSONResponse

from dashboard.dependencies import current_user, search_service
from dashboard.http import bounded_limit


router = APIRouter(prefix="/api")


@router.post("/search")
async def search(request: Request) -> JSONResponse:
    user = current_user(request)
    payload = await request.json()
    query = str(payload.get("query", "")).strip()
    if not query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query is required",
        )

    try:
        results = search_service(request).search(
            query=query,
            limit=bounded_limit(payload.get("limit", 10)),
            owner_user_id=user.id,
        )
    except Exception as error:
        return JSONResponse(
            {
                "ok": False,
                "error": str(error),
                "results": [],
            },
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    return JSONResponse({"ok": True, "results": results})
