from __future__ import annotations

import logging
from time import perf_counter

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import JSONResponse

from dashboard.api_models import SearchRequest, SearchResponse
from dashboard.dependencies import current_user, search_service


router = APIRouter(prefix="/api")
logger = logging.getLogger(__name__)


@router.post("/search", response_model=SearchResponse)
def search(request: Request, payload: SearchRequest) -> JSONResponse:
    user = current_user(request)
    query = payload.query.strip()
    if not query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query is required",
        )

    try:
        started_at = perf_counter()
        results = search_service(request).search(
            query=query,
            limit=payload.limit,
            owner_user_id=user.id,
        )
    except Exception as error:
        logger.exception("Retrieval search failed")
        return JSONResponse(
            {
                "ok": False,
                "query": query,
                "limit": payload.limit,
                "elapsed_ms": round((perf_counter() - started_at) * 1000, 2),
                "error": str(error),
                "results": [],
            },
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    return JSONResponse(
        {
            "ok": True,
            "query": query,
            "limit": payload.limit,
            "elapsed_ms": round((perf_counter() - started_at) * 1000, 2),
            "results": results,
        }
    )
