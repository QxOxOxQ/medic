from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from sqlalchemy import text

from backend.pipeline_use_cases import ListPipelineRunsUseCase
from dashboard.api_models import WorkspaceOverviewResponse
from dashboard.dependencies import (
    current_user,
    database_session_factory,
    document_catalog,
    document_settings,
)


router = APIRouter(prefix="/api/workspace")
logger = logging.getLogger(__name__)


@router.get("/overview", response_model=WorkspaceOverviewResponse)
def workspace_overview(request: Request) -> dict[str, object]:
    user = current_user(request)
    status = document_catalog(request).dashboard_status(
        document_settings(request),
        owner_user_id=user.id,
    )
    pipeline_use_case: ListPipelineRunsUseCase = (
        request.app.state.list_pipeline_runs_use_case
    )
    runs = pipeline_use_case.execute(owner_user_id=user.id, limit=1)
    conversations = request.app.state.chat_conversation_use_case.list_conversations(
        owner_user_id=user.id
    )
    return {
        "ok": True,
        "status": status.as_dict(),
        "postgres": _postgres_status(request),
        "latest_pipeline_run": runs[0].as_dict() if runs else None,
        "latest_conversation": (
            conversations[0].as_dict() if conversations else None
        ),
    }


def _postgres_status(request: Request) -> dict[str, object]:
    try:
        with database_session_factory(request)() as session:
            session.execute(text("SELECT 1"))
        return {"available": True, "error": None}
    except Exception as error:
        logger.exception("PostgreSQL health check failed")
        return {"available": False, "error": str(error)}
