from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session, sessionmaker

from agents.observability import AgentObservability
from backend.chat_use_cases import ChatConversationUseCase
from backend.factory import (
    build_answer_question_use_case,
    build_chat_conversation_use_case,
)
from backend.routes import router as backend_router
from backend.use_cases import AnswerQuestionUseCase
from dashboard.auth import AuthSettings, load_auth_settings
from dashboard.dependencies import current_user as resolve_current_user
from dashboard.jobs import JobStore
from dashboard.routes import documents, health, jobs, pages, search
from dashboard.services.document_catalog import DocumentCatalog
from dashboard.services.document_storage import DocumentStorage
from dashboard.services.process_detail import ProcessDetailService
from dashboard.services.qdrant_index import QdrantIndexService
from dashboard.services.search_service import SearchService
from observability import build_agent_observability
from rag.config import DocumentPreparationSettings, get_document_preparation_settings
from rag.database import get_session_factory
from rag.full_process import FullProcess
from rag.searcher import Searcher


BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"


def create_app(
    *,
    auth_settings: AuthSettings | None = None,
    document_settings: DocumentPreparationSettings | None = None,
    job_store: JobStore | None = None,
    searcher_factory: Callable[[], Any] | None = None,
    document_catalog: DocumentCatalog | None = None,
    document_storage: DocumentStorage | None = None,
    process_detail_service: ProcessDetailService | None = None,
    search_service: SearchService | None = None,
    answer_question_use_case: AnswerQuestionUseCase | None = None,
    chat_conversation_use_case: ChatConversationUseCase | None = None,
    database_session_factory: sessionmaker[Session] | None = None,
    agent_observability: AgentObservability | None = None,
) -> FastAPI:
    app = FastAPI(title="Medic RAG Dashboard", lifespan=_app_lifespan)
    qdrant_index = QdrantIndexService()
    session_factory = database_session_factory or get_session_factory()
    observability = agent_observability or build_agent_observability()
    catalog = document_catalog or DocumentCatalog(
        index_reader=qdrant_index,
        database_session_factory=session_factory,
    )

    app.state.auth_settings = auth_settings or load_auth_settings()
    app.state.agent_observability = observability
    app.state.document_settings = (
        document_settings or get_document_preparation_settings()
    )
    app.state.job_store = job_store or JobStore(
        process_factory=lambda settings: FullProcess(
            settings=settings,
            database_session_factory=session_factory,
        )
    )
    app.state.database_session_factory = session_factory
    app.state.document_catalog = catalog
    app.state.document_storage = document_storage or DocumentStorage(
        index_cleanup=qdrant_index,
        database_session_factory=session_factory,
    )
    app.state.process_detail_service = process_detail_service or ProcessDetailService(
        catalog=catalog,
        qdrant_index=qdrant_index,
        database_session_factory=session_factory,
    )
    app.state.search_service = search_service or _search_service(
        searcher_factory,
        database_session_factory=session_factory,
    )
    app.state.answer_question_use_case = (
        answer_question_use_case
        or build_answer_question_use_case(
            database_session_factory=session_factory,
            observability=observability,
        )
    )
    app.state.chat_conversation_use_case = (
        chat_conversation_use_case
        or build_chat_conversation_use_case(
            database_session_factory=session_factory,
            observability=observability,
        )
    )
    app.state.current_user_resolver = resolve_current_user
    app.state.searcher_factory = searcher_factory or Searcher
    app.state.templates = Jinja2Templates(directory=TEMPLATES_DIR)

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    app.include_router(health.router)
    app.include_router(pages.router)
    app.include_router(documents.router)
    app.include_router(jobs.router)
    app.include_router(search.router)
    app.include_router(backend_router)
    return app


@asynccontextmanager
async def _app_lifespan(app: FastAPI) -> AsyncIterator[None]:
    try:
        yield
    finally:
        observability = cast(AgentObservability, app.state.agent_observability)
        observability.close()


def _search_service(
    searcher_factory: Callable[[], Any] | None,
    *,
    database_session_factory: sessionmaker[Session],
) -> SearchService:
    if searcher_factory is None:
        return SearchService(database_session_factory=database_session_factory)
    return SearchService(
        search_provider=searcher_factory(),
        database_session_factory=database_session_factory,
    )
