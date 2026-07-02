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
from backend.chat_run_use_cases import (
    GetChatRunUseCase,
    StartChatRunUseCase,
    StreamChatRunEventsUseCase,
)
from backend.chat_use_cases import ChatConversationUseCase
from backend.pipeline_use_cases import (
    GetPipelineRunUseCase,
    ListPipelineRunsUseCase,
    StartPipelineRunUseCase,
    StreamPipelineEventsUseCase,
)
from backend.factory import (
    build_agent_runner_factory,
    build_chat_conversation_use_case,
    build_llm_provider_stats_use_case,
)
from backend.llm_provider_stats import GetLLMProviderStatsUseCase
from backend.routes import router as backend_router
from dashboard.admin import configure_admin
from dashboard.auth import AuthSettings, load_auth_settings
from dashboard.dependencies import current_user as resolve_current_user
from dashboard.jobs import JobStore
from dashboard.routes import (
    admin_stats,
    chat_runs,
    documents,
    health,
    jobs,
    pages,
    pipeline_runs,
    search,
    settings,
    workspace,
)
from dashboard.services.background_executor import ThreadBackgroundExecutor
from dashboard.services.document_catalog import DocumentCatalog
from dashboard.services.document_storage import DocumentStorage
from dashboard.services.process_detail import ProcessDetailService
from dashboard.services.qdrant_index import QdrantIndexService
from dashboard.services.search_service import SearchService
from observability import build_agent_observability
from rag.config import DocumentPreparationSettings, get_document_preparation_settings
from rag.database import get_session_factory
from rag.database.chat_store import SqlAlchemyChatConversationStore
from rag.database.pipeline_store import SqlAlchemyPipelineRunRepository
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
    chat_conversation_use_case: ChatConversationUseCase | None = None,
    database_session_factory: sessionmaker[Session] | None = None,
    agent_observability: AgentObservability | None = None,
    llm_provider_stats_use_case: GetLLMProviderStatsUseCase | None = None,
) -> FastAPI:
    app = FastAPI(title="Medic RAG Dashboard", lifespan=_app_lifespan)
    qdrant_index = QdrantIndexService()
    session_factory = database_session_factory or get_session_factory()
    observability = agent_observability or build_agent_observability()
    catalog = document_catalog or DocumentCatalog(
        index_reader=qdrant_index,
        database_session_factory=session_factory,
    )

    resolved_auth_settings = auth_settings or load_auth_settings()
    app.state.auth_settings = resolved_auth_settings
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
    app.state.chat_conversation_use_case = (
        chat_conversation_use_case
        or build_chat_conversation_use_case(
            database_session_factory=session_factory,
            observability=observability,
        )
    )
    app.state.current_user_resolver = resolve_current_user
    app.state.searcher_factory = searcher_factory or Searcher
    app.state.llm_provider_stats_use_case = (
        llm_provider_stats_use_case or build_llm_provider_stats_use_case()
    )
    app.state.templates = Jinja2Templates(directory=TEMPLATES_DIR)
    app.state.frontend_manifest_path = STATIC_DIR / "dist" / "manifest.json"
    pipeline_repository = SqlAlchemyPipelineRunRepository(session_factory)
    pipeline_executor = ThreadBackgroundExecutor()
    app.state.pipeline_run_repository = pipeline_repository
    app.state.start_pipeline_run_use_case = StartPipelineRunUseCase(
        repository=pipeline_repository,
        process_factory=lambda: FullProcess(
            settings=app.state.document_settings,
            database_session_factory=session_factory,
        ),
        executor=pipeline_executor,
    )
    app.state.list_pipeline_runs_use_case = ListPipelineRunsUseCase(
        pipeline_repository
    )
    app.state.get_pipeline_run_use_case = GetPipelineRunUseCase(pipeline_repository)
    app.state.stream_pipeline_events_use_case = StreamPipelineEventsUseCase(
        pipeline_repository
    )
    chat_run_store = SqlAlchemyChatConversationStore(session_factory)
    app.state.chat_run_store = chat_run_store
    app.state.start_chat_run_use_case = StartChatRunUseCase(
        store=chat_run_store,
        agent_runner_factory=build_agent_runner_factory(
            database_session_factory=session_factory,
            observability=observability,
        ),
        executor=pipeline_executor,
    )
    app.state.get_chat_run_use_case = GetChatRunUseCase(chat_run_store)
    app.state.stream_chat_run_events_use_case = StreamChatRunEventsUseCase(
        chat_run_store
    )

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    app.include_router(health.router)
    app.include_router(pages.router)
    app.include_router(documents.router)
    app.include_router(jobs.router)
    app.include_router(pipeline_runs.router)
    app.include_router(chat_runs.router)
    app.include_router(settings.router)
    app.include_router(workspace.router)
    app.include_router(search.router)
    app.include_router(admin_stats.router)
    app.include_router(backend_router)
    configure_admin(
        app,
        auth_settings=resolved_auth_settings,
        session_factory=session_factory,
    )
    return app


@asynccontextmanager
async def _app_lifespan(app: FastAPI) -> AsyncIterator[None]:
    pipeline_repository = cast(
        SqlAlchemyPipelineRunRepository,
        app.state.pipeline_run_repository,
    )
    pipeline_repository.interrupt_active_runs()
    chat_run_store = cast(
        SqlAlchemyChatConversationStore,
        app.state.chat_run_store,
    )
    chat_run_store.interrupt_active_runs()
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
