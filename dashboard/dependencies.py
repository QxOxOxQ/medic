from __future__ import annotations

from typing import Any, cast

from fastapi import Request
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session, sessionmaker

from dashboard.auth import AuthSettings, AuthenticatedUser, require_active_user
from dashboard.jobs import JobStore
from dashboard.services.document_catalog import DocumentCatalog
from dashboard.services.document_storage import DocumentStorage
from dashboard.services.process_detail import ProcessDetailService
from dashboard.services.search_service import SearchService
from rag.config import DocumentPreparationSettings


def auth_settings(request: Request) -> AuthSettings:
    return cast(AuthSettings, request.app.state.auth_settings)


def document_settings(request: Request) -> DocumentPreparationSettings:
    return cast(DocumentPreparationSettings, request.app.state.document_settings)


def job_store(request: Request) -> JobStore:
    return cast(JobStore, request.app.state.job_store)


def database_session_factory(request: Request) -> sessionmaker[Session]:
    return cast(sessionmaker[Session], request.app.state.database_session_factory)


def current_user(request: Request) -> AuthenticatedUser:
    return require_active_user(
        request,
        auth_settings(request),
        database_session_factory(request),
    )


def templates(request: Request) -> Jinja2Templates:
    return cast(Jinja2Templates, request.app.state.templates)


def document_catalog(request: Request) -> DocumentCatalog:
    return cast(DocumentCatalog, request.app.state.document_catalog)


def document_storage(request: Request) -> DocumentStorage:
    return cast(DocumentStorage, request.app.state.document_storage)


def process_detail_service(request: Request) -> ProcessDetailService:
    return cast(ProcessDetailService, request.app.state.process_detail_service)


def search_service(request: Request) -> SearchService:
    return cast(SearchService, request.app.state.search_service)


def searcher(request: Request) -> Any:
    return request.app.state.searcher_factory()
