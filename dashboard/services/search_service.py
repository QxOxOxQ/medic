from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy.orm import Session, sessionmaker

from rag.retrieval import (
    RetrievalService,
    SearchProvider,
    SearchResult,
    search_results_from_response,
)


class SearchService:
    def __init__(
        self,
        *,
        search_provider: SearchProvider | None = None,
        database_session_factory: sessionmaker[Session] | None = None,
    ) -> None:
        self._retrieval = RetrievalService(
            search_provider=search_provider,
            database_session_factory=database_session_factory,
        )

    def search(
        self,
        *,
        query: str,
        limit: int,
        owner_user_id: UUID | None = None,
    ) -> list[dict[str, Any]]:
        results = self._retrieval.search(
            query=query,
            limit=limit,
            owner_user_id=owner_user_id,
        )
        return [result.as_dict() for result in results]

__all__ = [
    "RetrievalService",
    "SearchProvider",
    "SearchResult",
    "SearchService",
    "search_results_from_response",
]
