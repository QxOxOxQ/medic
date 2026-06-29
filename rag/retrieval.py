from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID

from sqlalchemy.orm import Session, sessionmaker

from rag.database.repositories import (
    DocumentRepository,
    SearchDocumentMetadata,
    SearchOwnership,
)
from rag.searcher import Searcher


class SearchProvider(Protocol):
    def search(
        self,
        query: str,
        k: int = 3,
        *,
        owner_user_id: UUID | None = None,
    ) -> Any:
        ...


@dataclass(frozen=True)
class SearchResult:
    score: float | None
    source: str | None
    content_hash: str | None
    document_name: str | None
    excerpt: str
    qdrant_point_id: str | None = None
    document_id: UUID | None = None
    relative_raw_path: str | None = None
    chunk_index: int | None = None
    char_start: int | None = None
    char_end: int | None = None

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.document_id is not None:
            payload["document_id"] = str(self.document_id)
        return payload


class RetrievalService:
    def __init__(
        self,
        *,
        search_provider: SearchProvider | None = None,
        database_session_factory: sessionmaker[Session] | None = None,
    ) -> None:
        self._search_provider = search_provider
        self._database_session_factory = database_session_factory

    def search(
        self,
        *,
        query: str,
        limit: int,
        owner_user_id: UUID | None = None,
    ) -> list[SearchResult]:
        search_provider = self._search_provider or Searcher()
        response = search_provider.search(query, k=limit, owner_user_id=owner_user_id)
        return search_results_from_response(
            response,
            limit=limit,
            owner_user_id=owner_user_id,
            database_session_factory=self._database_session_factory,
        )


def search_results_from_response(
    response: Any,
    *,
    limit: int = 10,
    owner_user_id: UUID | None = None,
    database_session_factory: sessionmaker[Session] | None = None,
) -> list[SearchResult]:
    points = list(getattr(response, "points", response))
    if owner_user_id is not None and database_session_factory is not None:
        ownership = _search_ownership(
            points,
            owner_user_id=owner_user_id,
            database_session_factory=database_session_factory,
        )
        return _database_search_results(points, ownership=ownership, limit=limit)

    return []


def _search_ownership(
    points: list[Any],
    *,
    owner_user_id: UUID,
    database_session_factory: sessionmaker[Session],
) -> SearchOwnership:
    qdrant_point_ids: set[str] = set()
    for point in points:
        point_id = _point_id(point)
        if point_id:
            qdrant_point_ids.add(point_id)

    payloads = [getattr(point, "payload", {}) or {} for point in points]
    content_hashes: set[str] = set()
    sources: set[str] = set()
    for payload in payloads:
        content_hash = payload.get("content_hash")
        if isinstance(content_hash, str):
            content_hashes.add(content_hash)
        source = payload.get("source") or payload.get("file_name")
        if isinstance(source, str):
            sources.add(source)
    with database_session_factory() as session:
        return DocumentRepository(session).ownership_for_search(
            owner_user_id=owner_user_id,
            qdrant_point_ids=qdrant_point_ids,
            content_hashes=content_hashes,
            sources=sources,
        )


def _database_search_results(
    points: list[Any],
    *,
    ownership: SearchOwnership,
    limit: int,
) -> list[SearchResult]:
    results: list[SearchResult] = []
    for point in points:
        payload = getattr(point, "payload", {}) or {}
        source = payload.get("source") or payload.get("file_name")
        content_hash = payload.get("content_hash")
        if not _belongs_to_database(
            qdrant_point_id=_point_id(point),
            content_hash=content_hash,
            source=source,
            ownership=ownership,
        ):
            continue
        metadata = _document_metadata(
            qdrant_point_id=_point_id(point),
            content_hash=content_hash,
            source=source,
            ownership=ownership,
        )
        results.append(
            SearchResult(
                score=getattr(point, "score", None),
                source=source,
                content_hash=content_hash,
                document_name=metadata.document_name if metadata else None,
                excerpt=_excerpt(str(payload.get("content", ""))),
                qdrant_point_id=_point_id(point),
                document_id=metadata.document_id if metadata else None,
                relative_raw_path=metadata.relative_raw_path if metadata else None,
                chunk_index=metadata.chunk_index if metadata else None,
                char_start=metadata.char_start if metadata else None,
                char_end=metadata.char_end if metadata else None,
            )
        )
        if len(results) >= limit:
            break
    return results


def _belongs_to_database(
    *,
    qdrant_point_id: str | None,
    content_hash: str | None,
    source: str | None,
    ownership: SearchOwnership,
) -> bool:
    if qdrant_point_id:
        return qdrant_point_id in ownership.qdrant_point_ids
    source_name = Path(source).name if source else None
    return (
        content_hash in ownership.content_hashes
        or source in ownership.sources
        or source_name in ownership.sources
    )


def _point_id(point: Any) -> str | None:
    point_id = getattr(point, "id", None)
    return str(point_id) if point_id else None


def _document_metadata(
    *,
    qdrant_point_id: str | None,
    content_hash: str | None,
    source: str | None,
    ownership: SearchOwnership,
) -> SearchDocumentMetadata | None:
    if qdrant_point_id:
        return ownership.metadata_by_point_id.get(qdrant_point_id)
    if content_hash and content_hash in ownership.metadata_by_hash:
        return ownership.metadata_by_hash[content_hash]
    if source and source in ownership.metadata_by_source:
        return ownership.metadata_by_source[source]
    source_name = Path(source).name if source else None
    if source_name and source_name in ownership.metadata_by_source:
        return ownership.metadata_by_source[source_name]
    return None


def _excerpt(content: str, max_length: int = 1200) -> str:
    compact = " ".join(content.split())
    if len(compact) <= max_length:
        return compact
    return f"{compact[: max_length - 1]}..."
