from __future__ import annotations

from typing import Any, Protocol
from uuid import UUID

from sqlalchemy.orm import Session, sessionmaker

from dashboard.schemas import DashboardStatus, DocumentRecord
from dashboard.services.document_records import (
    build_document_record,
    raw_document_keys,
)
from dashboard.services.qdrant_index import QdrantIndexService
from rag.config import DocumentPreparationSettings, get_document_preparation_settings
from rag.database.models import Document
from rag.database.repositories import DocumentRepository
from rag.document_paths import safe_relative_markdown_path, safe_relative_pdf_path


class IndexStatusReader(Protocol):
    def indexed_content_hashes(
        self,
        content_hashes: set[str],
    ) -> tuple[set[str], str | None]:
        pass

    def status(self) -> dict[str, Any]:
        pass


class DocumentCatalog:
    def __init__(
        self,
        *,
        index_reader: IndexStatusReader | None = None,
        database_session_factory: sessionmaker[Session] | None = None,
    ) -> None:
        self._index_reader = index_reader or QdrantIndexService()
        self._database_session_factory = database_session_factory

    def dashboard_status(
        self,
        settings: DocumentPreparationSettings | None = None,
        owner_user_id: UUID | None = None,
    ) -> DashboardStatus:
        settings = settings or get_document_preparation_settings()
        records, qdrant_error = self.list_records(settings, owner_user_id=owner_user_id)
        qdrant = self._index_reader.status()
        if qdrant_error and qdrant.get("available"):
            qdrant["index_check_error"] = qdrant_error

        processed_times = [
            record.processed_at for record in records if record.processed_at
        ]
        return DashboardStatus(
            raw_pdf_count=sum(1 for record in records if record.raw_exists),
            parsed_markdown_count=sum(1 for record in records if record.parsed_exists),
            document_count=len(records),
            last_processed_at=max(processed_times) if processed_times else None,
            qdrant=qdrant,
        )

    def list_records(
        self,
        settings: DocumentPreparationSettings | None = None,
        owner_user_id: UUID | None = None,
    ) -> tuple[list[DocumentRecord], str | None]:
        settings = settings or get_document_preparation_settings()
        if owner_user_id is not None and self._database_session_factory is not None:
            return self._list_database_records(settings, owner_user_id)

        raw_keys = raw_document_keys(settings)
        records = [
            build_document_record(
                relative_raw_path=relative_raw_path,
                raw_keys=raw_keys,
                indexed_hashes=set(),
                qdrant_error=None,
                settings=settings,
            )
            for relative_raw_path in sorted(raw_keys)
        ]
        return records, None

    def _list_database_records(
        self,
        settings: DocumentPreparationSettings,
        owner_user_id: UUID,
    ) -> tuple[list[DocumentRecord], str | None]:
        session_factory = self._database_session_factory
        if session_factory is None:
            raise RuntimeError("Database session factory is not configured")
        with session_factory() as session:
            documents = DocumentRepository(session).list_for_owner(owner_user_id)
            content_hashes = {
                document.content_hash for document in documents if document.content_hash
            }
            indexed_hashes, qdrant_error = self._index_reader.indexed_content_hashes(
                content_hashes
            )
            return [
                _document_record_from_database(
                    document,
                    settings=settings,
                    indexed_hashes=indexed_hashes,
                    qdrant_error=qdrant_error,
                )
                for document in documents
            ], qdrant_error


def _document_record_from_database(
    document: Document,
    *,
    settings: DocumentPreparationSettings,
    indexed_hashes: set[str],
    qdrant_error: str | None,
) -> DocumentRecord:
    raw_exists = _relative_pdf_exists(
        settings=settings,
        relative_raw_path=document.relative_raw_path,
    )
    parsed_exists = _relative_markdown_exists(
        settings=settings,
        parsed_markdown_path=document.parsed_markdown_path,
    )
    indexed = _database_indexed_status(
        content_hash=document.content_hash,
        indexed_hashes=indexed_hashes,
        qdrant_error=qdrant_error,
    )
    return DocumentRecord(
        relative_raw_path=document.relative_raw_path,
        original_filename=document.original_filename,
        display_name=document.original_filename,
        raw_exists=raw_exists,
        parsed_markdown_path=document.parsed_markdown_path,
        parsed_exists=parsed_exists,
        content_hash=document.content_hash,
        processed_at=document.processed_at.isoformat() if document.processed_at else None,
        indexed=indexed,
        status=_database_document_status(
            persisted_status=document.status,
            raw_exists=raw_exists,
            parsed_exists=parsed_exists,
            indexed=indexed,
        ),
        processing_error=document.processing_error,
    )


def _database_indexed_status(
    *,
    content_hash: str | None,
    indexed_hashes: set[str],
    qdrant_error: str | None,
) -> bool | None:
    if qdrant_error is not None or content_hash is None:
        return None
    return content_hash in indexed_hashes


def _database_document_status(
    *,
    persisted_status: str,
    raw_exists: bool,
    parsed_exists: bool,
    indexed: bool | None,
) -> str:
    if not raw_exists:
        return "stale"
    if persisted_status == "failed":
        return "failed"
    if persisted_status == "raw" or not parsed_exists:
        return "raw"
    if indexed is True:
        return "indexed"
    if indexed is False:
        return "prepared"
    return "prepared_unverified"


def _relative_pdf_exists(
    *,
    settings: DocumentPreparationSettings,
    relative_raw_path: str,
) -> bool:
    try:
        relative_path = safe_relative_pdf_path(relative_raw_path)
    except ValueError:
        return False
    return (settings.raw_documents_dir / relative_path).exists()


def _relative_markdown_exists(
    *,
    settings: DocumentPreparationSettings,
    parsed_markdown_path: str | None,
) -> bool:
    if parsed_markdown_path is None:
        return False
    try:
        relative_path = safe_relative_markdown_path(parsed_markdown_path)
    except ValueError:
        return False
    return (settings.parsed_markdown_dir / relative_path).exists()
