from __future__ import annotations

from typing import Any

from dashboard.schemas import DocumentRecord, QdrantCleanupResult
from dashboard.services.document_catalog import DocumentCatalog
from dashboard.services.document_storage import DocumentStorage
from dashboard.services.process_detail import ProcessDetailService
from dashboard.services.qdrant_index import QdrantIndexService
from dashboard.services.qdrant_preview import (
    qdrant_index_preview_for_content_hash as qdrant_index_preview_for_content_hash,
)
from rag.config import DocumentPreparationSettings


def dashboard_status(
    settings: DocumentPreparationSettings | None = None,
) -> dict[str, Any]:
    return DocumentCatalog().dashboard_status(settings).as_dict()


def list_document_records(
    settings: DocumentPreparationSettings | None = None,
) -> tuple[list[DocumentRecord], str | None]:
    return DocumentCatalog().list_records(settings)


def save_uploaded_pdf(
    *,
    file_name: str | None,
    content: bytes,
    settings: DocumentPreparationSettings | None = None,
) -> dict[str, Any]:
    return DocumentStorage().save_uploaded_pdf(
        file_name=file_name,
        content=content,
        settings=settings,
    )


def delete_document(
    relative_raw_path: str,
    *,
    settings: DocumentPreparationSettings | None = None,
) -> dict[str, Any]:
    return DocumentStorage().delete_document(relative_raw_path, settings=settings)


def document_process_detail(
    relative_raw_path: str,
    *,
    settings: DocumentPreparationSettings | None = None,
) -> dict[str, Any]:
    return ProcessDetailService().document_process_detail(
        relative_raw_path,
        settings=settings,
    )


def qdrant_status() -> dict[str, Any]:
    return QdrantIndexService().status()


def delete_indexed_content_hash(content_hash: str | None) -> QdrantCleanupResult:
    return QdrantIndexService().delete_content_hash(content_hash)
