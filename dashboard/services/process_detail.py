from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy.orm import Session, sessionmaker

from dashboard.schemas import DocumentRecord
from dashboard.services.document_catalog import DocumentCatalog
from dashboard.services.qdrant_index import QdrantIndexService
from rag.config import DocumentPreparationSettings, get_document_preparation_settings
from rag.database.repositories import DocumentRepository
from rag.document_paths import safe_relative_markdown_path
from rag.indexer import chunks_from_text


class ProcessDetailService:
    def __init__(
        self,
        *,
        catalog: DocumentCatalog | None = None,
        qdrant_index: QdrantIndexService | None = None,
        database_session_factory: sessionmaker[Session] | None = None,
    ) -> None:
        self._catalog = catalog or DocumentCatalog()
        self._qdrant_index = qdrant_index or QdrantIndexService()
        self._database_session_factory = database_session_factory

    def document_process_detail(
        self,
        relative_raw_path: str,
        *,
        settings: DocumentPreparationSettings | None = None,
        owner_user_id: UUID | None = None,
    ) -> dict[str, Any]:
        settings = settings or get_document_preparation_settings()
        records, _ = self._catalog.list_records(settings, owner_user_id=owner_user_id)
        record = _find_record(records, relative_raw_path)
        if record is None:
            raise ValueError(f"Document not found: {relative_raw_path}")

        markdown = _read_markdown(record, settings=settings)
        chunks = _database_chunks_payload(
            session_factory=self._database_session_factory,
            relative_raw_path=relative_raw_path,
            owner_user_id=owner_user_id,
        )
        if not chunks:
            chunks = (
                _chunks_payload(
                    markdown,
                    parsed_markdown_path=record.parsed_markdown_path,
                    content_hash=record.content_hash,
                )
                if markdown is not None
                else []
            )
        return {
            "document": record.as_dict(),
            "markdown": markdown,
            "chunks": chunks,
            "chunk_count": len(chunks),
            "index": self._qdrant_index.preview_content_hash(record.content_hash),
        }


def _find_record(
    records: list[DocumentRecord],
    relative_raw_path: str,
) -> DocumentRecord | None:
    return next(
        (
            record
            for record in records
            if record.relative_raw_path == relative_raw_path
        ),
        None,
    )


def _read_markdown(
    record: DocumentRecord,
    *,
    settings: DocumentPreparationSettings,
) -> str | None:
    if not record.parsed_markdown_path or not record.parsed_exists:
        return None

    parsed_path = settings.parsed_markdown_dir / safe_relative_markdown_path(
        record.parsed_markdown_path
    )
    return parsed_path.read_text(encoding="utf-8")


def _chunks_payload(
    markdown: str,
    *,
    parsed_markdown_path: str | None,
    content_hash: str | None,
) -> list[dict[str, Any]]:
    source = parsed_markdown_path or "<unknown>"
    chunks = chunks_from_text(
        markdown,
        {
            "file_name": Path(source).name,
            "source": source,
            "content_hash": content_hash,
        },
    )
    return [
        {
            "index": index,
            "char_start": chunk.metadata.get("char_start"),
            "char_end": chunk.metadata.get("char_end"),
            "characters": len(chunk.content),
            "content": chunk.content,
        }
        for index, chunk in enumerate(chunks, start=1)
    ]


def _database_chunks_payload(
    *,
    session_factory: sessionmaker[Session] | None,
    relative_raw_path: str,
    owner_user_id: UUID | None,
) -> list[dict[str, Any]]:
    if session_factory is None or owner_user_id is None:
        return []
    with session_factory() as session:
        chunks = DocumentRepository(session).chunks_for_relative_raw_path(
            relative_raw_path=relative_raw_path,
            owner_user_id=owner_user_id,
        )
        return [
            {
                "index": chunk.chunk_index,
                "char_start": chunk.char_start,
                "char_end": chunk.char_end,
                "characters": len(chunk.content),
                "content": chunk.content,
            }
            for chunk in chunks
        ]
