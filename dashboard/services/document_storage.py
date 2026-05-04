from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Protocol
from uuid import UUID
from uuid import uuid4

from sqlalchemy.orm import Session, sessionmaker

from dashboard.auth import AuthenticatedUser
from dashboard.schemas import QdrantCleanupResult
from dashboard.services.qdrant_index import QdrantIndexService
from rag.config import DocumentPreparationSettings, get_document_preparation_settings
from rag.database.models import Document
from rag.database.repositories import DocumentRepository
from rag.database.session import session_scope
from rag.document_paths import (
    DocumentDeletionResult,
    delete_file_if_exists,
    parsed_markdown_relative_path,
    safe_relative_markdown_path,
    safe_relative_pdf_path,
)


class DocumentOperationError(ValueError):
    pass


class DocumentPermissionError(PermissionError):
    pass


class IndexCleanup(Protocol):
    def delete_content_hash(self, content_hash: str | None) -> QdrantCleanupResult:
        pass


class DocumentStorage:
    def __init__(
        self,
        *,
        index_cleanup: IndexCleanup | None = None,
        database_session_factory: sessionmaker[Session] | None = None,
    ) -> None:
        self._index_cleanup = index_cleanup or QdrantIndexService()
        self._database_session_factory = database_session_factory

    def save_uploaded_pdf(
        self,
        *,
        file_name: str | None,
        content: bytes,
        owner: AuthenticatedUser | None = None,
        settings: DocumentPreparationSettings | None = None,
    ) -> dict[str, Any]:
        settings = settings or get_document_preparation_settings()
        safe_name = _safe_upload_name(file_name)
        if not content:
            raise DocumentOperationError("Uploaded PDF is empty")

        settings.raw_documents_dir.mkdir(parents=True, exist_ok=True)
        session_factory = self._database_session_factory
        use_database = owner is not None and session_factory is not None
        relative_raw_path = _relative_upload_path(safe_name) if use_database else safe_name
        target_path = settings.raw_documents_dir / relative_raw_path
        if target_path.exists():
            raise DocumentOperationError(f"PDF already exists: {relative_raw_path}")

        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(content)
        if owner is None or session_factory is None:
            return {"relative_raw_path": relative_raw_path, "bytes": len(content)}

        try:
            with session_scope(session_factory) as session:
                document = DocumentRepository(session).create_uploaded_document(
                    owner_user_id=owner.id,
                    original_filename=safe_name,
                    relative_raw_path=relative_raw_path,
                    byte_size=len(content),
                )
                document_id = str(document.id)
        except Exception:
            target_path.unlink(missing_ok=True)
            _remove_empty_parent(target_path, stop_at=settings.raw_documents_dir)
            raise

        return {
            "document_id": document_id,
            "relative_raw_path": relative_raw_path,
            "bytes": len(content),
        }

    def save_uploaded_pdfs(
        self,
        files: Iterable[tuple[str | None, bytes]],
        *,
        owner: AuthenticatedUser | None = None,
        settings: DocumentPreparationSettings | None = None,
    ) -> list[dict[str, Any]]:
        return [
            self.save_uploaded_pdf(
                file_name=file_name,
                content=content,
                owner=owner,
                settings=settings,
            )
            for file_name, content in files
        ]

    def delete_document(
        self,
        relative_raw_path: str,
        *,
        owner: AuthenticatedUser | None = None,
        settings: DocumentPreparationSettings | None = None,
    ) -> dict[str, Any]:
        settings = settings or get_document_preparation_settings()
        safe_relative_raw_path = safe_relative_pdf_path(relative_raw_path).as_posix()
        document_content_hash = None
        document_id: UUID | None = None
        document_parsed_path = None
        if owner is not None and self._database_session_factory is not None:
            with self._database_session_factory() as session:
                repository = DocumentRepository(session)
                document = repository.get_by_relative_raw_path(safe_relative_raw_path)
                if document is None:
                    raise DocumentOperationError(
                        f"Document not found: {safe_relative_raw_path}"
                    )
                if document.owner_user_id != owner.id:
                    raise DocumentPermissionError("Document belongs to another user")
                document_id = document.id
                document_content_hash = document.content_hash
                document_parsed_path = document.parsed_markdown_path

        deletion = _delete_document_files(
            safe_relative_raw_path,
            parsed_markdown_path=document_parsed_path,
            content_hash=document_content_hash,
            settings=settings,
        )
        cleanup = self._index_cleanup.delete_content_hash(
            deletion.content_hash or document_content_hash
        )
        if document_id is not None and self._database_session_factory is not None:
            with session_scope(self._database_session_factory) as session:
                document = session.get(Document, document_id)
                if document is not None:
                    DocumentRepository(session).delete_document(document)
        return {
            "deletion": asdict(deletion),
            "qdrant_cleanup": cleanup.as_dict(),
        }

    def delete_documents(
        self,
        relative_raw_paths: Iterable[str],
        *,
        owner: AuthenticatedUser | None = None,
        settings: DocumentPreparationSettings | None = None,
    ) -> dict[str, Any]:
        settings = settings or get_document_preparation_settings()
        deletions = []
        cleanups = []
        for relative_raw_path in relative_raw_paths:
            result = self.delete_document(
                relative_raw_path,
                owner=owner,
                settings=settings,
            )
            deletions.append(result["deletion"])
            cleanups.append(result["qdrant_cleanup"])
        return {
            "deleted_count": len(deletions),
            "deletions": deletions,
            "qdrant_cleanups": cleanups,
        }


def _safe_upload_name(file_name: str | None) -> str:
    safe_name = Path(file_name or "").name
    if not safe_name or safe_name in {".", ".."}:
        raise DocumentOperationError("Missing uploaded file name")
    if Path(safe_name).suffix.lower() != ".pdf":
        raise DocumentOperationError("Only PDF files are supported")
    return safe_name


def _relative_upload_path(safe_name: str) -> str:
    return f"{uuid4()}/{safe_name}"


def _delete_document_files(
    relative_raw_path: str,
    *,
    parsed_markdown_path: str | None,
    content_hash: str | None,
    settings: DocumentPreparationSettings,
) -> DocumentDeletionResult:
    relative_path = safe_relative_pdf_path(relative_raw_path)
    parsed_relative_path = _parsed_relative_path(
        relative_path,
        parsed_markdown_path=parsed_markdown_path,
    )
    raw_deleted = delete_file_if_exists(settings.raw_documents_dir / relative_path)
    parsed_deleted = delete_file_if_exists(
        settings.parsed_markdown_dir / parsed_relative_path
    )
    return DocumentDeletionResult(
        relative_raw_path=relative_path.as_posix(),
        parsed_markdown_path=parsed_relative_path.as_posix(),
        content_hash=content_hash,
        raw_deleted=raw_deleted,
        parsed_deleted=parsed_deleted,
    )


def _parsed_relative_path(
    relative_raw_path: Path,
    *,
    parsed_markdown_path: str | None,
) -> Path:
    if parsed_markdown_path is None:
        return parsed_markdown_relative_path(relative_raw_path)
    return safe_relative_markdown_path(parsed_markdown_path)


def _remove_empty_parent(path: Path, *, stop_at: Path) -> None:
    parent = path.parent
    try:
        if parent != stop_at:
            parent.rmdir()
    except OSError:
        return
