from __future__ import annotations

from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Protocol
from uuid import UUID
from uuid import uuid4

import pymupdf
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


@dataclass(frozen=True)
class UploadPolicy:
    max_file_bytes: int = 25 * 1024 * 1024
    chunk_size: int = 1024 * 1024

    def __post_init__(self) -> None:
        if self.max_file_bytes < 1:
            raise ValueError("Upload max_file_bytes must be positive")
        if self.chunk_size < 1:
            raise ValueError("Upload chunk_size must be positive")


class IndexCleanup(Protocol):
    def delete_content_hash(self, content_hash: str | None) -> QdrantCleanupResult:
        pass


class DocumentStorage:
    def __init__(
        self,
        *,
        index_cleanup: IndexCleanup | None = None,
        database_session_factory: sessionmaker[Session] | None = None,
        upload_policy: UploadPolicy | None = None,
    ) -> None:
        self._index_cleanup = index_cleanup or QdrantIndexService()
        self._database_session_factory = database_session_factory
        self._upload_policy = upload_policy or UploadPolicy()

    def save_uploaded_pdf(
        self,
        *,
        file_name: str | None,
        content: bytes,
        owner: AuthenticatedUser | None = None,
        settings: DocumentPreparationSettings | None = None,
    ) -> dict[str, Any]:
        return self.save_uploaded_pdf_stream(
            file_name=file_name,
            stream=BytesIO(content),
            owner=owner,
            settings=settings,
        )

    def save_uploaded_pdf_stream(
        self,
        *,
        file_name: str | None,
        stream: BinaryIO,
        owner: AuthenticatedUser | None = None,
        settings: DocumentPreparationSettings | None = None,
    ) -> dict[str, Any]:
        settings = settings or get_document_preparation_settings()
        safe_name = _safe_upload_name(file_name)

        settings.raw_documents_dir.mkdir(parents=True, exist_ok=True)
        session_factory = self._database_session_factory
        use_database = owner is not None and session_factory is not None
        relative_raw_path = (
            _relative_upload_path(safe_name) if use_database else safe_name
        )
        target_path = settings.raw_documents_dir / relative_raw_path
        if target_path.exists():
            raise DocumentOperationError(f"PDF already exists: {relative_raw_path}")

        target_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = _temporary_upload_path(target_path)
        try:
            byte_size = _stream_upload_to_file(
                stream,
                temp_path,
                policy=self._upload_policy,
            )
            _validate_pdf(temp_path)
            temp_path.replace(target_path)
        except Exception:
            temp_path.unlink(missing_ok=True)
            _remove_empty_parent(temp_path, stop_at=settings.raw_documents_dir)
            raise

        if owner is None or session_factory is None:
            return {"relative_raw_path": relative_raw_path, "bytes": byte_size}

        try:
            with session_scope(session_factory) as session:
                document = DocumentRepository(session).create_uploaded_document(
                    owner_user_id=owner.id,
                    original_filename=safe_name,
                    relative_raw_path=relative_raw_path,
                    byte_size=byte_size,
                )
                document_id = str(document.id)
        except Exception:
            target_path.unlink(missing_ok=True)
            _remove_empty_parent(target_path, stop_at=settings.raw_documents_dir)
            raise

        return {
            "document_id": document_id,
            "relative_raw_path": relative_raw_path,
            "bytes": byte_size,
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

    def delete_document_by_id(
        self,
        document_id: UUID,
        *,
        owner: AuthenticatedUser,
        settings: DocumentPreparationSettings,
    ) -> dict[str, Any]:
        session_factory = self._database_session_factory
        if session_factory is None:
            raise DocumentOperationError("Database storage is not configured")
        with session_factory() as session:
            document = DocumentRepository(session).get_by_id_for_owner(
                document_id=document_id,
                owner_user_id=owner.id,
            )
            if document is None:
                raise DocumentOperationError(f"Document not found: {document_id}")
            relative_raw_path = document.relative_raw_path
        return self.delete_document(
            relative_raw_path,
            owner=owner,
            settings=settings,
        )

    def delete_documents_by_id(
        self,
        document_ids: Iterable[UUID],
        *,
        owner: AuthenticatedUser,
        settings: DocumentPreparationSettings,
    ) -> dict[str, Any]:
        deletions = [
            self.delete_document_by_id(
                document_id,
                owner=owner,
                settings=settings,
            )
            for document_id in document_ids
        ]
        return {
            "deleted_count": len(deletions),
            "deletions": [item["deletion"] for item in deletions],
            "qdrant_cleanups": [item["qdrant_cleanup"] for item in deletions],
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


def _stream_upload_to_file(
    stream: BinaryIO,
    target_path: Path,
    *,
    policy: UploadPolicy,
) -> int:
    byte_size = 0
    with target_path.open("xb") as target:
        while chunk := stream.read(policy.chunk_size):
            next_size = byte_size + len(chunk)
            if next_size > policy.max_file_bytes:
                raise DocumentOperationError(
                    f"Uploaded PDF exceeds {policy.max_file_bytes} bytes"
                )
            target.write(chunk)
            byte_size = next_size
    if byte_size == 0:
        raise DocumentOperationError("Uploaded PDF is empty")
    return byte_size


def _validate_pdf(path: Path) -> None:
    try:
        document = pymupdf.open(path)  # type: ignore[no-untyped-call]
        try:
            if document.page_count < 1:
                raise DocumentOperationError("Uploaded PDF has no pages")
        finally:
            document.close()  # type: ignore[no-untyped-call]
    except DocumentOperationError:
        raise
    except Exception as error:
        raise DocumentOperationError("Uploaded file is not a valid PDF") from error


def _temporary_upload_path(target_path: Path) -> Path:
    return target_path.with_name(f".{target_path.name}.{uuid4().hex}.upload")


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
