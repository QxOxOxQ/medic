from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.orm import Session, selectinload

from rag.database.models import Document, DocumentChunk, User
from rag.database.security import hash_password, verify_password


@dataclass(frozen=True)
class ChunkInput:
    chunk_index: int
    char_start: int | None
    char_end: int | None
    content: str
    qdrant_point_id: str


@dataclass(frozen=True)
class SearchDocumentMetadata:
    document_id: UUID
    document_name: str
    relative_raw_path: str
    chunk_index: int | None = None
    char_start: int | None = None
    char_end: int | None = None


@dataclass(frozen=True)
class SearchOwnership:
    qdrant_point_ids: set[str]
    content_hashes: set[str]
    sources: set[str]
    metadata_by_point_id: dict[str, SearchDocumentMetadata]
    metadata_by_hash: dict[str, SearchDocumentMetadata]
    metadata_by_source: dict[str, SearchDocumentMetadata]


class UserRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def get_by_id(self, user_id: UUID) -> User | None:
        return self._session.get(User, user_id)

    def get_by_username(self, username: str) -> User | None:
        normalized = normalize_username(username)
        return self._session.scalar(select(User).where(User.username == normalized))

    def create_user(
        self,
        *,
        username: str,
        password: str,
        is_admin: bool = False,
        is_active: bool = True,
    ) -> User:
        user = User(
            username=normalize_username(username),
            password_hash=hash_password(password),
            is_admin=is_admin,
            is_active=is_active,
        )
        self._session.add(user)
        self._session.flush()
        return user

    def seed_admin(self, *, username: str, password: str) -> User:
        existing = self.get_by_username(username)
        if existing is not None:
            return existing
        return self.create_user(username=username, password=password, is_admin=True)

    def authenticate(self, *, username: str, password: str) -> User | None:
        user = self.get_by_username(username)
        if user is None or not user.is_active:
            return None
        if not verify_password(password, user.password_hash):
            return None
        return user

    def first_active_admin(self) -> User | None:
        return self._session.scalar(
            select(User)
            .where(User.is_admin.is_(True), User.is_active.is_(True))
            .order_by(User.created_at, User.username)
            .limit(1)
        )

    def set_preferred_chat_model(
        self,
        *,
        user_id: UUID,
        model_key: str,
    ) -> User | None:
        user = self.get_by_id(user_id)
        if user is None:
            return None
        user.preferred_chat_model = model_key
        self._session.flush()
        return user

class DocumentRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def create_uploaded_document(
        self,
        *,
        owner_user_id: UUID,
        original_filename: str,
        relative_raw_path: str,
        byte_size: int,
    ) -> Document:
        document = Document(
            owner_user_id=owner_user_id,
            original_filename=original_filename,
            relative_raw_path=relative_raw_path,
            byte_size=byte_size,
            status="raw",
        )
        self._session.add(document)
        self._session.flush()
        return document

    def get_by_relative_raw_path(self, relative_raw_path: str) -> Document | None:
        return self._session.scalar(
            select(Document).where(Document.relative_raw_path == relative_raw_path)
        )

    def get_by_id_for_owner(
        self,
        *,
        document_id: UUID,
        owner_user_id: UUID,
    ) -> Document | None:
        return self._session.scalar(
            select(Document).where(
                Document.id == document_id,
                Document.owner_user_id == owner_user_id,
            )
        )

    def get_by_relative_raw_path_for_owner(
        self,
        *,
        relative_raw_path: str,
        owner_user_id: UUID,
    ) -> Document | None:
        return self._session.scalar(
            select(Document).where(
                Document.relative_raw_path == relative_raw_path,
                Document.owner_user_id == owner_user_id,
            )
        )

    def get_duplicate_by_content_hash(
        self,
        *,
        owner_user_id: UUID,
        content_hash: str,
        exclude_relative_raw_path: str,
    ) -> Document | None:
        return self._session.scalar(
            select(Document)
            .where(
                Document.owner_user_id == owner_user_id,
                Document.content_hash == content_hash,
                Document.relative_raw_path != exclude_relative_raw_path,
            )
            .limit(1)
        )

    def get_by_parsed_markdown_path(self, parsed_markdown_path: str) -> Document | None:
        return self._session.scalar(
            select(Document).where(Document.parsed_markdown_path == parsed_markdown_path)
        )

    def list_for_owner(self, owner_user_id: UUID) -> list[Document]:
        return list(
            self._session.scalars(
                select(Document)
                .where(Document.owner_user_id == owner_user_id)
                .order_by(Document.relative_raw_path)
            )
        )

    def list_with_chunks_for_owner(self, owner_user_id: UUID) -> list[Document]:
        return list(
            self._session.scalars(
                select(Document)
                .options(selectinload(Document.chunks))
                .where(Document.owner_user_id == owner_user_id)
                .order_by(Document.relative_raw_path)
            )
        )

    def list_existing_raw_paths_for_owner(self, owner_user_id: UUID) -> set[str]:
        return set(
            self._session.scalars(
                select(Document.relative_raw_path).where(
                    Document.owner_user_id == owner_user_id
                )
            )
        )

    def all_documents(self) -> list[Document]:
        return list(
            self._session.scalars(select(Document).order_by(Document.relative_raw_path))
        )

    def upsert_prepared_document(
        self,
        *,
        owner_user_id: UUID,
        relative_raw_path: str,
        original_filename: str,
        parsed_markdown_path: str | None,
        content_hash: str | None,
        byte_size: int | None,
        processed_at: datetime | None,
        status: str = "prepared",
    ) -> Document:
        document = self.get_by_relative_raw_path(relative_raw_path)
        if document is None:
            document = Document(
                owner_user_id=owner_user_id,
                original_filename=original_filename,
                relative_raw_path=relative_raw_path,
            )
            self._session.add(document)

        document.parsed_markdown_path = parsed_markdown_path
        document.content_hash = content_hash
        document.byte_size = byte_size
        document.processed_at = processed_at
        document.status = status
        document.processing_error = None
        self._session.flush()
        return document

    def mark_processing_failed(
        self,
        *,
        owner_user_id: UUID,
        relative_raw_path: str,
        original_filename: str,
        byte_size: int | None,
        processed_at: datetime | None,
        processing_error: str,
    ) -> Document:
        document = self.get_by_relative_raw_path(relative_raw_path)
        if document is None:
            document = Document(
                owner_user_id=owner_user_id,
                original_filename=original_filename,
                relative_raw_path=relative_raw_path,
            )
            self._session.add(document)

        document.parsed_markdown_path = None
        document.content_hash = None
        document.byte_size = byte_size
        document.processed_at = processed_at
        document.indexed_at = None
        document.status = "failed"
        document.processing_error = processing_error
        self._session.flush()
        return document

    def mark_stale_for_missing_raw_paths(
        self,
        *,
        owner_user_id: UUID,
        existing_relative_raw_paths: set[str],
    ) -> int:
        documents = self.list_for_owner(owner_user_id)
        stale_count = 0
        for document in documents:
            if document.relative_raw_path in existing_relative_raw_paths:
                continue
            if document.status == "stale":
                continue
            document.status = "stale"
            document.indexed_at = None
            stale_count += 1
        self._session.flush()
        return stale_count

    def mark_indexed_by_parsed_path(
        self,
        *,
        parsed_markdown_path: str,
        indexed_at: datetime | None = None,
    ) -> Document | None:
        document = self.get_by_parsed_markdown_path(parsed_markdown_path)
        if document is None:
            return None
        document.status = "indexed"
        document.indexed_at = indexed_at or _utc_now()
        document.processing_error = None
        self._session.flush()
        return document

    def upsert_chunks_for_parsed_path(
        self,
        *,
        parsed_markdown_path: str,
        chunks: Sequence[ChunkInput],
    ) -> int:
        document = self.get_by_parsed_markdown_path(parsed_markdown_path)
        if document is None:
            return 0

        self._session.execute(
            delete(DocumentChunk).where(DocumentChunk.document_id == document.id)
        )
        for chunk in chunks:
            self._session.add(
                DocumentChunk(
                    document_id=document.id,
                    chunk_index=chunk.chunk_index,
                    char_start=chunk.char_start,
                    char_end=chunk.char_end,
                    content=chunk.content,
                    qdrant_point_id=chunk.qdrant_point_id,
                )
            )
        document.status = "indexed"
        document.indexed_at = _utc_now()
        document.processing_error = None
        self._session.flush()
        return len(chunks)

    def chunks_for_relative_raw_path(
        self,
        *,
        relative_raw_path: str,
        owner_user_id: UUID,
    ) -> list[DocumentChunk]:
        document = self._session.scalar(
            select(Document)
            .options(selectinload(Document.chunks))
            .where(
                Document.relative_raw_path == relative_raw_path,
                Document.owner_user_id == owner_user_id,
            )
        )
        if document is None:
            return []
        return list(document.chunks)

    def delete_document(self, document: Document) -> None:
        self._session.delete(document)
        self._session.flush()

    def ownership_for_search(
        self,
        *,
        owner_user_id: UUID,
        qdrant_point_ids: Iterable[str],
        content_hashes: Iterable[str],
        sources: Iterable[str],
    ) -> SearchOwnership:
        point_ids = {value for value in qdrant_point_ids if value}
        hashes = {value for value in content_hashes if value}
        source_values = {value for value in sources if value}
        allowed_points: set[str] = set()
        allowed_hashes: set[str] = set()
        allowed_sources: set[str] = set()
        metadata_by_point_id: dict[str, SearchDocumentMetadata] = {}
        metadata_by_hash: dict[str, SearchDocumentMetadata] = {}
        metadata_by_source: dict[str, SearchDocumentMetadata] = {}

        if point_ids:
            rows = self._session.execute(
                select(
                    DocumentChunk.qdrant_point_id,
                    DocumentChunk.chunk_index,
                    DocumentChunk.char_start,
                    DocumentChunk.char_end,
                    Document.id,
                    Document.original_filename,
                    Document.relative_raw_path,
                )
                .join(Document)
                .where(
                    Document.owner_user_id == owner_user_id,
                    DocumentChunk.qdrant_point_id.in_(point_ids),
                )
            )
            for (
                point_id,
                chunk_index,
                char_start,
                char_end,
                document_id,
                original_filename,
                relative_raw_path,
            ) in rows:
                allowed_points.add(point_id)
                metadata_by_point_id[point_id] = SearchDocumentMetadata(
                    document_id=document_id,
                    document_name=original_filename,
                    relative_raw_path=relative_raw_path,
                    chunk_index=chunk_index,
                    char_start=char_start,
                    char_end=char_end,
                )

        if hashes or source_values:
            documents = self._session.scalars(
                select(Document).where(Document.owner_user_id == owner_user_id)
            )
            for document in documents:
                if not _document_matches_search_candidates(
                    document,
                    content_hashes=hashes,
                    sources=source_values,
                ):
                    continue
                if document.content_hash:
                    allowed_hashes.add(document.content_hash)
                    metadata_by_hash[document.content_hash] = (
                        _metadata_from_document(document)
                    )
                if document.parsed_markdown_path:
                    allowed_sources.add(document.parsed_markdown_path)
                    allowed_sources.add(Path(document.parsed_markdown_path).name)
                    metadata = _metadata_from_document(document)
                    metadata_by_source[document.parsed_markdown_path] = metadata
                    metadata_by_source[Path(document.parsed_markdown_path).name] = (
                        metadata
                    )

        return SearchOwnership(
            qdrant_point_ids=allowed_points,
            content_hashes=allowed_hashes,
            sources=allowed_sources,
            metadata_by_point_id=metadata_by_point_id,
            metadata_by_hash=metadata_by_hash,
            metadata_by_source=metadata_by_source,
        )


def normalize_username(username: str) -> str:
    return username.strip().lower()


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _metadata_from_document(document: Document) -> SearchDocumentMetadata:
    return SearchDocumentMetadata(
        document_id=document.id,
        document_name=document.original_filename,
        relative_raw_path=document.relative_raw_path,
    )


def _document_matches_search_candidates(
    document: Document,
    *,
    content_hashes: set[str],
    sources: set[str],
) -> bool:
    if document.content_hash in content_hashes:
        return True
    if not document.parsed_markdown_path:
        return False
    return (
        document.parsed_markdown_path in sources
        or Path(document.parsed_markdown_path).name in sources
    )
