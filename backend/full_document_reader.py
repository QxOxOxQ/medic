from __future__ import annotations

from uuid import UUID

from sqlalchemy.orm import Session, sessionmaker

from agents.models import AgentSource
from rag.config import DocumentPreparationSettings, get_document_preparation_settings
from rag.database.models import Document
from rag.database.repositories import DocumentRepository
from rag.document_paths import safe_relative_markdown_path


_DEFAULT_MAX_CHARS = 8000
_TRUNCATION_MARKER = "\n\n[document truncated]"


class ParsedMarkdownDocumentReader:
    """Reads a retrieved record's whole parsed-markdown document, owner-scoped.

    A SQL lookup authorizes the record and resolves its parsed-markdown path; the
    full text is then read from disk. Returns ``None`` when the record is missing,
    not owned by the user, or has no readable parsed document.
    """

    def __init__(
        self,
        *,
        database_session_factory: sessionmaker[Session],
        owner_user_id: UUID,
        settings: DocumentPreparationSettings | None = None,
        max_chars: int = _DEFAULT_MAX_CHARS,
    ) -> None:
        self._database_session_factory = database_session_factory
        self._owner_user_id = owner_user_id
        self._settings = settings or get_document_preparation_settings()
        self._max_chars = max(1, max_chars)

    def read(self, source: AgentSource) -> str | None:
        parsed_markdown_path = self._parsed_markdown_path(source)
        if not parsed_markdown_path:
            return None
        text = self._read_file(parsed_markdown_path)
        if text is None:
            return None
        return self._capped(text)

    def _parsed_markdown_path(self, source: AgentSource) -> str | None:
        with self._database_session_factory() as session:
            repository = DocumentRepository(session)
            document = self._lookup(repository, source)
            if document is None:
                return None
            return document.parsed_markdown_path

    def _lookup(
        self,
        repository: DocumentRepository,
        source: AgentSource,
    ) -> Document | None:
        if source.document_id is not None:
            return repository.get_by_id_for_owner(
                document_id=source.document_id,
                owner_user_id=self._owner_user_id,
            )
        if source.relative_raw_path:
            return repository.get_by_relative_raw_path_for_owner(
                relative_raw_path=source.relative_raw_path,
                owner_user_id=self._owner_user_id,
            )
        return None

    def _read_file(self, parsed_markdown_path: str) -> str | None:
        path = self._settings.parsed_markdown_dir / safe_relative_markdown_path(
            parsed_markdown_path
        )
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            return None

    def _capped(self, text: str) -> str:
        if len(text) <= self._max_chars:
            return text
        return text[: self._max_chars] + _TRUNCATION_MARKER
