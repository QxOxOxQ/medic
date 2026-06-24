from __future__ import annotations

from pathlib import Path

from dashboard.schemas import DocumentRecord
from rag.config import DocumentPreparationSettings
from rag.document_paths import parsed_markdown_relative_path, safe_relative_markdown_path
from rag.document_preparation import discover_raw_documents


def raw_document_keys(settings: DocumentPreparationSettings) -> set[str]:
    return {
        raw_document.relative_to(settings.raw_documents_dir).as_posix()
        for raw_document in discover_raw_documents(settings.raw_documents_dir)
    }


def build_document_record(
    *,
    relative_raw_path: str,
    raw_keys: set[str],
    indexed_hashes: set[str],
    qdrant_error: str | None,
    settings: DocumentPreparationSettings,
) -> DocumentRecord:
    parsed_markdown_path = parsed_markdown_relative_path(
        Path(relative_raw_path)
    ).as_posix()
    raw_exists = relative_raw_path in raw_keys
    parsed_exists = _parsed_markdown_exists(
        settings=settings,
        parsed_markdown_path=parsed_markdown_path,
    )
    content_hash = None
    indexed = _indexed_status(
        content_hash=content_hash,
        indexed_hashes=indexed_hashes,
        qdrant_error=qdrant_error,
    )
    return DocumentRecord(
        id=None,
        relative_raw_path=relative_raw_path,
        original_filename=Path(relative_raw_path).name,
        display_name=Path(relative_raw_path).name,
        byte_size=None,
        raw_exists=raw_exists,
        parsed_markdown_path=parsed_markdown_path,
        parsed_exists=parsed_exists,
        content_hash=content_hash,
        processed_at=None,
        indexed=indexed,
        status=_document_status(
            raw_exists=raw_exists,
            parsed_exists=parsed_exists,
            indexed=indexed,
        ),
        processing_error=None,
    )


def count_files(directory: Path, pattern: str) -> int:
    if not directory.exists():
        return 0
    return sum(1 for path in directory.rglob(pattern) if path.is_file())


def _indexed_status(
    *,
    content_hash: str | None,
    indexed_hashes: set[str],
    qdrant_error: str | None,
) -> bool | None:
    if qdrant_error is not None or content_hash is None:
        return None
    return content_hash in indexed_hashes


def _document_status(
    *,
    raw_exists: bool,
    parsed_exists: bool,
    indexed: bool | None,
) -> str:
    if not raw_exists:
        return "stale"
    if not parsed_exists:
        return "raw"
    if indexed is True:
        return "indexed"
    if indexed is False:
        return "prepared"
    return "prepared_unverified"


def _parsed_markdown_exists(
    *,
    settings: DocumentPreparationSettings,
    parsed_markdown_path: str,
) -> bool:
    try:
        safe_path = safe_relative_markdown_path(parsed_markdown_path)
    except ValueError:
        return False
    return (settings.parsed_markdown_dir / safe_path).exists()
