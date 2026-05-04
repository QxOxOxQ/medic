from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class QdrantCleanupResult:
    attempted: bool
    deleted: bool
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DocumentRecord:
    relative_raw_path: str
    original_filename: str
    display_name: str
    raw_exists: bool
    parsed_markdown_path: str | None
    parsed_exists: bool
    content_hash: str | None
    processed_at: str | None
    indexed: bool | None
    status: str
    processing_error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DashboardStatus:
    raw_pdf_count: int
    parsed_markdown_count: int
    document_count: int
    last_processed_at: str | None
    qdrant: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SearchResult:
    score: float | None
    source: str | None
    content_hash: str | None
    excerpt: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class IndexPreview:
    available: bool
    collection_name: str | None
    collection_exists: bool
    preview_limit: int
    points: list[dict[str, Any]] = field(default_factory=list)
    shown_points: int = 0
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
