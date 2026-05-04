from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DocumentDeletionResult:
    relative_raw_path: str
    parsed_markdown_path: str | None
    content_hash: str | None
    raw_deleted: bool
    parsed_deleted: bool


def relative_path_key(relative_path: Path) -> str:
    return relative_path.as_posix()


def parsed_markdown_relative_path(relative_raw_path: Path) -> Path:
    return relative_raw_path.with_suffix(".md")


def safe_relative_pdf_path(relative_raw_path: str) -> Path:
    return _safe_relative_path(
        relative_raw_path,
        suffix=".pdf",
        description="PDF path inside the raw documents directory",
    )


def safe_relative_markdown_path(relative_markdown_path: str) -> Path:
    return _safe_relative_path(
        relative_markdown_path,
        suffix=".md",
        description="markdown path inside the parsed documents directory",
    )


def delete_file_if_exists(file_path: Path) -> bool:
    if file_path.exists():
        file_path.unlink()
        return True
    return False


def _safe_relative_path(value: str, *, suffix: str, description: str) -> Path:
    relative_path = Path(value)
    if (
        not value
        or relative_path.is_absolute()
        or relative_path == Path(".")
        or ".." in relative_path.parts
        or relative_path.suffix.lower() != suffix
    ):
        raise ValueError(f"Unsafe {description}: {value}")
    return relative_path
