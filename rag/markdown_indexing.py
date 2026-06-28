from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

from rag.document_preparation import calculate_text_sha256
from rag.progress import ProgressCallback, ProgressEmitter


@dataclass(frozen=True)
class MarkdownIndexingResult:
    indexed_files: int
    indexed_chunks: int


class MarkdownIndexer:
    def __init__(
        self,
        *,
        parsed_markdown_dir: Path,
        indexer: Callable[..., int],
        accepts_progress_callback: bool,
        owner_user_id: UUID | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._parsed_markdown_dir = parsed_markdown_dir
        self._indexer = indexer
        self._accepts_progress_callback = accepts_progress_callback
        self._owner_user_id = owner_user_id
        self._logger = logger or logging.getLogger(__name__)

    def index_all(
        self,
        *,
        progress: ProgressEmitter,
        progress_callback: ProgressCallback | None,
        selected_sources: Iterable[str] | None = None,
    ) -> MarkdownIndexingResult:
        markdown_file_paths = sorted(self._parsed_markdown_dir.rglob("*.md"))
        selected_source_set = (
            set(selected_sources) if selected_sources is not None else None
        )
        if selected_source_set is not None:
            markdown_file_paths = [
                path
                for path in markdown_file_paths
                if path.relative_to(self._parsed_markdown_dir).as_posix()
                in selected_source_set
            ]
        self._emit_start(markdown_file_paths, progress)
        indexed_chunks = 0
        for index, markdown_file_path in enumerate(markdown_file_paths, start=1):
            indexed_chunks += self._index_one(
                markdown_file_path=markdown_file_path,
                index=index,
                total=len(markdown_file_paths),
                progress=progress,
                progress_callback=progress_callback,
            )
        return self._finish(
            indexed_files=len(markdown_file_paths),
            indexed_chunks=indexed_chunks,
            progress=progress,
        )

    def _emit_start(
        self,
        markdown_file_paths: list[Path],
        progress: ProgressEmitter,
    ) -> None:
        self._logger.info(
            "Indexing parsed markdown files: directory=%s files=%d",
            self._parsed_markdown_dir,
            len(markdown_file_paths),
        )
        progress.emit(
            step="index",
            status="running",
            message="Indexing parsed markdown files",
            counters={"files": len(markdown_file_paths)},
        )

    def _index_one(
        self,
        *,
        markdown_file_path: Path,
        index: int,
        total: int,
        progress: ProgressEmitter,
        progress_callback: ProgressCallback | None,
    ) -> int:
        source = markdown_file_path.relative_to(self._parsed_markdown_dir).as_posix()
        self._emit_file_start(source, index=index, total=total, progress=progress)
        try:
            saved_chunks = self._index_file(
                markdown_file_path=markdown_file_path,
                source=source,
                progress_callback=progress_callback,
            )
        except Exception as error:
            progress.emit(
                step="index",
                status="failed",
                message=f"Failed to index {source}",
                counters={"index": index, "total": total},
                result={"source": source, "error": str(error)},
            )
            raise
        self._emit_file_success(
            source,
            index=index,
            total=total,
            saved_chunks=saved_chunks,
            progress=progress,
        )
        return saved_chunks

    def _emit_file_start(
        self,
        source: str,
        *,
        index: int,
        total: int,
        progress: ProgressEmitter,
    ) -> None:
        self._logger.info(
            "Indexing parsed markdown file %d/%d: %s",
            index,
            total,
            source,
        )
        progress.emit(
            step="index",
            status="running",
            message=f"Indexing {source}",
            counters={"index": index, "total": total},
        )

    def _index_file(
        self,
        *,
        markdown_file_path: Path,
        source: str,
        progress_callback: ProgressCallback | None,
    ) -> int:
        document_content = markdown_file_path.read_text(encoding="utf-8")
        kwargs: dict[str, Any] = {
            "text": document_content,
            "source_metadata": source_metadata(
                markdown_file_path=markdown_file_path,
                source=source,
                document_content=document_content,
                owner_user_id=self._owner_user_id,
            ),
        }
        if self._accepts_progress_callback and progress_callback is not None:
            kwargs["progress_callback"] = progress_callback
        return self._indexer(**kwargs)

    def _emit_file_success(
        self,
        source: str,
        *,
        index: int,
        total: int,
        saved_chunks: int,
        progress: ProgressEmitter,
    ) -> None:
        self._logger.info(
            "Indexed parsed markdown file %d/%d: %s chunks=%d",
            index,
            total,
            source,
            saved_chunks,
        )
        progress.emit(
            step="index",
            status="succeeded",
            message=f"Indexed {source}",
            counters={"index": index, "total": total},
            result={"source": source, "chunks": saved_chunks},
        )

    def _finish(
        self,
        *,
        indexed_files: int,
        indexed_chunks: int,
        progress: ProgressEmitter,
    ) -> MarkdownIndexingResult:
        self._logger.info(
            "Finished ingestion: files=%d chunks=%d",
            indexed_files,
            indexed_chunks,
        )
        progress.emit(
            step="index",
            status="succeeded",
            message="Finished indexing parsed markdown files",
            counters={"files": indexed_files, "chunks": indexed_chunks},
        )
        return MarkdownIndexingResult(
            indexed_files=indexed_files,
            indexed_chunks=indexed_chunks,
        )


def source_metadata(
    *,
    markdown_file_path: Path,
    source: str,
    document_content: str,
    owner_user_id: UUID | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "file_name": markdown_file_path.name,
        "source": source,
        "content_hash": calculate_text_sha256(document_content),
    }
    if owner_user_id is not None:
        metadata["owner_user_id"] = str(owner_user_id)
    return metadata
