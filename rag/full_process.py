import logging
from collections.abc import Callable, Iterable
from uuid import UUID

from sqlalchemy.orm import Session, sessionmaker

from rag.config import DocumentPreparationSettings, get_document_preparation_settings
from rag.database.repositories import DocumentRepository
from rag.document_paths import (
    parsed_markdown_relative_path,
    relative_path_key,
    safe_relative_pdf_path,
)
from rag.document_preparation import (
    PreparationSummary,
    prepare_documents,
)
from rag.indexer import index_text
from rag.markdown_indexing import MarkdownIndexer
from rag.progress import ProgressCallback, ProgressEmitter

logger = logging.getLogger(__name__)


class FullProcess:
    def __init__(
        self,
        *,
        settings: DocumentPreparationSettings | None = None,
        database_session_factory: sessionmaker[Session] | None = None,
        indexer: Callable[..., int] | None = None,
    ) -> None:
        self._settings = settings
        self._database_session_factory = database_session_factory
        self._indexer = indexer

    def execute(
        self,
        *,
        progress_callback: ProgressCallback | None = None,
        print_summary: bool = True,
        selected_raw_paths: Iterable[str] | None = None,
        owner_user_id: UUID | None = None,
    ) -> PreparationSummary:
        selected_raw_path_list = (
            list(selected_raw_paths) if selected_raw_paths is not None else None
        )
        progress = ProgressEmitter(progress_callback)
        logger.info("Starting ingestion")
        progress.emit(
            step="pipeline",
            status="running",
            message="Starting ingestion",
        )
        settings = self._settings or get_document_preparation_settings()
        result = self._prepare_documents(
            settings=settings,
            progress_callback=progress_callback,
            print_summary=print_summary,
            selected_raw_paths=selected_raw_path_list,
            owner_user_id=owner_user_id,
        )
        indexing_result = self._markdown_indexer(
            settings, owner_user_id=owner_user_id
        ).index_all(
            progress=progress,
            progress_callback=progress_callback,
            selected_sources=self._selected_markdown_sources(
                selected_raw_path_list,
                owner_user_id=owner_user_id,
            ),
        )
        progress.emit(
            step="pipeline",
            status="failed" if result.failed else "succeeded",
            message="Finished ingestion",
            result={
                "summary": result.as_report_line(),
                "indexed_files": indexing_result.indexed_files,
                "indexed_chunks": indexing_result.indexed_chunks,
            },
        )
        return result

    def _prepare_documents(
        self,
        *,
        settings: DocumentPreparationSettings,
        progress_callback: ProgressCallback | None,
        print_summary: bool,
        selected_raw_paths: list[str] | None,
        owner_user_id: UUID | None,
    ) -> PreparationSummary:
        logger.info("Preparing documents")
        result = prepare_documents(
            settings=settings,
            progress_callback=progress_callback,
            selected_raw_paths=selected_raw_paths,
            database_session_factory=self._database_session_factory,
            owner_user_id=owner_user_id,
        )
        logger.info("Document preparation finished: %s", result.as_report_line())
        if print_summary:
            print(result.as_report_line())
        return result

    def _markdown_indexer(
        self,
        settings: DocumentPreparationSettings,
        *,
        owner_user_id: UUID | None,
    ) -> MarkdownIndexer:
        custom_indexer = self._indexer is not None
        return MarkdownIndexer(
            parsed_markdown_dir=settings.parsed_markdown_dir,
            indexer=self._indexer or index_text,
            accepts_progress_callback=not custom_indexer,
            owner_user_id=owner_user_id,
            logger=logger,
        )

    def _selected_markdown_sources(
        self,
        selected_raw_paths: Iterable[str] | None,
        *,
        owner_user_id: UUID | None,
    ) -> set[str] | None:
        if selected_raw_paths is not None:
            return _selected_markdown_sources(selected_raw_paths)
        if owner_user_id is None or self._database_session_factory is None:
            return None

        with self._database_session_factory() as session:
            documents = DocumentRepository(session).list_for_owner(owner_user_id)
            return {
                document.parsed_markdown_path
                for document in documents
                if document.parsed_markdown_path
                and document.status in {"prepared", "indexed"}
            }


def _selected_markdown_sources(
    selected_raw_paths: Iterable[str] | None,
) -> set[str] | None:
    if selected_raw_paths is None:
        return None
    return {
        relative_path_key(parsed_markdown_relative_path(safe_relative_pdf_path(path)))
        for path in selected_raw_paths
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    FullProcess().execute()
