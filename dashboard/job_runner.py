from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Protocol
from uuid import UUID

from dashboard.job_models import PipelineJob
from rag.config import DocumentPreparationSettings
from rag.document_preparation import PreparationSummary
from rag.full_process import FullProcess
from rag.progress import ProgressCallback


class PipelineProcess(Protocol):
    def execute(
        self,
        *,
        progress_callback: ProgressCallback | None = None,
        print_summary: bool = True,
        selected_raw_paths: Iterable[str] | None = None,
        owner_user_id: UUID | None = None,
    ) -> PreparationSummary:
        pass


ProcessFactory = Callable[[DocumentPreparationSettings], PipelineProcess]


class IngestJobRunner:
    def __init__(
        self,
        *,
        process_factory: ProcessFactory | None = None,
    ) -> None:
        self._process_factory = process_factory or (
            lambda settings: FullProcess(settings=settings)
        )

    def run(
        self,
        job: PipelineJob,
        settings: DocumentPreparationSettings,
        *,
        selected_raw_paths: Iterable[str] | None = None,
        owner_user_id: UUID | None = None,
    ) -> None:
        job.start()
        try:
            process = self._process_factory(settings)
            summary = process.execute(
                progress_callback=job.emit_progress,
                print_summary=False,
                selected_raw_paths=selected_raw_paths,
                owner_user_id=owner_user_id,
            )
            status = "failed" if summary.failed else "succeeded"
            job.emit(
                step="job",
                status=status,
                message="Ingestion job finished",
                result={"summary": summary.as_report_line()},
            )
            job.finish(status)
        except Exception as error:
            job.emit(
                step="job",
                status="failed",
                message="Ingestion job failed",
                result={"error": str(error)},
            )
            job.finish("failed", error=str(error))
