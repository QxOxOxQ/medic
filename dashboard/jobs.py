from __future__ import annotations

import threading
import uuid
from collections.abc import Iterable
from collections import OrderedDict
from typing import Any
from uuid import UUID

from dashboard.job_models import JobEvent, PipelineJob
from dashboard.job_runner import IngestJobRunner, ProcessFactory
from rag.config import DocumentPreparationSettings


MAX_JOB_HISTORY = 20
__all__ = [
    "JobAlreadyRunningError",
    "JobEvent",
    "JobStore",
    "PipelineJob",
    "ProcessFactory",
]


class JobAlreadyRunningError(RuntimeError):
    pass


class JobStore:
    def __init__(
        self,
        *,
        process_factory: ProcessFactory | None = None,
    ) -> None:
        self._runner = IngestJobRunner(process_factory=process_factory)
        self._jobs: OrderedDict[str, PipelineJob] = OrderedDict()
        self._active_job_id: str | None = None
        self._lock = threading.Lock()

    def start_ingest(
        self,
        settings: DocumentPreparationSettings,
        *,
        selected_raw_paths: Iterable[str] | None = None,
        owner_user_id: UUID | None = None,
    ) -> PipelineJob:
        selected_raw_path_list = (
            list(selected_raw_paths) if selected_raw_paths is not None else None
        )
        with self._lock:
            if self._active_job_id is not None:
                active_job = self._jobs.get(self._active_job_id)
                if active_job is not None and not active_job.is_terminal:
                    raise JobAlreadyRunningError("Ingestion is already running")

            job = PipelineJob(str(uuid.uuid4()))
            self._jobs[job.id] = job
            self._active_job_id = job.id
            self._prune_history()

        thread = threading.Thread(
            target=self._run_ingest,
            args=(job, settings, selected_raw_path_list, owner_user_id),
            daemon=True,
        )
        thread.start()
        return job

    def get_job(self, job_id: str) -> PipelineJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def recent_jobs(self) -> list[dict[str, Any]]:
        with self._lock:
            return [job.snapshot() for job in reversed(self._jobs.values())]

    def _run_ingest(
        self,
        job: PipelineJob,
        settings: DocumentPreparationSettings,
        selected_raw_paths: list[str] | None,
        owner_user_id: UUID | None,
    ) -> None:
        try:
            self._runner.run(
                job,
                settings,
                selected_raw_paths=selected_raw_paths,
                owner_user_id=owner_user_id,
            )
        finally:
            with self._lock:
                if self._active_job_id == job.id:
                    self._active_job_id = None

    def _prune_history(self) -> None:
        while len(self._jobs) > MAX_JOB_HISTORY:
            self._jobs.popitem(last=False)
