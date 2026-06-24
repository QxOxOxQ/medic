from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterable, Mapping
from typing import Any, Protocol
from uuid import UUID

from backend.execution import BackgroundExecutor
from backend.pipeline_models import (
    CreatedPipelineRun,
    PipelineEventView,
    PipelineRunView,
)


ProgressCallback = Callable[[dict[str, Any]], None]
logger = logging.getLogger(__name__)


class PipelineAlreadyRunningError(RuntimeError):
    pass


class PipelineRunNotFoundError(LookupError):
    pass


class PipelineProcess(Protocol):
    def execute(
        self,
        *,
        progress_callback: ProgressCallback | None = None,
        print_summary: bool = True,
        selected_raw_paths: Iterable[str] | None = None,
        owner_user_id: UUID | None = None,
    ) -> Any: ...


class PipelineRunRepository(Protocol):
    def create(
        self,
        *,
        owner_user_id: UUID,
        document_ids: tuple[UUID, ...],
    ) -> CreatedPipelineRun: ...

    def list_for_owner(
        self,
        *,
        owner_user_id: UUID,
        limit: int,
    ) -> tuple[PipelineRunView, ...]: ...

    def get_for_owner(
        self,
        *,
        owner_user_id: UUID,
        run_id: UUID,
    ) -> PipelineRunView | None: ...

    def events_after(
        self,
        *,
        owner_user_id: UUID,
        run_id: UUID,
        sequence: int,
    ) -> tuple[PipelineEventView, ...] | None: ...

    def has_active_run(self) -> bool: ...

    def start(self, *, run_id: UUID) -> None: ...

    def append_event(
        self,
        *,
        run_id: UUID,
        payload: Mapping[str, Any],
    ) -> None: ...

    def finish(
        self,
        *,
        run_id: UUID,
        status: str,
        summary: str | None,
        error: str | None,
    ) -> None: ...

    def interrupt_active_runs(self) -> int: ...


class StartPipelineRunUseCase:
    def __init__(
        self,
        *,
        repository: PipelineRunRepository,
        process_factory: Callable[[], PipelineProcess],
        executor: BackgroundExecutor,
    ) -> None:
        self._repository = repository
        self._process_factory = process_factory
        self._executor = executor
        self._lock = threading.Lock()

    def execute(
        self,
        *,
        owner_user_id: UUID,
        document_ids: tuple[UUID, ...],
    ) -> PipelineRunView:
        with self._lock:
            if self._repository.has_active_run():
                raise PipelineAlreadyRunningError("A pipeline run is already active")
            created = self._repository.create(
                owner_user_id=owner_user_id,
                document_ids=document_ids,
            )
            self._executor.submit(lambda: self._run(created))
            return created.run

    def _run(self, created: CreatedPipelineRun) -> None:
        run_id = created.run.id
        self._repository.start(run_id=run_id)
        try:
            summary = self._process_factory().execute(
                progress_callback=lambda payload: self._repository.append_event(
                    run_id=run_id,
                    payload=payload,
                ),
                print_summary=False,
                selected_raw_paths=created.selected_raw_paths,
                owner_user_id=created.run.owner_user_id,
            )
            failed = int(getattr(summary, "failed", 0))
            status = "failed" if failed else "succeeded"
            report = _summary_text(summary)
            self._repository.finish(
                run_id=run_id,
                status=status,
                summary=report,
                error=None,
            )
        except Exception as error:
            logger.exception("Pipeline run %s failed", run_id)
            self._repository.append_event(
                run_id=run_id,
                payload={
                    "step": "pipeline",
                    "status": "failed",
                    "message": "Pipeline execution failed",
                    "result": {"error": str(error)},
                },
            )
            self._repository.finish(
                run_id=run_id,
                status="failed",
                summary=None,
                error=str(error),
            )


class ListPipelineRunsUseCase:
    def __init__(self, repository: PipelineRunRepository) -> None:
        self._repository = repository

    def execute(
        self,
        *,
        owner_user_id: UUID,
        limit: int = 20,
    ) -> tuple[PipelineRunView, ...]:
        return self._repository.list_for_owner(
            owner_user_id=owner_user_id,
            limit=max(1, min(limit, 100)),
        )


class GetPipelineRunUseCase:
    def __init__(self, repository: PipelineRunRepository) -> None:
        self._repository = repository

    def execute(
        self,
        *,
        owner_user_id: UUID,
        run_id: UUID,
    ) -> PipelineRunView:
        run = self._repository.get_for_owner(
            owner_user_id=owner_user_id,
            run_id=run_id,
        )
        if run is None:
            raise PipelineRunNotFoundError("Pipeline run not found")
        return run


class StreamPipelineEventsUseCase:
    def __init__(self, repository: PipelineRunRepository) -> None:
        self._repository = repository

    def execute(
        self,
        *,
        owner_user_id: UUID,
        run_id: UUID,
        after_sequence: int,
    ) -> tuple[PipelineEventView, ...]:
        events = self._repository.events_after(
            owner_user_id=owner_user_id,
            run_id=run_id,
            sequence=max(0, after_sequence),
        )
        if events is None:
            raise PipelineRunNotFoundError("Pipeline run not found")
        return events


def _summary_text(summary: Any) -> str:
    as_report_line = getattr(summary, "as_report_line", None)
    if callable(as_report_line):
        return str(as_report_line())
    return str(summary)
