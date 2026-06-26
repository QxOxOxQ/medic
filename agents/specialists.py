from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from contextvars import copy_context

from agents.contracts import (
    CompletedConsultation,
    ConsultationReport,
    SpecialistTask,
)
from agents.models import AgentExecutionError, AgentRequest, AgentSource
from agents.ports import ProfessorModelPort
from agents.profiles import AgentRegistry
from agents.trace import AgentTraceRecorder


class SpecialistDispatcher:
    def __init__(
        self,
        *,
        model_gateway: ProfessorModelPort,
        registry: AgentRegistry,
        trace_recorder: AgentTraceRecorder,
        max_parallel_tasks: int = 2,
    ) -> None:
        self._model_gateway = model_gateway
        self._registry = registry
        self._trace_recorder = trace_recorder
        self._max_parallel_tasks = max_parallel_tasks

    def dispatch(
        self,
        request: AgentRequest,
        *,
        tasks: tuple[SpecialistTask, ...],
        sources: tuple[AgentSource, ...],
    ) -> tuple[CompletedConsultation, ...]:
        if len(tasks) <= 1:
            return tuple(
                self._consult(request, task=task, sources=sources) for task in tasks
            )

        worker_count = min(len(tasks), self._max_parallel_tasks)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                self._submit_consultation(
                    executor,
                    request,
                    task=task,
                    sources=sources,
                )
                for task in tasks
            ]
            return tuple(future.result() for future in futures)

    def revise(
        self,
        request: AgentRequest,
        *,
        consultation: CompletedConsultation,
        instructions: str,
        sources: tuple[AgentSource, ...],
    ) -> CompletedConsultation:
        return self._consult(
            request,
            task=consultation.task,
            sources=sources,
            previous_report=consultation.report,
            revision_instructions=instructions,
            revision_count=consultation.revision_count + 1,
        )

    def _submit_consultation(
        self,
        executor: ThreadPoolExecutor,
        request: AgentRequest,
        *,
        task: SpecialistTask,
        sources: tuple[AgentSource, ...],
    ) -> Future[CompletedConsultation]:
        context = copy_context()
        return executor.submit(
            context.run,
            self._consult,
            request,
            task,
            sources,
        )

    def _consult(
        self,
        request: AgentRequest,
        task: SpecialistTask,
        sources: tuple[AgentSource, ...],
        previous_report: ConsultationReport | None = None,
        revision_instructions: str | None = None,
        revision_count: int = 0,
    ) -> CompletedConsultation:
        profile = self._registry.get(task.profile)
        assigned_sources = _assigned_sources(task, sources)
        self._trace_recorder.record(
            event_type="agent",
            title=f"{profile.display_name} consultation started",
            status="running",
            agent_name=profile.name,
            payload={
                "task_id": task.id,
                "objective": task.objective,
                "independent": task.independent,
                "revision_count": revision_count,
            },
        )
        report = self._model_gateway.consultation_report(
            system_prompt=profile.system_prompt_text(),
            user_prompt=profile.consultation_prompt(
                task=task,
                question=request.question,
                conversation_context=_conversation_context(request),
                source_blocks=tuple(
                    source.prompt_block() for source in assigned_sources
                ),
                previous_report=previous_report,
                revision_instructions=revision_instructions,
            ),
            agent_name=profile.name,
            phase="consultation",
        )
        _validate_report(report, task=task)
        self._trace_recorder.record(
            event_type="agent",
            title=f"{profile.display_name} consultation finished",
            status="succeeded",
            agent_name=profile.name,
            payload={
                "task_id": task.id,
                "finding_count": len(report.findings),
                "uncertainty_count": len(report.uncertainties),
                "revision_count": revision_count,
            },
        )
        return CompletedConsultation(
            task=task,
            report=report,
            revision_count=revision_count,
        )


def _assigned_sources(
    task: SpecialistTask,
    sources: tuple[AgentSource, ...],
) -> tuple[AgentSource, ...]:
    by_id = {source.id: source for source in sources}
    try:
        return tuple(by_id[source_id] for source_id in task.source_ids)
    except KeyError as error:
        raise AgentExecutionError(
            f"Specialist task references unavailable source: {error.args[0]}"
        ) from error


def _validate_report(
    report: ConsultationReport,
    *,
    task: SpecialistTask,
) -> None:
    invalid_evidence = set(report.evidence) - set(task.source_ids)
    if invalid_evidence:
        invalid = ", ".join(sorted(invalid_evidence))
        raise AgentExecutionError(
            f"Specialist report references unassigned evidence: {invalid}"
        )


def _conversation_context(request: AgentRequest) -> str:
    return "\n".join(
        f"{message.role}: {message.content}"
        for message in request.conversation_messages
    )
