from __future__ import annotations

import logging
from collections.abc import Mapping

from langchain_core.language_models.chat_models import BaseChatModel

from agents.contracts import (
    CompletedConsultation,
    ResearchPlan,
    ReviewDecision,
    ReviewOutcome,
    SpecialistTask,
)
from agents.model_gateway import AgentModelGateway
from agents.model_router import RoutedModel
from agents.models import (
    AgentAnswer,
    AgentExecutionError,
    AgentRequest,
    AgentSource,
    UnknownAgentError,
)
from agents.observability import AgentObservability, NullAgentObservability
from agents.ports import FullDocumentReader, MedicalDocumentSearchPort
from agents.professor import (
    MedicalContextCollector,
    ProfessorResearchPlanner,
    ProfessorReviewer,
    ProfessorSourceExpander,
    ProfessorSynthesizer,
    ProfessorTaskPlanner,
    insufficient_reason,
)
from agents.profiles import AgentRegistry
from agents.specialists import SpecialistDispatcher
from agents.trace import AgentTraceRecorder


AGENT_PROMPT_VERSION = "agents-v2"
_AGENT_FAILURE_MESSAGE = "Agent execution failed. See server logs for details."

logger = logging.getLogger("medic.agents.graph")


class AgentGraph:
    def __init__(
        self,
        *,
        chat_model: BaseChatModel,
        search_port: MedicalDocumentSearchPort,
        max_retrieval_queries: int = 6,
        max_consultations: int = 4,
        max_review_rounds: int = 3,
        max_full_documents: int = 3,
        registry: AgentRegistry | None = None,
        trace_recorder: AgentTraceRecorder | None = None,
        observability: AgentObservability | None = None,
        full_document_reader: FullDocumentReader | None = None,
        model_overrides: Mapping[str, RoutedModel] | None = None,
        default_label: str | None = None,
    ) -> None:
        _validate_limits(
            max_retrieval_queries=max_retrieval_queries,
            max_consultations=max_consultations,
            max_review_rounds=max_review_rounds,
        )
        self._registry = registry or AgentRegistry()
        self._max_consultations = max_consultations
        self._max_review_rounds = max_review_rounds
        self._trace_recorder = trace_recorder or AgentTraceRecorder()
        self._observability = observability or NullAgentObservability()

        model_gateway = AgentModelGateway(
            chat_model=chat_model,
            observability=self._observability,
            trace_recorder=self._trace_recorder,
            model_overrides=model_overrides,
            default_label=default_label,
        )
        self._research_planner = ProfessorResearchPlanner(
            model_gateway=model_gateway,
            professor_prompt=self._registry.professor_prompt,
            max_initial_queries=min(4, max_retrieval_queries),
        )
        self._context_collector = MedicalContextCollector(
            search_port=search_port,
            trace_recorder=self._trace_recorder,
            max_queries=max_retrieval_queries,
        )
        self._source_expander = (
            ProfessorSourceExpander(
                model_gateway=model_gateway,
                professor_prompt=self._registry.professor_prompt,
                full_document_reader=full_document_reader,
                context_collector=self._context_collector,
                trace_recorder=self._trace_recorder,
                max_documents=max_full_documents,
            )
            if full_document_reader is not None
            else None
        )
        self._task_planner = ProfessorTaskPlanner(
            model_gateway=model_gateway,
            professor_prompt=self._registry.professor_prompt,
            registry=self._registry,
        )
        self._dispatcher = SpecialistDispatcher(
            model_gateway=model_gateway,
            registry=self._registry,
            trace_recorder=self._trace_recorder,
        )
        self._reviewer = ProfessorReviewer(
            model_gateway=model_gateway,
            professor_prompt=self._registry.professor_prompt,
            registry=self._registry,
        )
        self._synthesizer = ProfessorSynthesizer(
            model_gateway=model_gateway,
            professor_prompt=self._registry.professor_prompt,
        )

    def answer(self, request: AgentRequest) -> AgentAnswer:
        with self._observability.trace(request):
            return self._answer(request)

    def _answer(self, request: AgentRequest) -> AgentAnswer:
        try:
            result = self._coordinate(request)
        except (AgentExecutionError, UnknownAgentError):
            raise
        except Exception as error:
            logger.error(
                "Agent execution failed with %s: %s",
                type(error).__name__,
                error,
                exc_info=(type(error), error, error.__traceback__),
            )
            self._trace_recorder.record(
                event_type="error",
                title="Agent execution failed",
                status="failed",
                payload={"error": _AGENT_FAILURE_MESSAGE},
            )
            raise AgentExecutionError(_AGENT_FAILURE_MESSAGE) from error

        self._observability.complete(result)
        return result

    def _coordinate(self, request: AgentRequest) -> AgentAnswer:
        if request.requested_agent is not None:
            self._registry.canonical_name(request.requested_agent)
        research_plan = self._research_planner.plan(request)
        sources = self._initial_sources(research_plan)
        sources = self._expand_full_documents(request, sources)
        if self._requires_no_consultation(research_plan, sources):
            return self._direct_professor_answer(
                request,
                research_plan=research_plan,
                sources=sources,
            )

        try:
            tasks = self._task_planner.plan(
                request,
                research_plan=research_plan,
                sources=sources,
            )
        except AgentExecutionError as error:
            logger.warning(
                "Specialist planning failed (%s); answering directly from records",
                error,
            )
            self._record_planning_fallback(error)
            return self._direct_professor_answer(
                request,
                research_plan=research_plan,
                sources=sources,
            )
        self._record_coordination(research_plan, tasks=tasks)
        initial_tasks = tasks[: self._max_consultations]
        consultations = self._dispatcher.dispatch(
            request,
            tasks=initial_tasks,
            sources=sources,
        )
        consultations, review_outcome = self._review_loop(
            request,
            research_plan=research_plan,
            consultations=consultations,
        )
        final_sources = self._context_collector.sources()
        answer, insufficient_context = self._synthesizer.synthesize(
            request,
            research_plan=research_plan,
            consultations=consultations,
            sources=final_sources,
            review_outcome=review_outcome,
        )
        self._record_synthesis(
            consultations=consultations,
            sources=final_sources,
            insufficient_context=insufficient_context,
            reason=insufficient_reason(
                research_plan,
                sources=final_sources,
                review_outcome=review_outcome,
            ),
        )
        return AgentAnswer(
            answer=answer,
            agents=_consulted_profiles(consultations),
            sources=final_sources,
            insufficient_context=insufficient_context,
            trace_events=self._trace_recorder.events(),
        )

    def _initial_sources(
        self,
        research_plan: ResearchPlan,
    ) -> tuple[AgentSource, ...]:
        if research_plan.mode != "record_grounded":
            return ()
        return self._context_collector.collect(research_plan.queries)

    def _expand_full_documents(
        self,
        request: AgentRequest,
        sources: tuple[AgentSource, ...],
    ) -> tuple[AgentSource, ...]:
        if self._source_expander is None or not sources:
            return sources
        self._source_expander.expand(request)
        return self._context_collector.sources()

    @staticmethod
    def _requires_no_consultation(
        research_plan: ResearchPlan,
        sources: tuple[AgentSource, ...],
    ) -> bool:
        if research_plan.mode == "clarify":
            return True
        return research_plan.mode == "record_grounded" and not sources

    def _direct_professor_answer(
        self,
        request: AgentRequest,
        *,
        research_plan: ResearchPlan,
        sources: tuple[AgentSource, ...],
    ) -> AgentAnswer:
        self._record_coordination(research_plan, tasks=())
        review_outcome = ReviewOutcome(
            decision=None,
            rounds_completed=0,
            consultation_budget_exhausted=False,
            review_budget_exhausted=False,
        )
        answer, insufficient_context = self._synthesizer.synthesize(
            request,
            research_plan=research_plan,
            consultations=(),
            sources=sources,
            review_outcome=review_outcome,
        )
        self._record_synthesis(
            consultations=(),
            sources=sources,
            insufficient_context=insufficient_context,
            reason=insufficient_reason(
                research_plan,
                sources=sources,
                review_outcome=review_outcome,
            ),
        )
        return AgentAnswer(
            answer=answer,
            agents=(),
            sources=sources,
            insufficient_context=insufficient_context,
            trace_events=self._trace_recorder.events(),
        )

    def _review_loop(
        self,
        request: AgentRequest,
        *,
        research_plan: ResearchPlan,
        consultations: tuple[CompletedConsultation, ...],
    ) -> tuple[tuple[CompletedConsultation, ...], ReviewOutcome]:
        completed = consultations
        consultation_calls = len(completed)
        last_decision: ReviewDecision | None = None
        budget_exhausted = False
        review_budget_exhausted = False
        rounds_completed = 0

        for round_index in range(1, self._max_review_rounds + 1):
            rounds_completed = round_index
            try:
                last_decision = self._reviewer.review(
                    request,
                    research_plan=research_plan,
                    consultations=completed,
                    sources=self._context_collector.sources(),
                )
            except AgentExecutionError as error:
                self._record_review_failure(round_index=round_index, error=error)
                last_decision = None
                review_budget_exhausted = True
                break
            self._record_review(last_decision, round_index=round_index)
            if last_decision.status == "approved":
                return completed, ReviewOutcome(
                    decision=last_decision,
                    rounds_completed=round_index,
                    consultation_budget_exhausted=budget_exhausted,
                    review_budget_exhausted=False,
                )
            if round_index >= self._max_review_rounds:
                review_budget_exhausted = True
                break

            remaining_budget = self._max_consultations - consultation_calls
            if remaining_budget <= 0:
                budget_exhausted = True
                break
            if last_decision.status == "research":
                revised, calls = self._apply_research(
                    request,
                    consultations=completed,
                    decision=last_decision,
                    remaining_budget=remaining_budget,
                )
                if calls == 0:
                    break
                if calls < len(last_decision.revisions):
                    budget_exhausted = True
                completed = revised
                consultation_calls += calls
                continue
            if last_decision.status == "revise":
                revised, calls = self._apply_revisions(
                    request,
                    consultations=completed,
                    decision=last_decision,
                    remaining_budget=remaining_budget,
                )
                if calls == 0:
                    break
                if calls < len(last_decision.revisions):
                    budget_exhausted = True
                completed = revised
                consultation_calls += calls
                continue
            if last_decision.status == "consult":
                next_tasks = last_decision.next_tasks[:remaining_budget]
                if len(next_tasks) < len(last_decision.next_tasks):
                    budget_exhausted = True
                additional = self._dispatcher.dispatch(
                    request,
                    tasks=next_tasks,
                    sources=self._context_collector.sources(),
                )
                completed = (*completed, *additional)
                consultation_calls += len(additional)

        return completed, ReviewOutcome(
            decision=last_decision,
            rounds_completed=rounds_completed,
            consultation_budget_exhausted=budget_exhausted,
            review_budget_exhausted=review_budget_exhausted,
        )

    def _apply_research(
        self,
        request: AgentRequest,
        *,
        consultations: tuple[CompletedConsultation, ...],
        decision: ReviewDecision,
        remaining_budget: int,
    ) -> tuple[tuple[CompletedConsultation, ...], int]:
        previous_source_ids = {
            source.id for source in self._context_collector.sources()
        }
        sources = self._context_collector.collect(decision.additional_queries)
        new_source_ids = tuple(
            source.id for source in sources if source.id not in previous_source_ids
        )
        if not new_source_ids:
            return consultations, 0

        expanded = list(consultations)
        calls = 0
        by_task_id = {
            consultation.task.id: (index, consultation)
            for index, consultation in enumerate(consultations)
        }
        for revision in decision.revisions:
            if calls >= remaining_budget:
                break
            index, current = by_task_id[revision.task_id]
            if current.revision_count >= 1:
                continue
            expanded_task = SpecialistTask(
                id=current.task.id,
                profile=current.task.profile,
                objective=current.task.objective,
                source_ids=_merge_source_ids(
                    current.task.source_ids,
                    new_source_ids,
                ),
                response_language=current.task.response_language,
                independent=current.task.independent,
            )
            expanded[index] = self._dispatcher.revise(
                request,
                consultation=CompletedConsultation(
                    task=expanded_task,
                    report=current.report,
                    revision_count=current.revision_count,
                ),
                instructions=revision.instructions,
                sources=sources,
            )
            calls += 1
        return tuple(expanded), calls

    def _apply_revisions(
        self,
        request: AgentRequest,
        *,
        consultations: tuple[CompletedConsultation, ...],
        decision: ReviewDecision,
        remaining_budget: int,
    ) -> tuple[tuple[CompletedConsultation, ...], int]:
        revised = list(consultations)
        calls = 0
        by_task_id = {
            consultation.task.id: (index, consultation)
            for index, consultation in enumerate(consultations)
        }
        for revision in decision.revisions:
            if calls >= remaining_budget:
                break
            index, current = by_task_id[revision.task_id]
            if current.revision_count >= 1:
                continue
            revised[index] = self._dispatcher.revise(
                request,
                consultation=current,
                instructions=revision.instructions,
                sources=self._context_collector.sources(),
            )
            calls += 1
        return tuple(revised), calls

    def _record_coordination(
        self,
        research_plan: ResearchPlan,
        *,
        tasks: tuple[SpecialistTask, ...],
    ) -> None:
        self._trace_recorder.record(
            event_type="coordinator",
            title="Professor planned specialist consultations",
            status="succeeded",
            agent_name="professor",
            payload={
                "mode": research_plan.mode,
                "response_language": research_plan.response_language,
                "retrieval_queries": list(research_plan.queries),
                "selected_agents": [task.profile for task in tasks],
                "tasks": [
                    {
                        "id": task.id,
                        "profile": task.profile,
                        "objective": task.objective,
                        "source_ids": list(task.source_ids),
                        "independent": task.independent,
                    }
                    for task in tasks
                ],
            },
        )

    def _record_planning_fallback(self, error: Exception) -> None:
        self._trace_recorder.record(
            event_type="coordinator",
            title="Specialist planning unavailable; answered directly from records",
            status="degraded",
            agent_name="professor",
            payload={"reason": str(error)},
        )

    def _record_review(
        self,
        decision: ReviewDecision,
        *,
        round_index: int,
    ) -> None:
        self._trace_recorder.record(
            event_type="review",
            title="Professor reviewed specialist consultations",
            status=decision.status,
            agent_name="professor",
            payload={
                "round": round_index,
                "issues": list(decision.issues),
                "revision_task_ids": [
                    revision.task_id for revision in decision.revisions
                ],
                "next_task_ids": [task.id for task in decision.next_tasks],
                "additional_queries": list(decision.additional_queries),
            },
        )

    def _record_review_failure(
        self,
        *,
        round_index: int,
        error: Exception,
    ) -> None:
        self._trace_recorder.record(
            event_type="review",
            title="Professor review unavailable; answering from gathered evidence",
            status="failed",
            agent_name="professor",
            payload={"round": round_index, "error": str(error)},
        )

    def _record_synthesis(
        self,
        *,
        consultations: tuple[CompletedConsultation, ...],
        sources: tuple[AgentSource, ...],
        insufficient_context: bool,
        reason: str,
    ) -> None:
        self._trace_recorder.record(
            event_type="synthesis",
            title="Professor synthesized final answer",
            status="insufficient_context" if insufficient_context else "succeeded",
            agent_name="professor",
            payload={
                "consultation_count": len(consultations),
                "source_count": len(sources),
                "reason": reason,
            },
        )


def _consulted_profiles(
    consultations: tuple[CompletedConsultation, ...],
) -> tuple[str, ...]:
    names: list[str] = []
    for consultation in consultations:
        if consultation.task.profile not in names:
            names.append(consultation.task.profile)
    return tuple(names)


def _merge_source_ids(
    existing: tuple[str, ...],
    additional: tuple[str, ...],
) -> tuple[str, ...]:
    merged = list(existing)
    for source_id in additional:
        if source_id not in merged:
            merged.append(source_id)
    return tuple(merged)


def _validate_limits(
    *,
    max_retrieval_queries: int,
    max_consultations: int,
    max_review_rounds: int,
) -> None:
    if max_retrieval_queries < 1:
        raise ValueError("max_retrieval_queries must be at least 1")
    if max_consultations < 1:
        raise ValueError("max_consultations must be at least 1")
    if max_review_rounds < 1:
        raise ValueError("max_review_rounds must be at least 1")
