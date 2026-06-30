from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from agents.contracts import (
    ConsultationReport,
    ResearchPlan,
    ReviewDecision,
    RevisionRequest,
    SpecialistTask,
)


class StrictPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ResearchPlanPayload(StrictPayload):
    mode: Literal["record_grounded", "general_information", "clarify"]
    response_language: str = Field(min_length=1)
    queries: list[str] = Field(
        default_factory=list,
        description=(
            "Retrieval queries, each a terse phrase of concrete clinical terms "
            "for one concept (not a generic meta-phrase). The record language "
            "is unknown before searching and often differs from the question, "
            "so cover the most important concepts in both the question's "
            "language and English; decompose broad or whole-health questions "
            "into several specific facet queries."
        ),
    )

    def to_domain(self, *, max_queries: int) -> ResearchPlan:
        queries = _normalized_unique(self.queries)[:max_queries]
        return ResearchPlan(
            mode=self.mode,
            response_language=self.response_language.strip(),
            queries=queries,
        )


class SpecialistTaskPayload(StrictPayload):
    id: str = Field(min_length=1)
    profile: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    source_ids: list[str] = Field(default_factory=list)
    independent: bool = False

    def to_domain(self, *, response_language: str) -> SpecialistTask:
        return SpecialistTask(
            id=self.id.strip(),
            profile=self.profile.strip(),
            objective=self.objective.strip(),
            source_ids=_normalized_unique(self.source_ids),
            response_language=response_language,
            independent=self.independent,
        )


class TaskPlanPayload(StrictPayload):
    tasks: list[SpecialistTaskPayload] = Field(min_length=1, max_length=2)

    def to_domain(self, *, response_language: str) -> tuple[SpecialistTask, ...]:
        return tuple(
            task.to_domain(response_language=response_language)
            for task in self.tasks
        )


class DocumentExpansionPayload(StrictPayload):
    source_ids: list[str] = Field(default_factory=list)

    def to_domain(
        self,
        *,
        valid_source_ids: set[str],
        max_documents: int,
    ) -> tuple[str, ...]:
        selected: list[str] = []
        for value in self.source_ids:
            item = value.strip()
            if not item or item not in valid_source_ids or item in selected:
                continue
            selected.append(item)
            if len(selected) >= max_documents:
                break
        return tuple(selected)


class ConsultationReportPayload(StrictPayload):
    findings: list[str] = Field(min_length=1)
    evidence: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    red_flags: list[str] = Field(default_factory=list)
    missing_queries: list[str] = Field(default_factory=list)

    def to_domain(self) -> ConsultationReport:
        return ConsultationReport(
            findings=_normalized_nonempty(self.findings),
            evidence=_normalized_unique(self.evidence),
            uncertainties=_normalized_nonempty(self.uncertainties),
            red_flags=_normalized_nonempty(self.red_flags),
            missing_queries=_normalized_unique(self.missing_queries),
        )


class RevisionRequestPayload(StrictPayload):
    task_id: str = Field(min_length=1)
    instructions: str = Field(min_length=1)

    def to_domain(self) -> RevisionRequest:
        return RevisionRequest(
            task_id=self.task_id.strip(),
            instructions=self.instructions.strip(),
        )


class ReviewDecisionPayload(StrictPayload):
    status: Literal["approved", "revise", "consult", "research"]
    evidence_sufficient: bool
    issues: list[str] = Field(default_factory=list)
    revisions: list[RevisionRequestPayload] = Field(default_factory=list)
    next_tasks: list[SpecialistTaskPayload] = Field(default_factory=list)
    additional_queries: list[str] = Field(default_factory=list)

    def to_domain(self, *, response_language: str) -> ReviewDecision:
        return ReviewDecision(
            status=self.status,
            evidence_sufficient=self.evidence_sufficient,
            issues=_normalized_nonempty(self.issues),
            revisions=tuple(revision.to_domain() for revision in self.revisions),
            next_tasks=tuple(
                task.to_domain(response_language=response_language)
                for task in self.next_tasks
            ),
            additional_queries=_normalized_unique(self.additional_queries),
        )


def _normalized_unique(values: list[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for value in values:
        item = value.strip()
        if item and item not in normalized:
            normalized.append(item)
    return tuple(normalized)


def _normalized_nonempty(values: list[str]) -> tuple[str, ...]:
    return tuple(value.strip() for value in values if value.strip())
