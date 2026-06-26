from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


QuestionMode = Literal["record_grounded", "general_information", "clarify"]
ReviewStatus = Literal["approved", "revise", "consult", "research"]


@dataclass(frozen=True)
class ResearchPlan:
    mode: QuestionMode
    response_language: str
    queries: tuple[str, ...]


@dataclass(frozen=True)
class SpecialistTask:
    id: str
    profile: str
    objective: str
    source_ids: tuple[str, ...]
    response_language: str
    independent: bool


@dataclass(frozen=True)
class ConsultationReport:
    findings: tuple[str, ...]
    evidence: tuple[str, ...]
    uncertainties: tuple[str, ...]
    red_flags: tuple[str, ...]
    missing_queries: tuple[str, ...]


@dataclass(frozen=True)
class CompletedConsultation:
    task: SpecialistTask
    report: ConsultationReport
    revision_count: int = 0


@dataclass(frozen=True)
class RevisionRequest:
    task_id: str
    instructions: str


@dataclass(frozen=True)
class ReviewDecision:
    status: ReviewStatus
    evidence_sufficient: bool
    issues: tuple[str, ...]
    revisions: tuple[RevisionRequest, ...]
    next_tasks: tuple[SpecialistTask, ...]
    additional_queries: tuple[str, ...]


@dataclass(frozen=True)
class ReviewOutcome:
    decision: ReviewDecision | None
    rounds_completed: int
    consultation_budget_exhausted: bool
    review_budget_exhausted: bool
