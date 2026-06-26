from __future__ import annotations

from typing import Protocol

from agents.contracts import (
    ConsultationReport,
    ResearchPlan,
    ReviewDecision,
    SpecialistTask,
)
from agents.models import AgentSource


class ProfessorModelPort(Protocol):
    def research_plan(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_queries: int,
        agent_name: str,
        phase: str,
    ) -> ResearchPlan: ...

    def task_plan(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_language: str,
        agent_name: str,
        phase: str,
    ) -> tuple[SpecialistTask, ...]: ...

    def consultation_report(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        agent_name: str,
        phase: str,
    ) -> ConsultationReport: ...

    def review_decision(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_language: str,
        agent_name: str,
        phase: str,
    ) -> ReviewDecision: ...

    def select_full_documents(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        valid_source_ids: set[str],
        max_documents: int,
        agent_name: str,
        phase: str,
    ) -> tuple[str, ...]: ...

    def text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        agent_name: str,
        phase: str,
    ) -> str: ...


class MedicalDocumentSearchPort(Protocol):
    def search_sources(self, *, query: str) -> tuple[AgentSource, ...]: ...

    def sources(self) -> tuple[AgentSource, ...]: ...

    def attach_full_content(self, *, source_id: str, full_content: str) -> None: ...


class FullDocumentReader(Protocol):
    def read(self, source: AgentSource) -> str | None: ...
