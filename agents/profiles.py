from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agents.contracts import ConsultationReport, SpecialistTask
from agents.models import UnknownAgentError


AGENTS_DIR = Path(__file__).resolve().parent
DEFAULT_PROFILES_PATH = AGENTS_DIR / "profiles.json"
DEFAULT_PROMPTS_DIR = AGENTS_DIR / "prompts"


@dataclass(frozen=True)
class AgentProfile:
    name: str
    display_name: str
    aliases: tuple[str, ...]
    expertise: str
    system_prompt: str
    instructions_prompt: str

    def matches(self, value: str) -> bool:
        normalized = _normalize(value)
        return normalized in {_normalize(name) for name in self._names()}

    def consultation_prompt(
        self,
        *,
        task: SpecialistTask,
        question: str,
        conversation_context: str = "",
        source_blocks: tuple[str, ...] = (),
        previous_report: ConsultationReport | None = None,
        revision_instructions: str | None = None,
    ) -> str:
        return _consultation_prompt(
            task,
            question=question,
            conversation_context=conversation_context,
            source_blocks=source_blocks,
            previous_report=previous_report,
            revision_instructions=revision_instructions,
        )

    def _names(self) -> tuple[str, ...]:
        return (self.name, self.display_name, *self.aliases)

    def system_prompt_text(self) -> str:
        return f"{self.system_prompt}\n\n{self.instructions_prompt}"

@dataclass(frozen=True)
class AgentProfileSet:
    profiles: tuple[AgentProfile, ...]
    professor_prompt: str


class AgentRegistry:
    def __init__(
        self,
        profiles: tuple[AgentProfile, ...] | None = None,
        professor_prompt: str | None = None,
    ) -> None:
        if profiles is None:
            profile_set = load_profile_set()
            self._profiles = profile_set.profiles
            self._professor_prompt = professor_prompt or profile_set.professor_prompt
            return

        self._profiles = profiles
        self._professor_prompt = professor_prompt or ""

    @property
    def profiles(self) -> tuple[AgentProfile, ...]:
        return self._profiles

    @property
    def professor_prompt(self) -> str:
        return self._professor_prompt

    def get(self, name: str) -> AgentProfile:
        for profile in self._profiles:
            if profile.matches(name):
                return profile
        raise UnknownAgentError(f"Unknown agent: {name}")

    def canonical_name(self, name: str) -> str:
        return self.get(name).name

    def expertise_catalog(self) -> str:
        return "\n".join(
            f"- {profile.name}: {profile.expertise}" for profile in self._profiles
        )


def load_profiles(
    *,
    profiles_path: Path = DEFAULT_PROFILES_PATH,
    prompts_dir: Path = DEFAULT_PROMPTS_DIR,
) -> tuple[AgentProfile, ...]:
    return load_profile_set(
        profiles_path=profiles_path,
        prompts_dir=prompts_dir,
    ).profiles


def load_profile_set(
    *,
    profiles_path: Path = DEFAULT_PROFILES_PATH,
    prompts_dir: Path = DEFAULT_PROMPTS_DIR,
) -> AgentProfileSet:
    raw_config = json.loads(profiles_path.read_text(encoding="utf-8"))
    system_prompt = _read_prompt(prompts_dir, raw_config["system_prompt"])
    professor_prompt = _read_prompt(prompts_dir, raw_config["professor_prompt"])
    profiles = tuple(
        _profile_from_config(
            raw_profile,
            prompts_dir=prompts_dir,
            system_prompt=system_prompt,
        )
        for raw_profile in raw_config["profiles"]
    )
    return AgentProfileSet(
        profiles=profiles,
        professor_prompt=professor_prompt,
    )


def _profile_from_config(
    raw_profile: dict[str, Any],
    *,
    prompts_dir: Path,
    system_prompt: str,
) -> AgentProfile:
    return AgentProfile(
        name=raw_profile["name"],
        display_name=raw_profile["display_name"],
        aliases=tuple(raw_profile.get("aliases", ())),
        expertise=raw_profile["expertise"],
        system_prompt=system_prompt,
        instructions_prompt=_read_prompt(prompts_dir, raw_profile["prompt"]),
    )


def _read_prompt(prompts_dir: Path, prompt_path: str) -> str:
    path = prompts_dir / prompt_path
    return path.read_text(encoding="utf-8").strip()


def _normalize(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    return "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character)
    )


def _consultation_prompt(
    task: SpecialistTask,
    *,
    question: str,
    conversation_context: str,
    source_blocks: tuple[str, ...],
    previous_report: ConsultationReport | None,
    revision_instructions: str | None,
) -> str:
    previous_block = "-"
    if previous_report is not None:
        previous_block = _report_block(previous_report)
    return (
        "This is an internal consultation requested by the lead professor. "
        "Do not address the user directly. Return the requested structured "
        "consultation report.\n\n"
        f"Natural-language report language:\n{task.response_language}\n\n"
        f"Consultation objective:\n{task.objective}\n\n"
        f"User question:\n{question}\n\n"
        f"Recent conversation:\n{conversation_context or '-'}\n\n"
        f"Assigned sources:\n{chr(10).join(source_blocks) or '-'}\n\n"
        f"Previous version of this report:\n{previous_block}\n\n"
        f"Professor revision instructions:\n{revision_instructions or '-'}\n\n"
        "For document-grounded work, use only assigned source IDs in the "
        "evidence field. If evidence is missing, report the uncertainty and "
        "propose focused missing_queries instead of inventing facts."
    )


def _report_block(report: ConsultationReport) -> str:
    return (
        f"Findings: {list(report.findings)}\n"
        f"Evidence: {list(report.evidence)}\n"
        f"Uncertainties: {list(report.uncertainties)}\n"
        f"Red flags: {list(report.red_flags)}\n"
        f"Missing queries: {list(report.missing_queries)}"
    )
