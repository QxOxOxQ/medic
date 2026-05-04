from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agents.models import UnknownAgentError


AGENTS_DIR = Path(__file__).resolve().parent
DEFAULT_PROFILES_PATH = AGENTS_DIR / "profiles.json"
DEFAULT_PROMPTS_DIR = AGENTS_DIR / "prompts"


@dataclass(frozen=True)
class AgentProfile:
    name: str
    display_name: str
    aliases: tuple[str, ...]
    keywords: tuple[str, ...]
    system_prompt: str
    instructions_prompt: str

    def matches(self, value: str) -> bool:
        normalized = _normalize(value)
        return normalized in {_normalize(name) for name in self._names()}

    def keyword_score(self, question: str) -> int:
        normalized_question = _normalize(question)
        return sum(
            1
            for keyword in self.keywords
            if _normalize(keyword) in normalized_question
        )

    def build_messages(
        self,
        *,
        question: str,
        conversation_context: str = "",
    ) -> list[dict[str, str]]:
        user_prompt = self._user_prompt(
            question,
            conversation_context=conversation_context,
        )
        return [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": user_prompt},
        ]

    def _names(self) -> tuple[str, ...]:
        return (self.name, self.display_name, *self.aliases)

    def _system_prompt(self) -> str:
        return f"{self.system_prompt}\n\n{self.instructions_prompt}"

    def _user_prompt(
        self,
        question: str,
        conversation_context: str,
    ) -> str:
        context_block = ""
        if conversation_context:
            context_block = f"\n\nRecent conversation:\n{conversation_context}"
        return (
            "Response language: English"
            f"{context_block}"
            f"\n\nQuestion:\n{question}"
        )


@dataclass(frozen=True)
class AgentProfileSet:
    profiles: tuple[AgentProfile, ...]
    default_agent_name: str


class AgentRegistry:
    def __init__(
        self,
        profiles: tuple[AgentProfile, ...] | None = None,
        default_agent_name: str | None = None,
    ) -> None:
        if profiles is None:
            profile_set = load_profile_set()
            self._profiles = profile_set.profiles
            self._default_agent_name = (
                default_agent_name or profile_set.default_agent_name
            )
            return

        self._profiles = profiles
        self._default_agent_name = default_agent_name or profiles[0].name

    def get(self, name: str) -> AgentProfile:
        for profile in self._profiles:
            if profile.name == name:
                return profile
        raise UnknownAgentError(f"Unknown agent: {name}")

    def select(
        self,
        *,
        question: str,
        requested_agent: str | None,
    ) -> AgentProfile:
        if requested_agent:
            return self._select_requested(requested_agent)

        scored = [
            (profile.keyword_score(question), profile)
            for profile in self._profiles
        ]
        best_score, best_profile = max(scored, key=lambda item: item[0])
        if best_score > 0:
            return best_profile
        return self.get(self._default_agent_name)

    def select_many(
        self,
        *,
        question: str,
        requested_agent: str | None,
    ) -> tuple[AgentProfile, ...]:
        if requested_agent:
            return (self._select_requested(requested_agent),)

        scored = [
            (profile.keyword_score(question), index, profile)
            for index, profile in enumerate(self._profiles)
        ]
        matching = [
            item for item in scored
            if item[0] > 0
        ]
        if not matching:
            if _is_broad_question(question):
                return self._profiles
            return (self.get(self._default_agent_name),)
        return tuple(
            profile
            for _, _, profile in sorted(
                matching,
                key=lambda item: (-item[0], item[1]),
            )
        )

    def _select_requested(self, requested_agent: str) -> AgentProfile:
        for profile in self._profiles:
            if profile.matches(requested_agent):
                return profile
        raise UnknownAgentError(f"Unknown agent: {requested_agent}")


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
        default_agent_name=raw_config["default_agent"],
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
        keywords=tuple(raw_profile.get("keywords", ())),
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


def _is_broad_question(question: str) -> bool:
    normalized = _normalize(question)
    broad_terms = (
        "all",
        "any",
        "everything",
        "overview",
        "summary",
        "summarize",
        "broad",
        "wszystkie",
        "wszystko",
        "calosc",
        "cala",
        "podsumuj",
        "przeglad",
    )
    return any(term in normalized for term in broad_terms)
