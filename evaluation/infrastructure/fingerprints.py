from __future__ import annotations

import hashlib
import json
from importlib.metadata import version
from typing import Any

from clients.chat_models import ChatModelSettings
from evaluation.config import EvaluationSettings
from evaluation.domain.suite import EvaluationProfile
from evaluation.infrastructure.profile_json import profile_payload


class ProfileFingerprintCalculator:
    def calculate(self, profile: EvaluationProfile) -> str:
        return _fingerprint(profile_payload(profile))


class SystemFingerprintCalculator:
    def __init__(
        self,
        chat_settings: ChatModelSettings,
        *,
        agent_prompt_version: str,
    ) -> None:
        self._chat_settings = chat_settings
        self._agent_prompt_version = agent_prompt_version

    def calculate(
        self,
        profile: EvaluationProfile,
        *,
        corpus_fingerprint: str,
    ) -> str:
        if profile.agent_prompt_version != self._agent_prompt_version:
            raise ValueError("Evaluation profile agent prompt version is stale")
        return _fingerprint(
            {
                "corpus_fingerprint": corpus_fingerprint,
                "provider": self._chat_settings.provider,
                "model": self._chat_settings.model,
                "temperature": self._chat_settings.temperature,
                "max_retrieval_queries": (
                    self._chat_settings.max_retrieval_queries
                ),
                "max_consultations": self._chat_settings.max_consultations,
                "max_review_rounds": self._chat_settings.max_review_rounds,
                "retrieval_limit": profile.retrieval_limit,
                "agent_prompt_version": self._agent_prompt_version,
            }
        )


class JudgeFingerprintCalculator:
    def __init__(self, settings: EvaluationSettings) -> None:
        self._settings = settings

    def calculate(self) -> str:
        return _fingerprint(
            {
                "model": self._settings.judge_model,
                "provider": self._settings.judge_provider,
                "prompt_version": self._settings.judge_prompt_version,
                "embedding_model": self._settings.embedding_model,
                "ragas_version": version("ragas"),
            }
        )


def _fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
