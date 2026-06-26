from __future__ import annotations

from dataclasses import dataclass

from evaluation.domain.errors import InvalidSuiteError
from evaluation.domain.values import SourceKey, Threshold


@dataclass(frozen=True)
class EvaluationCase:
    id: str
    question: str
    reference_answer: str
    expected_source_keys: tuple[SourceKey, ...]
    answerable: bool
    requested_agent: str | None = None
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise InvalidSuiteError("Evaluation case id cannot be empty")
        if not self.question.strip():
            raise InvalidSuiteError(f"Evaluation case {self.id} has an empty question")
        if not self.reference_answer.strip():
            raise InvalidSuiteError(
                f"Evaluation case {self.id} has no reference answer"
            )
        if self.answerable and not self.expected_source_keys:
            raise InvalidSuiteError(
                f"Answerable evaluation case {self.id} must declare expected sources"
            )
        if not self.answerable and self.expected_source_keys:
            raise InvalidSuiteError(
                f"Unanswerable evaluation case {self.id} cannot expect sources"
            )


@dataclass(frozen=True)
class EvaluationProfile:
    id: str
    version: str
    dataset_name: str
    document_paths: tuple[str, ...]
    thresholds: tuple[Threshold, ...]
    gate_version: str
    agent_prompt_version: str
    retrieval_limit: int = 5

    def __post_init__(self) -> None:
        required = (
            self.id,
            self.version,
            self.dataset_name,
            self.gate_version,
            self.agent_prompt_version,
        )
        if any(not value.strip() for value in required):
            raise InvalidSuiteError("Evaluation profile fields cannot be empty")
        if not self.document_paths:
            raise InvalidSuiteError("Evaluation profile must declare corpus documents")
        if self.retrieval_limit < 1:
            raise InvalidSuiteError("Retrieval limit must be positive")
        self._ensure_unique_thresholds()

    def _ensure_unique_thresholds(self) -> None:
        keys = [(threshold.metric, threshold.scope) for threshold in self.thresholds]
        if len(keys) != len(set(keys)):
            raise InvalidSuiteError("Metric thresholds must be unique per scope")
