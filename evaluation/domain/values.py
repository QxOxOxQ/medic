from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from evaluation.domain.errors import InvalidScoreError, InvalidSuiteError


class MetricName(StrEnum):
    HIT_RATE_AT_5 = "hit_rate_at_5"
    MRR_AT_5 = "mrr_at_5"
    CITATION_VALIDITY = "citation_validity"
    ABSTENTION_ACCURACY = "abstention_accuracy"
    CONTEXT_PRECISION = "context_precision"
    CONTEXT_RECALL = "context_recall"
    FAITHFULNESS = "faithfulness"
    ANSWER_CORRECTNESS = "answer_correctness"
    ANSWER_RELEVANCY = "answer_relevancy"


class ThresholdScope(StrEnum):
    AGGREGATE = "aggregate"
    CASE = "case"


class EvaluationRunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    PASSED = "passed"
    FAILED_QUALITY = "failed_quality"
    FAILED_ERROR = "failed_error"


@dataclass(frozen=True, order=True)
class Score:
    value: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.value) or not 0.0 <= self.value <= 1.0:
            raise InvalidScoreError("Metric score must be finite and between 0 and 1")


@dataclass(frozen=True)
class Threshold:
    metric: MetricName
    minimum: Score
    scope: ThresholdScope = ThresholdScope.AGGREGATE


@dataclass(frozen=True, order=True)
class SourceKey:
    value: str

    def __post_init__(self) -> None:
        normalized = self.value.strip()
        if not normalized:
            raise InvalidSuiteError("Source key cannot be empty")
        object.__setattr__(self, "value", normalized)
