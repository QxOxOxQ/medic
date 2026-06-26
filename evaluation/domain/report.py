from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from evaluation.domain.samples import (
    AnswerEvaluationSample,
    RetrievalEvaluationSample,
)
from evaluation.domain.values import MetricName, Score


@dataclass(frozen=True)
class MetricResult:
    metric: MetricName
    score: Score
    case_id: str | None = None
    raw_result_json: str | None = None


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    retrieval: RetrievalEvaluationSample
    answer: AnswerEvaluationSample
    metrics: tuple[MetricResult, ...]


@dataclass(frozen=True)
class EvaluationReport:
    run_id: str
    profile_id: str
    profile_version: str
    started_at: datetime
    finished_at: datetime
    cases: tuple[CaseResult, ...]
    aggregate_metrics: tuple[MetricResult, ...]
