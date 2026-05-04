from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from evaluation.domain.quality import GateDecision
from evaluation.domain.report import CaseResult, EvaluationReport, MetricResult


@dataclass(frozen=True)
class SeededEvaluationCorpus:
    document_ids: frozenset[UUID]
    relative_raw_paths: frozenset[str]
    source_keys: frozenset[str]


@dataclass(frozen=True)
class ReadyEvaluationCorpus:
    owner_user_id: UUID
    collection_name: str
    fingerprint: str
    seeded: SeededEvaluationCorpus


@dataclass(frozen=True)
class EvaluationFingerprints:
    profile: str
    corpus: str
    system: str
    judge: str


@dataclass(frozen=True)
class EvaluationRunSummary:
    aggregate_metrics: tuple[MetricResult, ...]
    decision: GateDecision


@dataclass(frozen=True)
class PublishedScore:
    name: str
    value: float
    data_type: str


@dataclass(frozen=True)
class ScorePublication:
    trace_id: str | None
    dataset_run_id: str | None
    scores: tuple[PublishedScore, ...]


@dataclass(frozen=True)
class ExperimentExecution:
    run_id: str
    run_name: str
    dataset_version: datetime
    dataset_name: str
    dataset_run_id: str
    dataset_run_url: str | None
    expected_item_count: int
    cases: tuple[CaseResult, ...]
    summary: EvaluationRunSummary
    score_publications: tuple[ScorePublication, ...]


@dataclass(frozen=True)
class DatasetBootstrapResult:
    dataset_name: str
    created_items: int
    verified_items: int


@dataclass(frozen=True)
class CalibrationResult:
    passed: bool
    good_score: float
    bad_score: float


@dataclass(frozen=True)
class EvaluationOutcome:
    execution: ExperimentExecution
    report: EvaluationReport
    fingerprints: EvaluationFingerprints
