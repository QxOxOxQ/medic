from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Protocol
from uuid import UUID

from evaluation.application.models import (
    DatasetBootstrapResult,
    ExperimentExecution,
    EvaluationRunSummary,
    CalibrationResult,
    ReadyEvaluationCorpus,
    SeededEvaluationCorpus,
)
from evaluation.domain.report import CaseResult, MetricResult
from evaluation.domain.samples import (
    AnswerEvaluationSample,
    RetrievalEvaluationSample,
    RetrievalItem,
)
from evaluation.domain.suite import EvaluationCase, EvaluationProfile


class ProfileRepository(Protocol):
    def get(self, profile_id: str) -> EvaluationProfile: ...


class RetrieverUnderTest(Protocol):
    def retrieve(
        self,
        case: EvaluationCase,
        *,
        corpus: ReadyEvaluationCorpus,
        limit: int,
    ) -> tuple[RetrievalItem, ...]: ...


class AnswerSystemUnderTest(Protocol):
    def answer(
        self,
        case: EvaluationCase,
        *,
        corpus: ReadyEvaluationCorpus,
        retrieval_limit: int,
    ) -> AnswerEvaluationSample: ...


class RagasEvaluator(Protocol):
    def evaluate(self, sample: AnswerEvaluationSample) -> tuple[MetricResult, ...]: ...


class CorpusFingerprintProvider(Protocol):
    def calculate(self, profile: EvaluationProfile) -> str: ...


class ProfileFingerprintProvider(Protocol):
    def calculate(self, profile: EvaluationProfile) -> str: ...


class SystemFingerprintProvider(Protocol):
    def calculate(
        self,
        profile: EvaluationProfile,
        *,
        corpus_fingerprint: str,
    ) -> str: ...


class JudgeFingerprintProvider(Protocol):
    def calculate(self) -> str: ...


class JudgeCalibration(Protocol):
    def execute(self) -> CalibrationResult: ...


class CollectionGuard(Protocol):
    def validate(self, collection_name: str) -> None: ...


class TenantProvisioner(Protocol):
    def ensure_tenant(self) -> UUID: ...


class DocumentSeeder(Protocol):
    def seed(
        self,
        profile: EvaluationProfile,
        *,
        corpus_fingerprint: str,
        owner_user_id: UUID,
    ) -> SeededEvaluationCorpus: ...


class IndexRebuilder(Protocol):
    def rebuild(
        self,
        profile: EvaluationProfile,
        *,
        corpus_fingerprint: str,
        owner_user_id: UUID,
        collection_name: str,
    ) -> None: ...


class CollectionInspector(Protocol):
    def is_ready(
        self,
        *,
        collection_name: str,
        document_ids: frozenset[UUID],
    ) -> bool: ...


CaseTask = Callable[[EvaluationCase], CaseResult]
RunSummarizer = Callable[[tuple[CaseResult, ...]], EvaluationRunSummary]


class DatasetBootstrapGateway(Protocol):
    def authenticate(self) -> None: ...

    def bootstrap(
        self,
        *,
        profile: EvaluationProfile,
        manifest_path: str,
    ) -> DatasetBootstrapResult: ...


class ExperimentGateway(Protocol):
    def authenticate(self) -> None: ...

    def execute(
        self,
        *,
        profile: EvaluationProfile,
        dataset_version: datetime,
        run_name: str,
        metadata: dict[str, str],
        task: CaseTask,
        summarize: RunSummarizer,
    ) -> ExperimentExecution: ...

    def flush_and_verify(self, execution: ExperimentExecution) -> None: ...


def retrieval_sample(
    case: EvaluationCase,
    items: tuple[RetrievalItem, ...],
) -> RetrievalEvaluationSample:
    return RetrievalEvaluationSample(
        case_id=case.id,
        question=case.question,
        expected_source_keys=case.expected_source_keys,
        items=items,
    )
