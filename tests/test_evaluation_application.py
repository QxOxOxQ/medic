from __future__ import annotations

from uuid import uuid4

import pytest

from evaluation.application.errors import CorpusIsolationError, EvaluationExecutionError
from evaluation.application.guard import SyntheticBoundaryGuard
from evaluation.application.models import ReadyEvaluationCorpus, SeededEvaluationCorpus
from evaluation.application.run import EvaluationRunSummarizer
from evaluation.application.scoring import ScoreAggregator
from evaluation.domain.report import CaseResult, MetricResult
from evaluation.domain.samples import (
    AnswerContext,
    AnswerEvaluationSample,
    RetrievalEvaluationSample,
    RetrievalItem,
)
from evaluation.domain.suite import EvaluationProfile
from evaluation.domain.values import MetricName, Score, SourceKey, Threshold


def test_synthetic_guard_accepts_only_seeded_document_identity() -> None:
    document_id = uuid4()
    path = "medical-demo-v1/fingerprint/document.pdf"
    corpus = ReadyEvaluationCorpus(
        owner_user_id=uuid4(),
        collection_name="medic_eval_test_fingerprint",
        fingerprint="fingerprint",
        seeded=SeededEvaluationCorpus(
            document_ids=frozenset({document_id}),
            relative_raw_paths=frozenset({path}),
            source_keys=frozenset({"document.pdf"}),
        ),
    )

    SyntheticBoundaryGuard().validate(
        _result(document_id=document_id, relative_raw_path=path),
        corpus=corpus,
    )


def test_synthetic_guard_rejects_unknown_document_before_publication() -> None:
    corpus = ReadyEvaluationCorpus(
        owner_user_id=uuid4(),
        collection_name="medic_eval_test_fingerprint",
        fingerprint="fingerprint",
        seeded=SeededEvaluationCorpus(
            document_ids=frozenset({uuid4()}),
            relative_raw_paths=frozenset({"allowed/document.pdf"}),
            source_keys=frozenset({"document.pdf"}),
        ),
    )

    with pytest.raises(CorpusIsolationError):
        SyntheticBoundaryGuard().validate(
            _result(document_id=uuid4(), relative_raw_path="allowed/document.pdf"),
            corpus=corpus,
        )


def test_run_summarizer_preserves_gate_threshold_snapshot() -> None:
    profile = EvaluationProfile(
        id="profile",
        version="1",
        dataset_name="medic/test",
        document_paths=("document.pdf",),
        thresholds=(Threshold(MetricName.FAITHFULNESS, Score(0.9)),),
        gate_version="gate-v1",
        agent_prompt_version="agents-v1",
    )

    summary = EvaluationRunSummarizer(profile, ScoreAggregator())(
        (_result(metric_score=0.8),)
    )

    assert summary.decision.passed is False
    assert summary.aggregate_metrics[0].score == Score(0.8)


def test_run_summarizer_treats_missing_required_metric_as_execution_error() -> None:
    profile = EvaluationProfile(
        id="profile",
        version="1",
        dataset_name="medic/test",
        document_paths=("document.pdf",),
        thresholds=(Threshold(MetricName.ANSWER_CORRECTNESS, Score(0.9)),),
        gate_version="gate-v1",
        agent_prompt_version="agents-v1",
    )

    with pytest.raises(EvaluationExecutionError):
        EvaluationRunSummarizer(profile, ScoreAggregator())((_result(),))


def _result(
    *,
    document_id=None,
    relative_raw_path: str | None = None,
    metric_score: float = 1.0,
) -> CaseResult:
    source = SourceKey("document.pdf")
    retrieval = RetrievalEvaluationSample(
        case_id="case",
        question="Question?",
        expected_source_keys=(source,),
        items=(
            RetrievalItem(
                source,
                "context",
                0.9,
                1,
                document_id=document_id,
                relative_raw_path=relative_raw_path,
            ),
        ),
    )
    answer = AnswerEvaluationSample(
        case_id="case",
        question="Question?",
        reference_answer="Answer.",
        answer="Answer [S1].",
        contexts=(
            AnswerContext(
                id="S1",
                source_key=source,
                excerpt="context",
                score=0.9,
                retrieval_query="query",
                document_id=document_id,
                relative_raw_path=relative_raw_path,
            ),
        ),
        insufficient_context=False,
        answerable=True,
        latency_ms=10,
    )
    return CaseResult(
        case_id="case",
        retrieval=retrieval,
        answer=answer,
        metrics=(
            MetricResult(MetricName.FAITHFULNESS, Score(metric_score), "case"),
        ),
    )
