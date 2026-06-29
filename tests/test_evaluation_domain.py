from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from evaluation.application.scoring import RetrievalScorer, ScoreAggregator
from evaluation.domain.errors import InvalidScoreError, InvalidSuiteError
from evaluation.domain.quality import QualityGate
from evaluation.domain.report import CaseResult, EvaluationReport, MetricResult
from evaluation.domain.samples import AnswerEvaluationSample, RetrievalEvaluationSample
from evaluation.domain.samples import RetrievalItem
from evaluation.domain.suite import EvaluationProfile
from evaluation.domain.values import (
    MetricName,
    Score,
    SourceKey,
    Threshold,
    ThresholdScope,
)
from evaluation.infrastructure.profile_json import JsonProfileRepository
from evaluation.infrastructure.ragas_adapter import RagasMetricEvaluator


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_medical_demo_profile_and_bootstrap_manifest_are_complete() -> None:
    profile = JsonProfileRepository(PROJECT_ROOT / "evaluation" / "profiles").get(
        "medical-demo-v1"
    )
    manifest = json.loads(
        (PROJECT_ROOT / "evaluation" / "suites" / "medical_demo_v1.json").read_text()
    )

    assert profile.dataset_name == "medic/medical-demo-v1"
    assert profile.retrieval_limit == 5
    assert len(manifest["cases"]) == 24


def test_score_rejects_nan_and_values_outside_unit_interval() -> None:
    with pytest.raises(InvalidScoreError):
        Score(float("nan"))
    with pytest.raises(InvalidScoreError):
        Score(1.01)


def test_profile_rejects_duplicate_threshold_scope() -> None:
    threshold = Threshold(MetricName.FAITHFULNESS, Score(0.8))
    with pytest.raises(InvalidSuiteError):
        EvaluationProfile(
            id="profile",
            version="1",
            dataset_name="medic/test",
            document_paths=("document.pdf",),
            thresholds=(threshold, threshold),
            gate_version="v1",
            agent_prompt_version="v1",
        )


def test_retrieval_scorer_enforces_at_five_boundary() -> None:
    sample = RetrievalEvaluationSample(
        case_id="ranking",
        question="Question?",
        expected_source_keys=(SourceKey("expected.pdf"),),
        items=tuple(
            RetrievalItem(
                SourceKey("expected.pdf" if rank == 6 else f"other-{rank}.pdf"),
                "excerpt",
                0.9,
                rank,
            )
            for rank in range(1, 7)
        ),
    )

    results = {
        result.metric: result.score.value for result in RetrievalScorer().score(sample)
    }

    assert results[MetricName.HIT_RATE_AT_5] == 0.0
    assert results[MetricName.MRR_AT_5] == 0.0


def test_quality_gate_applies_snapshot_thresholds() -> None:
    now = datetime.now(UTC)
    metric = MetricResult(MetricName.FAITHFULNESS, Score(0.7), "case")
    retrieval = RetrievalEvaluationSample("case", "Question?", (), ())
    answer = AnswerEvaluationSample(
        "case", "Question?", "Answer.", "Answer.", (), False, True, 1
    )
    report = EvaluationReport(
        run_id="run",
        profile_id="profile",
        profile_version="1",
        started_at=now,
        finished_at=now,
        cases=(CaseResult("case", retrieval, answer, (metric,)),),
        aggregate_metrics=(),
    )
    gate = QualityGate(
        (
            Threshold(
                MetricName.FAITHFULNESS,
                Score(0.8),
                ThresholdScope.CASE,
            ),
        )
    )

    decision = gate.evaluate(report)

    assert decision.passed is False
    assert decision.violations[0].case_id == "case"


def test_recorded_unscorable_metric_fails_gate_instead_of_aborting() -> None:
    now = datetime.now(UTC)
    unscorable = RagasMetricEvaluator._unscored_result(
        MetricName.FAITHFULNESS,
        "case",
        ValueError("retrieved_contexts cannot be empty"),
    )
    retrieval = RetrievalEvaluationSample("case", "Question?", (), ())
    answer = AnswerEvaluationSample(
        "case", "Question?", "Answer.", "Answer.", (), False, True, 1
    )
    report = EvaluationReport(
        run_id="run",
        profile_id="profile",
        profile_version="1",
        started_at=now,
        finished_at=now,
        cases=(CaseResult("case", retrieval, answer, (unscorable,)),),
        aggregate_metrics=(),
    )
    gate = QualityGate(
        (Threshold(MetricName.FAITHFULNESS, Score(0.8), ThresholdScope.CASE),)
    )

    decision = gate.evaluate(report)

    assert unscorable.score == Score(0.0)
    assert decision.passed is False
    assert decision.violations[0].case_id == "case"


def test_score_aggregator_keeps_metrics_separate() -> None:
    results = (
        MetricResult(MetricName.FAITHFULNESS, Score(1.0), "a"),
        MetricResult(MetricName.FAITHFULNESS, Score(0.8), "b"),
        MetricResult(MetricName.MRR_AT_5, Score(0.5), "a"),
    )

    aggregated = {
        result.metric: result.score.value
        for result in ScoreAggregator().aggregate(results)
    }

    assert aggregated[MetricName.FAITHFULNESS] == pytest.approx(0.9)
    assert aggregated[MetricName.MRR_AT_5] == 0.5
