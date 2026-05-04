from __future__ import annotations

import json
from datetime import UTC, datetime
from uuid import uuid4

from evaluation.application.case_runner import EvaluationCaseRunner
from evaluation.application.corpus import EnsureEvaluationCorpus
from evaluation.application.errors import EvaluationExecutionError, JudgeNotCalibratedError
from evaluation.application.guard import SyntheticBoundaryGuard
from evaluation.application.models import (
    EvaluationFingerprints,
    EvaluationOutcome,
    EvaluationRunSummary,
    ReadyEvaluationCorpus,
)
from evaluation.application.ports import (
    ExperimentGateway,
    JudgeCalibration,
    JudgeFingerprintProvider,
    ProfileFingerprintProvider,
    ProfileRepository,
    SystemFingerprintProvider,
)
from evaluation.application.scoring import SampleScoringPipeline, ScoreAggregator
from evaluation.domain.quality import ANSWERABLE_ONLY_METRICS, QualityGate
from evaluation.domain.report import CaseResult, EvaluationReport
from evaluation.domain.suite import EvaluationCase, EvaluationProfile
from evaluation.domain.values import MetricName, ThresholdScope


class EvaluationRunSummarizer:
    def __init__(self, profile: EvaluationProfile, aggregator: ScoreAggregator) -> None:
        self._profile = profile
        self._aggregator = aggregator

    def __call__(self, cases: tuple[CaseResult, ...]) -> EvaluationRunSummary:
        self._validate_required_metrics(cases)
        metrics = tuple(metric for case in cases for metric in case.metrics)
        aggregate = self._aggregator.aggregate(metrics)
        now = datetime.now(UTC)
        report = EvaluationReport(
            run_id="pending",
            profile_id=self._profile.id,
            profile_version=self._profile.version,
            started_at=now,
            finished_at=now,
            cases=cases,
            aggregate_metrics=aggregate,
        )
        decision = QualityGate(self._profile.thresholds).evaluate(report)
        return EvaluationRunSummary(aggregate_metrics=aggregate, decision=decision)

    def _validate_required_metrics(self, cases: tuple[CaseResult, ...]) -> None:
        available = {metric.metric for case in cases for metric in case.metrics}
        missing = {
            threshold.metric
            for threshold in self._profile.thresholds
            if threshold.metric not in available
        }
        if missing:
            names = ", ".join(sorted(metric.value for metric in missing))
            raise EvaluationExecutionError(f"Missing required evaluation metrics: {names}")
        for threshold in self._profile.thresholds:
            if threshold.scope is not ThresholdScope.CASE:
                continue
            for case in cases:
                if not _metric_applies_to_case(threshold.metric, case):
                    continue
                if any(metric.metric is threshold.metric for metric in case.metrics):
                    continue
                raise EvaluationExecutionError(
                    f"Missing {threshold.metric.value} for case {case.case_id}"
                )


class RunEvaluation:
    def __init__(
        self,
        *,
        profiles: ProfileRepository,
        corpus: EnsureEvaluationCorpus,
        case_runner: EvaluationCaseRunner,
        scoring: SampleScoringPipeline,
        aggregator: ScoreAggregator,
        guard: SyntheticBoundaryGuard,
        experiments: ExperimentGateway,
        calibration: JudgeCalibration,
        profile_fingerprint: ProfileFingerprintProvider,
        system_fingerprint: SystemFingerprintProvider,
        judge_fingerprint: JudgeFingerprintProvider,
        configuration_metadata: dict[str, str],
    ) -> None:
        self._profiles = profiles
        self._corpus = corpus
        self._case_runner = case_runner
        self._scoring = scoring
        self._aggregator = aggregator
        self._guard = guard
        self._experiments = experiments
        self._calibration = calibration
        self._profile_fingerprint = profile_fingerprint
        self._system_fingerprint = system_fingerprint
        self._judge_fingerprint = judge_fingerprint
        self._configuration_metadata = dict(configuration_metadata)

    def execute(
        self,
        *,
        profile_id: str,
        dataset_version: datetime,
    ) -> EvaluationOutcome:
        started_at = datetime.now(UTC)
        profile = self._profiles.get(profile_id)
        self._experiments.authenticate()
        calibration = self._calibration.execute()
        if not calibration.passed:
            raise JudgeNotCalibratedError("Configured evaluation judge failed calibration")
        corpus = self._corpus.execute(profile)
        fingerprints = self._fingerprints(profile, corpus_fingerprint=corpus.fingerprint)
        run_name = _run_name(profile.id, fingerprints.system)
        execution = self._experiments.execute(
            profile=profile,
            dataset_version=dataset_version,
            run_name=run_name,
            metadata=self._metadata(
                profile,
                corpus.collection_name,
                dataset_version,
                fingerprints,
            ),
            task=lambda case: self._execute_case(case, profile=profile, corpus=corpus),
            summarize=EvaluationRunSummarizer(profile, self._aggregator),
        )
        self._experiments.flush_and_verify(execution)
        report = EvaluationReport(
            run_id=execution.run_id,
            profile_id=profile.id,
            profile_version=profile.version,
            started_at=started_at,
            finished_at=datetime.now(UTC),
            cases=execution.cases,
            aggregate_metrics=execution.summary.aggregate_metrics,
        )
        return EvaluationOutcome(
            execution=execution,
            report=report,
            fingerprints=fingerprints,
        )

    def _execute_case(
        self,
        case: EvaluationCase,
        *,
        profile: EvaluationProfile,
        corpus: ReadyEvaluationCorpus,
    ) -> CaseResult:
        executed = self._case_runner.execute(
            case,
            corpus=corpus,
            retrieval_limit=profile.retrieval_limit,
        )
        provisional = CaseResult(case.id, executed.retrieval, executed.answer, ())
        self._guard.validate(provisional, corpus=corpus)
        return CaseResult(
            case.id,
            executed.retrieval,
            executed.answer,
            self._scoring.score(executed),
        )

    def _fingerprints(
        self,
        profile: EvaluationProfile,
        *,
        corpus_fingerprint: str,
    ) -> EvaluationFingerprints:
        return EvaluationFingerprints(
            profile=self._profile_fingerprint.calculate(profile),
            corpus=corpus_fingerprint,
            system=self._system_fingerprint.calculate(
                profile,
                corpus_fingerprint=corpus_fingerprint,
            ),
            judge=self._judge_fingerprint.calculate(),
        )

    def _metadata(
        self,
        profile: EvaluationProfile,
        collection_name: str,
        dataset_version: datetime,
        fingerprints: EvaluationFingerprints,
    ) -> dict[str, str]:
        thresholds = [
            {
                "metric": threshold.metric.value,
                "minimum": threshold.minimum.value,
                "scope": threshold.scope.value,
            }
            for threshold in profile.thresholds
        ]
        return {
            **self._configuration_metadata,
            "profile_id": profile.id,
            "profile_version": profile.version,
            "profile_fingerprint": fingerprints.profile,
            "corpus_fingerprint": fingerprints.corpus,
            "system_fingerprint": fingerprints.system,
            "judge_fingerprint": fingerprints.judge,
            "collection_name": collection_name,
            "dataset_version": dataset_version.isoformat(),
            "retrieval_limit": str(profile.retrieval_limit),
            "agent_prompt_version": profile.agent_prompt_version,
            "thresholds": json.dumps(thresholds, sort_keys=True),
            "gate_version": profile.gate_version,
        }


def _run_name(profile_id: str, system_fingerprint: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{profile_id}-{timestamp}-{system_fingerprint[:8]}-{uuid4().hex[:8]}"


def _metric_applies_to_case(metric: MetricName, case: CaseResult) -> bool:
    if metric in ANSWERABLE_ONLY_METRICS:
        return case.answer.answerable
    if metric in {MetricName.HIT_RATE_AT_5, MetricName.MRR_AT_5}:
        return bool(case.retrieval.expected_source_keys)
    if metric is MetricName.ABSTENTION_ACCURACY:
        return not case.answer.answerable
    return True
