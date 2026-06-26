from __future__ import annotations

from dataclasses import dataclass

from evaluation.domain.report import EvaluationReport, MetricResult
from evaluation.domain.values import MetricName, Score, Threshold, ThresholdScope


ANSWERABLE_ONLY_METRICS = {
    MetricName.CONTEXT_PRECISION,
    MetricName.CONTEXT_RECALL,
    MetricName.FAITHFULNESS,
    MetricName.ANSWER_CORRECTNESS,
    MetricName.ANSWER_RELEVANCY,
}


@dataclass(frozen=True)
class GateViolation:
    metric: MetricName
    actual: Score | None
    required: Score
    case_id: str | None = None


@dataclass(frozen=True)
class GateDecision:
    passed: bool
    violations: tuple[GateViolation, ...]


@dataclass(frozen=True)
class QualityGate:
    thresholds: tuple[Threshold, ...]

    def evaluate(self, report: EvaluationReport) -> GateDecision:
        violations = tuple(self._violations(report))
        return GateDecision(passed=not violations, violations=violations)

    def _violations(self, report: EvaluationReport) -> list[GateViolation]:
        aggregate_by_metric = {
            result.metric: result for result in report.aggregate_metrics
        }
        violations: list[GateViolation] = []
        for threshold in self.thresholds:
            if threshold.scope is ThresholdScope.AGGREGATE:
                result = aggregate_by_metric.get(threshold.metric)
                if result is None:
                    violations.append(
                        GateViolation(
                            metric=threshold.metric,
                            actual=None,
                            required=threshold.minimum,
                        )
                    )
                else:
                    self._append_violation(result, threshold, violations)
                continue
            for case in report.cases:
                if (
                    threshold.metric in ANSWERABLE_ONLY_METRICS
                    and not case.answer.answerable
                ):
                    continue
                result = next(
                    (
                        metric
                        for metric in case.metrics
                        if metric.metric is threshold.metric
                    ),
                    None,
                )
                if result is None:
                    violations.append(
                        GateViolation(
                            metric=threshold.metric,
                            actual=None,
                            required=threshold.minimum,
                            case_id=case.case_id,
                        )
                    )
                    continue
                self._append_violation(result, threshold, violations)
        return violations

    @staticmethod
    def _append_violation(
        result: MetricResult,
        threshold: Threshold,
        violations: list[GateViolation],
    ) -> None:
        if result.score >= threshold.minimum:
            return
        violations.append(
            GateViolation(
                metric=result.metric,
                actual=result.score,
                required=threshold.minimum,
                case_id=result.case_id,
            )
        )
