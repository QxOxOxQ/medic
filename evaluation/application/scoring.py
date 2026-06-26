from __future__ import annotations

import re
from collections import defaultdict
from statistics import fmean

from evaluation.application.case_runner import ExecutedEvaluationCase
from evaluation.application.ports import RagasEvaluator
from evaluation.domain.report import MetricResult
from evaluation.domain.samples import (
    AnswerEvaluationSample,
    RetrievalEvaluationSample,
)
from evaluation.domain.values import MetricName, Score


class RetrievalScorer:
    def score(self, sample: RetrievalEvaluationSample) -> tuple[MetricResult, ...]:
        if not sample.expected_source_keys:
            return ()
        expected = {source.value for source in sample.expected_source_keys}
        first_rank = next(
            (
                item.rank
                for item in sample.items
                if item.rank <= 5 and item.source_key.value in expected
            ),
            None,
        )
        hit = 1.0 if first_rank is not None else 0.0
        reciprocal_rank = 1.0 / first_rank if first_rank is not None else 0.0
        return (
            MetricResult(MetricName.HIT_RATE_AT_5, Score(hit), sample.case_id),
            MetricResult(MetricName.MRR_AT_5, Score(reciprocal_rank), sample.case_id),
        )


class CitationScorer:
    _citation_pattern = re.compile(r"\[(S\d+)\]")

    def score(self, sample: AnswerEvaluationSample) -> MetricResult:
        citations = set(self._citation_pattern.findall(sample.answer))
        available = {context.id for context in sample.contexts}
        valid = citations.issubset(available)
        if sample.answerable:
            valid = valid and bool(citations)
        return MetricResult(
            MetricName.CITATION_VALIDITY,
            Score(1.0 if valid else 0.0),
            sample.case_id,
        )


class AbstentionScorer:
    def score(self, sample: AnswerEvaluationSample) -> MetricResult | None:
        if sample.answerable:
            return None
        return MetricResult(
            MetricName.ABSTENTION_ACCURACY,
            Score(1.0 if sample.insufficient_context else 0.0),
            sample.case_id,
        )


class AnswerScoringPipeline:
    def __init__(
        self,
        *,
        ragas_evaluator: RagasEvaluator,
        citation_scorer: CitationScorer | None = None,
        abstention_scorer: AbstentionScorer | None = None,
    ) -> None:
        self._ragas_evaluator = ragas_evaluator
        self._citation_scorer = citation_scorer or CitationScorer()
        self._abstention_scorer = abstention_scorer or AbstentionScorer()

    def score(self, sample: AnswerEvaluationSample) -> tuple[MetricResult, ...]:
        results = [self._citation_scorer.score(sample)]
        abstention = self._abstention_scorer.score(sample)
        if abstention is not None:
            results.append(abstention)
            return tuple(results)
        results.extend(self._ragas_evaluator.evaluate(sample))
        return tuple(results)


class SampleScoringPipeline:
    def __init__(
        self,
        *,
        retrieval_scorer: RetrievalScorer,
        answer_scorer: AnswerScoringPipeline,
    ) -> None:
        self._retrieval_scorer = retrieval_scorer
        self._answer_scorer = answer_scorer

    def score(self, executed: ExecutedEvaluationCase) -> tuple[MetricResult, ...]:
        return (
            *self._retrieval_scorer.score(executed.retrieval),
            *self._answer_scorer.score(executed.answer),
        )


class ScoreAggregator:
    def aggregate(self, results: tuple[MetricResult, ...]) -> tuple[MetricResult, ...]:
        values: dict[MetricName, list[float]] = defaultdict(list)
        for result in results:
            values[result.metric].append(result.score.value)
        return tuple(
            MetricResult(metric=metric, score=Score(fmean(metric_values)))
            for metric, metric_values in sorted(
                values.items(), key=lambda item: item[0]
            )
        )
