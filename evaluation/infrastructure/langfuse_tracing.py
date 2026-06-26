from __future__ import annotations

from langfuse import Langfuse

from evaluation.application.models import ReadyEvaluationCorpus
from evaluation.application.ports import (
    AnswerSystemUnderTest,
    RagasEvaluator,
    RetrieverUnderTest,
)
from evaluation.domain.report import MetricResult
from evaluation.domain.samples import AnswerEvaluationSample, RetrievalItem
from evaluation.domain.suite import EvaluationCase


class TracingRetrieverUnderTest:
    def __init__(self, wrapped: RetrieverUnderTest, client: Langfuse) -> None:
        self._wrapped = wrapped
        self._client = client

    def retrieve(
        self,
        case: EvaluationCase,
        *,
        corpus: ReadyEvaluationCorpus,
        limit: int,
    ) -> tuple[RetrievalItem, ...]:
        with self._client.start_as_current_observation(
            name="retrieval",
            as_type="retriever",
            metadata={"case_id": case.id, "limit": limit},
        ) as observation:
            items = self._wrapped.retrieve(case, corpus=corpus, limit=limit)
            observation.update(output={"item_count": len(items)})
            return items


class TracingAnswerSystemUnderTest:
    def __init__(self, wrapped: AnswerSystemUnderTest, client: Langfuse) -> None:
        self._wrapped = wrapped
        self._client = client

    def answer(
        self,
        case: EvaluationCase,
        *,
        corpus: ReadyEvaluationCorpus,
        retrieval_limit: int,
    ) -> AnswerEvaluationSample:
        with self._client.start_as_current_observation(
            name="answer",
            as_type="agent",
            metadata={"case_id": case.id, "retrieval_limit": retrieval_limit},
        ) as observation:
            answer = self._wrapped.answer(
                case,
                corpus=corpus,
                retrieval_limit=retrieval_limit,
            )
            observation.update(
                output={
                    "source_count": len(answer.contexts),
                    "insufficient_context": answer.insufficient_context,
                }
            )
            return answer


class TracingRagasEvaluator:
    def __init__(self, wrapped: RagasEvaluator, client: Langfuse) -> None:
        self._wrapped = wrapped
        self._client = client

    def evaluate(self, sample: AnswerEvaluationSample) -> tuple[MetricResult, ...]:
        with self._client.start_as_current_observation(
            name="scoring",
            as_type="evaluator",
            metadata={"case_id": sample.case_id},
        ) as observation:
            metrics = self._wrapped.evaluate(sample)
            observation.update(
                output={"metrics": [metric.metric.value for metric in metrics]}
            )
            return metrics
