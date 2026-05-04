from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from langfuse.openai import AsyncOpenAI  # type: ignore[attr-defined]

from evaluation.application.errors import MetricEvaluationError
from evaluation.application.models import CalibrationResult
from evaluation.application.ports import RagasEvaluator
from evaluation.domain.report import MetricResult
from evaluation.domain.samples import AnswerContext, AnswerEvaluationSample
from evaluation.domain.values import MetricName, Score, SourceKey


logger = logging.getLogger(__name__)


class RagasMetricEvaluator:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        judge_model: str,
        embedding_model: str,
    ) -> None:
        from ragas.embeddings.base import embedding_factory
        from ragas.embeddings.base import BaseRagasEmbedding
        from ragas.llms import llm_factory
        from ragas.metrics.collections import (
            AnswerCorrectness,
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
        )

        client = AsyncOpenAI(api_key=api_key, base_url=base_url, max_retries=2)
        llm = llm_factory(
            judge_model,
            client=client,
            temperature=0,
            extra_body={"provider": {"order": ["OpenAI"], "allow_fallbacks": False}},
        )
        embeddings = cast(
            BaseRagasEmbedding,
            embedding_factory(
                "openai",
                model=embedding_model,
                client=client,
            ),
        )
        self._metrics: tuple[
            tuple[MetricName, Any, Callable[..., dict[str, Any]]], ...
        ] = (
            (
                MetricName.CONTEXT_PRECISION,
                ContextPrecision(llm=llm),
                lambda sample: {
                    "user_input": sample.question,
                    "reference": sample.reference_answer,
                    "retrieved_contexts": _contexts(sample),
                },
            ),
            (
                MetricName.CONTEXT_RECALL,
                ContextRecall(llm=llm),
                lambda sample: {
                    "user_input": sample.question,
                    "reference": sample.reference_answer,
                    "retrieved_contexts": _contexts(sample),
                },
            ),
            (
                MetricName.FAITHFULNESS,
                Faithfulness(llm=llm),
                lambda sample: {
                    "user_input": sample.question,
                    "response": sample.answer,
                    "retrieved_contexts": _contexts(sample),
                },
            ),
            (
                MetricName.ANSWER_CORRECTNESS,
                AnswerCorrectness(llm=llm, embeddings=embeddings),
                lambda sample: {
                    "user_input": sample.question,
                    "response": sample.answer,
                    "reference": sample.reference_answer,
                },
            ),
            (
                MetricName.ANSWER_RELEVANCY,
                AnswerRelevancy(llm=llm, embeddings=embeddings),
                lambda sample: {
                    "user_input": sample.question,
                    "response": sample.answer,
                },
            ),
        )

    def evaluate(self, sample: AnswerEvaluationSample) -> tuple[MetricResult, ...]:
        results: list[MetricResult] = []
        for metric_name, metric, arguments in self._metrics:
            try:
                result = self._score_with_retry(
                    metric_name,
                    metric,
                    arguments(sample),
                    sample.case_id,
                )
            except Exception as error:
                raise MetricEvaluationError(
                    f"RAGAS {metric_name.value} failed for case {sample.case_id}"
                ) from error
            results.append(result)
        return tuple(results)

    @classmethod
    def _score_with_retry(
        cls,
        metric_name: MetricName,
        metric: Any,
        arguments: dict[str, Any],
        case_id: str,
    ) -> MetricResult:
        try:
            return cls._score(metric_name, metric, arguments, case_id)
        except Exception:
            logger.warning(
                "Retrying RAGAS metric after failure: metric=%s case=%s",
                metric_name.value,
                case_id,
                exc_info=True,
            )
        return cls._score(metric_name, metric, arguments, case_id)

    @staticmethod
    def _score(
        metric_name: MetricName,
        metric: Any,
        arguments: dict[str, Any],
        case_id: str,
    ) -> MetricResult:
        result = metric.score(**arguments)
        raw = {
            "value": result.value,
            "reason": result.reason,
            "traces": result.traces,
        }
        return MetricResult(
            metric=metric_name,
            score=Score(float(result.value)),
            case_id=case_id,
            raw_result_json=json.dumps(raw, ensure_ascii=False, default=str),
        )


def _contexts(sample: AnswerEvaluationSample) -> list[str]:
    return [context.excerpt for context in sample.contexts]


class RagasJudgeCalibration:
    def __init__(self, evaluator: RagasEvaluator, path: Path) -> None:
        self._evaluator = evaluator
        self._path = path

    def execute(self) -> CalibrationResult:
        payload = json.loads(self._path.read_text(encoding="utf-8"))
        scores: dict[str, float] = {}
        for item in payload["cases"]:
            sample = self._sample(item)
            results = self._evaluator.evaluate(sample)
            selected = [
                result.score.value
                for result in results
                if result.metric
                in {MetricName.FAITHFULNESS, MetricName.ANSWER_CORRECTNESS}
            ]
            scores[str(item["expected"])] = sum(selected) / len(selected)
        good = scores["high"]
        bad = scores["low"]
        return CalibrationResult(
            passed=good >= 0.8 and bad <= 0.5,
            good_score=good,
            bad_score=bad,
        )

    @staticmethod
    def _sample(item: dict[str, Any]) -> AnswerEvaluationSample:
        return AnswerEvaluationSample(
            case_id=str(item["id"]),
            question=str(item["question"]),
            reference_answer=str(item["reference_answer"]),
            answer=str(item["answer"]),
            contexts=tuple(
                AnswerContext(
                    id=f"S{index}",
                    source_key=SourceKey("calibration"),
                    excerpt=str(context),
                    score=None,
                    retrieval_query=None,
                )
                for index, context in enumerate(item["contexts"], start=1)
            ),
            insufficient_context=False,
            answerable=True,
            latency_ms=0,
        )
