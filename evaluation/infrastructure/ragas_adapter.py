from __future__ import annotations

import importlib
import json
import logging
import sys
import types
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

_VERTEXAI_MODULE = "langchain_community.chat_models.vertexai"


def _ensure_ragas_importable() -> None:
    """Register a stand-in for a langchain module ragas still imports eagerly.

    ragas 0.4.3 imports ``langchain_community.chat_models.vertexai.ChatVertexAI``
    at module load, but that module was dropped from the langchain 1.x community
    line this project targets (``langchain >= 1.3``). ragas only references the
    symbol in ``isinstance`` checks to detect a VertexAI judge; here the judge is
    always an OpenAI model, so an inert placeholder is sufficient and is never
    instantiated. Installing it lets ragas import without pinning the whole
    langchain stack back to the 0.3.x line. If a future langchain-community
    restores the real module, that one is imported instead.
    """
    try:
        importlib.import_module(_VERTEXAI_MODULE)
        return
    except ModuleNotFoundError:
        pass

    shim = types.ModuleType(_VERTEXAI_MODULE)

    class ChatVertexAI:
        """Inert stand-in; ragas references this type only in isinstance checks."""

    setattr(shim, "ChatVertexAI", ChatVertexAI)
    sys.modules.setdefault(_VERTEXAI_MODULE, shim)


class RagasMetricEvaluator:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        judge_model: str,
        embedding_model: str,
    ) -> None:
        _ensure_ragas_importable()
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
