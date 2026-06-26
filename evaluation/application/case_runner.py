from __future__ import annotations

import time
from dataclasses import dataclass

from evaluation.application.ports import (
    AnswerSystemUnderTest,
    RetrieverUnderTest,
    retrieval_sample,
)
from evaluation.application.models import ReadyEvaluationCorpus
from evaluation.domain.samples import (
    AnswerEvaluationSample,
    RetrievalEvaluationSample,
)
from evaluation.domain.suite import EvaluationCase


@dataclass(frozen=True)
class ExecutedEvaluationCase:
    retrieval: RetrievalEvaluationSample
    answer: AnswerEvaluationSample


class EvaluationCaseRunner:
    def __init__(
        self,
        *,
        retriever: RetrieverUnderTest,
        answer_system: AnswerSystemUnderTest,
    ) -> None:
        self._retriever = retriever
        self._answer_system = answer_system

    def execute(
        self,
        case: EvaluationCase,
        *,
        corpus: ReadyEvaluationCorpus,
        retrieval_limit: int,
    ) -> ExecutedEvaluationCase:
        items = self._retriever.retrieve(
            case,
            corpus=corpus,
            limit=retrieval_limit,
        )
        started = time.perf_counter()
        answer = self._answer_system.answer(
            case,
            corpus=corpus,
            retrieval_limit=retrieval_limit,
        )
        latency_ms = int((time.perf_counter() - started) * 1000)
        measured_answer = AnswerEvaluationSample(
            case_id=answer.case_id,
            question=answer.question,
            reference_answer=answer.reference_answer,
            answer=answer.answer,
            contexts=answer.contexts,
            insufficient_context=answer.insufficient_context,
            answerable=answer.answerable,
            latency_ms=latency_ms,
        )
        return ExecutedEvaluationCase(
            retrieval=retrieval_sample(case, items),
            answer=measured_answer,
        )
