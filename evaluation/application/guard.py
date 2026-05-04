from __future__ import annotations

from uuid import UUID

from evaluation.application.errors import CorpusIsolationError
from evaluation.application.models import ReadyEvaluationCorpus
from evaluation.domain.report import CaseResult
from evaluation.domain.samples import AnswerContext, RetrievalItem


class SyntheticBoundaryGuard:
    def validate(
        self,
        result: CaseResult,
        *,
        corpus: ReadyEvaluationCorpus,
    ) -> None:
        for item in result.retrieval.items:
            self._validate_retrieval(item, corpus=corpus)
        for context in result.answer.contexts:
            self._validate_context(context, corpus=corpus)

    @staticmethod
    def _validate_retrieval(
        item: RetrievalItem,
        *,
        corpus: ReadyEvaluationCorpus,
    ) -> None:
        SyntheticBoundaryGuard._validate_source(
            document_id=item.document_id,
            relative_raw_path=item.relative_raw_path,
            source_key=item.source_key.value,
            corpus=corpus,
        )

    @staticmethod
    def _validate_context(
        context: AnswerContext,
        *,
        corpus: ReadyEvaluationCorpus,
    ) -> None:
        SyntheticBoundaryGuard._validate_source(
            document_id=context.document_id,
            relative_raw_path=context.relative_raw_path,
            source_key=context.source_key.value,
            corpus=corpus,
        )

    @staticmethod
    def _validate_source(
        *,
        document_id: UUID | None,
        relative_raw_path: str | None,
        source_key: str,
        corpus: ReadyEvaluationCorpus,
    ) -> None:
        seeded = corpus.seeded
        if document_id not in seeded.document_ids:
            raise CorpusIsolationError("Evaluation result contains an unknown document")
        if relative_raw_path not in seeded.relative_raw_paths:
            raise CorpusIsolationError("Evaluation result contains an unknown path")
        if source_key not in seeded.source_keys:
            raise CorpusIsolationError("Evaluation result contains an unknown source")
