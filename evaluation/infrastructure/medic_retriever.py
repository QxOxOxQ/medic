from __future__ import annotations

from pathlib import Path

from sqlalchemy.orm import Session, sessionmaker

from evaluation.application.errors import EvaluationExecutionError
from evaluation.application.models import ReadyEvaluationCorpus
from evaluation.domain.samples import RetrievalItem
from evaluation.domain.suite import EvaluationCase
from evaluation.domain.values import SourceKey
from rag.qdrant import Qdrant
from rag.retrieval import RetrievalService, SearchResult
from rag.searcher import Searcher


class MedicRetrieverUnderTest:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def retrieve(
        self,
        case: EvaluationCase,
        *,
        corpus: ReadyEvaluationCorpus,
        limit: int,
    ) -> tuple[RetrievalItem, ...]:
        retriever = self._retriever(corpus.collection_name)
        try:
            results = retriever.search(
                query=case.question,
                limit=limit,
                owner_user_id=corpus.owner_user_id,
            )
        except Exception as error:
            raise EvaluationExecutionError("Direct retrieval failed") from error
        return tuple(
            RetrievalItem(
                source_key=_source_key(result),
                excerpt=result.excerpt,
                score=result.score,
                rank=rank,
                document_id=result.document_id,
                relative_raw_path=result.relative_raw_path,
            )
            for rank, result in enumerate(results, start=1)
        )

    def _retriever(self, collection_name: str) -> RetrievalService:
        return RetrievalService(
            search_provider=Searcher(Qdrant(collection_name=collection_name)),
            database_session_factory=self._session_factory,
        )


def _source_key(result: SearchResult) -> SourceKey:
    value = result.document_name or Path(result.source or "unknown").name
    return SourceKey(value)
