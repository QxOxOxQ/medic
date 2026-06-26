from __future__ import annotations

from pathlib import Path

from sqlalchemy.orm import Session, sessionmaker

from agents.observability import AgentObservability
from backend.factory import build_answer_question_use_case
from evaluation.application.errors import EvaluationExecutionError
from evaluation.application.models import ReadyEvaluationCorpus
from evaluation.domain.samples import AnswerContext, AnswerEvaluationSample
from evaluation.domain.suite import EvaluationCase
from evaluation.domain.values import SourceKey
from rag.qdrant import Qdrant
from rag.retrieval import RetrievalService
from rag.searcher import Searcher


class MedicAnswerSystemUnderTest:
    def __init__(
        self,
        session_factory: sessionmaker[Session],
        observability: AgentObservability,
    ) -> None:
        self._session_factory = session_factory
        self._observability = observability

    def answer(
        self,
        case: EvaluationCase,
        *,
        corpus: ReadyEvaluationCorpus,
        retrieval_limit: int,
    ) -> AnswerEvaluationSample:
        retriever = RetrievalService(
            search_provider=Searcher(Qdrant(collection_name=corpus.collection_name)),
            database_session_factory=self._session_factory,
        )
        use_case = build_answer_question_use_case(
            database_session_factory=self._session_factory,
            retriever=retriever,
            observability=self._observability,
        )
        try:
            answer = use_case.execute(
                question=case.question,
                limit=retrieval_limit,
                owner_user_id=corpus.owner_user_id,
                requested_agent=case.requested_agent,
            )
        except Exception as error:
            raise EvaluationExecutionError("Agent answer execution failed") from error
        contexts = tuple(
            AnswerContext(
                id=source.id,
                source_key=SourceKey(
                    source.document_name or Path(source.source or "unknown").name
                ),
                excerpt=source.excerpt,
                score=source.score,
                retrieval_query=source.retrieval_query,
                document_id=source.document_id,
                relative_raw_path=source.relative_raw_path,
            )
            for source in answer.sources
        )
        return AnswerEvaluationSample(
            case_id=case.id,
            question=case.question,
            reference_answer=case.reference_answer,
            answer=answer.answer,
            contexts=contexts,
            insufficient_context=answer.insufficient_context,
            answerable=case.answerable,
            latency_ms=0,
        )
