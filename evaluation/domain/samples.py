from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from evaluation.domain.values import SourceKey


@dataclass(frozen=True)
class RetrievalItem:
    source_key: SourceKey
    excerpt: str
    score: float | None
    rank: int
    document_id: UUID | None = None
    relative_raw_path: str | None = None


@dataclass(frozen=True)
class RetrievalEvaluationSample:
    case_id: str
    question: str
    expected_source_keys: tuple[SourceKey, ...]
    items: tuple[RetrievalItem, ...]


@dataclass(frozen=True)
class AnswerContext:
    id: str
    source_key: SourceKey
    excerpt: str
    score: float | None
    retrieval_query: str | None
    document_id: UUID | None = None
    relative_raw_path: str | None = None


@dataclass(frozen=True)
class AnswerEvaluationSample:
    case_id: str
    question: str
    reference_answer: str
    answer: str
    contexts: tuple[AnswerContext, ...]
    insufficient_context: bool
    answerable: bool
    latency_ms: int
