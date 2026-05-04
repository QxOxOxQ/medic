from __future__ import annotations

from collections.abc import Iterable

from agents.models import AgentSource
from rag.retrieval import SearchResult


class SourceLedger:
    def __init__(self) -> None:
        self._sources: list[AgentSource] = []
        self._keys: dict[tuple[str | None, str | None, str | None, str], str] = {}

    def record_results(
        self,
        results: Iterable[SearchResult],
        *,
        retrieval_query: str | None = None,
    ) -> tuple[AgentSource, ...]:
        recorded: list[AgentSource] = []
        for result in results:
            key = (
                result.qdrant_point_id,
                result.source,
                result.content_hash,
                result.excerpt,
            )
            source_id = self._keys.get(key)
            if source_id is None:
                source_id = f"S{len(self._sources) + 1}"
                self._keys[key] = source_id
                self._sources.append(
                    AgentSource(
                        id=source_id,
                        source=result.source,
                        content_hash=result.content_hash,
                        document_name=result.document_name,
                        score=result.score,
                        excerpt=result.excerpt,
                        qdrant_point_id=result.qdrant_point_id,
                        document_id=result.document_id,
                        relative_raw_path=result.relative_raw_path,
                        chunk_index=result.chunk_index,
                        char_start=result.char_start,
                        char_end=result.char_end,
                        retrieval_query=retrieval_query,
                    )
                )
            recorded.append(self._source_by_id(source_id))
        return tuple(recorded)

    def sources(self) -> tuple[AgentSource, ...]:
        return tuple(self._sources)

    def _source_by_id(self, source_id: str) -> AgentSource:
        for source in self._sources:
            if source.id == source_id:
                return source
        raise KeyError(source_id)
