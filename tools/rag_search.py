from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Protocol
from uuid import UUID

from langchain_core.tools import StructuredTool

from agents.observability import AgentObservability
from agents.models import AgentSource
from agents.trace import AgentTraceRecorder
from rag.retrieval import SearchResult
from tools.source_ledger import SourceLedger


class RagRetriever(Protocol):
    def search(
        self,
        *,
        query: str,
        limit: int,
        owner_user_id: UUID | None = None,
    ) -> Sequence[SearchResult]:
        ...


class RagSearchTool:
    name = "search_user_medical_documents"
    description = (
        "Search the current user's indexed medical documents. Write the query "
        "as a terse phrase of concrete clinical terms — anatomy, imaging "
        "modality, test/analyte names, medications, diagnoses — covering one "
        "concept, not a generic meta-phrase. Records may be in a different "
        "language than the question (often English or Latin), so cover key "
        "concepts in both the question's language and English. The user scope "
        "is enforced by the backend."
    )

    def __init__(
        self,
        *,
        retriever: RagRetriever,
        owner_user_id: UUID,
        source_ledger: SourceLedger,
        default_limit: int,
        trace_recorder: AgentTraceRecorder | None = None,
        max_limit: int = 20,
    ) -> None:
        self._retriever = retriever
        self._owner_user_id = owner_user_id
        self._source_ledger = source_ledger
        self._default_limit = default_limit
        self._trace_recorder = trace_recorder
        self._max_limit = max_limit

    def search_user_medical_documents(self, query: str, limit: int | None = None) -> str:
        normalized_query = query.strip()
        sources = self.search_sources(query=query, limit=limit)
        if not normalized_query:
            return json.dumps(
                {
                    "query": normalized_query,
                    "sources": [],
                    "message": "Empty search query.",
                }
            )
        return json.dumps(
            {
                "query": normalized_query,
                "sources": [source.as_dict() for source in sources],
            },
            ensure_ascii=False,
        )

    def search_sources(
        self,
        *,
        query: str,
        limit: int | None = None,
    ) -> tuple[AgentSource, ...]:
        normalized_query = query.strip()
        bounded_limit = self._bounded_limit(limit)
        if not normalized_query:
            self._record_trace(
                status="skipped",
                payload={
                    "query": normalized_query,
                    "limit": bounded_limit,
                    "message": "Empty search query.",
                },
            )
            return ()

        results = self._retriever.search(
            query=normalized_query,
            limit=bounded_limit,
            owner_user_id=self._owner_user_id,
        )
        sources = self._source_ledger.record_results(
            results,
            retrieval_query=normalized_query,
        )
        self._record_trace(
            status="succeeded",
            payload={
                "query": normalized_query,
                "limit": bounded_limit,
                "source_count": len(sources),
                "sources": [source.as_dict() for source in sources],
            },
        )
        return sources

    def sources(self) -> tuple[AgentSource, ...]:
        return self._source_ledger.sources()

    def attach_full_content(self, *, source_id: str, full_content: str) -> None:
        self._source_ledger.attach_full_content(
            source_id=source_id,
            full_content=full_content,
        )

    def to_langchain_tool(self) -> StructuredTool:
        default_limit = self._default_limit

        def search_user_medical_documents(
            query: str,
            limit: int = default_limit,
        ) -> str:
            return self.search_user_medical_documents(query=query, limit=limit)

        return StructuredTool.from_function(
            func=search_user_medical_documents,
            name=self.name,
            description=self.description,
        )

    def _bounded_limit(self, limit: int | None) -> int:
        if limit is None:
            return self._default_limit
        try:
            parsed_limit = int(limit)
        except (TypeError, ValueError):
            parsed_limit = self._default_limit
        return max(1, min(parsed_limit, self._max_limit))

    def _record_trace(self, *, status: str, payload: dict[str, object]) -> None:
        if self._trace_recorder is None:
            return
        self._trace_recorder.record(
            event_type="tool",
            title="RAG search",
            status=status,
            tool_name=self.name,
            payload=payload,
        )


class ObservedRagSearchPort:
    def __init__(
        self,
        *,
        tool: RagSearchTool,
        observability: AgentObservability,
        agent_name: str,
    ) -> None:
        self._tool = tool
        self._observability = observability
        self._agent_name = agent_name
        self._langchain_tool = tool.to_langchain_tool()

    def search_sources(self, *, query: str) -> tuple[AgentSource, ...]:
        config = self._observability.tool_config(
            agent_name=self._agent_name,
            tool_name=self._langchain_tool.name,
        )
        if config is None:
            return self._tool.search_sources(query=query)
        self._langchain_tool.invoke({"query": query}, config=config)
        return self._tool.sources()

    def sources(self) -> tuple[AgentSource, ...]:
        return self._tool.sources()

    def attach_full_content(self, *, source_id: str, full_content: str) -> None:
        self._tool.attach_full_content(source_id=source_id, full_content=full_content)
