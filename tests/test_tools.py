from __future__ import annotations

import json
from collections.abc import Sequence
from uuid import UUID, uuid4

from rag.retrieval import SearchResult
from tools import RagSearchTool, SourceLedger


class RecordingRetriever:
    def __init__(self, results: Sequence[SearchResult]) -> None:
        self._results = tuple(results)
        self.calls: list[dict[str, object]] = []

    def search(
        self,
        *,
        query: str,
        limit: int,
        owner_user_id: UUID | None = None,
    ) -> Sequence[SearchResult]:
        self.calls.append(
            {
                "query": query,
                "limit": limit,
                "owner_user_id": owner_user_id,
            }
        )
        return self._results


def test_rag_search_tool_uses_agent_query_and_backend_user_scope() -> None:
    owner_user_id = uuid4()
    retriever = RecordingRetriever(
        [
            SearchResult(
                score=0.82,
                source="report.md",
                document_name="Clinical Report",
                content_hash="hash",
                excerpt="LDL cholesterol is elevated.",
            )
        ]
    )
    ledger = SourceLedger()
    tool = RagSearchTool(
        retriever=retriever,
        owner_user_id=owner_user_id,
        source_ledger=ledger,
        default_limit=5,
    )

    payload = json.loads(
        tool.search_user_medical_documents(
            query="focused lipid panel query",
            limit=2,
        )
    )

    assert retriever.calls == [
        {
            "query": "focused lipid panel query",
            "limit": 2,
            "owner_user_id": owner_user_id,
        }
    ]
    assert payload["sources"][0]["id"] == "S1"
    assert payload["sources"][0]["source"] == "report.md"
    assert payload["sources"][0]["document_name"] == "Clinical Report"
    assert ledger.sources()[0].id == "S1"


def test_source_ledger_keeps_stable_ids_for_repeated_results() -> None:
    result = SearchResult(
        score=0.82,
        source="report.md",
        document_name="Clinical Report",
        content_hash="hash",
        excerpt="LDL cholesterol is elevated.",
    )
    ledger = SourceLedger()

    first = ledger.record_results([result])
    second = ledger.record_results([result])

    assert first[0].id == "S1"
    assert second[0].id == "S1"
    assert len(ledger.sources()) == 1


def test_rag_search_tool_converts_to_langchain_structured_tool() -> None:
    retriever = RecordingRetriever([])
    tool = RagSearchTool(
        retriever=retriever,
        owner_user_id=uuid4(),
        source_ledger=SourceLedger(),
        default_limit=3,
    )

    langchain_tool = tool.to_langchain_tool()
    payload = json.loads(langchain_tool.invoke({"query": "missing source"}))

    assert langchain_tool.name == "search_user_medical_documents"
    assert payload["sources"] == []
    assert retriever.calls[0]["limit"] == 3
