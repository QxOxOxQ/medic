from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from qdrant_client import models

import rag.qdrant as qdrant_module
from rag.embedding.embedder import DENSE_EMBEDDING, EmbeddingModelConfig
from rag.qdrant import Qdrant
from rag.searcher import Searcher


def test_searcher_delegates_to_qdrant_hybrid_rrf() -> None:
    class FakeQdrant:
        def __init__(self) -> None:
            self.hybrid_calls: list[dict[str, Any]] = []

        def hybrid_search_with_rrf(self, query_text: str, limit: int) -> list[str]:
            self.hybrid_calls.append({"query_text": query_text, "limit": limit})
            return ["result"]

    qdrant = FakeQdrant()

    result = Searcher(qdrant=qdrant).search("query text", k=5)

    assert result == ["result"]
    assert qdrant.hybrid_calls == [{"query_text": "query text", "limit": 5}]


def test_qdrant_hybrid_search_uses_dense_sparse_prefetch_and_rrf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeEmbedder:
        created_with: list[dict[str, str]] = []

        def __init__(self, *, provider: str, model: str) -> None:
            self.created_with.append({"provider": provider, "model": model})

        def embed_text(self, text: str) -> list[float]:
            assert text == "query text"
            return [0.1, 0.2, 0.3]

    class FakeClient:
        def __init__(self) -> None:
            self.query_calls: list[dict[str, Any]] = []

        def query_points(self, **kwargs: Any) -> SimpleNamespace:
            self.query_calls.append(kwargs)
            return SimpleNamespace(points=["point"])

    monkeypatch.setattr(qdrant_module, "Embedder", FakeEmbedder)

    client = FakeClient()
    qdrant = Qdrant.__new__(Qdrant)
    qdrant.client = client
    qdrant.settings = SimpleNamespace(
        qdrant_collection_name="documents",
        dense_vector_name="dense",
        sparse_vector_name="sparse",
        sparse_vector_model="Qdrant/bm25",
        prefetch_limit=20,
    )
    qdrant.embedding_model_config = EmbeddingModelConfig(
        provider="openrouter",
        model="openai/text-embedding-3-small",
        vector_size=1536,
        kind=DENSE_EMBEDDING,
    )

    result = qdrant.hybrid_search_with_rrf("query text", limit=5)

    assert result == ["point"]
    assert FakeEmbedder.created_with == [
        {
            "provider": "openrouter",
            "model": "openai/text-embedding-3-small",
        }
    ]
    assert len(client.query_calls) == 1
    query_call = client.query_calls[0]
    assert query_call["collection_name"] == "documents"
    assert query_call["limit"] == 5
    assert query_call["query"].fusion == models.Fusion.RRF

    dense_prefetch, sparse_prefetch = query_call["prefetch"]
    assert dense_prefetch.query == [0.1, 0.2, 0.3]
    assert dense_prefetch.using == "dense"
    assert dense_prefetch.limit == 20

    assert isinstance(sparse_prefetch.query, models.Document)
    assert sparse_prefetch.query.text == "query text"
    assert sparse_prefetch.query.model == "Qdrant/bm25"
    assert sparse_prefetch.using == "sparse"
    assert sparse_prefetch.limit == 20
