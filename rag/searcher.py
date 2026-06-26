from __future__ import annotations

from typing import Any

from rag.qdrant import Qdrant


class Searcher:
    def __init__(self, qdrant: Qdrant | None = None) -> None:
        self._qdrant = qdrant or Qdrant()

    def search(self, query: str, k: int = 3) -> Any:
        return self._qdrant.hybrid_search_with_rrf(query_text=query, limit=k)


if __name__ == "__main__":
    searcher = Searcher()
    results = searcher.search(query="blood test results")
    print(results)
