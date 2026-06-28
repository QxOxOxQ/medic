from __future__ import annotations

from typing import Any
from uuid import UUID

from rag.qdrant import Qdrant


class Searcher:
    def __init__(self, qdrant: Qdrant | None = None) -> None:
        self._qdrant = qdrant or Qdrant()

    def search(
        self,
        query: str,
        k: int = 3,
        *,
        owner_user_id: UUID | None = None,
    ) -> Any:
        if owner_user_id is None:
            return self._qdrant.hybrid_search_with_rrf(query_text=query, limit=k)
        return self._qdrant.hybrid_search_with_rrf(
            query_text=query, limit=k, owner_user_id=owner_user_id
        )


if __name__ == "__main__":
    searcher = Searcher()
    results = searcher.search(query="blood test results")
    print(results)
