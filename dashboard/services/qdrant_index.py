from __future__ import annotations

from typing import Any

from qdrant_client.http import models

from dashboard.schemas import IndexPreview, QdrantCleanupResult
from dashboard.services.qdrant_filters import content_hash_filter
from dashboard.services.qdrant_preview import (
    INDEX_PREVIEW_LIMIT,
    qdrant_index_preview_for_content_hash,
)
from rag.qdrant import Qdrant


class QdrantIndexService:
    def status(self) -> dict[str, Any]:
        try:
            qdrant = Qdrant()
            collection_name = qdrant.settings.qdrant_collection_name
            exists = qdrant.collection_exists(collection_name)
            points_count = None
            if exists:
                collection = qdrant.get_collection(collection_name)
                points_count = getattr(collection, "points_count", None)
            return {
                "available": True,
                "collection_name": collection_name,
                "collection_exists": exists,
                "points_count": points_count,
                "error": None,
            }
        except Exception as error:
            return {
                "available": False,
                "collection_name": None,
                "collection_exists": False,
                "points_count": None,
                "error": str(error),
            }

    def indexed_content_hashes(
        self,
        content_hashes: set[str],
    ) -> tuple[set[str], str | None]:
        if not content_hashes:
            return set(), None

        try:
            qdrant = Qdrant()
            collection_name = qdrant.settings.qdrant_collection_name
            if not qdrant.collection_exists(collection_name):
                return set(), None

            indexed_hashes = {
                content_hash
                for content_hash in content_hashes
                if _content_hash_exists(
                    client=qdrant.client,
                    collection_name=collection_name,
                    content_hash=content_hash,
                )
            }
            return indexed_hashes, None
        except Exception as error:
            return set(), str(error)

    def delete_content_hash(self, content_hash: str | None) -> QdrantCleanupResult:
        if not content_hash:
            return QdrantCleanupResult(attempted=False, deleted=False)

        try:
            qdrant = Qdrant()
            collection_name = qdrant.settings.qdrant_collection_name
            if not qdrant.collection_exists(collection_name):
                return QdrantCleanupResult(attempted=True, deleted=False)

            qdrant.client.delete(
                collection_name=collection_name,
                points_selector=models.FilterSelector(
                    filter=content_hash_filter(content_hash)
                ),
            )
            return QdrantCleanupResult(attempted=True, deleted=True)
        except Exception as error:
            return QdrantCleanupResult(
                attempted=True,
                deleted=False,
                error=str(error),
            )

    def preview_content_hash(self, content_hash: str | None) -> dict[str, Any]:
        if not content_hash:
            return IndexPreview(
                available=False,
                collection_name=None,
                collection_exists=False,
                preview_limit=INDEX_PREVIEW_LIMIT,
                error="Document has no content hash yet",
            ).as_dict()

        try:
            qdrant = Qdrant()
            return qdrant_index_preview_for_content_hash(
                client=qdrant.client,
                collection_name=qdrant.settings.qdrant_collection_name,
                content_hash=content_hash,
                vector_name=qdrant.settings.dense_vector_name,
            )
        except Exception as error:
            return IndexPreview(
                available=False,
                collection_name=None,
                collection_exists=False,
                preview_limit=INDEX_PREVIEW_LIMIT,
                error=str(error),
            ).as_dict()


def _content_hash_exists(
    *,
    client: Any,
    collection_name: str,
    content_hash: str,
) -> bool:
    records, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter=content_hash_filter(content_hash),
        limit=1,
        with_payload=False,
        with_vectors=False,
    )
    return bool(records)
