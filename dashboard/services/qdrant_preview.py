from __future__ import annotations

from typing import Any

from dashboard.schemas import IndexPreview
from dashboard.services.qdrant_filters import content_hash_filter
from rag.indexer import vector_previews


INDEX_PREVIEW_LIMIT = 20


def qdrant_index_preview_for_content_hash(
    *,
    client: Any,
    collection_name: str,
    content_hash: str,
    vector_name: str | None = None,
    limit: int = INDEX_PREVIEW_LIMIT,
) -> dict[str, Any]:
    if not client.collection_exists(collection_name):
        return IndexPreview(
            available=True,
            collection_name=collection_name,
            collection_exists=False,
            preview_limit=limit,
        ).as_dict()

    records, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter=content_hash_filter(content_hash),
        limit=limit,
        with_payload=True,
        with_vectors=True,
    )
    points = sorted(
        (_point_preview(record, vector_name=vector_name) for record in records),
        key=_point_sort_key,
    )
    return IndexPreview(
        available=True,
        collection_name=collection_name,
        collection_exists=True,
        preview_limit=limit,
        points=points,
        shown_points=len(points),
    ).as_dict()


def _point_preview(point: Any, *, vector_name: str | None) -> dict[str, Any]:
    payload = getattr(point, "payload", {}) or {}
    embeddings = vector_previews(
        getattr(point, "vector", None),
        preferred_name=vector_name,
    )
    return {
        "id": str(getattr(point, "id", "")),
        "source": payload.get("source") or payload.get("file_name"),
        "content_hash": payload.get("content_hash"),
        "char_start": payload.get("char_start"),
        "char_end": payload.get("char_end"),
        "content": _excerpt(str(payload.get("content", ""))),
        "embedding": embeddings[0] if embeddings else None,
        "embeddings": embeddings,
    }


def _point_sort_key(point: dict[str, Any]) -> tuple[bool, int, str]:
    return (
        point.get("char_start") is None,
        point.get("char_start") or 0,
        point.get("id") or "",
    )


def _excerpt(content: str, max_length: int = 320) -> str:
    compact = " ".join(content.split())
    if len(compact) <= max_length:
        return compact
    return f"{compact[: max_length - 1]}..."
