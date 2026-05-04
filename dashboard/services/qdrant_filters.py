from __future__ import annotations

from qdrant_client.http import models


def content_hash_filter(content_hash: str) -> models.Filter:
    return models.Filter(
        must=[
            models.FieldCondition(
                key="content_hash",
                match=models.MatchValue(value=content_hash),
            )
        ]
    )
