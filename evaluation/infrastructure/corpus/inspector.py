from __future__ import annotations

from uuid import UUID

from qdrant_client import models
from sqlalchemy.orm import Session, sessionmaker

from rag.database.models import Document
from rag.qdrant import Qdrant


class EvaluationCollectionInspector:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def is_ready(
        self,
        *,
        collection_name: str,
        document_ids: frozenset[UUID],
    ) -> bool:
        content_hashes = self._content_hashes(document_ids)
        if content_hashes is None:
            return False
        qdrant = Qdrant(collection_name=collection_name)
        if not qdrant.collection_exists(collection_name):
            return False
        return all(
            self._contains(qdrant, collection_name, content_hash)
            for content_hash in content_hashes
        )

    def _content_hashes(
        self,
        document_ids: frozenset[UUID],
    ) -> tuple[str, ...] | None:
        with self._session_factory() as session:
            documents = tuple(
                session.get(Document, document_id) for document_id in document_ids
            )
        if any(document is None or not document.content_hash for document in documents):
            return None
        return tuple(
            document.content_hash
            for document in documents
            if document is not None and document.content_hash
        )

    @staticmethod
    def _contains(qdrant: Qdrant, collection_name: str, content_hash: str) -> bool:
        points, _ = qdrant.scroll(
            collection_name=collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="content_hash",
                        match=models.MatchValue(value=content_hash),
                    )
                ]
            ),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        return bool(points)
