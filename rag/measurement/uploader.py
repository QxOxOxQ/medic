from __future__ import annotations

import uuid
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, TypeGuard

from qdrant_client.http import models

from rag.embedding.embedder import EmbeddingVector, embed_texts
from rag.qdrant import Qdrant


EmbedFunction = Callable[[str], EmbeddingVector]


@dataclass(frozen=True)
class CollectionInfo:
    collection_name: str
    vector_size: int
    count: int
    distance: models.Distance


class Uploader:
    def __init__(
        self,
        client: Any | None = None,
        embed_fn: EmbedFunction | None = None,
    ) -> None:
        self.client = client or Qdrant()
        self.embed_fn = embed_fn or _embed_one
        self._created_collections: list[CollectionInfo] = []

    def list_collections(self) -> list[dict[str, str]]:
        return [
            {
                "name": "AG News small",
                "slug": "ag_news_small",
                "source_url": "https://huggingface.co/datasets/fancyzhx/ag_news",
            },
            {
                "name": "DBpedia test",
                "slug": "dbpedia_test",
                "source_url": "https://huggingface.co/datasets/fancyzhx/dbpedia_14",
            },
            {
                "name": "TREC small",
                "slug": "trec_small",
                "source_url": "https://huggingface.co/datasets/CogComp/trec",
            },
            {
                "name": "MS MARCO 1536",
                "slug": "msmarco_1536",
                "source_url": "https://huggingface.co/datasets/microsoft/ms_marco",
            },
            {
                "name": "Medical QA small",
                "slug": "medical_qa_small",
                "source_url": "https://huggingface.co/datasets/medalpaca/medical_meadow_medqa",
            },
            {
                "name": "Synthetic clinical",
                "slug": "synthetic_clinical",
                "source_url": "rag/documents",
            },
        ]

    def ensure_collection(
        self,
        collection_name: str,
        vector_size: int,
        is_multivector: bool = False,
    ) -> None:
        if self.client.collection_exists(collection_name):
            return

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=_vector_params(vector_size, is_multivector),
        )

    def upsert_texts(
        self,
        *,
        collection_name: str,
        items: Iterable[dict[str, Any]],
    ) -> int:
        item_list = list(items)
        if not item_list:
            return 0

        vectors = [self.embed_fn(str(item["text"])) for item in item_list]
        self.ensure_collection(
            collection_name,
            vector_size=_embedding_vector_size(vectors[0]),
            is_multivector=_is_multivector_embedding(vectors[0]),
        )
        self.client.upsert(
            collection_name=collection_name,
            points=self._points(item_list, vectors),
        )
        return len(item_list)

    def create_and_upsert(
        self,
        *,
        dataset_name: str,
        items: Iterable[dict[str, Any]],
    ) -> CollectionInfo:
        item_list = list(items)
        if not item_list:
            raise ValueError("items must not be empty")

        vectors = [self.embed_fn(str(item["text"])) for item in item_list]
        vector_size = _embedding_vector_size(vectors[0])
        self.ensure_collection(
            dataset_name,
            vector_size=vector_size,
            is_multivector=_is_multivector_embedding(vectors[0]),
        )
        self.client.upsert(
            collection_name=dataset_name,
            points=self._points(item_list, vectors),
        )
        info = CollectionInfo(
            collection_name=dataset_name,
            vector_size=vector_size,
            count=len(item_list),
            distance=models.Distance.COSINE,
        )
        self._created_collections.append(info)
        return info

    def created_collections_info(self) -> list[CollectionInfo]:
        return list(self._created_collections)

    def _points(
        self,
        items: list[dict[str, Any]],
        vectors: list[EmbeddingVector],
    ) -> list[models.PointStruct]:
        return [
            models.PointStruct(
                id=_item_id(index, item),
                vector=vector,
                payload={key: value for key, value in item.items() if key != "text"},
            )
            for index, (item, vector) in enumerate(zip(items, vectors, strict=True))
        ]


def _embed_one(text: str) -> EmbeddingVector:
    return embed_texts([text])[0]


def _vector_params(vector_size: int, is_multivector: bool) -> models.VectorParams:
    params: dict[str, Any] = {
        "size": vector_size,
        "distance": models.Distance.COSINE,
    }
    if is_multivector:
        params["multivector_config"] = models.MultiVectorConfig(
            comparator=models.MultiVectorComparator.MAX_SIM
        )
    return models.VectorParams(**params)


def _embedding_vector_size(embedding: EmbeddingVector) -> int:
    if _is_multivector_embedding(embedding):
        return len(embedding[0])
    return len(embedding)


def _is_multivector_embedding(
    embedding: EmbeddingVector,
) -> TypeGuard[list[list[float]]]:
    return bool(embedding) and isinstance(embedding[0], Sequence)


def _item_id(index: int, item: dict[str, Any]) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{index}:{item.get('text', '')}"))
