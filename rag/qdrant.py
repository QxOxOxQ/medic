from __future__ import annotations

from dataclasses import replace
from typing import Any
from uuid import UUID

from qdrant_client import QdrantClient, models
from qdrant_client.http.models import CollectionInfo, ScoredPoint

from rag.config import QdrantSettings, get_qdrant_settings
from rag.embedding.embedder import (
    Embedder,
    EmbeddingModelConfig,
    get_selected_embedding_model,
)


def get_qdrant_client() -> QdrantClient:
    return Qdrant().client


class Qdrant:
    def __init__(
        self,
        *,
        settings: QdrantSettings | None = None,
        collection_name: str | None = None,
        client: QdrantClient | None = None,
    ) -> None:
        resolved_settings = settings or get_qdrant_settings()
        self.settings = (
            replace(resolved_settings, qdrant_collection_name=collection_name)
            if collection_name is not None
            else resolved_settings
        )
        self.embedding_model_config = get_selected_embedding_model()
        self.client = client or QdrantClient(
            url=self.settings.qdrant_url,
            api_key=self.settings.qdrant_api_key,
            timeout=self.settings.client_timeout_seconds,
        )

    @property
    def models(self) -> Any:
        return models

    def query_points(self, **kwargs: Any) -> Any:
        return self.client.query_points(**kwargs)

    def create_collection(self, **kwargs: Any) -> Any:
        return self.client.create_collection(**kwargs)

    def upsert(self, **kwargs: Any) -> Any:
        return self.client.upsert(**kwargs)

    def upload_points(self, **kwargs: Any) -> Any:
        return self.client.upload_points(**kwargs)

    def get_collection(self, collection_name: str) -> CollectionInfo:
        return self.client.get_collection(collection_name)

    def scroll(self, **kwargs: Any) -> Any:
        return self.client.scroll(**kwargs)

    def delete_collection(self, collection_name: str) -> bool:
        return self.client.delete_collection(collection_name)

    def collection_exists(self, collection_name: str | None = None) -> bool:
        if collection_name is None:
            collection_name = self.settings.qdrant_collection_name
        return self.client.collection_exists(collection_name)

    def setup_db(self) -> None:
        collection_name = self.settings.qdrant_collection_name
        if not self.collection_exists(collection_name):
            print(f"Collection {collection_name} does not exist. Creating...")
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    self.settings.dense_vector_name: _vector_params(
                        self.embedding_model_config
                    )
                },
                sparse_vectors_config={
                    self.settings.sparse_vector_name: models.SparseVectorParams(
                        index=models.SparseIndexParams(
                            on_disk=self.settings.sparse_vector_on_disk
                        )
                    )
                },
                quantization_config=models.BinaryQuantization(
                    binary=models.BinaryQuantizationConfig(
                        encoding=getattr(
                            models.BinaryQuantizationEncoding,
                            self.settings.quantization_encoding,
                        ),
                        always_ram=self.settings.quantization_always_ram,
                    )
                ),
            )
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name=_OWNER_PAYLOAD_FIELD,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            print(f"Collection {collection_name} created successfully.")
        else:
            collection = self.client.get_collection(collection_name)
            _validate_collection_vectors(
                collection_name=collection_name,
                vectors_config=collection.config.params.vectors,
                vector_name=self.settings.dense_vector_name,
                model_config=self.embedding_model_config,
            )
            print(f"Collection {collection_name} already exists.")

    def hybrid_search_with_rrf(
        self,
        query_text: str,
        limit: int = 10,
        *,
        owner_user_id: UUID | None = None,
    ) -> list[ScoredPoint]:
        embedder = Embedder(
            provider=self.embedding_model_config.provider,
            model=self.embedding_model_config.model,
        )
        query_dense = embedder.embed_text(query_text)
        query_sparse = models.Document(
            text=query_text,
            model=self.settings.sparse_vector_model,
        )
        owner_filter = _owner_filter(owner_user_id)
        response = self.client.query_points(
            collection_name=self.settings.qdrant_collection_name,
            prefetch=[
                models.Prefetch(
                    query=query_dense,
                    using=self.settings.dense_vector_name,
                    limit=self.settings.prefetch_limit,
                    filter=owner_filter,
                ),
                models.Prefetch(
                    query=query_sparse,
                    using=self.settings.sparse_vector_name,
                    limit=self.settings.prefetch_limit,
                    filter=owner_filter,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            query_filter=owner_filter,
            limit=limit,
        )

        return response.points


_OWNER_PAYLOAD_FIELD = "owner_user_id"


def _owner_filter(owner_user_id: UUID | None) -> models.Filter | None:
    if owner_user_id is None:
        return None
    return models.Filter(
        must=[
            models.FieldCondition(
                key=_OWNER_PAYLOAD_FIELD,
                match=models.MatchValue(value=str(owner_user_id)),
            )
        ]
    )


def _vector_params(model_config: EmbeddingModelConfig) -> models.VectorParams:
    params: dict[str, Any] = {
        "size": model_config.vector_size,
        "distance": models.Distance.COSINE,
    }
    if model_config.is_multivector:
        params["multivector_config"] = models.MultiVectorConfig(
            comparator=models.MultiVectorComparator.MAX_SIM
        )
    return models.VectorParams(**params)


def _validate_collection_vectors(
    *,
    collection_name: str,
    vectors_config: Any,
    vector_name: str,
    model_config: EmbeddingModelConfig,
) -> None:
    if not isinstance(vectors_config, dict) or vector_name not in vectors_config:
        raise ValueError(
            f"Collection {collection_name} does not contain vector '{vector_name}'. "
            "Reindex the collection or configure another collection name."
        )

    vector_params = vectors_config[vector_name]
    existing_size = getattr(vector_params, "size", None)
    existing_is_multivector = (
        getattr(vector_params, "multivector_config", None) is not None
    )
    if (
        existing_size != model_config.vector_size
        or existing_is_multivector != model_config.is_multivector
    ):
        expected_type = "multivector" if model_config.is_multivector else "dense"
        existing_type = "multivector" if existing_is_multivector else "dense"
        raise ValueError(
            f"Collection {collection_name} has incompatible vector configuration: "
            f"{existing_type} size {existing_size}; selected embedding model requires "
            f"{expected_type} size {model_config.vector_size}. Reindex the collection "
            "or configure another collection name."
        )

    if model_config.is_multivector:
        comparator = vector_params.multivector_config.comparator
        if comparator != models.MultiVectorComparator.MAX_SIM:
            raise ValueError(
                f"Collection {collection_name} has incompatible multivector comparator. "
                "Reindex the collection or configure another collection name."
            )


def hybrid_search_with_rrf(
    query_text: str,
    limit: int = 10,
    *,
    owner_user_id: UUID | None = None,
) -> list[ScoredPoint]:
    return Qdrant().hybrid_search_with_rrf(
        query_text, limit, owner_user_id=owner_user_id
    )


def create_collections() -> None:
    Qdrant().setup_db()


if __name__ == "__main__":
    qdrant = Qdrant()
    qdrant.setup_db()
    print(f"Connected to Qdrant. Collection {qdrant.settings.qdrant_collection_name} is ready.")

    # Test hybrid search
    results = qdrant.hybrid_search_with_rrf("knee")
    for i, point in enumerate(results, 1):
        print(f"{i}. (Score: {point.score:.3f})  {point.payload} ")
