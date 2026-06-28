from __future__ import annotations

import logging
import uuid
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, TypeGuard

from qdrant_client.http import models
from sqlalchemy.orm import Session, sessionmaker

from rag.chunking import TextChunk
from rag.chunking.process_text import MARKDOWN_CHUNK_OVERLAP, ProcessText
from rag.database.repositories import ChunkInput, DocumentRepository
from rag.database.session import get_session_factory, session_scope
from rag.embedding.embedder import Embedder, EmbeddingModelConfig, EmbeddingVector
from rag.progress import ProgressCallback, emit_progress as _emit_progress
from rag.qdrant import Qdrant


_CONTENT_HASH_PAYLOAD_FIELD = "content_hash"
_OWNER_PAYLOAD_FIELD = "owner_user_id"
_PREVIEW_CHUNK_LIMIT = 3
_VECTOR_SAMPLE_SIZE = 12
logger = logging.getLogger(__name__)


def index_text(
    text: str,
    source_metadata: dict[str, Any],
    *,
    progress_callback: ProgressCallback | None = None,
    database_session_factory: sessionmaker[Session] | None = None,
    qdrant: Qdrant | None = None,
    embedder: Embedder | None = None,
) -> int:
    database_session_factory = database_session_factory or _optional_session_factory()
    source = _source_label(source_metadata)
    logger.info("Chunking document: source=%s", source)
    _emit_progress(
        progress_callback,
        step="chunk",
        status="running",
        message=f"Chunking {source}",
        result={"source": source},
    )
    chunks = chunks_from_text(text, source_metadata)
    logger.info("Chunked document: source=%s chunks=%d", source, len(chunks))
    _emit_progress(
        progress_callback,
        step="chunk",
        status="succeeded" if chunks else "skipped",
        message=f"Chunked {source}",
        counters={"chunks": len(chunks)},
        result={"source": source, "chunks": _chunk_previews(chunks)},
    )
    if not chunks:
        logger.info("Skipping empty document: source=%s", source)
        return 0

    qdrant = qdrant or Qdrant()
    content_hash = source_metadata.get("content_hash")
    owner_user_id = _metadata_owner(source_metadata)
    if isinstance(content_hash, str) and _content_hash_exists(
        client=qdrant.client,
        collection_name=qdrant.settings.qdrant_collection_name,
        content_hash=content_hash,
        owner_user_id=owner_user_id,
    ):
        _sync_indexed_chunks(
            session_factory=database_session_factory,
            source_metadata=source_metadata,
            chunks=chunks,
        )
        logger.info(
            "Skipping already indexed document: source=%s collection=%s",
            source,
            qdrant.settings.qdrant_collection_name,
        )
        _emit_progress(
            progress_callback,
            step="index",
            status="skipped",
            message=f"Already indexed {source}",
            result={
                "source": source,
                "content_hash": content_hash,
                "collection": qdrant.settings.qdrant_collection_name,
            },
        )
        return 0

    logger.info("Loading embedder: source=%s", source)
    embedder = embedder or Embedder()
    logger.info(
        "Embedding document chunks: source=%s chunks=%d",
        source,
        len(chunks),
    )
    _emit_progress(
        progress_callback,
        step="embed",
        status="running",
        message=f"Embedding chunks for {source}",
        counters={"chunks": len(chunks)},
        result={"source": source},
    )
    embeddings = embedder.embed_texts([chunk.content for chunk in chunks])
    _validate_embedding_count(chunks, embeddings)
    _emit_progress(
        progress_callback,
        step="embed",
        status="succeeded",
        message=f"Embedded chunks for {source}",
        counters={"chunks": len(embeddings)},
        result={
            "source": source,
            "embedding": vector_preview(
                embeddings[0],
                vector_name=qdrant.settings.dense_vector_name,
            ),
        },
    )
    logger.info(
        "Upserting document chunks: source=%s collection=%s chunks=%d",
        source,
        qdrant.settings.qdrant_collection_name,
        len(chunks),
    )
    _emit_progress(
        progress_callback,
        step="index",
        status="running",
        message=f"Indexing chunks for {source}",
        counters={"chunks": len(chunks)},
        result={
            "source": source,
            "collection": qdrant.settings.qdrant_collection_name,
            "points": _point_previews(
                chunks,
                embeddings,
                vector_name=qdrant.settings.dense_vector_name,
            ),
        },
    )
    saved_chunks = _index_chunks(
        chunks=chunks,
        embeddings=embeddings,
        embedding_model_config=embedder.model_config,
        client=qdrant.client,
        collection_name=qdrant.settings.qdrant_collection_name,
        vector_name=qdrant.settings.dense_vector_name,
        sparse_vector_name=qdrant.settings.sparse_vector_name,
        sparse_vector_model=qdrant.settings.sparse_vector_model,
        sparse_vector_on_disk=qdrant.settings.sparse_vector_on_disk,
    )
    _sync_indexed_chunks(
        session_factory=database_session_factory,
        source_metadata=source_metadata,
        chunks=chunks,
    )
    logger.info(
        "Finished indexing document: source=%s collection=%s chunks=%d",
        source,
        qdrant.settings.qdrant_collection_name,
        saved_chunks,
    )
    _emit_progress(
        progress_callback,
        step="index",
        status="succeeded",
        message=f"Indexed {source}",
        counters={"chunks": saved_chunks},
        result={
            "source": source,
            "collection": qdrant.settings.qdrant_collection_name,
            "points": _point_previews(
                chunks,
                embeddings,
                vector_name=qdrant.settings.dense_vector_name,
            ),
        },
    )
    return saved_chunks


def chunks_from_text(
    text: str,
    source_metadata: dict[str, Any] | None = None,
) -> list[TextChunk]:
    metadata = source_metadata or {}
    chunks: list[TextChunk] = []
    cursor = 0
    for content in ProcessText(document=text).markdown_chunking():
        search_start = max(0, cursor - MARKDOWN_CHUNK_OVERLAP)
        start = text.find(content, search_start)
        if start < 0:
            start = cursor
        end = start + len(content)
        chunks.append(
            TextChunk(
                content=content,
                metadata={
                    **metadata,
                    "char_start": start,
                    "char_end": end,
                },
            )
        )
        cursor = end
    return chunks


def vector_preview(
    vector: Any,
    *,
    vector_name: str | None = None,
    sample_size: int = _VECTOR_SAMPLE_SIZE,
) -> dict[str, Any]:
    selected_name = vector_name
    selected_vector = _plain_value(vector)
    if isinstance(selected_vector, dict) and not _is_sparse_vector(selected_vector):
        selected_name = _selected_vector_name(selected_vector, vector_name)
        selected_vector = selected_vector.get(selected_name) if selected_name else None

    selected_vector = _plain_value(selected_vector)
    kind = _vector_kind(selected_vector)
    preview: dict[str, Any] = {
        "vector_name": selected_name,
        "kind": kind,
        "dimensions": _preview_vector_dimensions(selected_vector, kind),
        "rows": _preview_vector_rows(selected_vector, kind),
        "sample": _numeric_sample(selected_vector, sample_size),
    }
    if kind == "sparse":
        preview["indices_sample"] = _sparse_indices_sample(
            selected_vector,
            sample_size,
        )
    return preview


def vector_previews(
    vector: Any,
    *,
    preferred_name: str | None = None,
    sample_size: int = _VECTOR_SAMPLE_SIZE,
) -> list[dict[str, Any]]:
    vector = _plain_value(vector)
    if not isinstance(vector, dict) or _is_sparse_vector(vector):
        return [
            vector_preview(
                vector,
                vector_name=preferred_name,
                sample_size=sample_size,
            )
        ]

    vector_names = list(vector)
    if preferred_name in vector:
        vector_names.remove(preferred_name)
        vector_names.insert(0, preferred_name)

    return [
        vector_preview(
            vector[name],
            vector_name=name,
            sample_size=sample_size,
        )
        for name in vector_names
    ]


def _source_label(source_metadata: dict[str, Any]) -> str:
    return str(
        source_metadata.get("source")
        or source_metadata.get("file_name")
        or "<unknown>"
    )


def _chunk_previews(chunks: Sequence[TextChunk]) -> list[dict[str, Any]]:
    previews = []
    for index, chunk in enumerate(chunks[:_PREVIEW_CHUNK_LIMIT], start=1):
        previews.append(
            {
                "index": index,
                "char_start": chunk.metadata.get("char_start"),
                "char_end": chunk.metadata.get("char_end"),
                "characters": len(chunk.content),
                "content": _preview_text(chunk.content),
            }
        )
    return previews


def _point_previews(
    chunks: Sequence[TextChunk],
    embeddings: Sequence[EmbeddingVector],
    *,
    vector_name: str | None,
) -> list[dict[str, Any]]:
    previews = []
    preview_pairs = list(zip(chunks, embeddings, strict=True))[:_PREVIEW_CHUNK_LIMIT]
    for index, (chunk, embedding) in enumerate(preview_pairs, start=1):
        previews.append(
            {
                "id": _chunk_id(chunk),
                "chunk_index": index,
                "char_start": chunk.metadata.get("char_start"),
                "char_end": chunk.metadata.get("char_end"),
                "embedding": vector_preview(embedding, vector_name=vector_name),
            }
        )
    return previews


def _validate_embedding_count(
    chunks: Sequence[TextChunk],
    embeddings: Sequence[EmbeddingVector],
) -> None:
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings must have the same length")


def _preview_text(text: str, max_length: int = 240) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_length:
        return compact
    return f"{compact[: max_length - 1]}..."


def _selected_vector_name(
    vectors: dict[str, Any],
    preferred_name: str | None,
) -> str | None:
    if preferred_name is not None and preferred_name in vectors:
        return preferred_name
    return next(iter(vectors), None)


def _vector_kind(vector: Any) -> str:
    if _is_sparse_vector(vector):
        return "sparse"
    if _is_sequence_value(vector) and vector:
        first_value = _plain_value(vector[0])
        if _is_sequence_value(first_value):
            return "multivector"
        return "dense"
    return "unknown"


def _preview_vector_dimensions(vector: Any, kind: str) -> int | None:
    if kind == "sparse":
        values = _sparse_values(vector)
        return len(values) if values is not None else None
    if kind == "dense" and _is_sequence_value(vector):
        return len(vector)
    if kind == "multivector" and _is_sequence_value(vector) and vector:
        first_row = _plain_value(vector[0])
        if _is_sequence_value(first_row):
            return len(first_row)
    return None


def _preview_vector_rows(vector: Any, kind: str) -> int | None:
    if kind == "multivector" and _is_sequence_value(vector):
        return len(vector)
    if kind in {"dense", "sparse"}:
        return 1
    return None


def _numeric_sample(vector: Any, sample_size: int) -> list[float]:
    values: list[float] = []
    for value in _flatten_vector_values(vector):
        try:
            values.append(round(float(value), 6))
        except (TypeError, ValueError):
            continue
        if len(values) >= sample_size:
            break
    return values


def _flatten_vector_values(vector: Any) -> Iterator[Any]:
    vector = _plain_value(vector)
    sparse_values = _sparse_values(vector)
    if sparse_values is not None:
        yield from sparse_values
        return
    if not _is_sequence_value(vector):
        return
    for value in vector:
        value = _plain_value(value)
        if _is_sequence_value(value):
            for nested_value in value:
                yield nested_value
        else:
            yield value


def _plain_value(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _is_sparse_vector(vector: Any) -> bool:
    return _sparse_indices(vector) is not None and _sparse_values(vector) is not None


def _sparse_indices(vector: Any) -> Any:
    vector = _plain_value(vector)
    if isinstance(vector, dict):
        indices = vector.get("indices")
    else:
        indices = getattr(vector, "indices", None)
    indices = _plain_value(indices)
    return indices if _is_sequence_value(indices) else None


def _sparse_values(vector: Any) -> Any:
    vector = _plain_value(vector)
    if isinstance(vector, dict):
        values = vector.get("values")
    else:
        values = getattr(vector, "values", None)
    values = _plain_value(values)
    return values if _is_sequence_value(values) else None


def _sparse_indices_sample(vector: Any, sample_size: int) -> list[int]:
    indices = _sparse_indices(vector)
    if indices is None:
        return []
    sample = []
    for index in indices:
        try:
            sample.append(int(index))
        except (TypeError, ValueError):
            continue
        if len(sample) >= sample_size:
            break
    return sample


def _is_sequence_value(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    )


def _index_chunks(
    *,
    chunks: Sequence[TextChunk],
    embeddings: Sequence[EmbeddingVector],
    client: Any,
    collection_name: str,
    embedding_model_config: EmbeddingModelConfig | None = None,
    vector_name: str | None = None,
    sparse_vector_name: str | None = None,
    sparse_vector_model: str | None = None,
    sparse_vector_on_disk: bool | None = None,
) -> int:
    _validate_embedding_count(chunks, embeddings)

    if not chunks:
        return 0

    vector_size = _embedding_vector_size(embeddings[0])
    is_multivector = _is_multivector_embedding(embeddings[0])
    if embedding_model_config is not None:
        if vector_size != embedding_model_config.vector_size:
            raise ValueError(
                "embedding vector size does not match selected embedding model"
            )
        if is_multivector != embedding_model_config.is_multivector:
            raise ValueError(
                "embedding vector type does not match selected embedding model"
            )

    if sparse_vector_name is not None and sparse_vector_model is None:
        raise ValueError("sparse_vector_model is required when sparse_vector_name is set")

    _ensure_collection(
        client=client,
        collection_name=collection_name,
        vector_size=vector_size,
        is_multivector=is_multivector,
        vector_name=vector_name,
        sparse_vector_name=sparse_vector_name,
        sparse_vector_on_disk=sparse_vector_on_disk,
    )
    if any(chunk.metadata.get(_OWNER_PAYLOAD_FIELD) for chunk in chunks):
        _ensure_owner_payload_index(client=client, collection_name=collection_name)

    points = []
    for chunk, embedding in zip(chunks, embeddings, strict=True):
        vector: Any = _qdrant_vector(embedding)
        if vector_name is not None:
            vector = {vector_name: _qdrant_vector(embedding)}
            if sparse_vector_name is not None:
                assert sparse_vector_model is not None
                vector[sparse_vector_name] = models.Document(
                    text=chunk.content,
                    model=sparse_vector_model,
                )

        points.append(
            models.PointStruct(
                id=_chunk_id(chunk),
                vector=vector,
                payload={**chunk.metadata, "content": chunk.content},
            )
        )

    client.upsert(collection_name=collection_name, points=points)
    return len(points)


def _content_hash_exists(
    *,
    client: Any,
    collection_name: str,
    content_hash: str,
    owner_user_id: str | None = None,
) -> bool:
    if not client.collection_exists(collection_name):
        return False

    _ensure_content_hash_payload_index(
        client=client,
        collection_name=collection_name,
    )
    must: list[models.Condition] = [
        models.FieldCondition(
            key=_CONTENT_HASH_PAYLOAD_FIELD,
            match=models.MatchValue(value=content_hash),
        )
    ]
    if owner_user_id is not None:
        _ensure_owner_payload_index(
            client=client,
            collection_name=collection_name,
        )
        must.append(
            models.FieldCondition(
                key=_OWNER_PAYLOAD_FIELD,
                match=models.MatchValue(value=owner_user_id),
            )
        )
    records, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter=models.Filter(must=must),
        limit=1,
        with_payload=False,
        with_vectors=False,
    )
    return bool(records)


def _metadata_owner(source_metadata: dict[str, Any]) -> str | None:
    owner = source_metadata.get(_OWNER_PAYLOAD_FIELD)
    return owner if isinstance(owner, str) and owner else None


_chunks_from_text = chunks_from_text


def _ensure_collection(
    *,
    client: Any,
    collection_name: str,
    vector_size: int,
    vector_name: str | None,
    sparse_vector_name: str | None,
    sparse_vector_on_disk: bool | None = None,
    is_multivector: bool = False,
) -> None:
    if client.collection_exists(collection_name):
        collection = client.get_collection(collection_name)
        _validate_collection_vectors(
            collection_name=collection_name,
            vectors_config=collection.config.params.vectors,
            vector_size=vector_size,
            is_multivector=is_multivector,
            vector_name=vector_name,
        )
        return

    if vector_name is None:
        vectors_config: Any = _vector_params(vector_size, is_multivector)
    else:
        vectors_config = {vector_name: _vector_params(vector_size, is_multivector)}

    create_kwargs: dict[str, Any] = {
        "collection_name": collection_name,
        "vectors_config": vectors_config,
    }
    if sparse_vector_name is not None:
        if sparse_vector_on_disk is None:
            raise ValueError(
                "sparse_vector_on_disk is required when sparse_vector_name is set"
            )
        create_kwargs["sparse_vectors_config"] = {
            sparse_vector_name: models.SparseVectorParams(
                index=models.SparseIndexParams(on_disk=sparse_vector_on_disk)
            )
        }

    client.create_collection(**create_kwargs)
    _ensure_content_hash_payload_index(
        client=client,
        collection_name=collection_name,
    )


def _ensure_content_hash_payload_index(*, client: Any, collection_name: str) -> None:
    collection = client.get_collection(collection_name)
    if _payload_schema_has_keyword_content_hash_index(collection):
        return

    client.create_payload_index(
        collection_name=collection_name,
        field_name=_CONTENT_HASH_PAYLOAD_FIELD,
        field_schema=models.PayloadSchemaType.KEYWORD,
    )


def _ensure_owner_payload_index(*, client: Any, collection_name: str) -> None:
    collection = client.get_collection(collection_name)
    if _payload_schema_has_keyword_index(collection, _OWNER_PAYLOAD_FIELD):
        return

    client.create_payload_index(
        collection_name=collection_name,
        field_name=_OWNER_PAYLOAD_FIELD,
        field_schema=models.PayloadSchemaType.KEYWORD,
    )


def _payload_schema_has_keyword_index(collection: Any, field_name: str) -> bool:
    payload_schema = getattr(collection, "payload_schema", None) or {}
    field_schema = payload_schema.get(field_name)
    if field_schema is None:
        return False

    data_type = getattr(field_schema, "data_type", field_schema)
    if isinstance(data_type, list):
        return any(_is_keyword_payload_schema_type(item) for item in data_type)
    return _is_keyword_payload_schema_type(data_type)


def _payload_schema_has_keyword_content_hash_index(collection: Any) -> bool:
    payload_schema = getattr(collection, "payload_schema", None) or {}
    field_schema = payload_schema.get(_CONTENT_HASH_PAYLOAD_FIELD)
    if field_schema is None:
        return False

    data_type = getattr(field_schema, "data_type", field_schema)
    if isinstance(data_type, list):
        return any(_is_keyword_payload_schema_type(item) for item in data_type)
    return _is_keyword_payload_schema_type(data_type)


def _is_keyword_payload_schema_type(data_type: Any) -> bool:
    return bool(
        data_type == models.PayloadSchemaType.KEYWORD
        or data_type == models.PayloadSchemaType.KEYWORD.value
    )


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


def _validate_collection_vectors(
    *,
    collection_name: str,
    vectors_config: Any,
    vector_size: int,
    is_multivector: bool,
    vector_name: str | None,
) -> None:
    vector_params = _existing_vector_params(
        collection_name=collection_name,
        vectors_config=vectors_config,
        vector_name=vector_name,
    )
    existing_size = getattr(vector_params, "size", None)
    existing_is_multivector = (
        getattr(vector_params, "multivector_config", None) is not None
    )

    if existing_size != vector_size or existing_is_multivector != is_multivector:
        expected_type = "multivector" if is_multivector else "dense"
        existing_type = "multivector" if existing_is_multivector else "dense"
        raise ValueError(
            f"Collection {collection_name} has incompatible vector configuration: "
            f"{existing_type} size {existing_size}; selected embedding model requires "
            f"{expected_type} size {vector_size}. Reindex the collection or configure "
            "another collection name."
        )

    if is_multivector:
        comparator = vector_params.multivector_config.comparator
        if comparator != models.MultiVectorComparator.MAX_SIM:
            raise ValueError(
                f"Collection {collection_name} has incompatible multivector comparator. "
                "Reindex the collection or configure another collection name."
            )


def _existing_vector_params(
    *,
    collection_name: str,
    vectors_config: Any,
    vector_name: str | None,
) -> Any:
    if vector_name is None:
        if isinstance(vectors_config, dict):
            raise ValueError(
                f"Collection {collection_name} uses named vectors. Reindex the "
                "collection or configure another collection name."
            )
        return vectors_config

    if not isinstance(vectors_config, dict) or vector_name not in vectors_config:
        raise ValueError(
            f"Collection {collection_name} does not contain vector '{vector_name}'. "
            "Reindex the collection or configure another collection name."
        )

    return vectors_config[vector_name]


def _embedding_vector_size(embedding: EmbeddingVector) -> int:
    if not embedding:
        raise ValueError("embeddings must not be empty")

    if _is_multivector_embedding(embedding):
        first_vector = embedding[0]
        if not first_vector:
            raise ValueError("embeddings must not be empty")
        return len(first_vector)

    return len(embedding)


def _is_multivector_embedding(
    embedding: EmbeddingVector,
) -> TypeGuard[list[list[float]]]:
    return bool(embedding) and isinstance(embedding[0], Sequence)


def _qdrant_vector(embedding: EmbeddingVector) -> Any:
    if _is_multivector_embedding(embedding):
        return [list(vector) for vector in embedding]
    return list(embedding)


def _chunk_id(chunk: TextChunk) -> str:
    source = chunk.metadata.get("source") or chunk.metadata.get("file_name") or ""
    raw_id = "|".join(
        [
            str(source),
            str(chunk.metadata.get("char_start", "")),
            str(chunk.metadata.get("char_end", "")),
            chunk.content,
        ]
    )
    return str(uuid.uuid5(uuid.NAMESPACE_URL, raw_id))


def _sync_indexed_chunks(
    *,
    session_factory: sessionmaker[Session] | None,
    source_metadata: dict[str, Any],
    chunks: Sequence[TextChunk],
) -> None:
    source = source_metadata.get("source")
    if session_factory is None or not isinstance(source, str):
        return

    chunk_inputs = [
        ChunkInput(
            chunk_index=index,
            char_start=_metadata_int(chunk.metadata.get("char_start")),
            char_end=_metadata_int(chunk.metadata.get("char_end")),
            content=chunk.content,
            qdrant_point_id=_chunk_id(chunk),
        )
        for index, chunk in enumerate(chunks, start=1)
    ]
    with session_scope(session_factory) as session:
        DocumentRepository(session).upsert_chunks_for_parsed_path(
            parsed_markdown_path=source,
            chunks=chunk_inputs,
        )


def _metadata_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_session_factory() -> sessionmaker[Session] | None:
    try:
        return get_session_factory()
    except ValueError:
        return None


if __name__ == "__main__":
    markdown_file_path = Path("../data/parsed/synthetic_demo.md")
    document_content = markdown_file_path.read_text(encoding="utf-8")
    index_text(
        text=document_content,
        source_metadata={"file_name": "synthetic_demo.md"},
    )
