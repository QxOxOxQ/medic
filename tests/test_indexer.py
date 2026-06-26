from __future__ import annotations

from dataclasses import replace
import logging
import os
import uuid
from types import SimpleNamespace

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sqlalchemy.orm import sessionmaker

import rag.indexer as indexer_module
import rag.config as settings_module
import rag.qdrant as qdrant_module
from rag.chunking import TextChunk
from rag.database.migrations import upgrade_database
from rag.database.repositories import DocumentRepository, UserRepository
from rag.database.session import create_database_engine
from rag.embedding.embedder import DENSE_EMBEDDING, EmbeddingModelConfig
from rag.indexer import (
    _content_hash_exists,
    _ensure_collection,
    _index_chunks,
    chunks_from_text,
    index_text,
)


ENV_NAMES = settings_module.SETTINGS["env"]


class PayloadIndexRecordingClient:
    def __init__(self, records: list | None = None) -> None:
        self.records = records or []
        self.events = []

    def collection_exists(self, collection_name: str) -> bool:
        self.events.append(("collection_exists", collection_name))
        return True

    def get_collection(self, collection_name: str):
        self.events.append(("get_collection", collection_name))
        return SimpleNamespace(payload_schema={})

    def create_payload_index(
        self,
        *,
        collection_name: str,
        field_name: str,
        field_schema: models.PayloadSchemaType,
    ) -> None:
        self.events.append(
            ("create_payload_index", collection_name, field_name, field_schema)
        )

    def scroll(self, **kwargs):
        self.events.append(("scroll", kwargs["collection_name"], kwargs))
        return self.records, None


class NewCollectionRecordingClient:
    def __init__(self) -> None:
        self.events = []

    def collection_exists(self, collection_name: str) -> bool:
        self.events.append(("collection_exists", collection_name))
        return False

    def create_collection(self, **kwargs) -> None:
        self.events.append(("create_collection", kwargs))

    def get_collection(self, collection_name: str):
        self.events.append(("get_collection", collection_name))
        return SimpleNamespace(payload_schema={})

    def create_payload_index(
        self,
        *,
        collection_name: str,
        field_name: str,
        field_schema: models.PayloadSchemaType,
    ) -> None:
        self.events.append(
            ("create_payload_index", collection_name, field_name, field_schema)
        )


def test_index_chunks_creates_collection_and_upserts_payload() -> None:
    client = QdrantClient(":memory:")
    chunks = [
        TextChunk(
            content="Clinical note for vector indexing.",
            metadata={
                "source": "tests/clinical.md",
                "file_name": "clinical.md",
                "content_hash": "parsed-hash",
                "char_start": 0,
                "char_end": 34,
                "content": "metadata must not overwrite chunk content",
            },
        )
    ]

    saved_count = _index_chunks(
        chunks=chunks,
        embeddings=[[0.1, 0.2, 0.3]],
        client=client,
        collection_name="test_documents",
    )

    collection = client.get_collection("test_documents")
    vectors_config = collection.config.params.vectors
    records, _ = client.scroll("test_documents", limit=10, with_payload=True)

    assert saved_count == 1
    assert vectors_config.size == 3
    assert vectors_config.distance == models.Distance.COSINE
    assert len(records) == 1
    assert records[0].payload == {
        "content": "Clinical note for vector indexing.",
        "source": "tests/clinical.md",
        "file_name": "clinical.md",
        "content_hash": "parsed-hash",
        "char_start": 0,
        "char_end": 34,
    }


def test_index_chunks_is_idempotent_for_same_source_and_content() -> None:
    client = QdrantClient(":memory:")
    chunks = [
        TextChunk(
            content="Repeatable content.",
            metadata={"source": "tests/repeat.md", "char_start": 0, "char_end": 19},
        )
    ]
    embeddings = [[0.1, 0.2]]

    first_count = _index_chunks(
        chunks=chunks,
        embeddings=embeddings,
        client=client,
        collection_name="test_documents",
    )
    second_count = _index_chunks(
        chunks=chunks,
        embeddings=embeddings,
        client=client,
        collection_name="test_documents",
    )

    assert first_count == 1
    assert second_count == 1
    assert client.count("test_documents").count == 1


def test_index_text_skips_existing_content_hash_before_embedding(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="rag.indexer")
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="test_documents",
        vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE),
    )
    client.upsert(
        collection_name="test_documents",
        points=[
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=[0.1, 0.2],
                payload={
                    "content": "Already indexed.",
                    "content_hash": "duplicate-hash",
                    "file_name": "existing.md",
                },
            )
        ],
    )

    class LocalQdrant:
        def __init__(self) -> None:
            self.client = client
            self.settings = SimpleNamespace(
                qdrant_collection_name="test_documents",
                dense_vector_name=None,
                sparse_vector_name=None,
                sparse_vector_model=None,
                sparse_vector_on_disk=None,
            )

    class FailingEmbedder:
        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            raise AssertionError("duplicate content should not be embedded")

    monkeypatch.setattr(indexer_module, "Qdrant", LocalQdrant)
    monkeypatch.setattr(indexer_module, "Embedder", FailingEmbedder)

    saved_count = index_text(
        "Already indexed.",
        {
            "source": "tests/new-copy.md",
            "file_name": "new-copy.md",
            "content_hash": "duplicate-hash",
        },
    )

    assert saved_count == 0
    assert client.count("test_documents").count == 1
    messages = [
        record.getMessage() for record in caplog.records if record.name == "rag.indexer"
    ]
    assert "Chunking document: source=tests/new-copy.md" in messages
    assert "Chunked document: source=tests/new-copy.md chunks=1" in messages
    assert (
        "Skipping already indexed document: source=tests/new-copy.md collection=test_documents"
        in messages
    )


def test_index_text_logs_embedding_and_upsert(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="rag.indexer")
    client = QdrantClient(":memory:")

    class LocalQdrant:
        def __init__(self) -> None:
            self.client = client
            self.settings = SimpleNamespace(
                qdrant_collection_name="test_documents",
                dense_vector_name=None,
                sparse_vector_name=None,
                sparse_vector_model=None,
                sparse_vector_on_disk=None,
            )

    class FakeEmbedder:
        model_config = EmbeddingModelConfig(
            provider="openrouter",
            model="test-model",
            vector_size=3,
            kind=DENSE_EMBEDDING,
        )

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            return [[0.1, 0.2, 0.3] for _ in texts]

    monkeypatch.setattr(indexer_module, "Qdrant", LocalQdrant)
    monkeypatch.setattr(indexer_module, "Embedder", FakeEmbedder)

    saved_count = index_text(
        "Loggable indexing document.",
        {"source": "tests/logged.md", "content_hash": "logged-hash"},
    )

    assert saved_count == 1
    messages = [
        record.getMessage() for record in caplog.records if record.name == "rag.indexer"
    ]
    assert "Chunking document: source=tests/logged.md" in messages
    assert "Chunked document: source=tests/logged.md chunks=1" in messages
    assert "Loading embedder: source=tests/logged.md" in messages
    assert "Embedding document chunks: source=tests/logged.md chunks=1" in messages
    assert (
        "Upserting document chunks: source=tests/logged.md collection=test_documents chunks=1"
        in messages
    )
    assert (
        "Finished indexing document: source=tests/logged.md collection=test_documents chunks=1"
        in messages
    )


def test_index_text_rejects_embedding_count_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = QdrantClient(":memory:")

    class LocalQdrant:
        def __init__(self) -> None:
            self.client = client
            self.settings = SimpleNamespace(
                qdrant_collection_name="test_documents",
                dense_vector_name=None,
                sparse_vector_name=None,
                sparse_vector_model=None,
                sparse_vector_on_disk=None,
            )

    class EmptyEmbedder:
        model_config = EmbeddingModelConfig(
            provider="openrouter",
            model="test-model",
            vector_size=3,
            kind=DENSE_EMBEDDING,
        )

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            assert texts == ["Mismatched embedding document."]
            return []

    monkeypatch.setattr(indexer_module, "Qdrant", LocalQdrant)
    monkeypatch.setattr(indexer_module, "Embedder", EmptyEmbedder)

    with pytest.raises(ValueError, match="chunks and embeddings"):
        index_text(
            "Mismatched embedding document.",
            {"source": "tests/mismatch.md", "content_hash": "mismatch-hash"},
        )


def test_index_text_stores_chunk_metadata_in_database(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    client = QdrantClient(":memory:")
    database_url = f"sqlite:///{tmp_path / 'indexer.db'}"
    upgrade_database(database_url)
    factory = sessionmaker(
        bind=create_database_engine(database_url),
        expire_on_commit=False,
        future=True,
    )
    with factory() as session:
        user = UserRepository(session).create_user(username="admin", password="secret")
        DocumentRepository(session).upsert_prepared_document(
            owner_user_id=user.id,
            relative_raw_path="tests/logged.pdf",
            original_filename="logged.pdf",
            parsed_markdown_path="tests/logged.md",
            content_hash="db-hash",
            byte_size=1,
            processed_at=None,
        )
        session.commit()

    class LocalQdrant:
        def __init__(self) -> None:
            self.client = client
            self.settings = SimpleNamespace(
                qdrant_collection_name="test_documents",
                dense_vector_name=None,
                sparse_vector_name=None,
                sparse_vector_model=None,
                sparse_vector_on_disk=None,
            )

    class FakeEmbedder:
        model_config = EmbeddingModelConfig(
            provider="openrouter",
            model="test-model",
            vector_size=3,
            kind=DENSE_EMBEDDING,
        )

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            assert texts == ["Database synced chunk."]
            return [[0.1, 0.2, 0.3]]

    monkeypatch.setattr(indexer_module, "Qdrant", LocalQdrant)
    monkeypatch.setattr(indexer_module, "Embedder", FakeEmbedder)

    saved_count = index_text(
        "Database synced chunk.",
        {
            "source": "tests/logged.md",
            "file_name": "logged.md",
            "content_hash": "db-hash",
        },
        database_session_factory=factory,
    )

    assert saved_count == 1
    with factory() as session:
        document = DocumentRepository(session).get_by_parsed_markdown_path(
            "tests/logged.md"
        )
        assert document is not None
        assert document.status == "indexed"
        assert len(document.chunks) == 1
        assert document.chunks[0].content == "Database synced chunk."
        assert document.chunks[0].qdrant_point_id


def test_chunks_from_text_exposes_indexer_chunk_offsets() -> None:
    chunks = chunks_from_text(
        "Clinical note for dashboard chunk preview.",
        {"source": "dashboard.md"},
    )

    assert len(chunks) == 1
    assert chunks[0].content == "Clinical note for dashboard chunk preview."
    assert chunks[0].metadata == {
        "source": "dashboard.md",
        "char_start": 0,
        "char_end": 42,
    }


def test_chunks_from_text_keeps_markdown_table_context() -> None:
    text = (
        "## Wyniki laboratoryjne\n\n"
        "|Badanie|Wynik|Jedn.|MIN|MAX|\n"
        "|---|---:|---|---:|---:|\n"
        "|ALT (ICD-9: 117)|15|U/l|0|41|\n"
        "|CRP (ICD-9: 181)|0,3|mg/l|0,0|5,0|\n\n"
        "Wniosek: wyniki w zakresie referencyjnym."
    )

    chunks = chunks_from_text(text, {"source": "table.md"})

    assert len(chunks) == 1
    assert "ALT (ICD-9: 117)" in chunks[0].content
    assert "CRP (ICD-9: 181)" in chunks[0].content
    assert chunks[0].metadata["char_start"] == 0
    assert chunks[0].metadata["char_end"] == len(text)


def test_chunks_from_text_exposes_exact_offsets_with_overlap() -> None:
    text = " ".join(
        f"Sentence {index} describes clinical finding and treatment response."
        for index in range(80)
    )

    chunks = chunks_from_text(text, {"source": "long.md"})

    assert len(chunks) > 1
    assert any(
        chunks[index].metadata["char_start"]
        < chunks[index - 1].metadata["char_end"]
        for index in range(1, len(chunks))
    )
    assert all(
        text[chunk.metadata["char_start"] : chunk.metadata["char_end"]]
        == chunk.content
        for chunk in chunks
    )


def test_index_text_emits_process_preview_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = QdrantClient(":memory:")
    events = []

    class LocalQdrant:
        def __init__(self) -> None:
            self.client = client
            self.settings = SimpleNamespace(
                qdrant_collection_name="test_documents",
                dense_vector_name="dense",
                sparse_vector_name=None,
                sparse_vector_model=None,
                sparse_vector_on_disk=None,
            )

    class FakeEmbedder:
        model_config = EmbeddingModelConfig(
            provider="openrouter",
            model="test-model",
            vector_size=3,
            kind=DENSE_EMBEDDING,
        )

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            assert texts == ["Progress event document."]
            return [[0.1, 0.2, 0.3]]

    monkeypatch.setattr(indexer_module, "Qdrant", LocalQdrant)
    monkeypatch.setattr(indexer_module, "Embedder", FakeEmbedder)

    saved_count = index_text(
        "Progress event document.",
        {"source": "progress.md", "content_hash": "progress-hash"},
        progress_callback=events.append,
    )

    chunk_event = next(
        event
        for event in events
        if event["step"] == "chunk" and event["status"] == "succeeded"
    )
    embed_event = next(
        event
        for event in events
        if event["step"] == "embed" and event["status"] == "succeeded"
    )
    index_event = next(
        event
        for event in events
        if event["step"] == "index" and event["status"] == "succeeded"
    )

    assert saved_count == 1
    assert chunk_event["result"]["chunks"][0]["content"] == "Progress event document."
    assert chunk_event["result"]["chunks"][0]["char_start"] == 0
    assert embed_event["result"]["embedding"] == {
        "vector_name": "dense",
        "kind": "dense",
        "dimensions": 3,
        "rows": 1,
        "sample": [0.1, 0.2, 0.3],
    }
    assert index_event["result"]["points"][0]["id"]
    assert index_event["result"]["points"][0]["embedding"]["sample"] == [
        0.1,
        0.2,
        0.3,
    ]


def test_content_hash_lookup_creates_payload_index_before_scroll() -> None:
    client = PayloadIndexRecordingClient()

    exists = _content_hash_exists(
        client=client,
        collection_name="test_documents",
        content_hash="duplicate-hash",
    )

    assert exists is False
    assert [event[0] for event in client.events] == [
        "collection_exists",
        "get_collection",
        "create_payload_index",
        "scroll",
    ]
    assert client.events[2] == (
        "create_payload_index",
        "test_documents",
        "content_hash",
        models.PayloadSchemaType.KEYWORD,
    )
    scroll_filter = client.events[3][2]["scroll_filter"]
    assert scroll_filter.must[0].key == "content_hash"
    assert scroll_filter.must[0].match.value == "duplicate-hash"


def test_ensure_collection_creates_content_hash_payload_index() -> None:
    client = NewCollectionRecordingClient()

    _ensure_collection(
        client=client,
        collection_name="test_documents",
        vector_size=3,
        vector_name=None,
        sparse_vector_name=None,
    )

    assert [event[0] for event in client.events] == [
        "collection_exists",
        "create_collection",
        "get_collection",
        "create_payload_index",
    ]
    assert client.events[1][1]["collection_name"] == "test_documents"
    assert client.events[3] == (
        "create_payload_index",
        "test_documents",
        "content_hash",
        models.PayloadSchemaType.KEYWORD,
    )


def test_ensure_collection_uses_vector_settings() -> None:
    client = QdrantClient(":memory:")
    qdrant_config = settings_module.SETTINGS["qdrant"]
    dense_vector_name = qdrant_config["dense_vector"]["name"]
    sparse_vector_name = qdrant_config["sparse_vector"]["name"]
    sparse_vector_on_disk = qdrant_config["sparse_vector"]["on_disk"]

    _ensure_collection(
        client=client,
        collection_name="test_documents",
        vector_size=3,
        vector_name=dense_vector_name,
        sparse_vector_name=sparse_vector_name,
        sparse_vector_on_disk=sparse_vector_on_disk,
    )

    collection = client.get_collection("test_documents")
    vectors_config = collection.config.params.vectors
    sparse_vectors_config = collection.config.params.sparse_vectors

    assert vectors_config[dense_vector_name].size == 3
    assert sparse_vector_name in sparse_vectors_config
    sparse_index = sparse_vectors_config[sparse_vector_name].index
    assert sparse_index.on_disk == sparse_vector_on_disk


def test_ensure_collection_uses_multivector_settings() -> None:
    client = QdrantClient(":memory:")
    qdrant_config = settings_module.SETTINGS["qdrant"]
    dense_vector_name = qdrant_config["dense_vector"]["name"]

    _ensure_collection(
        client=client,
        collection_name="test_documents",
        vector_size=128,
        vector_name=dense_vector_name,
        sparse_vector_name=None,
        is_multivector=True,
    )

    collection = client.get_collection("test_documents")
    vector_params = collection.config.params.vectors[dense_vector_name]

    assert vector_params.size == 128
    assert vector_params.multivector_config.comparator == (
        models.MultiVectorComparator.MAX_SIM
    )


def test_ensure_collection_rejects_existing_vector_size_mismatch() -> None:
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="test_documents",
        vectors_config=models.VectorParams(size=3, distance=models.Distance.COSINE),
    )

    with pytest.raises(ValueError, match="Reindex the collection"):
        _ensure_collection(
            client=client,
            collection_name="test_documents",
            vector_size=4,
            vector_name=None,
            sparse_vector_name=None,
        )


def test_index_chunks_creates_multivector_collection() -> None:
    client = QdrantClient(":memory:")
    chunks = [
        TextChunk(
            content="Clinical note for multivector indexing.",
            metadata={"source": "tests/clinical.md"},
        )
    ]
    config = EmbeddingModelConfig(
        provider="fast_embedding",
        model="colbert-ir/colbertv2.0",
        vector_size=2,
        kind="multivector",
        fast_embedding_provider="late_interaction_text_embedding",
    )

    saved_count = _index_chunks(
        chunks=chunks,
        embeddings=[[[0.1, 0.2], [0.3, 0.4]]],
        embedding_model_config=config,
        client=client,
        collection_name="test_documents",
        vector_name="dense",
        sparse_vector_name=None,
    )

    collection = client.get_collection("test_documents")
    vector_params = collection.config.params.vectors["dense"]

    assert saved_count == 1
    assert vector_params.size == 2
    assert vector_params.multivector_config.comparator == (
        models.MultiVectorComparator.MAX_SIM
    )


def test_index_text_live_writes_to_running_qdrant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    live_test_env = ENV_NAMES["run_qdrant_live_tests"]
    if os.getenv(live_test_env) != "1":
        pytest.skip(f"set {live_test_env}=1 to run live Qdrant tests")

    collection_name = f"test_documents_{uuid.uuid4().hex}"
    qdrant_settings = qdrant_module.get_qdrant_settings()
    monkeypatch.setattr(
        qdrant_module,
        "get_qdrant_settings",
        lambda: replace(qdrant_settings, qdrant_collection_name=collection_name),
    )

    class FakeEmbedder:
        model_config = EmbeddingModelConfig(
            provider="openrouter",
            model="test-model",
            vector_size=3,
            kind=DENSE_EMBEDDING,
        )

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            return [[0.1, 0.2, 0.3] for _ in texts]

    monkeypatch.setattr(
        indexer_module,
        "Embedder",
        FakeEmbedder,
    )
    qdrant = indexer_module.Qdrant()
    client = qdrant.client

    try:
        saved_count = index_text(
            "Live Qdrant indexing smoke test.",
            {"source": "live.md"},
        )
        records, _ = client.scroll(collection_name, limit=10, with_payload=True)

        assert saved_count == 1
        assert client.count(collection_name).count == 1
        assert records[0].payload == {
            "content": "Live Qdrant indexing smoke test.",
            "source": "live.md",
            "char_start": 0,
            "char_end": 32,
        }
    finally:
        if client.collection_exists(collection_name):
            client.delete_collection(collection_name)
